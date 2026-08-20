"""Tests for the eval pipeline."""
import json
import random
import os
import tempfile
from pathlib import Path

import pytest

from ms_moe_maker.eval import (
    EvalResult,
    EvalReport,
    _tokenize_simple,
    _exact_match,
    _rouge1,
    _bleu_simple,
    _load_or_split,
    _torch_available,
    detect_dead_experts,
    eval_generation,
    probe_router_discrimination,
    run_eval,
    save_eval_report,
    eval_from_manifest,
)
from ms_moe_maker.evalrecord import PASS, FAIL, UNMEASURABLE


# -- metrics ------------------------------------------------------------------

class TestMetrics:
    """Metric helper functions."""

    def test_tokenize_simple(self):
        assert _tokenize_simple("Hello World") == ["hello", "world"]

    def test_tokenize_empty(self):
        assert _tokenize_simple("") == []
        assert _tokenize_simple("  ") == []

    def test_exact_match_identical(self):
        assert _exact_match("hello world", "hello world") == 1.0

    def test_exact_match_different(self):
        assert _exact_match("hello world", "hello there") == 0.0

    def test_exact_match_case_insensitive(self):
        assert _exact_match("Hello World", "hello world") == 1.0

    def test_rouge1_overlap(self):
        score = _rouge1("the cat sat", "the cat")
        assert 0.0 < score <= 1.0

    def test_rouge1_no_overlap(self):
        assert _rouge1("hello world", "goodbye universe") == 0.0

    def test_rouge1_empty_ref(self):
        assert _rouge1("hello", "") == 0.0

    def test_bleu_simple(self):
        assert 0.0 <= _bleu_simple("hello world", "hello") <= 1.0

    def test_bleu_simple_empty(self):
        assert _bleu_simple("", "") == 0.0


# -- data splitting -----------------------------------------------------------

class TestDataSplitting:
    """Train/held-out split."""

    def test_load_or_split_creates_files(self, tmp_path):
        data = tmp_path / "test.jsonl"
        lines = [f'{{"prompt": "q{i}", "answer": "a{i}"}}' for i in range(20)]
        data.write_text("\n".join(lines) + "\n", encoding="utf-8")

        train, held = _load_or_split(str(data), 0.2)

        assert Path(train).exists()
        assert Path(held).exists()
        held_lines = Path(held).read_text(encoding="utf-8").strip().splitlines()
        train_lines = Path(train).read_text(encoding="utf-8").strip().splitlines()

        # 20% of 20 = 4 held-out
        assert len(held_lines) == 4
        assert len(train_lines) == 16

    def test_load_or_split_zero_holdout_min_one(self, tmp_path):
        """With 0% holdout, at least 1 line is held out."""
        data = tmp_path / "tiny.jsonl"
        data.write_text('{"prompt": "q", "answer": "a"}\n', encoding="utf-8")

        train, held = _load_or_split(str(data), 0.0)

        # min 1 line held out regardless of percentage
        held_lines = Path(held).read_text(encoding="utf-8").strip().splitlines()
        train_lines = Path(train).read_text(encoding="utf-8").strip().splitlines()
        assert len(held_lines) >= 1  # minimum 1 held out
        assert held != str(data)  # held is a new file, not the original


# -- eval result / report ----------------------------------------------------

class TestEvalResult:
    """EvalResult dataclass."""

    def test_default_result(self):
        r = EvalResult(expert_name="python", domain="py")
        assert r.exact_match == 0.0
        assert r.status == "pending"

    def test_done_result(self):
        r = EvalResult(expert_name="py", domain="py", status="done",
                       exact_match=0.9, rouge1=0.85)
        assert r.status == "done"


class TestEvalReport:
    """EvalReport dataclass."""

    def test_empty_report(self):
        r = EvalReport()
        assert r.stages == {}
        assert r.dead_experts == []
        assert r.ok is False

    def test_ok_report(self):
        r = EvalReport(ok=True, message="done")
        assert r.ok is True


# -- dead expert detection ----------------------------------------------------
#
# "Dead" is now a ROUTING fact, read off report.routing, not a generation score.
# These tests build the routing dict directly - the same trick the old suite
# used, except now it builds the shape run_eval actually produces.


def _routing(status=PASS, n_experts=None, top_k=1, **experts):
    """Build a routing dict.

    n_experts and top_k are NOT decoration: "dead" is measured against what
    uniform routing would give (top_k / n_experts), so a fixture that omits
    them is a fixture testing a different threshold than production uses. Two
    tests in this class used to pass because a single-expert fixture made
    uniform=1.0 and dragged the floor up to meet the value under test.
    """
    return {"status": status, "reason": "",
            "n_experts": n_experts if n_experts is not None else len(experts),
            "top_k": top_k,
            "experts": dict(experts)}


class TestDeadExpertDetection:

    def test_healthy_expert_not_dead(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=5, top_k=2,
            python={"enrichment": 2.1, "own_share": 0.31,
                    "marginal_share": 0.30,
                    "top_competitor": "csharp", "top_competitor_share": 0.19,
                    "outranked": False})
        assert detect_dead_experts(report, threshold=1.2) == []

    def test_low_enrichment_is_not_specialised_not_dead(self):
        """THE FALSE ALARM FROM THE FIRST REAL RUN.

        Both 0.5B experts sat at ~0.50 selection share with top-1 of 2 - which
        IS uniform, i.e. used on half of every source's tokens - and enrichment
        1.02x. The report said DEAD EXPERTS: python, csharp, which reads as a
        broken stitch and is not what happened. The stitch was fine; the router
        had not learned to prefer anything.
        """
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.02, "own_share": 0.508,
                    "marginal_share": 0.503, "top_competitor": "csharp",
                    "top_competitor_share": 0.498, "outranked": False},
            csharp={"enrichment": 1.02, "own_share": 0.502,
                    "marginal_share": 0.497, "top_competitor": "python",
                    "top_competitor_share": 0.492, "outranked": False})
        assert detect_dead_experts(report, threshold=1.2) == []
        assert sorted(report.undiscriminating) == ["csharp", "python"]

    def test_an_expert_the_router_never_picks_is_dead(self):
        """The real thing. 0.02 share against 0.40 for uniform: a passenger."""
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=5, top_k=2,
            powershell={"enrichment": 1.11, "own_share": 0.02,
                        "marginal_share": 0.019, "top_competitor": "csharp",
                        "top_competitor_share": 0.278, "outranked": True})
        assert detect_dead_experts(report, threshold=1.2) == ["powershell"]
        assert report.undiscriminating == []

    def test_two_experts_carries_a_power_caveat(self):
        """p=0.25 by construction. The headline cannot be evidence at E=2."""
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 3.0, "own_share": 0.75,
                    "marginal_share": 0.5, "outranked": False},
            csharp={"enrichment": 3.0, "own_share": 0.75,
                    "marginal_share": 0.5, "outranked": False})
        detect_dead_experts(report)
        assert any("cannot reach significance" in c for c in report.caveats)

    def test_a_wide_moe_carries_no_power_caveat(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=5, top_k=2,
            python={"enrichment": 3.0, "own_share": 0.6,
                    "marginal_share": 0.4, "outranked": False})
        detect_dead_experts(report)
        assert not report.caveats

    def test_outranked_on_own_domain_is_dead_even_if_enriched(self):
        """The 0.5B rung's actual weak spot: an expert can clear the enrichment
        bar on the column comparison and still lose its own domain to a
        neighbour. Column-only reads miss this."""
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=5, top_k=2,
            powershell={"enrichment": 1.9, "own_share": 0.192,
                        "marginal_share": 0.19, "top_competitor": "csharp",
                        "top_competitor_share": 0.278, "outranked": True})
        assert detect_dead_experts(report, threshold=1.2) == []
        assert report.undiscriminating == ["powershell"], (
            "outranked on its own ground is a specialisation failure, and it "
            "must still be reported - just not as a dead expert")

    def test_unmeasurable_routing_is_not_a_clean_bill(self):
        """The regression that matters. The old detector defaulted its MoE
        lookup to the expert itself, making the comparison a tautology that
        could never fire - so it always reported no dead experts, on any input,
        and that read like good news. Absence of measurement must never
        present as absence of problems."""
        report = EvalReport(ok=True)
        report.routing = {"status": UNMEASURABLE,
                          "reason": "torch not importable", "experts": {}}
        assert detect_dead_experts(report) == []
        assert report.unmeasured, "unmeasurable routing must be recorded"
        assert "torch not importable" in report.unmeasured[0]

    def test_no_routing_at_all_is_recorded(self):
        report = EvalReport(ok=True)
        assert detect_dead_experts(report) == []
        assert report.unmeasured


# -- the router probe ---------------------------------------------------------

class TestRouterProbe:

    def test_missing_moe_is_unmeasurable_not_fail(self, tmp_path):
        out = probe_router_discrimination(
            moe_dir=str(tmp_path / "nope"),
            held_paths={"python": str(tmp_path / "p.jsonl")},
            expert_order=["python"])
        assert out["status"] == UNMEASURABLE
        assert out["experts"] == {}

    def test_torch_probe_returns_a_reason_when_absent(self):
        ok, reason = _torch_available()
        if not ok:
            assert reason, "an unavailable backend must say why"


# -- generation eval ----------------------------------------------------------

class TestGenerationEval:

    def test_missing_model_is_unmeasurable_not_zero(self, tmp_path):
        """A score of 0.0 and 'we could not load the model' are different
        claims. The proxy scorer this replaced could not tell them apart
        because it never loaded a model at all."""
        data = tmp_path / "held.jsonl"
        data.write_text(json.dumps({"prompt": "hi", "answer": "there"}) + "\n")
        res = eval_generation(model_dir=str(tmp_path / "nope"),
                              test_data_path=str(data),
                              label="python", domain="python")
        assert res.status == UNMEASURABLE
        assert res.exact_match == 0.0
        assert res.note


# -- run_eval -----------------------------------------------------------------

class TestRunEval:

    def test_unknown_mode_refused(self, tmp_path):
        cfg = _cfg(tmp_path)
        report = run_eval(cfg, spec={"mode": "sideways"})
        assert report.ok is False
        assert "sideways" in report.message

    def test_no_data_is_an_error_not_a_pass(self, tmp_path):
        cfg = _cfg(tmp_path)
        report = run_eval(cfg, spec={})
        assert report.ok is False
        assert "no data files" in report.message

    def test_routing_only_records_that_quality_was_not_run(self, tmp_path):
        cfg = _cfg(tmp_path, with_data=True)
        report = run_eval(cfg, spec={"mode": "routing"})
        assert report.stages == {}, "routing mode must not fabricate quality rows"

    def test_quality_only_records_that_dead_check_was_skipped(self, tmp_path):
        cfg = _cfg(tmp_path, with_data=True)
        report = run_eval(cfg, spec={"mode": "quality"})
        assert any("dead-expert check" in u for u in report.unmeasured)

    def test_custom_script_missing_is_refused(self, tmp_path):
        cfg = _cfg(tmp_path, with_data=True)
        report = run_eval(cfg, spec={"script": str(tmp_path / "nope.py")})
        assert report.ok is False
        assert "does not exist" in report.message


class _Cfg:
    def __init__(self, data_root, output_root, expert_names, base=""):
        self.data_root = data_root
        self.output_root = output_root
        self.expert_names = expert_names
        self.base = base


def _cfg(tmp_path, with_data=False):
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    names = []
    if with_data:
        for name in ("python", "csharp"):
            (data / f"{name}.jsonl").write_text(
                "\n".join(json.dumps({"prompt": f"q{i}", "answer": f"a{i}",
                                      "text": f"body {i}"})
                          for i in range(10)) + "\n", encoding="utf-8")
            names.append(name)
    return _Cfg(str(data), str(out), names)


class TestReportSerialization:
    """save_eval_report and eval_from_manifest."""

    def test_save_and_load_report(self, tmp_path):
        report = EvalReport(
            ok=True,
            message="all good",
            dead_experts=["rust"],
        )
        report.stages["python"] = EvalResult(
            expert_name="python", domain="py",
            status="done", exact_match=0.9)

        out_path = tmp_path / "eval_report.json"
        save_eval_report(report, out_path)

        loaded = eval_from_manifest(tmp_path)

        assert loaded.ok
        assert loaded.message == "all good"
        assert "rust" in loaded.dead_experts
        assert loaded.stages["python"].exact_match == 0.9

    def test_load_missing_report(self, tmp_path):
        report = eval_from_manifest(tmp_path)
        assert not report.ok
        assert "no eval report" in report.message.lower()


class TestDegenerateRoutingIsUnmeasurable:
    """top-k == expert count is not a finding, it is an absence of one.

    The first real run reported "INPUT-BLIND - the router ignores its input
    entirely" for a router that had never been asked to choose: the recipe set
    experts_per_tok=2 on a 2-expert MoE, so topk() returned BOTH experts on
    every token. Shares 0.5/0.5, enrichment 1.00x, JS exactly 0.000 - all
    arithmetic, none of it measurement, and numerically identical to the
    genuine failure it was misreported as.

    A false diagnosis backed by decisive-looking numbers is the worst output
    this tool can produce; it is the same sin as the fabricated scores this
    module was built to replace.
    """

    def test_topk_equal_to_expert_count_selects_everything(self):
        """The arithmetic, stated plainly so the reason survives."""
        def topk(vals, k):
            return [i for i, _ in sorted(enumerate(vals), key=lambda p: -p[1])[:k]]
        E = K = 2
        counts = [0] * E
        for seed in range(200):
            random.seed(seed)
            logits = [random.gauss(0, 10) for _ in range(E)]
            for e in topk(logits, K):
                counts[e] += 1
        total = sum(counts)
        assert [c / total for c in counts] == [0.5, 0.5], (
            "with K == E every expert is selected every time, whatever the "
            "logits - so the shares cannot carry information")

    def test_topk_below_expert_count_can_differ(self):
        def topk(vals, k):
            return [i for i, _ in sorted(enumerate(vals), key=lambda p: -p[1])[:k]]
        counts = [0, 0]
        for seed in range(200):
            random.seed(seed)
            logits = [random.gauss(0, 10) for _ in range(2)]
            for e in topk(logits, 1):
                counts[e] += 1
        assert counts[0] != counts[1], "top-1 of 2 must be able to prefer one"

    def test_validate_warns_before_the_build(self):
        from ms_moe_maker.recipe import parse, validate
        rec, _ = parse({
            "schema_version": 1, "name": "t", "moe": {"experts_per_tok": 2},
            "experts": [
                {"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]})
        errs, warns = validate(rec)
        assert errs == [], "a dense ensemble is legal, just not measurable"
        assert any("experts_per_tok" in w and "cannot discriminate" in w
                   for w in warns), warns

    def test_the_shipped_flow_recipe_is_measurable(self):
        import pathlib
        from ms_moe_maker.recipe import load
        p = (pathlib.Path(__file__).resolve().parent.parent
             / "recipe.flow-0.5B.yaml")
        if not p.is_file():
            pytest.skip("flow recipe not in this checkout")
        rec, _ = load(str(p))
        assert rec.moe.experts_per_tok < len(rec.experts), (
            "the recipe we hand people must produce a measurable router")


class TestQualityOnRawTextCorpora:
    """A stack/gh corpus is {"text": ...} and has no answer key.

    Every quality row came back "no sample had both a prompt and a reference",
    which is true and useless: the corpus most people will actually build an
    expert from could not be scored at all. Raw text has an answer key hiding
    in it - hold back the second half of a document.
    """

    def test_qa_shaped_rows_are_used_directly(self):
        from ms_moe_maker.eval import _prompt_and_reference
        for keys in (("prompt", "answer"), ("input", "output"),
                     ("question", "reference")):
            row = {keys[0]: "Q", keys[1]: "A"}
            assert _prompt_and_reference(row) == ("Q", "A"), keys

    def test_text_rows_become_a_completion_task(self):
        from ms_moe_maker.eval import _prompt_and_reference
        text = "".join(f"line{i}\n" for i in range(10))
        prompt, ref = _prompt_and_reference({"text": text})
        assert prompt and ref
        assert prompt + ref == text, "the split must lose nothing"
        assert prompt.endswith("\n"), "split on a line boundary, not mid-token"

    def test_content_field_works_too(self):
        from ms_moe_maker.eval import _prompt_and_reference
        text = "".join(f"line{i}\n" for i in range(10))
        assert all(_prompt_and_reference({"content": text}))

    def test_a_document_too_short_to_split_is_skipped(self):
        """Two lines cannot be halved into a prompt and a meaningful reference."""
        from ms_moe_maker.eval import _prompt_and_reference
        assert _prompt_and_reference({"text": "a\nb\n"}) == ("", "")

    def test_an_empty_row_is_skipped(self):
        from ms_moe_maker.eval import _prompt_and_reference
        assert _prompt_and_reference({}) == ("", "")


class TestEvalMemoryDiscipline:
    """The eval path OOM-killed a 121 GB Spark. Three causes, all here.

    The load trace told the story: 387 weights, 290, 387 - the MoE was being
    loaded once per expert on top of once for routing, and nothing was ever
    actually freed between them.
    """

    def test_cleanup_takes_no_model_argument(self):
        """`model_cleanup(model)` did `del model` on its own PARAMETER.

        That drops one name inside the function while the caller still holds
        the object, so the gc.collect() right after ran with the model fully
        reachable and freed nothing. Same wrong-scope `del` as the `del torch`
        in builder.py - a function cannot drop a reference it does not own.
        """
        import inspect
        from ms_moe_maker import eval as ev
        assert hasattr(ev, "release_memory")
        assert not hasattr(ev, "model_cleanup"), (
            "model_cleanup could not free anything; taking the object is the "
            "bug, not an implementation detail")
        assert list(inspect.signature(ev.release_memory).parameters) == []

    def test_a_del_on_a_parameter_frees_nothing(self):
        """The mechanism, pinned down so nobody reintroduces it."""
        freed = []

        class Big:
            def __del__(self):
                freed.append(True)

        def bad_cleanup(obj):
            import gc
            del obj
            gc.collect()

        def caller():
            model = Big()
            bad_cleanup(model)
            return not freed          # still alive after "cleanup"?

        assert caller(), "the object must still be alive - that is the bug"

    def test_the_loader_does_not_use_device_map_auto_or_float32(self):
        """Both are traps on unified memory, and both were in this file.

        Checks CODE, not prose: the docstrings deliberately name both traps to
        explain them, and a grep over the raw source flags its own
        explanation. Stripping docstrings and comments first is the difference
        between a guard and a gag order on the comments.
        """
        import ast
        import inspect
        from ms_moe_maker.eval import _load_model, probe_router_discrimination

        def code_only(fn):
            tree = ast.parse(inspect.getsource(fn).lstrip())
            for node in ast.walk(tree):
                # drop docstrings
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Module)):
                    body = node.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        node.body = body[1:]
            return ast.unparse(tree)

        for fn in (_load_model, probe_router_discrimination):
            src = code_only(fn)
            assert "device_map" not in src, f"{fn.__name__} uses device_map"
            assert "float32" not in src, f"{fn.__name__} loads in float32"

    def test_the_moe_is_loaded_once_not_once_per_expert(self):
        import inspect
        from ms_moe_maker.eval import run_eval
        src = inspect.getsource(run_eval)
        body = src[src.index("if do_quality:"):]
        # the MoE load happens outside the per-expert loop
        assert "_load_model(moe_dir)" in body
        assert body.count("_load_model(moe_dir)") == 1, (
            "loading the MoE per expert is what filled 121 GB")

    def test_eval_generation_can_borrow_a_model(self):
        import inspect
        from ms_moe_maker.eval import eval_generation
        assert "loaded" in inspect.signature(eval_generation).parameters

    def test_a_borrowed_model_is_not_freed_by_the_borrower(self):
        """Whoever loads it frees it. Releasing a shared model mid-loop would
        pull it out from under the next call."""
        import inspect
        from ms_moe_maker.eval import eval_generation
        src = inspect.getsource(eval_generation)
        assert "owns_model" in src
        assert "if owns_model:" in src
