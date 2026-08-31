"""Tests for the eval pipeline."""
import json
import random
import os
import tempfile
from pathlib import Path

import pytest

from ms_moe_maker.eval.harness import (
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
from ms_moe_maker.eval.record import PASS, FAIL, UNMEASURABLE


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

    def test_a_reasoning_experts_corpus_is_found(self, tmp_path):
        """THE EXPERT THAT VANISHED.

        `reasoning: true` writes `<name>_reasoning.jsonl`, not `<name>.jsonl`.
        eval re-derived the expert name from the filename and stripped only
        `_code`, so the expert resolved as "math_reasoning", matched nothing in
        expert_names, and disappeared from every table with no note anywhere.
        """
        data = tmp_path / "data"; out = tmp_path / "out"
        data.mkdir(); out.mkdir()
        rows = "\n".join(json.dumps({"prompt": f"q{i}", "answer": f"a{i}",
                                     "text": f"body {i}"}) for i in range(10))
        (data / "python_code.jsonl").write_text(rows + "\n", encoding="utf-8")
        (data / "math_reasoning.jsonl").write_text(rows + "\n", encoding="utf-8")
        cfg = _Cfg(str(data), str(out), ["python", "math"])
        report = run_eval(cfg, spec={"mode": "quality"})
        assert "math" in report.stages, (
            f"reasoning expert dropped; stages={list(report.stages)}")
        assert not any("corpus/math" in u for u in report.unmeasured)

    def test_a_named_expert_with_no_corpus_is_stated_not_skipped(self, tmp_path):
        """Silence is signal. A narrowed test has to say it narrowed."""
        cfg = _cfg(tmp_path, with_data=True)
        cfg.expert_names = list(cfg.expert_names) + ["ghost"]
        report = run_eval(cfg, spec={"mode": "quality"})
        assert any("corpus/ghost" in u for u in report.unmeasured), report.unmeasured

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
        # run_eval reads these off the config; PipelineConfig declares them, so
        # the stub has to as well or the test asserts a shape nothing ships.
        self.reasoning_type = ""
        self.reasoning_experts = []


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
        from ms_moe_maker.config.recipe import parse, validate
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
        from ms_moe_maker.config.recipe import load
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
        from ms_moe_maker.eval.harness import _prompt_and_reference
        for keys in (("prompt", "answer"), ("input", "output"),
                     ("question", "reference")):
            row = {keys[0]: "Q", keys[1]: "A"}
            assert _prompt_and_reference(row) == ("Q", "A"), keys

    def test_text_rows_become_a_completion_task(self):
        from ms_moe_maker.eval.harness import _prompt_and_reference
        text = "".join(f"line{i}\n" for i in range(10))
        prompt, ref = _prompt_and_reference({"text": text})
        assert prompt and ref
        assert prompt + ref == text, "the split must lose nothing"
        assert prompt.endswith("\n"), "split on a line boundary, not mid-token"

    def test_content_field_works_too(self):
        from ms_moe_maker.eval.harness import _prompt_and_reference
        text = "".join(f"line{i}\n" for i in range(10))
        assert all(_prompt_and_reference({"content": text}))

    def test_a_document_too_short_to_split_is_skipped(self):
        """Two lines cannot be halved into a prompt and a meaningful reference."""
        from ms_moe_maker.eval.harness import _prompt_and_reference
        assert _prompt_and_reference({"text": "a\nb\n"}) == ("", "")

    def test_an_empty_row_is_skipped(self):
        from ms_moe_maker.eval.harness import _prompt_and_reference
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
        from ms_moe_maker.eval import harness as ev
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
        from ms_moe_maker.eval.harness import _load_model, probe_router_discrimination

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
        from ms_moe_maker.eval.harness import run_eval
        src = inspect.getsource(run_eval)
        body = src[src.index("if do_quality:"):]
        # the MoE load happens outside the per-expert loop
        assert "_load_model(moe_dir)" in body
        assert body.count("_load_model(moe_dir)") == 1, (
            "loading the MoE per expert is what filled 121 GB")

    def test_eval_generation_can_borrow_a_model(self):
        import inspect
        from ms_moe_maker.eval.harness import eval_generation
        assert "loaded" in inspect.signature(eval_generation).parameters

    def test_a_borrowed_model_is_not_freed_by_the_borrower(self):
        """Whoever loads it frees it. Releasing a shared model mid-loop would
        pull it out from under the next call."""
        import inspect
        from ms_moe_maker.eval.harness import eval_generation
        src = inspect.getsource(eval_generation)
        assert "owns_model" in src
        assert "if owns_model:" in src


class TestCollapsedRouter:
    """A router that puts everything through one expert, reported as clean.

    These are the real numbers from the first build after the top-1 gradient
    fix: the gate finally MOVED (uniform 0.50/0.50 -> 0.96/0.04) and then
    collapsed onto one expert instead of specialising. The eval printed

        csharp  own 0.045  share 0.043
        python  own 0.957  share 0.955
        INPUT-BLIND - the router ignores its input entirely
        Eval complete. 0 dead expert(s).

    Four percent of uniform, called alive, because the input-blind guard
    returned early and discarded the share column along with the enrichment
    column. Only one of those two depends on the input.
    """

    def _collapsed(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.00, "own_share": 0.957,
                    "marginal_share": 0.955, "top_competitor": "csharp",
                    "top_competitor_share": 0.955, "outranked": False},
            csharp={"enrichment": 1.04, "own_share": 0.045,
                    "marginal_share": 0.043, "top_competitor": "python",
                    "top_competitor_share": 0.043, "outranked": False})
        report.routing["mean_js_bits"] = 0.0007
        return report

    def test_the_starved_expert_is_dead_even_when_routing_is_input_blind(self):
        report = self._collapsed()
        assert detect_dead_experts(report) == ["csharp"], (
            "4.3% of tokens against 50% for uniform is a passenger, and "
            "whether the router reads its input has no bearing on that")

    def test_blindness_still_suppresses_the_specialisation_verdict(self):
        report = self._collapsed()
        detect_dead_experts(report)
        assert report.undiscriminating == [], (
            "enrichment is a per-source ratio; with input-blind routing both "
            "terms are the same number and the ratio is noise")
        assert any("input-blind" in u for u in report.unmeasured)

    def test_collapse_is_named_as_collapse_not_just_as_a_dead_expert(self):
        """'one dead expert' points at the stitch. 'the router put 96% of every
        source through one expert' points at the aux-loss coefficient."""
        report = self._collapsed()
        detect_dead_experts(report)
        assert any("collapsed onto python" in c for c in report.caveats), \
            report.caveats
        assert any("aux_loss_coef" in c for c in report.caveats)

    def test_a_healthy_blind_router_reports_nothing_dead(self):
        """Input-blind but BALANCED is the earlier failure, and it has no dead
        expert in it - both are used exactly as often as uniform predicts."""
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.00, "own_share": 0.499,
                    "marginal_share": 0.498, "outranked": False},
            csharp={"enrichment": 1.00, "own_share": 0.502,
                    "marginal_share": 0.501, "outranked": False})
        report.routing["mean_js_bits"] = 0.0007
        assert detect_dead_experts(report) == []
        assert not any("collapsed" in c for c in report.caveats)


class TestGateConfidence:
    """Confidence answers a question share and enrichment cannot.

    With norm_topk_prob=false the selected weight MULTIPLIES the frozen
    expert's output, so the gate probability is a free scalar gain. At init
    p=1/E, which scales down an FFN contribution the base model expects at
    full strength, and the cheapest repair is p -> 1 on every token regardless
    of input. The result is a collapsed, input-blind router that got there for
    reasons unrelated to routing - indistinguishable from any other collapse
    by share alone, obvious the moment confidence is next to JS.
    """

    def test_saturated_confidence_is_surfaced(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.0, "own_share": 0.958,
                    "marginal_share": 0.957, "outranked": False},
            csharp={"enrichment": 1.0, "own_share": 0.043,
                    "marginal_share": 0.042, "outranked": False})
        report.routing.update({"mean_js_bits": 0.0001, "moe_layers": 24,
                               "mean_gate_confidence": 0.998,
                               "uniform_confidence": 0.5})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "mean gate confidence" in out
        assert "SATURATED" in out

    def test_healthy_confidence_is_reported_without_alarm(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.4, "own_share": 0.6,
                    "marginal_share": 0.5, "outranked": False},
            csharp={"enrichment": 1.4, "own_share": 0.6,
                    "marginal_share": 0.5, "outranked": False})
        report.routing.update({"mean_js_bits": 0.4, "moe_layers": 24,
                               "mean_gate_confidence": 0.62,
                               "uniform_confidence": 0.5})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "0.620" in out
        assert "SATURATED" not in out

    def test_a_probe_without_confidence_prints_nothing(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.4, "own_share": 0.6,
                    "marginal_share": 0.5, "outranked": False})
        report.routing.update({"mean_js_bits": 0.4, "moe_layers": 24})
        _print_eval_report(report)
        assert "gate confidence" not in capsys.readouterr().out


def test_a_floating_point_tie_is_not_an_outranking():
    """Both experts printed OUTRANKED ON ITS OWN GROUND with own and rival
    equal to four decimals. A tie is not a finding."""
    report = EvalReport(ok=True)
    report.routing = _routing(
        n_experts=2, top_k=1,
        python={"enrichment": 1.00, "own_share": 0.4999, "marginal_share": 0.4999,
                "top_competitor": "csharp", "top_competitor_share": 0.5001,
                "outranked": False},
        csharp={"enrichment": 1.00, "own_share": 0.5001, "marginal_share": 0.5001,
                "top_competitor": "python", "top_competitor_share": 0.4999,
                "outranked": False})
    report.routing["mean_js_bits"] = 0.4
    assert detect_dead_experts(report) == []


class TestStarvedEnrichment:
    """An abandoned expert's enrichment is one noise over another.

    A collapsed run printed `python own 0.001 others 0.000 enrich 2.15x` - the
    largest enrichment in the table, computed from a handful of selections,
    on the expert the router had abandoned. It also pulled the reported MEAN
    from ~1.0 to 1.57, so the single number a reader is most likely to quote
    was set by the expert with the least evidence behind it.
    """

    def test_a_starved_expert_prints_noise_not_a_ratio(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            csharp={"enrichment": 1.00, "own_share": 1.000,
                    "others_share": 0.999, "marginal_share": 0.999,
                    "enrichment_reliable": True, "own_is_column_max": True},
            python={"enrichment": 2.15, "own_share": 0.001,
                    "others_share": 0.000, "marginal_share": 0.001,
                    "enrichment_reliable": False, "own_is_column_max": True})
        report.routing.update({"mean_js_bits": 0.0001, "moe_layers": 24,
                               "named_experts": 2, "own_is_max_count": 2,
                               "mean_enrichment": 1.00, "p_value": 0.25})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "2.15" not in out, "a starved expert must not advertise a ratio"
        assert "noise" in out
        assert "STARVED" in out

    def test_a_healthy_expert_still_prints_its_enrichment(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=3, top_k=2,
            a={"enrichment": 2.10, "own_share": 0.55, "others_share": 0.26,
               "marginal_share": 0.35, "enrichment_reliable": True,
               "own_is_column_max": True})
        report.routing.update({"mean_js_bits": 0.4, "moe_layers": 24,
                               "named_experts": 1, "own_is_max_count": 1,
                               "mean_enrichment": 2.10, "p_value": 0.037})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "2.10x" in out
        assert "STARVED" not in out


class TestTheOwnColumnPValue:
    """A p-value for an event that did not happen is worse than no p-value.

    The probe hardcoded 1/n^n - the probability that ALL n experts top their
    own column - and printed it whatever the count was. On a table where one
    expert of five won its column it still announced p=0.00032, a decisive
    looking significance figure for something that never occurred, directly
    under a failing table. And `hits` was counted over every expert with a
    column while `n` counted only the readable ones, so three experts with one
    starved-but-topping-its-own-row printed "column maximum for 3/2".
    """

    def test_all_n_of_n_is_unchanged(self):
        from ms_moe_maker.eval.harness import _own_column_p
        # the proven 0.5B headline: five experts, five columns, p=0.00032
        assert _own_column_p(5, 5) == pytest.approx(1 / 5 ** 5)
        assert _own_column_p(2, 2) == pytest.approx(0.25)

    def test_a_partial_result_gets_a_bigger_p_not_the_n_of_n_one(self):
        from ms_moe_maker.eval.harness import _own_column_p
        assert _own_column_p(1, 5) > _own_column_p(3, 5) > _own_column_p(5, 5)
        assert _own_column_p(1, 5) > 0.05, (
            "one of five winning its column is not a significant result")
        assert _own_column_p(0, 5) == pytest.approx(1.0)
        assert _own_column_p(1, 0) is None

    def test_hits_are_counted_over_the_same_set_as_n(self):
        """A starved expert has no readable enrichment and no vote here."""
        import ast
        import inspect
        from ms_moe_maker.eval.harness import probe_router_discrimination
        src = inspect.getsource(probe_router_discrimination)
        tree = ast.parse(src.lstrip())
        for node in ast.walk(tree):
            if (isinstance(node, ast.If)
                    and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.op, ast.Not)
                    and getattr(node.test.operand, "id", "") == "starved"):
                body = ast.unparse(node)
                assert "hits" in body, (
                    "hits must be counted inside `if not starved`, over the "
                    "same set n and the p-value are computed over")
                break
        else:
            pytest.fail("the starved branch went missing")

    def test_the_printed_sentence_and_the_number_agree(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        from ms_moe_maker.eval.harness import _own_column_p
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=5, top_k=2,
            a={"enrichment": 1.4, "own_share": 0.30, "others_share": 0.21,
               "marginal_share": 0.20, "enrichment_reliable": True,
               "own_is_column_max": True})
        report.routing.update({
            "mean_js_bits": 0.3, "moe_layers": 24, "named_experts": 5,
            "own_is_max_count": 1, "mean_enrichment": 1.10,
            "p_value": _own_column_p(1, 5),
            "p_value_event": "at least 1 of 5 by chance"})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "column maximum for 1/5" in out
        assert "for 5/5 by chance" not in out, (
            "a significance figure for all-five-of-five, printed under a "
            "table where one expert won its column")
        assert "0.00032" not in out
        assert "at least 1 of 5" in out


class TestAnExpertThatLostItsHeldOutSet:
    """Narrowing the test is allowed. Narrowing it quietly is not.

    THE REAL RUN. router_mix_total went 1200 -> 4000 and consumed every usable
    python held-out row. The probe carried on with what was left:

        csharp     0.377  1.21x  <- own is top
        markdown   0.375  1.15x  <- own is top
          own-expert is the column maximum for 2/2
          p = 0.25000
        [ok] No dead experts. Every check measured.

    Three experts, two rows, and a p-value that had silently regressed from
    0.037 to 0.250 because the width of the test changed. Python was reported
    as neither dead nor undiscriminating - it passed a check that never ran,
    under a line claiming every check measured.
    """

    def _narrowed(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=3, top_k=2,
            csharp={"enrichment": 1.21, "own_share": 0.377,
                    "marginal_share": 0.330, "enrichment_reliable": True,
                    "own_is_column_max": True},
            markdown={"enrichment": 1.15, "own_share": 0.375,
                      "marginal_share": 0.335, "enrichment_reliable": True,
                      "own_is_column_max": True})
        report.routing.update({
            "mean_js_bits": 0.008, "moe_layers": 24, "named_experts": 2,
            "own_is_max_count": 2, "mean_enrichment": 1.18, "p_value": 0.25,
            "excluded": ["python"],
            "excluded_reason": "every held-out row was consumed by the "
                               "router's training mix"})
        return report

    def test_the_missing_expert_is_recorded_as_unmeasured(self):
        report = self._narrowed()
        detect_dead_experts(report)
        assert any("routing/python" in u for u in report.unmeasured), (
            "an expert with no held-out rows must be reported as unmeasured, "
            "not omitted")

    def test_the_narrowed_width_is_stated_next_to_the_p_value(self):
        report = self._narrowed()
        detect_dead_experts(report)
        assert any("2 of 3" in c and "p-value" in c for c in report.caveats), \
            report.caveats

    def test_the_missing_expert_is_not_called_dead_or_undiscriminating(self):
        """It was not measured. Neither verdict is available."""
        report = self._narrowed()
        dead = detect_dead_experts(report)
        assert "python" not in dead
        assert "python" not in report.undiscriminating

    def test_a_complete_table_carries_no_width_caveat(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            a={"enrichment": 1.4, "own_share": 0.6, "marginal_share": 0.5,
               "enrichment_reliable": True},
            b={"enrichment": 1.4, "own_share": 0.6, "marginal_share": 0.5,
               "enrichment_reliable": True})
        report.routing.update({"mean_js_bits": 0.4, "excluded": []})
        detect_dead_experts(report)
        assert not any("not scored" in c for c in report.caveats)


class TestTheScoredSampleCount:
    """A mean over 3 rows and a mean over 20 used to print identically.

    The count lived only in `result.note` - which _print_eval_report never
    printed, and which detect_dead_experts overwrote with its routing verdict,
    so the JSON lost it too. A `stack` corpus of mostly short files yields 3
    usable rows out of num_samples: 20, because _prompt_and_reference needs
    4+ lines to build a completion task.
    """

    def test_the_count_is_a_field_and_it_is_printed(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.stages["stack"] = EvalResult(
            expert_name="stack", domain="stack", status="done",
            exact_match=0.0, rouge1=0.4, bleu=0.3,
            scored_samples=3, attempted_samples=20)
        report.stages["python"] = EvalResult(
            expert_name="python", domain="python", status="done",
            exact_match=0.0, rouge1=0.4, bleu=0.3,
            scored_samples=20, attempted_samples=20)
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "3/20" in out and "20/20" in out
        assert "thin" in out, "a 3-of-20 average must not read like a 20-of-20"

    def test_the_routing_verdict_does_not_erase_the_count(self):
        report = EvalReport(ok=True)
        report.stages["python"] = EvalResult(
            expert_name="python", domain="python", status="done",
            scored_samples=3, attempted_samples=20,
            note="3 of 20 sampled rows scored (completion: second half)")
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 1.01, "own_share": 0.50,
                    "marginal_share": 0.50, "enrichment_reliable": True},
            csharp={"enrichment": 1.01, "own_share": 0.50,
                    "marginal_share": 0.50, "enrichment_reliable": True})
        report.routing.update({"mean_js_bits": 0.4})
        detect_dead_experts(report, threshold=1.2)
        note = report.stages["python"].note
        assert "3 of 20" in note, (
            "the routing verdict deleted a quality fact it did not write")
        assert "not specialised" in note, "and it must still say its own piece"

    def test_the_count_survives_a_round_trip_through_json(self, tmp_path):
        from ms_moe_maker.eval.harness import save_eval_report, eval_from_manifest
        report = EvalReport(ok=True, message="")
        report.stages["stack"] = EvalResult(
            expert_name="stack", domain="stack", status="done",
            scored_samples=3, attempted_samples=20)
        save_eval_report(report, tmp_path / "eval_report.json")
        back = eval_from_manifest(tmp_path)
        assert back.stages["stack"].scored_samples == 3
        assert back.stages["stack"].attempted_samples == 20


class TestTheProbeSpeaksTheRoutersLanguage:
    """The probe asked `Write csharp:` in a chat template. Training never did.

    router.format_fn uses _make_code_prompt with the DISPLAY name and one of
    six templates for code sources, and returns ex["text"] RAW - no chat
    template at all - for the tools expert and the reasoning experts. Probing
    everything through one made-up template measured formatting, and measured
    it WORSE for the tools expert than for the code ones.
    """

    def test_the_probe_uses_the_trainers_own_prompt_builder(self):
        import inspect
        from ms_moe_maker.eval.harness import probe_router_discrimination
        src = inspect.getsource(probe_router_discrimination)
        assert "_make_code_prompt" in src, (
            "the probe must build its prompt the way the trainer does, not "
            "reimplement it and drift")
        assert 'f"Write {s}:"' not in src, (
            "that string is a format the router has never seen once")
        assert "raw_sources" in src, (
            "tools/reasoning experts trained on raw text and must be probed "
            "on raw text")

    def test_display_names_are_what_training_used(self):
        import random as _random
        from ms_moe_maker.config.pipeline import DISPLAY_LANG
        from ms_moe_maker.train.router import _make_code_prompt
        prompts = [_make_code_prompt("csharp", DISPLAY_LANG["csharp"],
                                     unnamed_fraction=0.0,
                                     rnd=_random.Random(i))
                   for i in range(25)]
        assert all("C#" in p for p in prompts), prompts
        assert not any("csharp" in p for p in prompts), (
            "the safe_name is a filename, not a language a router was taught")

    def test_run_eval_tells_the_probe_which_sources_were_raw(self):
        import inspect
        from ms_moe_maker.eval.harness import run_eval
        src = inspect.getsource(run_eval)
        assert "raw_text_sources=" in src
        assert "tools_expert_name" in src and "reasoning_experts" in src, (
            "the probe needs the same list router.format_fn branched on")

    def test_a_template_failure_is_recorded_not_swallowed(self):
        import inspect
        from ms_moe_maker.eval.harness import probe_router_discrimination
        src = inspect.getsource(probe_router_discrimination)
        assert "format_errors" in src
        assert "except Exception:\n                    pass" not in src, (
            "a silent fallback means one column was measured in a different "
            "format and nobody can tell")


def test_the_router_mix_prefers_the_train_split():
    """The mix drew from the WHOLE corpus, held-out rows included, so a large
    enough mix ate the very rows the routing probe needs. `.train` is written
    at a fixed seed and is exactly the complement of what eval holds out."""
    import inspect
    from ms_moe_maker.train import router
    src = inspect.getsource(router.train_router)
    assert 'path + ".train"' in src, (
        "the router mix must prefer the .train split so held-out stays held "
        "out")


class TestTheGenerationBudgetReachesGeneration:
    """`max_new_tokens` was a function default no recipe key could reach.

    A `<think>` block alone routinely runs past 256 tokens, so a reasoning eval
    was cut off mid-thought, never reached the answer, and reported "does not
    reliably reason" about a model that reasons fine.
    """

    @staticmethod
    def _budgets(cfg, spec, monkeypatch):
        from ms_moe_maker.eval import harness as H
        seen = []

        def _fake(**kw):
            seen.append(kw.get("max_new_tokens"))
            return EvalResult(expert_name=kw["label"], domain=kw["domain"],
                              status="done", scored_samples=5,
                              attempted_samples=5)

        monkeypatch.setattr(H, "eval_generation", _fake)
        H.run_eval(cfg, spec=spec)
        return seen

    def test_a_plain_run_keeps_the_old_cap(self, tmp_path, monkeypatch):
        seen = self._budgets(_cfg(tmp_path, with_data=True),
                             {"mode": "quality"}, monkeypatch)
        assert seen and set(seen) == {256}

    def test_a_reasoning_run_gets_room_for_the_block_and_the_answer(
            self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path, with_data=True)
        cfg.reasoning_experts = ["python"]
        seen = self._budgets(cfg, {"mode": "quality"}, monkeypatch)
        assert seen and set(seen) == {1024}

    def test_the_recipe_still_wins(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path, with_data=True)
        cfg.reasoning_experts = ["python"]
        seen = self._budgets(cfg, {"mode": "quality", "max_new_tokens": 320},
                             monkeypatch)
        assert seen and set(seen) == {320}


class TestTheThinkBlockSegmenter:
    """Token offsets, not character offsets — see think_token_segments."""

    STYLE = type("S", (), {"open": "<think>", "close": "</think>"})()

    def test_it_finds_the_block_and_what_follows_it(self):
        from ms_moe_maker.eval.harness import think_token_segments
        text = "<think>abc</think>xyz"
        #        0      7   10      18
        offsets = [(0, 0), (0, 7), (7, 10), (10, 18), (18, 21)]
        inside, after = think_token_segments(text, offsets, self.STYLE)
        assert inside == [(2, 3)], "only the interior counts as thinking"
        assert after == [(4, 5)]

    def test_the_delimiters_themselves_belong_to_neither_segment(self):
        from ms_moe_maker.eval.harness import think_token_segments
        inside, after = think_token_segments(
            "<think>abc</think>xyz",
            [(0, 7), (7, 10), (10, 18), (18, 21)], self.STYLE)
        assert 0 not in [t for a, b in inside for t in range(a, b)]
        assert 2 not in [t for a, b in after for t in range(a, b)]

    def test_no_block_is_none_not_an_empty_answer(self):
        from ms_moe_maker.eval.harness import think_token_segments
        assert think_token_segments("just an answer", [(0, 14)],
                                    self.STYLE) is None
        assert think_token_segments("<think>never closed", [(0, 7), (7, 19)],
                                    self.STYLE) is None
        assert think_token_segments("<think>a</think>b", [(0, 17)], None) is None

    def test_an_interwoven_trace_pools_every_block(self):
        """`interwoven` styles emit many blocks around tool calls; 'after' is
        what follows the LAST close tag, which is what the splitter reads as
        the answer."""
        from ms_moe_maker.eval.harness import think_token_segments
        text = "<think>a</think>b<think>c</think>d"
        offsets = [(0, 7), (7, 8), (8, 16), (16, 17), (17, 24), (24, 25),
                   (25, 33), (33, 34)]
        inside, after = think_token_segments(text, offsets, self.STYLE)
        flat = [t for a, b in inside for t in range(a, b)]
        assert flat == [1, 5], "both interiors, and nothing else"
        assert after == [(7, 8)], "only what follows the last close tag"


class TestTheSwingIsTheFinding:
    """Two share tables make the reader subtract. Report the subtraction."""

    def test_a_relay_names_who_takes_over_on_each_side(self):
        from ms_moe_maker.eval.harness import summarize_think_swing
        out = summarize_think_swing({"deliberation": 0.72, "python": 0.28},
                                    {"deliberation": 0.19, "python": 0.81})
        assert out["verdict"] == "relay"
        assert out["swing_to"] == "deliberation"
        assert out["yields_to"] == "python"
        assert round(out["swing"], 3) == 0.53
        assert round(out["delta"]["python"], 3) == -0.53

    def test_a_duet_is_reported_as_a_finding_not_as_silence(self):
        from ms_moe_maker.eval.harness import summarize_think_swing
        out = summarize_think_swing({"deliberation": 0.50, "python": 0.50},
                                    {"deliberation": 0.49, "python": 0.51})
        assert out["verdict"] == "duet"
        assert out["swing_to"] == "", "nothing to attribute below min_swing"
        assert out["delta"]["deliberation"] == pytest.approx(0.01)

    def test_the_segment_shares_use_the_pooled_arithmetic(self):
        """Slots, not tokens: shares sum to 1.0 and uniform is 1/E, which is
        the denominator a second copy of this loop already got wrong once."""
        from ms_moe_maker.eval.harness import _pooled_shares
        counts = [[6, 2], [4, 4]]      # two layers, two experts
        totals = [4, 4]                # four tokens per layer
        shares = _pooled_shares(counts, totals, 2, 2)
        assert shares == pytest.approx([10 / 16, 6 / 16])
        assert sum(shares) == pytest.approx(1.0)


class TestTheDisciplineDiagnosis:
    """High enrichment + low `reasoned` has a name, and the reader needs it."""

    def test_it_fires_on_register_without_discipline(self):
        from ms_moe_maker.eval.harness import reasoning_discipline_caveat
        msg = reasoning_discipline_caveat(enrichment=2.4, reasoned=0.12,
                                          capped_fraction=0.0,
                                          expert="deliberation")
        assert "deliberation" in msg
        assert "REGISTER" in msg and "DISCIPLINE" in msg
        assert "attention" in msg, "say why a routed FFN expert cannot fix it"

    def test_truncation_is_not_indiscipline(self):
        """A trace cut off by the token budget has no close tag because it was
        CUT. Calling that a discipline failure would be its own lie."""
        from ms_moe_maker.eval.harness import reasoning_discipline_caveat
        assert reasoning_discipline_caveat(enrichment=2.4, reasoned=0.12,
                                           capped_fraction=0.9) == ""

    def test_it_stays_quiet_when_routing_is_not_working(self):
        from ms_moe_maker.eval.harness import reasoning_discipline_caveat
        assert reasoning_discipline_caveat(enrichment=1.01, reasoned=0.12,
                                           capped_fraction=0.0) == ""

    def test_it_stays_quiet_when_the_model_reasons_fine(self):
        from ms_moe_maker.eval.harness import reasoning_discipline_caveat
        assert reasoning_discipline_caveat(enrichment=2.4, reasoned=0.95,
                                           capped_fraction=0.0) == ""
        assert reasoning_discipline_caveat(enrichment=2.4, reasoned=-1.0,
                                           capped_fraction=0.0) == "", (
            "-1 is 'not a reasoning run', not 'reasoned zero percent'")


class TestTheDiagnosisReachesTheReport:
    """The pure helper is the diagnosis; this is the wiring that fires it."""

    @staticmethod
    def _run(tmp_path, monkeypatch, reasoned, capped):
        from ms_moe_maker.eval import harness as H
        cfg = _cfg(tmp_path, with_data=True)
        cfg.reasoning_experts = ["python"]

        def _fake_gen(**kw):
            return EvalResult(expert_name=kw["label"], domain=kw["domain"],
                              status="done", scored_samples=10,
                              attempted_samples=10, reasoned=reasoned,
                              capped_generations=capped)

        def _fake_probe(**kw):
            r = _routing(n_experts=2, top_k=1,
                         python={"enrichment": 2.4, "own_share": 0.72,
                                 "marginal_share": 0.50,
                                 "enrichment_reliable": True,
                                 "top_competitor": "csharp",
                                 "top_competitor_share": 0.28,
                                 "outranked": False},
                         csharp={"enrichment": 2.0, "own_share": 0.66,
                                 "marginal_share": 0.50,
                                 "enrichment_reliable": True,
                                 "top_competitor": "python",
                                 "top_competitor_share": 0.34,
                                 "outranked": False})
            r["mean_js_bits"] = 0.4
            return r

        monkeypatch.setattr(H, "eval_generation", _fake_gen)
        monkeypatch.setattr(H, "probe_router_discrimination", _fake_probe)
        return H.run_eval(cfg, spec={"mode": "all"})

    def test_register_without_discipline_is_said_out_loud(
            self, tmp_path, monkeypatch):
        report = self._run(tmp_path, monkeypatch, reasoned=0.1, capped=0)
        assert any("REGISTER" in c for c in report.caveats), report.caveats

    def test_a_truncated_run_gets_the_other_caveat_and_only_that_one(
            self, tmp_path, monkeypatch):
        report = self._run(tmp_path, monkeypatch, reasoned=0.1, capped=9)
        assert not any("REGISTER" in c for c in report.caveats), report.caveats
        assert any("Raise eval.max_new_tokens" in c for c in report.caveats), (
            report.caveats)

    def test_a_model_that_reasons_gets_neither(self, tmp_path, monkeypatch):
        report = self._run(tmp_path, monkeypatch, reasoned=0.98, capped=0)
        assert not any("REGISTER" in c for c in report.caveats)
        assert not any("Raise eval.max_new_tokens" in c for c in report.caveats)


class TestReasonedIsInTheTable:
    """It existed, in a block under the table where it read as a footnote.

    On its own the number says nothing; it only becomes a diagnosis beside the
    routing enrichment, so it belongs in the row with the scores it qualifies.
    """

    def test_the_column_is_printed_and_a_plain_run_shows_a_dash(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.stages["deliberation"] = EvalResult(
            expert_name="deliberation", domain="deliberation", status="done",
            rouge1=0.4, bleu=0.3, scored_samples=20, attempted_samples=20,
            reasoned=0.15)
        report.stages["python"] = EvalResult(
            expert_name="python", domain="python", status="done",
            rouge1=0.4, bleu=0.3, scored_samples=20, attempted_samples=20)
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "reasoned" in out
        assert "0.15" in out
        assert "does not reliably reason" in out
        assert any(ln.strip().startswith("python") and "-" in ln
                   for ln in out.splitlines()), (
            "a non-reasoning row was never asked the question; 0.00 would be "
            "a different claim")

    def test_the_swing_table_prints_the_delta_not_two_columns(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            deliberation={"enrichment": 2.4, "own_share": 0.72,
                          "others_share": 0.30, "marginal_share": 0.50,
                          "enrichment_reliable": True,
                          "top_competitor": "python",
                          "top_competitor_share": 0.28, "outranked": False})
        report.routing["think_segments"] = {
            "deliberation": {
                "samples": 12,
                "think": {"deliberation": 0.72, "python": 0.28},
                "after": {"deliberation": 0.19, "python": 0.81},
                "delta": {"deliberation": 0.53, "python": -0.53},
                "swing_to": "deliberation", "yields_to": "python",
                "swing": 0.53, "verdict": "relay"}}
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "in think" in out and "delta" in out
        assert "+0.530" in out, "the swing is the finding, so print it"
        assert "RELAY" in out

    def test_a_run_without_think_blocks_prints_no_swing_table(self, capsys):
        from ms_moe_maker.__main__ import _print_eval_report
        report = EvalReport(ok=True)
        report.routing = _routing(
            n_experts=2, top_k=1,
            python={"enrichment": 2.4, "own_share": 0.72,
                    "others_share": 0.30, "marginal_share": 0.50,
                    "enrichment_reliable": True, "top_competitor": "csharp",
                    "top_competitor_share": 0.28, "outranked": False})
        _print_eval_report(report)
        out = capsys.readouterr().out
        assert "in think" not in out and "RELAY" not in out
