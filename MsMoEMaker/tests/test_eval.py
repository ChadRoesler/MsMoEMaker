"""Tests for the eval pipeline."""
import json
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


def _routing(status=PASS, **experts):
    return {"status": status, "reason": "", "experts": dict(experts)}


class TestDeadExpertDetection:

    def test_healthy_expert_not_dead(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            python={"enrichment": 2.1, "own_share": 0.31,
                    "top_competitor": "csharp", "top_competitor_share": 0.19,
                    "outranked": False})
        assert detect_dead_experts(report, threshold=1.2) == []

    def test_low_enrichment_is_dead(self):
        report = EvalReport(ok=True)
        report.routing = _routing(
            powershell={"enrichment": 1.11, "own_share": 0.192,
                        "top_competitor": "csharp",
                        "top_competitor_share": 0.278, "outranked": True})
        assert detect_dead_experts(report, threshold=1.2) == ["powershell"]

    def test_outranked_on_own_domain_is_dead_even_if_enriched(self):
        """The 0.5B rung's actual weak spot: an expert can clear the enrichment
        bar on the column comparison and still lose its own domain to a
        neighbour. Column-only reads miss this."""
        report = EvalReport(ok=True)
        report.routing = _routing(
            powershell={"enrichment": 1.9, "own_share": 0.192,
                        "top_competitor": "csharp",
                        "top_competitor_share": 0.278, "outranked": True})
        assert detect_dead_experts(report, threshold=1.2) == ["powershell"]

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
