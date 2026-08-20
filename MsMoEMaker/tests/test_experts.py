"""The expert gate — three probes that each cost a build before they existed.

Every case here is a real failure this project shipped into, not a
hypothetical:

  * cos=1.000 from a base nobody checked          -> base is a lookup now
  * two experts that were "identical" but weren't -> base-free pairwise check
  * a router severed from the loss by one config field
  * an expert pair that no router could ever separate
"""
import json

import pytest

from ms_moe_maker import experts as ex
from ms_moe_maker import stages as st


def _write_cfg(tmp_path, **cfg):
    d = tmp_path / "moe"
    d.mkdir(exist_ok=True)
    base = {"num_experts": 3, "num_experts_per_tok": 2,
            "norm_topk_prob": True, "use_cache": True,
            "expert_names": ["a", "b", "c"]}
    base.update(cfg)
    (d / "config.json").write_text(json.dumps(base), encoding="utf-8")
    return str(d)


# ── the config audit ──────────────────────────────────────────────────────

class TestConfigAudit:

    def test_top1_with_normalisation_is_flagged(self, tmp_path):
        """The one that cost the most. top-1 normalisation divides a weight by
        itself, so the gate gets ZERO gradient from the LM loss and can only be
        moved by the aux loss, which pulls toward uniform. Every other check in
        this tool passes on this model: the tensors verify, it generates, eval
        reports balanced routing and no dead experts."""
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=2, num_experts_per_tok=1,
                       norm_topk_prob=True))
        assert r["status"] == ex.WARN
        assert any("severs the router" in f for f in r["findings"])

    def test_top1_without_normalisation_is_clean(self, tmp_path):
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=4, num_experts_per_tok=1,
                       norm_topk_prob=False))
        assert not any("severs" in f for f in r["findings"]), r["findings"]

    def test_topk_equal_to_expert_count_is_flagged(self, tmp_path):
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=2, num_experts_per_tok=2))
        assert any("selected on every token" in f for f in r["findings"])

    def test_two_experts_carries_the_power_floor(self, tmp_path):
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=2, num_experts_per_tok=1,
                       norm_topk_prob=False))
        assert any("p=0.250" in f for f in r["findings"]), r["findings"]

    def test_four_experts_has_no_power_floor_finding(self, tmp_path):
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=4, num_experts_per_tok=2))
        assert not any("significance" in f for f in r["findings"])

    def test_use_cache_false_is_flagged(self, tmp_path):
        r = ex.audit_moe_config(
            _write_cfg(tmp_path, num_experts=4, use_cache=False))
        assert any("use_cache=false" in f for f in r["findings"])

    def test_a_missing_config_is_unmeasurable_not_clean(self, tmp_path):
        """Absence of measurement must never present as absence of problems."""
        r = ex.audit_moe_config(str(tmp_path / "nope"))
        assert r["status"] == ex.UNMEASURABLE
        assert r["reason"]


# ── interpretation ────────────────────────────────────────────────────────

class TestInterpretation:

    def _rep(self, **kw):
        r = ex.ExpertsReport(**kw)
        ex._interpret(r)
        return r

    def test_identical_experts_are_named(self):
        r = self._rep(pairwise={"status": ex.OK, "pairs": {"a|b": 0.0},
                                "identical_pairs": ["a|b"],
                                "closest_pair": "a|b", "closest": 0.0})
        assert any("identical experts" in f for f in r.findings)
        assert r.status == ex.WARN

    def test_barely_trained_invalidates_the_cosine(self):
        """A low cosine on near-zero deltas is noise, not independence. The
        probe says so in prose; the gate has to act on it."""
        r = self._rep(divergence={"status": ex.OK,
                                  "movement": {"a": 0.0001, "b": 0.0001},
                                  "cos": {"a|b": 0.01}, "most_aligned": 0.01,
                                  "most_aligned_pair": "a|b",
                                  "low_movement": ["a", "b"],
                                  "chance_cos": 0.0005, "dims": 4000000})
        assert any("barely trained" in f for f in r.findings)

    def test_aligned_deltas_are_flagged(self):
        r = self._rep(divergence={"status": ex.OK,
                                  "movement": {"a": 0.05, "b": 0.05},
                                  "cos": {"a|b": 0.94}, "most_aligned": 0.94,
                                  "most_aligned_pair": "a|b",
                                  "low_movement": [], "chance_cos": 0.0005,
                                  "dims": 4000000})
        assert any("learned the same thing twice" in f for f in r.findings)

    def test_no_router_signal_is_the_loudest_finding(self):
        r = self._rep(cross_loss={
            "status": ex.OK, "signal": False, "domains": ["py", "cs"],
            "matrix": {"py": {"py": 2.0, "cs": 1.0}, "cs": {"py": 1.5, "cs": 0.5}},
            "gaps": {"py": {"own": 2.0, "best_rival": "cs",
                            "best_rival_loss": 1.5, "gap": -0.5,
                            "own_wins": False},
                     "cs": {"own": 0.5, "best_rival": "py",
                            "best_rival_loss": 1.0, "gap": 0.5,
                            "own_wins": True}},
            "mean_gap": 0.0})
        assert any("NO ROUTER SIGNAL" in f for f in r.findings)
        assert any("py" in f for f in r.findings)

    def test_healthy_everything_produces_no_findings(self):
        r = self._rep(
            pairwise={"status": ex.OK, "pairs": {"a|b": 0.06},
                      "identical_pairs": [], "closest_pair": "a|b",
                      "closest": 0.06},
            divergence={"status": ex.OK, "movement": {"a": 0.04, "b": 0.05},
                        "cos": {"a|b": 0.03}, "most_aligned": 0.03,
                        "most_aligned_pair": "a|b", "low_movement": [],
                        "chance_cos": 0.0005, "dims": 4000000},
            cross_loss={"status": ex.OK, "signal": True, "domains": ["a", "b"],
                        "matrix": {"a": {"a": 1.0, "b": 2.0},
                                   "b": {"a": 2.0, "b": 1.0}},
                        "gaps": {"a": {"own": 1.0, "best_rival": "b",
                                       "best_rival_loss": 2.0, "gap": 1.0,
                                       "own_wins": True},
                                 "b": {"own": 1.0, "best_rival": "a",
                                       "best_rival_loss": 2.0, "gap": 1.0,
                                       "own_wins": True}},
                        "mean_gap": 1.0},
            config_audit={"status": ex.OK, "findings": []})
        assert r.findings == []
        assert r.status == ex.OK

    def test_unmeasurable_blocks_are_recorded_not_swallowed(self):
        r = self._rep(cross_loss={"status": ex.UNMEASURABLE,
                                  "reason": "no held-out data"})
        assert any("cross-loss" in u and "no held-out data" in u
                   for u in r.unmeasured)


# ── the gate itself ───────────────────────────────────────────────────────

class _Cfg:
    base = "some/base"


def test_run_experts_on_an_empty_run_dir_never_raises(tmp_path):
    """Partial input is the normal case - before the stitch there is no MoE.
    Every block must report why it could not run rather than vanishing."""
    rep = ex.run_experts(_Cfg(), {"a": str(tmp_path / "a"),
                                  "b": str(tmp_path / "b")})
    assert rep.pairwise["status"] == ex.UNMEASURABLE
    assert rep.cross_loss["status"] == ex.UNMEASURABLE
    assert rep.config_audit["status"] == ex.UNMEASURABLE
    assert len(rep.unmeasured) >= 3
    assert "a" in " ".join(rep.unmeasured)


def test_format_report_survives_a_fully_unmeasured_run(tmp_path):
    rep = ex.run_experts(_Cfg(), {"a": str(tmp_path / "a")})
    text = ex.format_report(rep)
    assert "NOT MEASURED" in text
    assert "unmeasurable" in text.lower()


def test_the_base_is_never_a_parameter_of_the_gate():
    """THE WRONG-BASE TRAP, closed structurally.

    movement_and_direction takes a base because it is a primitive; run_experts
    does NOT, because it is the entry point everything real goes through, and
    it reads config.base. A measurement whose reference can be supplied by
    hand will eventually be supplied wrong - it was, and it produced a
    confident 'these experts are identical' that was pure artifact.
    """
    import inspect
    params = inspect.signature(ex.run_experts).parameters
    assert "base" not in params, (
        "run_experts must read the base from config, never accept one")


def test_movement_without_a_base_is_unmeasurable_not_zero():
    r = ex.movement_and_direction("", {"a": "/x", "b": "/y"})
    assert r["status"] == ex.UNMEASURABLE


# ── wiring ────────────────────────────────────────────────────────────────

def test_the_gate_is_planned_for_multi_expert_builds():
    ids = [sid for sid, _ in st.plan(["a", "b"])]
    assert st.GATE_EXPERTS in ids
    assert ids.index(st.GATE_EXPERTS) < ids.index(st.STITCH), (
        "the gate is worthless after the stitch it was meant to save")


def test_a_single_expert_build_has_nothing_to_compare():
    assert st.GATE_EXPERTS not in [sid for sid, _ in st.plan(["a"])]


def test_the_legacy_runner_plans_no_gate():
    """fraunkenstein_universal.py cannot emit it, and a planned stage nothing
    can close hangs in the manifest forever."""
    assert st.GATE_EXPERTS not in [
        sid for sid, _ in st.plan(["a", "b"], gates=False)]


def test_experts_is_a_declared_eval_mode():
    from ms_moe_maker import _describe
    assert "experts" in _describe.EVAL_MODES
    assert list(_describe.DESCRIBE["eval_modes"]) == list(_describe.EVAL_MODES)


def test_unknown_eval_mode_names_the_new_one():
    from ms_moe_maker.eval import run_eval

    class _C:
        data_root = "/nope"
        output_root = "/nope"
    rep = run_eval(_C(), {"mode": "nonsense"})
    assert not rep.ok
    assert "experts" in rep.message


def test_gates_experts_accepts_three_settings():
    from ms_moe_maker.recipe import parse, validate
    for value, should_error in (("auto", False), ("cheap", False),
                                ("skip", False), ("sometimes", True)):
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a",
                         "source": {"kind": "stack", "language": "Python"}}],
            "gates": {"experts": value},
        })
        errs, _ = validate(rec)
        hit = any("gates.experts" in e for e in errs)
        assert hit == should_error, (value, errs)
