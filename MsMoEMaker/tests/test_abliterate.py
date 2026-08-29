"""Tests for the `abliterate:` recipe block and the abliterate.base stage.

No torch: these exercise recipe parsing, validation, config resolution and
stage planning only. The vendored Heretic core (ms_moe_maker.heretic) is
imported lazily by the stage and is deliberately NOT exercised here — that
half needs a GPU and the training stack.
"""

import pytest

from ms_moe_maker.run import stages
from ms_moe_maker.config.recipe import Abliterate, _build_abliterate, parse, validate


BASE = {
    "schema_version": 1,
    "name": "t",
    "experts": [
        {"name": "python", "source": {"kind": "stack", "language": "Python"}},
        {"name": "csharp", "source": {"kind": "stack", "language": "C#"}},
    ],
}


class TestBuildAbliterate:
    def test_true_enables_with_defaults(self):
        abl = _build_abliterate(True, [])
        assert abl.enabled is True
        assert abl.n_trials == -1            # -1 = Heretic default (200)
        assert abl.quantization == "none"
        assert abl.export == "merge"
        assert abl.checkpoint_action == "continue"

    def test_false_and_none_disable(self):
        assert _build_abliterate(False, []).enabled is False
        assert _build_abliterate(None, []).enabled is False

    def test_mapping_enables_and_overrides(self):
        abl = _build_abliterate({"n_trials": 100, "quantization": "bnb_4bit"}, [])
        assert abl.enabled is True
        assert abl.n_trials == 100
        assert abl.quantization == "bnb_4bit"
        assert abl.export == "merge"         # untouched fields keep defaults

    def test_unknown_key_warns(self):
        warns = []
        abl = _build_abliterate({"bogus": 1}, warns)
        assert abl.enabled is True
        assert any("abliterate.bogus" in w for w in warns)


class TestParse:
    def test_absent_is_disabled(self):
        rec, _ = parse(dict(BASE))
        assert rec.abliterate.enabled is False

    def test_true_is_enabled(self):
        rec, _ = parse({**BASE, "abliterate": True})
        assert rec.abliterate.enabled is True

    def test_mapping_is_enabled(self):
        rec, _ = parse({**BASE, "abliterate": {"n_trials": 50}})
        assert rec.abliterate.enabled is True
        assert rec.abliterate.n_trials == 50


class TestValidate:
    def test_valid_block_passes(self):
        rec, _ = parse({**BASE, "abliterate": True})
        errs, _ = validate(rec)
        assert not any("abliterate" in e for e in errs)

    def test_bad_quantization(self):
        rec, _ = parse({**BASE, "abliterate": {"quantization": "int8"}})
        errs, _ = validate(rec)
        assert any("abliterate.quantization" in e for e in errs)

    def test_bad_export(self):
        rec, _ = parse({**BASE, "abliterate": {"export": "safetensors"}})
        errs, _ = validate(rec)
        assert any("abliterate.export" in e for e in errs)

    def test_bad_checkpoint_action(self):
        rec, _ = parse({**BASE, "abliterate": {"checkpoint_action": "always"}})
        errs, _ = validate(rec)
        assert any("abliterate.checkpoint_action" in e for e in errs)

    def test_zero_trials(self):
        rec, _ = parse({**BASE, "abliterate": {"n_trials": 0}})
        errs, _ = validate(rec)
        assert any("abliterate.n_trials" in e for e in errs)


class TestBuildConfig:
    def test_abliterate_fields_resolve(self):
        from ms_moe_maker.config import pipeline as config

        rec, _ = parse({**BASE, "abliterate": {"n_trials": 123}})
        cfg = config.build_config(rec)
        assert cfg.abliterate_enabled is True
        assert cfg.abliterate_n_trials == 123
        assert cfg.abliterate_export == "merge"

    def test_abliterate_off_by_default(self):
        from ms_moe_maker.config import pipeline as config

        rec, _ = parse(dict(BASE))
        cfg = config.build_config(rec)
        assert cfg.abliterate_enabled is False


class TestPlan:
    def test_stage_included_when_enabled(self):
        plan = stages.plan(["python", "csharp"], abliterate=True)
        ids = [sid for sid, _ in plan]
        assert "abliterate.base" in ids
        assert ids.index("abliterate.base") < ids.index("data.corpus")

    def test_stage_absent_by_default(self):
        plan = stages.plan(["python", "csharp"])
        assert "abliterate.base" not in [sid for sid, _ in plan]
