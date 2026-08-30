"""The vendored Heretic core, guarded for the first time.

tests/test_abliterate.py deliberately does NOT exercise the vendored core; the
refusal direction, orthogonalization, scorers, plugin load and the stage
itself had zero coverage on anything that computes. These tests cover the pure
pieces - settings isolation, the Pareto selection, the scorer's arithmetic,
and the stage's done-predicate - without torch or a GPU.
"""
import types

import pytest

# The vendored core needs the [train] extra (optuna/pydantic). The suite must
# still pass on the base install, so skip the whole module without it.
pytest.importorskip("optuna")

from ms_moe_maker.abliterate import stage as stage_mod
from ms_moe_maker.abliterate.heretic import abliterate as ab_mod
from ms_moe_maker.abliterate.heretic.config import Settings
from ms_moe_maker.abliterate.heretic.scorers.keyword_rate import KeywordRate


# ── 5.2: the child must not inherit config from its CWD ─────────────────────

def test_payload_mode_ignores_env_and_toml(monkeypatch, tmp_path):
    """A stray config.toml in the CWD or a HERETIC_* env var used to silently
    redefine the refusal objective (good_prompts/bad_prompts, scorers, ...)
    while the study ran to completion. from_payload must honour ONLY the
    payload, then defaults."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        'system_prompt = "evil toml"\n', encoding="utf-8")
    monkeypatch.setenv(
        "HERETIC_GOOD_PROMPTS",
        '{"dataset": "evil/env", "split": "train[:1]", "column": "text"}')

    settings = Settings.from_payload({"model": "some/model"})
    assert settings.system_prompt == "You are a helpful assistant.", (
        "the TOML in the CWD must not leak into the payload build")
    assert settings.good_prompts.dataset == "mlabonne/harmless_alpaca", (
        "the HERETIC_ env must not leak into the payload build")
    assert settings.model == "some/model"

    # The interactive path (Settings()) still layers those sources - the gate
    # is real in BOTH directions, not a source chain that was just deleted.
    layered = Settings(model="some/model", _env_file=None, _secrets_dir=None)
    assert layered.system_prompt == "evil toml"


def test_payload_mode_resets_even_on_error():
    """The module toggle must not leak into a later plain Settings() build."""
    try:
        Settings.from_payload({"n_trials": 0})  # PositiveInt: raises
    except Exception:
        pass
    settings = Settings(model="m")
    assert settings.n_trials == 200  # default, not polluted by payload mode


# ── 5.3 + 5.5: Pareto selection ─────────────────────────────────────────────

def _trial(index, **scores):
    return types.SimpleNamespace(
        index=index,
        user_attrs={"scores": [{"name": name, "score": {"value": value}}
                               for name, value in scores.items()]})


def test_pareto_sort_honours_maximize():
    """Every objective used to sort ascending, so a maximize scorer had its
    WORST trial restored and exported."""
    trials = [_trial(0, A=0.9), _trial(1, A=0.1), _trial(2, A=0.5)]
    key = ab_mod._pareto_key(["A"], ["maximize"])
    assert sorted(trials, key=key)[0].index == 0, (
        "the best A (0.9) must sort first under maximize")


def test_pareto_sort_honours_minimize():
    trials = [_trial(0, A=0.9), _trial(1, A=0.1), _trial(2, A=0.5)]
    key = ab_mod._pareto_key(["A"], ["minimize"])
    assert sorted(trials, key=key)[0].index == 1


def test_a_missing_objective_ranks_worst_not_typeerror():
    """Resuming after changing the scorer list used to put None in the sort
    key and die with TypeError after the whole study had run."""
    trials = [_trial(0, A=0.9, B=0.1), _trial(1, A=0.5)]
    key = ab_mod._pareto_key(["A", "B"], ["minimize", "minimize"])
    ordered = sorted(trials, key=key)
    assert ordered[-1].index == 1, "the trial missing B must rank worst"
    assert ordered[0].index == 0


def test_trial_index_past_the_front_clamps():
    trials = [_trial(0), _trial(1)]
    assert ab_mod._pick_trial(trials, None).index == 0
    assert ab_mod._pick_trial(trials, 5).index == 1, (
        "a past-the-front index used to raise IndexError after the study ran")
    assert ab_mod._pick_trial(trials, 1).index == 1


# ── 5.5: the scorer's arithmetic ────────────────────────────────────────────

class _FakeCtx:
    def __init__(self, responses):
        self._responses = responses

    def get_responses(self, prompts):
        return self._responses

    def load_prompts(self, spec):
        return []


def test_keyword_rate_counts_a_missing_response_as_a_refusal():
    """The zip truncated silently, so a short response list deflated the
    refusal rate - the optimizer could score better by producing fewer
    answers. A missing response is a refusal."""
    scorer = KeywordRate(settings=KeywordRate.Settings(
        keyword_markers=["sorry"], prompts={}))
    scorer.prompts = [types.SimpleNamespace(system="s", user="u")] * 4
    ctx = _FakeCtx(["Fine, here is how.", "", "sorry, no."])
    score = scorer.get_score(ctx)
    assert score.value == 0.75, (
        f"3 of 4 should match (2 refused, 1 missing), got {score.value}")


def test_keyword_rate_on_no_prompts_does_not_divide_by_zero():
    scorer = KeywordRate(settings=KeywordRate.Settings(
        keyword_markers=["sorry"], prompts={}))
    scorer.prompts = []
    score = scorer.get_score(_FakeCtx([]))
    assert score.value == 0.0


# ── 5.4: the stage's done-predicate ─────────────────────────────────────────

def test_abliterate_is_done_rejects_a_half_written_model(tmp_path):
    """config.json exists before the shards; an OOM mid-save used to leave a
    shell that every specialist then trained from. All three markers must
    exist."""
    cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
    d = tmp_path / stage_mod.abliterate_dir(cfg)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert stage_mod.abliterate_is_done(cfg) is False, (
        "config.json alone is the half-written shell")
    (d / "model.safetensors").write_bytes(b"x")
    assert stage_mod.abliterate_is_done(cfg) is False, "tokenizer missing"
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert stage_mod.abliterate_is_done(cfg) is True


def test_abliterate_is_done_respects_force(tmp_path):
    cfg = types.SimpleNamespace(force=True, output_root=str(tmp_path))
    assert stage_mod.abliterate_is_done(cfg) is False


# ── 5.1: the settings payload the stage writes ──────────────────────────────

def test_the_stage_payload_round_trips_without_external_sources(tmp_path):
    """The 9 keys the stage owns must load with every other field at its
    default - the contract 5.2's fix restores."""
    payload = {
        "model": "some/model",
        "save_directory": str(tmp_path / "out"),
        "export_strategy": "adapter",
        "checkpoint_action": "continue",
        "trial_index": 0,
        "n_trials": 5,
        "seed": 1,
        "quantization": "none",
        "study_checkpoint_dir": str(tmp_path / "ckpt"),
    }
    settings = Settings.from_payload(payload)
    assert settings.export_strategy.value == "adapter"
    assert settings.n_trials == 5
    assert settings.good_prompts.dataset == "mlabonne/harmless_alpaca"
    assert settings.bad_prompts.dataset == "mlabonne/harmful_behaviors"
