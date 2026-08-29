"""A perfectly symmetric router has no way out of symmetry except the aux loss.

Zero-init makes the untrained MoE reproduce one expert exactly, and lets
verify_stitch assert bit-equality. It is also a starting point where every
expert's logit is identical on every token, and a top-k router leaving that
state depends entirely on which side the first optimizer step happens to
favour.

Measured across three router trainings on one zero-init 0.5B skeleton: all
three collapsed onto a single expert, the winner differed between runs
(python, csharp, csharp), and JS divergence stayed at 0.0001 every time -
while the experts genuinely differed at 263x the chance floor and correct
routing was worth 0.43 nats.

Switch and Mixtral seed their routers with small noise for this reason. Random
is now the DEFAULT because a Ms.MoE built with the defaults has to route, and
zero-init collapses. `zero` remains for verify_stitch's bit-equality check:
it is the "is the skeleton well-formed" answer, not a training starting point.
"""
import pytest

from ms_moe_maker.config.recipe import parse, validate


def _rec(**moe):
    base = {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "experts": [
            {"name": "a", "source": {"kind": "stack", "language": "Python"}},
            {"name": "b", "source": {"kind": "stack", "language": "C#"}},
            {"name": "c", "source": {"kind": "stack", "language": "Go"}},
        ],
    }
    if moe:
        base["moe"] = moe
    rec, _ = parse(base)
    return rec


def test_random_is_the_default():
    assert _rec().moe.router_init == "random"


def test_random_is_accepted():
    errs, _ = validate(_rec(router_init="random"))
    assert not any("router_init" in e for e in errs), errs


def test_a_typo_is_refused_not_silently_zeroed():
    errs, _ = validate(_rec(router_init="randon"))
    assert any("router_init must be zero | random" in e for e in errs), errs


@pytest.mark.parametrize("std", [0.0, -0.1, 0.9])
def test_an_untrainable_noise_scale_is_refused(std):
    errs, _ = validate(_rec(router_init="random", router_init_std=std))
    assert any("router_init_std" in e for e in errs), errs


def test_a_sane_noise_scale_passes():
    errs, _ = validate(_rec(router_init="random", router_init_std=0.02))
    assert not any("router_init_std" in e for e in errs), errs


def test_the_knob_reaches_the_pipeline_config():
    from ms_moe_maker.config import pipeline as cfg_mod
    c = cfg_mod.build_config(_rec(router_init="random", router_init_std=0.05),
                             dryrun=True)
    assert c.router_init == "random"
    assert c.router_init_std == 0.05
    assert cfg_mod.build_config(_rec(), dryrun=True).router_init == "random"


class TestVerifyKnowsWhatWasAskedFor:
    """verify_stitch must check the router against the init that was REQUESTED.

    A zero check on a randomly-initialised gate fails a correct build; no check
    at all on it would pass a gate that silently never got initialised. Both
    are the same error - a verifier that does not know what it is verifying.
    """

    def test_the_zero_check_is_still_strict_by_default(self):
        import inspect
        from ms_moe_maker.moe import stitch
        src = inspect.getsource(stitch.verify_stitch)
        assert 'router_init == "random"' in src
        assert "router not zero" in src, (
            "the strict zero assertion must survive for the default path")

    def test_random_still_refuses_an_uninitialised_gate(self):
        import inspect
        from ms_moe_maker.moe import stitch
        src = inspect.getsource(stitch.verify_stitch)
        assert "the init did not run" in src, (
            "all-zeros under router_init=random means the init silently did "
            "nothing, and that must fail rather than pass as 'not checked'")

    def test_random_bounds_the_noise(self):
        import inspect
        from ms_moe_maker.moe import stitch
        src = inspect.getsource(stitch.verify_stitch)
        assert "route on this" in src, (
            "unbounded noise means the untrained MoE routes on garbage")
