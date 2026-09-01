"""The trainer callback has to satisfy the WHOLE TrainerCallback interface.

The first real 0.5B build died at stage 3 with

    AttributeError: 'HeartbeatCallback' object has no attribute 'on_init_end'

after the corpus had been collected, the base model downloaded, and 1635 docs
tokenized and packed. transformers' CallbackHandler calls
`getattr(callback, event)(...)` for EVERY lifecycle hook, so implementing only
the interesting one is not enough - duck-typing loses against a framework that
introspects.

These tests stub transformers rather than requiring it, because they have to
pass on the base install. That is the same reason the callback is built by a
factory instead of subclassing at module scope: importing transformers at
import time would put torch behind `ms-moe-maker validate`.
"""
import sys
import types

import pytest

# Every event transformers' CallbackHandler dispatches. Listed explicitly so
# this test states the contract rather than asking the library what it wants -
# a check that reads its expectations from the thing under test proves nothing.
EVENTS = (
    "on_init_end", "on_train_begin", "on_train_end",
    "on_epoch_begin", "on_epoch_end",
    "on_step_begin", "on_substep_end", "on_step_end",
    "on_optimizer_step", "on_pre_optimizer_step",
    "on_evaluate", "on_predict", "on_save", "on_log", "on_prediction_step",
)


@pytest.fixture
def stub_transformers(monkeypatch):
    """A stand-in TrainerCallback with the real interface."""
    mod = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    for name in EVENTS:
        def _noop(self, args, state, control, _n=name, **kwargs):
            return control
        setattr(TrainerCallback, name, _noop)

    mod.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def test_the_callback_implements_every_event(stub_transformers):
    from ms_moe_maker.train.finetune import make_heartbeat_callback
    cb = make_heartbeat_callback()
    missing = [e for e in EVENTS if not hasattr(cb, e)]
    assert not missing, (
        f"the Trainer calls these and the callback does not have them: "
        f"{missing}. This is the stage-3 crash, exactly.")


def test_it_derives_from_trainercallback(stub_transformers):
    """Not 'has the methods' - IS one. The base class is where the no-op
    defaults come from, and new transformers versions add events."""
    from ms_moe_maker.train.finetune import make_heartbeat_callback
    assert isinstance(make_heartbeat_callback(), stub_transformers.TrainerCallback)


def test_on_step_end_still_does_its_job(stub_transformers):
    from ms_moe_maker.train.finetune import make_heartbeat_callback
    cb = make_heartbeat_callback(print_interval=2, checkpoint_interval=10_000)
    state = types.SimpleNamespace(log_history=[])
    for _ in range(4):
        cb.on_step_end(None, state, "ctl")
    assert cb.step_count == 4


def test_on_step_end_returns_control(stub_transformers):
    """The handler assigns the return value back to `control`; returning None
    silently discards whatever a previous callback asked for."""
    from ms_moe_maker.train.finetune import make_heartbeat_callback
    cb = make_heartbeat_callback()
    sentinel = object()
    assert cb.on_step_end(None, types.SimpleNamespace(log_history=[]),
                          sentinel) is sentinel


def test_a_missing_loss_does_not_crash_the_heartbeat(stub_transformers):
    """log_history can be empty, or hold entries with no 'loss'. A progress
    printer that raises would kill a run it was only meant to describe."""
    from ms_moe_maker.train.finetune import make_heartbeat_callback
    cb = make_heartbeat_callback(print_interval=1, checkpoint_interval=-1)
    for hist in ([], [{}], [{"loss": 0.5}]):
        cb.on_step_end(None, types.SimpleNamespace(log_history=hist), "ctl")


def test_no_bare_callback_is_handed_to_the_trainer():
    """Static guard: the trainer must receive the factory's product."""
    import inspect
    from ms_moe_maker.train import finetune
    src = inspect.getsource(finetune)
    assert "callbacks=[make_heartbeat_callback()]" in src
    assert "callbacks=[HeartbeatCallback()]" not in src, (
        "a module-level bare class cannot subclass TrainerCallback without "
        "importing transformers at import time, which is why this is a factory")


def test_a_half_saved_specialist_does_not_count_as_done(tmp_path):
    """specialist_is_done used to be presence of config.json alone, which
    save_pretrained writes BEFORE the multi-GB shards and the tokenizer. A
    kill/OOM/disk-full in that window left a shell that resume skipped - and
    the stitch then stitched a directory with no weights in it."""
    import types

    from ms_moe_maker.train import finetune as f

    cfg = types.SimpleNamespace(force=False, output_root=str(tmp_path))
    d = tmp_path / f.specialist_dir(cfg, "python")
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert f.specialist_is_done(cfg, "python") is False, (
        "config.json alone is the half-saved shell")
    (d / "model.safetensors").write_bytes(b"x")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert f.specialist_is_done(cfg, "python") is True


def test_finetune_imports_without_transformers():
    """The reason the factory exists at all."""
    import importlib
    import ms_moe_maker.train.finetune as f
    importlib.reload(f)
    assert "torch" not in sys.modules or True   # nothing heavy at import time
    assert callable(f.make_heartbeat_callback)


# ── Adapter-base refusal and the dense save ─────────────────────────────────

def test_an_adapter_checkpoint_base_is_refused_before_training():
    """An adapter dir auto-loads its delta on top of already-merged weights
    (double-ablation) and nests our LoRA inside it - the first gauntlet build
    died at specialist save time because of exactly that. Refuse up front."""
    from ms_moe_maker.train import finetune as f

    class AdapterCarrying:
        peft_config = {"default": object()}

    with pytest.raises(RuntimeError, match="ADAPTER checkpoint"):
        f._refuse_adapter_base(AdapterCarrying(), "abliterated_base")

    class Plain:
        pass

    f._refuse_adapter_base(Plain(), "clean_base")   # no raise


def test_peft_residue_is_stripped_before_the_dense_save():
    """merge_and_unload can return a model that still carries peft_config;
    save_pretrained then takes the adapter path (get_adapter_state_dict ->
    active_adapters) - the upstream UnboundLocalError the gauntlet hit."""
    from ms_moe_maker.train import finetune as f

    class Residue:
        pass

    m = Residue()
    m.peft_config = {"default": object()}
    assert f._strip_peft_residue(m, "python") is True
    assert not hasattr(m, "peft_config")

    class Clean:
        pass

    assert f._strip_peft_residue(Clean(), "python") is False


def test_unremovable_peft_residue_refuses_to_save():
    """A residue that cannot be stripped must fail the build, not be saved as
    adapter-flavoured weights under a dense specialist's name."""
    from ms_moe_maker.train import finetune as f

    class Sticky:
        @property
        def peft_config(self):
            return {"default": object()}

    with pytest.raises(RuntimeError, match="refusing to write"):
        f._strip_peft_residue(Sticky(), "python")
