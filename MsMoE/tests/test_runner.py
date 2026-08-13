"""The runner, driven end to end against a FAKE pipeline.

No torch, no GPU, no 23-hour wait. The fake prints the exact lines the real
`fraunkenstein_universal.py` prints - quoted from its source, not invented -
and the runner is required to turn them into the right stage transitions.

This is the shakedown-cruise argument applied to the wrapper: the point of a
fake pipeline is that every mapping bug gets caught in 0.2 seconds instead of
six hours into a real build, at the exact moment you least want to discover
that your dashboard has been lying since stage three.

The fake is also the honest test of the coupling. Every string below is a
promise the real script currently keeps. When the carve starts changing those
prints into structured output, THESE tests are what fail first, which is
precisely where you want to be told.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ms_moe import manifest as mf
from ms_moe import stages as st
from ms_moe.events import Events
from ms_moe.levers import Translation
from ms_moe.runner import Runner


# -- a stand-in recipe -------------------------------------------------------

class _Budget:
    target_steps = 150


class _MoE:
    dense_layers = "auto"


class _Runtime:
    direct_load = False
    alloc_conf = ""


class _Expert:
    def __init__(self, name):
        self.name = name


class FakeRecipe:
    name = "test-recipe"
    size = "0.5B"
    base = "Qwen/Qwen2.5-Coder-0.5B"
    budget = _Budget()
    moe = _MoE()
    runtime = _Runtime()

    def __init__(self, experts):
        self.experts = [_Expert(e) for e in experts]

    def recipe_id(self):
        return "deadbeef"


# The lines the real pipeline prints, in order. Quoted from
# fraunkenstein_universal.py - see the module docstring.
SCRIPT = r'''
import sys
print("[cfg] batch=4x2 (eff 8)  lora_r=64")
print("[cfg] rung: size=0.5B  target_steps=150")
print("[skip] Python dataset already present at /x/dryrun_data/python_code.jsonl")
print("Agent dataset ready: 2000 samples")
print("")
print("Fine-tuning python...")
print("some tqdm noise 100%|xxx| 579/579 [00:01<00:00, 2398it/s]")
print("Dense specialist saved to /x/dryrun_0.5B/qwen_coder_python")
print("Fine-tuning csharp...")
print("Dense specialist saved to /x/dryrun_0.5B/qwen_coder_csharp")
print("")
print("Stitching 2 experts into MoE...")
print("MoE skeleton saved to /x/dryrun_0.5B/fraunkenstein_moe_untrained")
print("   router-only training: 1,234,567 trainable params")
print("Fraunkenstein Agent MoE is ALIVE! Saved to /x/dryrun_0.5B/final")
print("")
print("Exporting GGUF -> /x/dryrun_0.5B/model.gguf")
print("   converted OK (3.78 GB)")
print("smoke test PASSED")
sys.exit(0)
'''


@pytest.fixture
def lab(tmp_path):
    """A directory with a fake pipeline in it."""
    script = tmp_path / "fraunkenstein_universal.py"
    script.write_text(SCRIPT, encoding="utf-8")
    return script


def run(lab, experts=("python", "csharp"), capsys=None, **kw):
    recipe = FakeRecipe(list(experts))
    ev = Events(enabled=False)
    runner = Runner(recipe, lab, Translation(), ev, cwd=lab.parent,
                    dryrun=True, **kw)
    code = runner.run()
    return code, runner


# -- the happy path ----------------------------------------------------------

def test_a_clean_run_exits_zero_and_writes_a_manifest(lab):
    code, runner = run(lab)
    assert code == 0
    written = mf.read(runner.run_dir)
    assert written is not None
    assert written.ok is True
    assert written.finished is not None


def test_every_stage_reaches_a_terminal_state(lab):
    _, runner = run(lab)
    m = mf.read(runner.run_dir)
    unfinished = [s.id for s in m.stages if s.status not in mf.TERMINAL]
    assert unfinished == [], f"stages left hanging: {unfinished}"


def test_each_expert_gets_its_own_stage_and_artifact(lab):
    _, runner = run(lab)
    m = mf.read(runner.run_dir)
    for expert in ("python", "csharp"):
        stage = m.stage(st.finetune_id(expert))
        assert stage is not None, f"no stage for {expert}"
        assert stage.status == mf.DONE
        assert stage.artifact == f"qwen_coder_{expert}", (
            "the artifact should be stored RELATIVE to the run dir so the "
            "manifest survives the directory being moved or mounted elsewhere")


def test_a_skipped_dataset_reports_skipped_not_done(lab):
    """`_done()` firing means the artifact was already there. That is a
    different fact from 'we just built it' and the manifest keeps them apart -
    a resumed run should not look like a fresh one."""
    _, runner = run(lab)
    stage = mf.read(runner.run_dir).stage(st.DATA_CODE)
    assert stage.status == mf.SKIPPED
    assert "already present" in (stage.note or "")


def test_converted_is_not_proven_until_the_smoke_test(lab, tmp_path):
    """The pipeline learned this the hard way: a GGUF that converts and then
    hangs its smoke test would be marked finished forever."""
    # Truncate the script right after "converted OK".
    script = tmp_path / "fraunkenstein_universal.py"
    body = SCRIPT.split('print("smoke test PASSED")')[0] + "sys.exit(1)\n"
    script.write_text(body, encoding="utf-8")
    code, runner = run(script)
    assert code != 0
    stage = mf.read(runner.run_dir).stage(st.EXPORT_GGUF)
    assert stage.status != mf.DONE, (
        "GGUF was marked done on conversion alone - conversion is not proof")
    assert "smoke test pending" in (stage.note or "")


# -- failure is reported as failure ------------------------------------------

def test_a_crashing_pipeline_marks_the_running_stage_failed(tmp_path):
    script = tmp_path / "fraunkenstein_universal.py"
    script.write_text(
        'print("Fine-tuning python...")\n'
        'print("Traceback (most recent call last)")\n'
        'import sys; sys.exit(1)\n', encoding="utf-8")
    code, runner = run(script, experts=("python",))
    assert code == 1
    m = mf.read(runner.run_dir)
    stage = m.stage(st.finetune_id("python"))
    assert stage.status == mf.FAILED
    assert m.ok is False


def test_a_stage_never_reported_is_not_claimed_as_done(tmp_path):
    """Silence is not success. If the pipeline finished without ever mentioning
    a stage we do not get to say it happened."""
    script = tmp_path / "fraunkenstein_universal.py"
    script.write_text('print("Fine-tuning python...")\n'
                      'print("Dense specialist saved to /x/qwen_coder_python")\n',
                      encoding="utf-8")
    _, runner = run(script, experts=("python",))
    m = mf.read(runner.run_dir)
    stitch = m.stage(st.STITCH)
    assert stitch.status == mf.PENDING
    assert stitch.note == "never reported by the pipeline"


# -- the event stream --------------------------------------------------------

def test_json_events_are_one_object_per_line(lab, capsys):
    recipe = FakeRecipe(["python", "csharp"])
    ev = Events(enabled=True)
    Runner(recipe, lab, Translation(), ev, cwd=lab.parent, dryrun=True).run()
    out = capsys.readouterr().out
    kinds = []
    for line in out.splitlines():
        if not line.strip():
            continue
        kinds.append(json.loads(line)["event"])   # raises if prose leaked
    assert kinds[0] == "started"
    assert kinds[-1] == "done"
    assert "stage" in kinds


def test_the_child_log_goes_to_stderr_not_the_event_stream(lab, capsys):
    """A wrapper that swallows the log it is summarising makes debugging
    strictly harder than not having the wrapper."""
    recipe = FakeRecipe(["python"])
    Runner(recipe, lab, Translation(), Events(enabled=True),
           cwd=lab.parent, dryrun=True).run()
    captured = capsys.readouterr()
    assert "Fine-tuning python..." in captured.err
    assert "Fine-tuning python..." not in captured.out


# -- refusals ride along ------------------------------------------------------

def test_refusals_are_recorded_in_the_manifest(lab):
    """The person who needs to know a lever was ignored is the one reading the
    dashboard six hours later, not the one who saw the terminal at kickoff."""
    recipe = FakeRecipe(["python"])
    tr = Translation(refusals=["gates.main_evals='manual' cannot be honoured"])
    runner = Runner(recipe, lab, tr, Events(enabled=False), cwd=lab.parent,
                    dryrun=True)
    runner.run()
    assert mf.read(runner.run_dir).refusals == tr.refusals
