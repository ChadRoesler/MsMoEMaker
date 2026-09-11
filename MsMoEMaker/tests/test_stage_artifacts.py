"""Where each stage put what it made, in the manifest, on the in-process path.

THE FIELD WAS DECLARED, DOCUMENTED, WRITTEN, PARSED, SHIPPED AND DRAWN - AND
ALWAYS BLANK. `manifest.Stage.artifact` carries a docstring about surviving the
run directory being moved or read through a mount, `manifest.py` parses it,
seren_theatre ships it in /api/state, and `scripts.js` renders
`artifact <code>...</code>` on the stage row.

`builder.py` computed every one of those paths and recorded them in
`BuildResult.artifacts` at ten sites. And `StageCallback.notify(stage_id, status,
note)` had no fourth parameter, so the callback dropped them; nothing copied that
dict into the manifest afterwards either. The producer computed it, the consumer
drew it, and the wire between them was three arguments wide.

Which is why the tests that matter here are the WIRING ones - a widened callback
that nothing passes an artifact to would be the same defect with a longer
signature. TestItReachesTheManifest drives a real Runner.

ONE HONEST LIMIT, ON PURPOSE. `BuildResult.artifacts` is looser than its own
`stage_id -> path` comment: the expert gate stores a status word there and a
skipped GGUF export stores the sentence "skipped (no llama.cpp)". Those are not
paths and are not passed, because Stage.artifact promises one. See
TestOnlyRealPathsTravel.
"""
from __future__ import annotations

from ms_moe_maker.config.levers import Translation
from ms_moe_maker.run import builder as bld
from ms_moe_maker.run import manifest as mf
from ms_moe_maker.run import stages as st
from ms_moe_maker.run.events import Events
from ms_moe_maker.run.runner import Runner


class _Budget:
    target_steps = 10
    token_budget_per_expert = 1000


class _MoE:
    dense_layers = 2


class _Runtime:
    pass


class _Roots:
    data = "d/{size}"
    output = "o/{size}"


class _Expert:
    def __init__(self, name):
        self.name = name
        self.source = type("S", (), {"kind": "hf", "id": "z"})()


class FakeRecipe:
    name = "test-recipe"
    size = "0.5B"
    base = "Qwen/Qwen2.5-Coder-0.5B"
    budget = _Budget()
    moe = _MoE()
    runtime = _Runtime()
    roots = _Roots()

    def __init__(self, experts=("python",)):
        self.experts = [_Expert(e) for e in experts]

    def recipe_id(self):
        return "deadbeef"


class TestTheCallbackCarriesIt:
    def test_the_listener_is_handed_the_artifact(self):
        seen = []
        cb = bld.StageCallback(notify=lambda s, st_, n, a: seen.append(a))
        cb.stage(st.ROUTER, mf.DONE, "trained", artifact="/x/moe_trained")
        assert seen == ["/x/moe_trained"]

    def test_a_stage_with_nothing_to_show_hands_over_none(self):
        seen = []
        cb = bld.StageCallback(notify=lambda s, st_, n, a: seen.append(a))
        cb.stage(st.PREFLIGHT, mf.DONE)
        assert seen == [None]


class TestItReachesTheManifest:
    """THE WIRE, END TO END. `_on_stage` is a closure inside `run_builder`, so
    the only honest way at it is to run that method - which also exercises
    `_relative`, the seam where an absolute builder path becomes the relative one
    the manifest promises."""

    def _run(self, tmp_path, monkeypatch, emit):
        recipe = FakeRecipe()
        runner = Runner(recipe, None, Translation(), Events(enabled=False),
                        cwd=tmp_path, dryrun=True)

        def fake_pipeline(_recipe, force=False, dryrun=False, callback=None,
                          only=()):
            emit(callback, runner.run_dir)
            out = bld.BuildResult()
            out.ok = True
            out.message = "done"
            return out

        monkeypatch.setattr(bld, "run_pipeline", fake_pipeline)
        runner.run_builder()
        return runner

    def test_an_absolute_builder_path_lands_relative(self, tmp_path,
                                                     monkeypatch):
        def emit(cb, run_dir):
            cb.stage(st.ROUTER, mf.DONE, "trained",
                     artifact=str(run_dir / "moe_trained"))

        runner = self._run(tmp_path, monkeypatch, emit)
        stage = runner.manifest.stage(st.ROUTER)
        assert stage is not None
        assert stage.artifact == "moe_trained", (
            f"expected a path relative to the run dir, got {stage.artifact!r} - "
            f"one absolute path in there and the manifest stops surviving being "
            f"read through a mount")

    def test_every_specialist_records_its_own_directory(self, tmp_path,
                                                        monkeypatch):
        def emit(cb, run_dir):
            for name in ("python", "csharp"):
                cb.stage(f"finetune.{name}", mf.DONE, "saved",
                         artifact=str(run_dir / f"specialist_{name}"))

        runner = self._run(tmp_path, monkeypatch, emit)
        got = {s.id: s.artifact for s in runner.manifest.stages
               if s.id.startswith("finetune.")}
        assert got.get("finetune.python") == "specialist_python"
        assert got.get("finetune.csharp") == "specialist_csharp"

    def test_a_later_transition_does_not_blank_it(self, tmp_path, monkeypatch):
        """`_set` does setattr, so passing artifact=None on a later transition
        of the same stage would erase what an earlier one recorded. A stage that
        reports RUNNING after DONE is unusual; a stage that loses its artifact
        because of it is a silent hole."""
        def emit(cb, run_dir):
            cb.stage(st.STITCH, mf.DONE, "skeleton",
                     artifact=str(run_dir / "moe_untrained"))
            cb.stage(st.STITCH, mf.RUNNING, "second thoughts")

        runner = self._run(tmp_path, monkeypatch, emit)
        assert runner.manifest.stage(st.STITCH).artifact == "moe_untrained"

    def test_it_survives_the_round_trip_to_disk(self, tmp_path, monkeypatch):
        """The manifest is what Theatre reads, so the field has to be there
        after a write and a read - not just on the in-memory object."""
        def emit(cb, run_dir):
            cb.stage(st.ROUTER, mf.DONE, "trained",
                     artifact=str(run_dir / "moe_trained"))

        runner = self._run(tmp_path, monkeypatch, emit)
        back = mf.read(runner.run_dir)
        assert back is not None
        assert back.stage(st.ROUTER).artifact == "moe_trained"


class TestOnlyRealPathsTravel:
    """`BuildResult.artifacts` is looser than Stage.artifact, and the difference
    is not a rounding error: two of its ten entries are prose."""

    def test_a_stage_that_reports_prose_keeps_it_as_a_note(self, tmp_path,
                                                           monkeypatch):
        recipe = FakeRecipe()
        runner = Runner(recipe, None, Translation(), Events(enabled=False),
                        cwd=tmp_path, dryrun=True)

        def fake_pipeline(_recipe, force=False, dryrun=False, callback=None,
                          only=()):
            # The real builder's shape for a skipped export: a sentence in the
            # NOTE and no artifact, because there is no file.
            callback.stage(st.EXPORT_GGUF, mf.WARNED,
                           "GGUF export skipped (no llama.cpp)")
            out = bld.BuildResult()
            out.ok = True
            return out

        monkeypatch.setattr(bld, "run_pipeline", fake_pipeline)
        runner.run_builder()
        stage = runner.manifest.stage(st.EXPORT_GGUF)
        assert stage.artifact is None, (
            f"a sentence reached a field documented as a path: "
            f"{stage.artifact!r}")
        assert "llama.cpp" in (stage.note or "")

    def test_the_builder_passes_a_path_for_every_artifact_it_names(self):
        """READ OFF THE REAL SOURCE. The five stages that produce ONE file each
        must hand it over; the ones whose `artifacts` entry is prose or a
        comma-joined list must not. Checked against builder.py itself rather
        than against a list retyped here, because a retyped list agrees with
        itself and this is the pairing that already drifted once.
        """
        import inspect
        import re

        src = inspect.getsource(bld.run_pipeline)
        # Every cb.stage(...) call that passes an artifact, and every one that
        # records into result.artifacts, by stage expression.
        passes = set(re.findall(r"artifact=(\w+)", src))
        assert {"ablated_dir", "out_dir", "moe_dir", "router_dir",
                "gguf_path"} <= passes, (
            f"a single-file stage stopped naming its artifact: {sorted(passes)}")
        # And the prose entries are NOT passed as artifacts anywhere.
        assert "artifact=gate.status" not in src
        assert 'artifact="skipped' not in src
