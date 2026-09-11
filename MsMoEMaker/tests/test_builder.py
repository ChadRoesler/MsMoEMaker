"""Tests for ms_moe_maker/builder.py — pipeline orchestrator."""

import pytest
from ms_moe_maker.run import builder
from ms_moe_maker.config import pipeline as config


class TestStageCallback:
    """Test the StageCallback progress reporter."""

    def test_stage_records_status(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "running", "starting scan")
        info = cb._stages.get("data.corpus", {})
        assert info["status"] == "running"
        assert info["note"] == "starting scan"

    def test_stage_overwrites_status(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "running", "starting")
        cb.stage("data.corpus", "done", "scan complete")
        info = cb._stages.get("data.corpus", {})
        assert info["status"] == "done"
        assert info["note"] == "scan complete"

    def test_notify_callback_called(self):
        notified = []
        cb = builder.StageCallback(
            notify=lambda s, st, n, a: notified.append((s, st, n, a)))
        cb.stage("data.corpus", "running", "hello")
        assert len(notified) == 1
        assert notified[0] == ("data.corpus", "running", "hello", None)

    def test_notify_callback_for_done(self):
        notified = []
        cb = builder.StageCallback(
            notify=lambda s, st, n, a: notified.append((s, st, n, a)))
        cb.stage("data.corpus", "done", "finished")
        assert notified[0] == ("data.corpus", "done", "finished", None)

    def test_the_artifact_reaches_the_listener(self):
        """THE FOURTH ARGUMENT, AND THE REASON IT EXISTS.

        The pipeline knew every artifact path and recorded it in
        BuildResult.artifacts; this callback had nowhere to put one, and nothing
        copied that dict into the manifest afterwards. So manifest.Stage.artifact
        - declared, documented, parsed by Theatre, shipped in /api/state and
        drawn on the card - was blank on every in-process build.
        """
        notified = []
        cb = builder.StageCallback(
            notify=lambda s, st, n, a: notified.append((s, st, n, a)))
        cb.stage("stitch", "done", "skeleton", artifact="/runs/0.5B/moe_untrained")
        assert notified[0][3] == "/runs/0.5B/moe_untrained"
        assert cb._stages["stitch"]["artifact"] == "/runs/0.5B/moe_untrained"

    def test_a_stage_with_no_artifact_reports_none_not_an_empty_string(self):
        """None and "" are different answers. A stage that produced nothing has
        no artifact; a stage whose artifact came back empty is a bug upstream,
        and the manifest reader has to be able to tell them apart."""
        notified = []
        cb = builder.StageCallback(
            notify=lambda s, st, n, a: notified.append(a))
        cb.stage("preflight", "done")
        assert notified == [None]
        assert "artifact" not in cb._stages["preflight"]

    def test_multi_stage_tracking(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "running")
        cb.stage("finetune.python", "running")
        cb.stage("data.corpus", "done")
        assert len(cb._stages) == 2
        assert cb._stages["data.corpus"]["status"] == "done"
        assert cb._stages["finetune.python"]["status"] == "running"

    def test_no_notify_when_not_provided(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "running")
        info = cb._stages.get("data.corpus", {})
        assert info["status"] == "running"

    def test_empty_note_not_stored(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "done", "")
        info = cb._stages.get("data.corpus", {})
        assert info["note"] == ""

    def test_pending_initial_status(self):
        cb = builder.StageCallback()
        cb.stage("data.corpus", "pending")
        info = cb._stages.get("data.corpus", {})
        assert info["status"] == "pending"
        assert info["note"] == ""


class TestBuildResult:
    """Test the BuildResult data container."""

    def test_build_result_default(self):
        result = builder.BuildResult()
        assert result.ok is False
        assert result.message == ""
        assert result.failed_stage is None
        assert result.stages_completed == []
        assert result.artifacts == {}

    def test_build_result_set_success(self):
        result = builder.BuildResult()
        result.ok = True
        result.message = "All stages completed"
        result.stages_completed.append("data.corpus")
        assert result.ok is True
        assert "data.corpus" in result.stages_completed

    def test_build_result_set_failed(self):
        result = builder.BuildResult()
        result.ok = False
        result.failed_stage = "finetune.python"
        result.message = "Training failed"
        assert result.ok is False
        assert result.failed_stage == "finetune.python"

    def test_build_result_artifacts(self):
        result = builder.BuildResult()
        result.artifacts["data.corpus"] = "/path/to/data.jsonl"
        assert result.artifacts["data.corpus"] == "/path/to/data.jsonl"

    def test_build_result_multi_stages(self):
        result = builder.BuildResult()
        result.ok = True
        result.stages_completed = ["data.corpus", "finetune.python", "stitch"]
        assert len(result.stages_completed) == 3


class TestPipelineStagesIntegration:
    """Test that the builder tracks stages through callbacks."""

    def test_full_pipeline_simulation(self):
        """Simulate what run_pipeline does with callbacks."""
        cb = builder.StageCallback()

        # Preflight
        cb.stage("preflight", "running")
        cb.stage("preflight", "done", "config stamped")

        # Data corpus
        cb.stage("data.corpus", "running", "collecting expert corpora")
        cb.stage("data.corpus", "done", "collected 2 corpora")

        # Finetune specialists
        cb.stage("finetune.python", "running", "training")
        cb.stage("finetune.python", "done", "trained")
        cb.stage("finetune.csharp", "running", "training")
        cb.stage("finetune.csharp", "done", "trained")

        # Stitch
        cb.stage("stitch", "running", "stitching 2 experts")
        cb.stage("stitch", "done", "stitched")

        # Router
        cb.stage("router", "running", "training router")
        cb.stage("router", "done", "trained router")

        # Verify all stages tracked (6 stages: preflight, data.corpus, finetune.python, finetune.csharp, stitch, router)
        assert len(cb._stages) == 6

        # Verify terminal states
        for stage_name in cb._stages:
            assert cb._stages[stage_name]["status"] in ("done", "running")

        # Verify specific stages
        assert cb._stages["preflight"]["status"] == "done"
        assert cb._stages["data.corpus"]["status"] == "done"
        assert cb._stages["stitch"]["status"] == "done"
        assert cb._stages["router"]["status"] == "done"

    def test_failed_stage_simulation(self):
        """Simulate a pipeline failure."""
        cb = builder.StageCallback()
        result = builder.BuildResult()

        cb.stage("data.corpus", "running")
        cb.stage("data.corpus", "failed", "no corpora found")

        result.failed_stage = "data.corpus"
        result.message = "No corpora were collected"
        result.ok = False

        assert result.ok is False
        assert result.failed_stage == "data.corpus"
        assert cb._stages["data.corpus"]["status"] == "failed"

    def test_skipped_synth_stage(self):
        """Simulate a skipped synth stage."""
        cb = builder.StageCallback()
        cb.stage("data.synth", "skipped", "agent traces already present")
        info = cb._stages.get("data.synth", {})
        assert info["status"] == "skipped"


class TestConfigConstants:
    """Test that builder imports and uses config constants correctly."""

    def test_display_lang_accessible(self):
        """DISPLAY_LANG should be importable from config."""
        from ms_moe_maker.config.pipeline import DISPLAY_LANG
        assert isinstance(DISPLAY_LANG, dict)
        assert "python" in DISPLAY_LANG

    def test_model_sizes_accessible(self):
        from ms_moe_maker.config.pipeline import MODEL_SIZES
        assert isinstance(MODEL_SIZES, dict)
        assert "0.5B" in MODEL_SIZES

    def test_code_languages_accessible(self):
        from ms_moe_maker.config.pipeline import CODE_LANGUAGES
        assert isinstance(CODE_LANGUAGES, list)
        assert len(CODE_LANGUAGES) == 4
