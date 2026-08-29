"""A recipe names a build; it does not determine one.

`recipe_id` excludes `runtime`, so it never did — and once defaults moved into
a file on the box, the values that decide what gets trained can live somewhere
the recipe never mentions. These tests pin the three things that keep that
honest: a build id over the RESOLVED config, the fingerprint stored so a
drifted resume can name the field, and a refusal before finished artifacts get
inherited under settings that have since changed.
"""
import json
from pathlib import Path

import pytest

from ms_moe_maker.config import pipeline as C
from ms_moe_maker.run import manifest as mf
from ms_moe_maker.config import recipe as R


def _rec(**extra):
    body = {"schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]}
    body.update(extra)
    rec, _ = R.parse(body)
    return rec


class TestBuildFingerprint:
    def test_the_same_resolved_config_is_the_same_id(self):
        a = C.build_config(_rec(), dryrun=False)
        b = C.build_config(_rec(), dryrun=False)
        assert C.build_id(a) == C.build_id(b)

    def test_a_knob_that_changes_the_build_changes_the_id(self):
        a = C.build_config(_rec(), dryrun=False)
        b = C.build_config(_rec(budget={"target_steps": 999}), dryrun=False)
        assert C.build_id(a) != C.build_id(b)

    def test_the_name_does_not_change_the_id(self):
        """_auto_name embeds a timestamp. A fingerprint that moves every time
        you run it fingerprints nothing."""
        a = C.build_config(_rec(name="one"), dryrun=False)
        b = C.build_config(_rec(name="two"), dryrun=False)
        assert C.build_id(a) == C.build_id(b)

    def test_force_does_not_change_the_id(self):
        a = C.build_config(_rec(), force=False, dryrun=False)
        b = C.build_config(_rec(), force=True, dryrun=False)
        assert C.build_id(a) == C.build_id(b)

    def test_the_fingerprint_is_fail_closed(self):
        """Every PipelineConfig field is IN unless it is on the exclusion list.
        A hand-picked include list is how a fingerprint quietly stops
        fingerprinting the year somebody adds a field and forgets."""
        import dataclasses
        cfg = C.build_config(_rec(), dryrun=False)
        declared = {f.name for f in dataclasses.fields(cfg)}
        covered = set(C.build_fingerprint(cfg))
        assert covered == declared - C._FINGERPRINT_EXCLUDE
        assert C._FINGERPRINT_EXCLUDE <= declared, (
            "the exclusion list names fields that no longer exist")

    def test_the_box_moves_the_id(self, tmp_path):
        """The whole point: same recipe, different box, different build."""
        box = tmp_path / "box.yaml"
        box.write_text("budget:\n  target_steps: 777\n", encoding="utf-8")
        r = tmp_path / "r.json"
        r.write_text(json.dumps({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}],
        }), encoding="utf-8")
        plain, _ = R.load(str(r), include_user_defaults=False)
        boxed, _ = R.load(str(r), defaults_path=str(box),
                          include_user_defaults=False)
        assert plain.recipe_id() == boxed.recipe_id(), (
            "recipe_id is about the recipe and must NOT move")
        assert (C.build_id(C.build_config(plain, dryrun=False))
                != C.build_id(C.build_config(boxed, dryrun=False))), (
            "build_id is about the build and MUST move")

    def test_the_diff_names_the_field(self):
        a = C.build_fingerprint(C.build_config(_rec(), dryrun=False))
        b = C.build_fingerprint(
            C.build_config(_rec(budget={"target_steps": 999}), dryrun=False))
        diff = dict((k, (was, now)) for k, was, now in C.fingerprint_diff(a, b))
        assert "target_steps" in diff
        assert diff["target_steps"][1] == 999


class TestManifestCarriesTheBuild:
    def test_the_new_fields_round_trip(self, tmp_path):
        m = mf.Manifest(recipe_id="abc", build_id="deadbeef",
                        resolved={"target_steps": 400},
                        defaults_files={"/box.yaml": "aaaa1111"})
        mf.write(tmp_path, m)
        back = mf.read(tmp_path)
        assert back.build_id == "deadbeef"
        assert back.resolved["target_steps"] == 400
        assert back.defaults_files["/box.yaml"] == "aaaa1111"

    def test_an_old_manifest_still_reads(self, tmp_path):
        """Additive only — seren-theatre has an independent reader for this
        format and a schema bump would blind it."""
        (tmp_path / mf.MANIFEST_NAME).write_text(json.dumps({
            "schema_version": 1, "recipe_id": "abc", "name": "t",
            "size": "0.5B", "base": "", "experts": ["a"], "stages": [],
        }), encoding="utf-8")
        back = mf.read(tmp_path)
        assert back.build_id == ""
        assert back.resolved == {} and back.defaults_files == {}
        assert mf.SCHEMA_VERSION == 1, "these fields must not need a bump"


class TestDrift:
    """THE FAILURE THIS SLICE EXISTS FOR.

    Stages self-skip on artifacts found on disk. Build at 400 steps, stop
    halfway, change the number — in the recipe, or now just as easily in a
    defaults file three directories away — and resume: the finished expert is
    kept at 400 while the rest train at 1200, and nothing says a word.
    """

    def _runner(self, tmp_path, rec):
        from ms_moe_maker.run.events import Events
        from ms_moe_maker.config.levers import translate
        from ms_moe_maker.run.runner import Runner
        return Runner(rec, None, translate(rec), Events(enabled=False),
                      cwd=tmp_path, builder=True)

    def _seed(self, run_dir, build_id, resolved, finished=("data.corpus",)):
        run_dir.mkdir(parents=True, exist_ok=True)
        mf.write(run_dir, mf.Manifest(
            recipe_id="abc", build_id=build_id, resolved=resolved,
            stages=[mf.Stage(id=s, label=s, status=mf.DONE) for s in finished]))

    def test_an_unchanged_resume_is_silent(self, tmp_path):
        r = self._runner(tmp_path, _rec())
        self._seed(r.run_dir, r.manifest.build_id, r.manifest.resolved)
        assert r.drift() == ([], [])

    def test_a_changed_build_is_reported_with_the_field_and_the_stages(
            self, tmp_path):
        r = self._runner(tmp_path, _rec(budget={"target_steps": 1200}))
        old = dict(r.manifest.resolved)
        old["target_steps"] = 400
        self._seed(r.run_dir, "0000deadbeef", old, finished=("data.corpus",))
        changed, finished = r.drift()
        assert finished == ["data.corpus"]
        assert any("target_steps" in c and "400" in c and "1200" in c
                   for c in changed), changed

    def test_a_fresh_directory_is_not_drift(self, tmp_path):
        r = self._runner(tmp_path, _rec())
        assert r.drift() == ([], [])

    def test_a_changed_defaults_file_is_named(self, tmp_path):
        r = self._runner(tmp_path, _rec())
        r.manifest.defaults_files = {"/box.yaml": "nnnn2222"}
        old = dict(r.manifest.resolved)
        old["target_steps"] = 400
        self._seed(r.run_dir, "0000deadbeef", old)
        r.manifest.build_id = "1111feedface"
        # rewrite the seeded manifest with the OLD digest
        prev = mf.read(r.run_dir)
        prev.defaults_files = {"/box.yaml": "oooo1111"}
        mf.write(r.run_dir, prev)
        changed, _ = r.drift()
        assert any("/box.yaml" in c and "oooo1111" in c and "nnnn2222" in c
                   for c in changed), changed


class TestDescribeSurfacesTheBox:
    """--describe is how Starwright and Theatre ask a machine what it offers.

    Reporting the built-in floor there would describe a different install than
    the one answering — so it reports what THIS box resolves, defensively
    enough that a broken file can never take the one-line-JSON contract down.
    """

    def _describe(self, tmp_path=None):
        """--describe in a subprocess, with the USER layer pointed at nothing.

        `--describe` deliberately includes the box's own files - that is the
        whole point of it - which makes it the one thing in this suite that
        could fail because of a file in somebody's home directory. A test that
        depends on whose laptop it runs on is not a test, so it gets the same
        isolation everything else here gets: point the user layer at a path
        that does not exist.
        """
        import json
        import os
        import subprocess
        import sys
        env = dict(os.environ)
        env["MSMOE_DEFAULTS"] = "/nonexistent/msmoe-defaults.yaml"
        env["MSMOE_REASONING"] = "/nonexistent/msmoe-reasoning.yaml"
        out = subprocess.run([sys.executable, "-m", "ms_moe_maker", "--describe"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    def test_the_contract_still_holds(self):
        d = self._describe()
        assert d["requires"] == []
        assert list(d["modes"]) == list(d["eval_modes"])
        assert d["manifest_schema_version"] == 1

    def test_it_reports_the_layers_and_what_they_set(self):
        d = self._describe()
        labels = [l["label"] for l in d["defaults"]["layers"]]
        assert labels[:2] == ["packaged", "user"]
        packaged = d["defaults"]["layers"][0]
        assert packaged["present"] is True and packaged["sha256"]
        assert "tools_expert" in d["defaults"]["blocks"]

    def test_it_reports_the_reasoning_table(self):
        d = self._describe()
        keys = {s["key"] for s in d["reasoning"]["styles"]}
        assert {"xml", "agentic_xml"} <= keys
        assert d["reasoning"]["warnings"] == []
        assert any(s["interwoven"] for s in d["reasoning"]["styles"])

    def test_defaults_is_a_declared_event_kind(self):
        """Additive by the rule in _describe: a consumer that does not know a
        kind ignores it. Removing or renaming one is the breaking change."""
        d = self._describe()
        assert "defaults" in d["events"]
        for older in ("started", "stage", "progress", "refused", "warning",
                      "error", "done"):
            assert older in d["events"], f"{older} must not disappear"


class TestExpertSeedDerivation:
    """`seed` is in the fingerprint, so build_id claims two runs trained the
    same way. These pin the part of that claim the training path can keep:
    the per-expert stream is a function of (seed, expert) and nothing else."""

    def test_two_experts_do_not_share_a_stream(self):
        from ms_moe_maker.train import finetune as F
        assert F.expert_seed(42, "python") != F.expert_seed(42, "rust")

    def test_a_different_build_seed_moves_every_stream(self):
        from ms_moe_maker.train import finetune as F
        assert F.expert_seed(42, "python") != F.expert_seed(43, "python")

    def test_the_same_pair_is_the_same_number(self):
        from ms_moe_maker.train import finetune as F
        assert F.expert_seed(42, "python") == F.expert_seed(42, "python")

    def test_it_survives_a_different_interpreter(self):
        """The reason it is sha256 and not hash(): the builtin is salted per
        process, so a hash()-derived seed would differ between the run that
        stamped build_id and the run that resumed it."""
        import os
        import subprocess
        import sys
        code = ("from ms_moe_maker.train import finetune as F;"
                "print(F.expert_seed(42, 'python'))")
        vals = set()
        for salt in ("0", "1", "12345"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = salt
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, env=env)
            assert out.returncode == 0, out.stderr
            vals.add(out.stdout.strip())
        assert len(vals) == 1, f"seed moved with PYTHONHASHSEED: {vals}"


class TestStaleCheckpointsAreNotResumed:
    """--force plus a leftover tmp_{expert} used to resume a schedule that was
    already over, take zero optimiser steps, and save an untrained adapter as
    'done'. These cover the filesystem half, which needs no torch."""

    def _ckpt(self, tmp_path, step, global_step=None):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if global_step is not None:
            (d / "trainer_state.json").write_text(
                json.dumps({"global_step": global_step}))
        return d

    def test_latest_is_the_highest_step_not_the_last_alphabetically(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        self._ckpt(tmp_path, 9)
        self._ckpt(tmp_path, 100)
        assert Path(F.latest_checkpoint(str(tmp_path))).name == "checkpoint-100"

    def test_nothing_there_is_none_not_a_crash(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        assert F.latest_checkpoint(str(tmp_path)) is None
        assert F.latest_checkpoint(str(tmp_path / "never-made")) is None
        assert F.clear_checkpoints(str(tmp_path / "never-made")) == []

    def test_a_dir_that_only_looks_like_a_checkpoint_is_not_one(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        (tmp_path / "checkpoint-notes").mkdir()
        assert F.latest_checkpoint(str(tmp_path)) is None

    def test_it_reads_the_step_a_resume_would_start_at(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        d = self._ckpt(tmp_path, 200, global_step=200)
        assert F.checkpoint_global_step(str(d)) == 200

    def test_an_unreadable_state_reads_as_zero(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        bare = self._ckpt(tmp_path, 5)
        assert F.checkpoint_global_step(str(bare)) == 0
        (bare / "trainer_state.json").write_text("{not json")
        assert F.checkpoint_global_step(str(bare)) == 0

    def test_clearing_removes_the_checkpoints_and_only_those(self, tmp_path):
        from ms_moe_maker.train import finetune as F
        self._ckpt(tmp_path, 100, global_step=100)
        self._ckpt(tmp_path, 200, global_step=200)
        (tmp_path / "runs").mkdir()
        (tmp_path / "note.txt").write_text("mine")
        assert F.clear_checkpoints(str(tmp_path)) == ["checkpoint-100",
                                                      "checkpoint-200"]
        assert F.latest_checkpoint(str(tmp_path)) is None
        assert (tmp_path / "runs").is_dir()
        assert (tmp_path / "note.txt").read_text() == "mine"


class TestTheSeedActuallySeedsSomething:
    """`seed` is not in _FINGERPRINT_EXCLUDE, so it is hashed into build_id.

    Nothing in data/synth.py read it. The generators drew from the global,
    unseeded `random`, so three runs produced three different tool surfaces
    while advertising the same build id - and an id that does not predict the
    data is worse than no id at all.
    """

    def _cfg(self, seed=42):
        import types
        return types.SimpleNamespace(seed=seed)

    def _surface(self, cfg, expert="agentcore"):
        from ms_moe_maker.data import synth as d
        rnd = d._expert_rng(cfg, expert)
        return [sorted(t["name"] for t in tools)
                for _, tools, _, _ in d._agent_batch_specs(6, None, "sys", rnd)]

    def test_the_same_seed_generates_the_same_tool_surface(self):
        assert self._surface(self._cfg()) == self._surface(self._cfg())

    def test_a_different_seed_generates_a_different_one(self):
        assert self._surface(self._cfg(1)) != self._surface(self._cfg(2))

    def test_two_experts_do_not_get_identical_draws(self):
        """One shared stream would make each expert's data depend on which
        expert ran first; one seed reused per expert would hand both the same
        surface."""
        cfg = self._cfg()
        assert self._surface(cfg, "agentcore") != self._surface(cfg, "toolsy")

    def test_the_generators_do_not_reach_for_the_global_random(self):
        """The module RNG is process-wide and nothing seeds it, so a single
        call anywhere in the generation path un-does the whole thing."""
        import inspect

        from ms_moe_maker.data import synth as d
        src = inspect.getsource(d._agent_batch_specs)
        assert "random.randint" not in src
        assert "random.sample" not in src
        assert "random.choice" not in src
