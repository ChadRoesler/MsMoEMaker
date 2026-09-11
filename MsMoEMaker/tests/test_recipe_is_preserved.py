"""The run's own copy of the recipe that produced it.

WHAT THE RUN DIRECTORY COULD NOT SAY. The manifest stamps a sha256 of every
DEFAULTS file a run inherits, plus `resolved` (the whole config fingerprint) and
`build_id` (a digest of it). So a finished run could tell you, exactly, the
fingerprint of the configuration it resolved to - and nothing at all about the
recipe text that produced it.

That gap is the difference between a record that VERIFIES a rebuild and one that
makes a rebuild possible. `build_id` lets you check you got the same model back;
it does not hand you the thing to run. And it is a gap in this project's own
terms, before any viewer is involved: a build directory that cannot say what
recipe built it is incomplete, and the run that most needs to answer that
question is the one somebody is looking at six months later.

So `load()` remembers where it read the recipe from and what it said, and the
Runner copies it in beside the manifest on the first flush. The builder learns
nothing about anything else in the process - it writes a file next to its own
outputs, which is the same instinct that writes the manifest.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ms_moe_maker.config import recipe as R
from ms_moe_maker.config.levers import Translation
from ms_moe_maker.run import manifest as mf
from ms_moe_maker.run import stages as st
from ms_moe_maker.run.events import Events
from ms_moe_maker.run.runner import Runner

EXAMPLE = Path(__file__).resolve().parent.parent / "recipe.example.yaml"


def loaded(tmp_path: Path, name: str = "mine.yaml"):
    """A recipe read off a path, which is the only way it has a source."""
    target = tmp_path / name
    shutil.copy(EXAMPLE, target)
    rec, _warns = R.load(str(target))
    return rec, target


class TestLoadRemembersWhereItRead:
    def test_the_path_and_the_text_both_travel(self, tmp_path):
        rec, target = loaded(tmp_path)
        assert getattr(rec, "source_path", None) == str(target)
        assert getattr(rec, "source_text", "") == target.read_text(
            encoding="utf-8")

    def test_parse_stays_pure(self):
        """`parse()` must not acquire a source. It is the pure half on purpose -
        a unit test calling parse() should not inherit a box, or a path."""
        rec, _ = R.parse({"name": "x", "base": "b",
                          "experts": [{"name": "e", "source": {"kind": "hf",
                                                               "id": "z"}}]})
        assert getattr(rec, "source_path", None) is None
        assert getattr(rec, "source_text", None) is None


class TestTheFileItWrites:
    def test_the_text_round_trips(self, tmp_path):
        rec, target = loaded(tmp_path)
        run = tmp_path / "runs" / "0.5B"
        written = mf.write_recipe(run, rec.source_text, ".yaml")
        assert written.read_text(encoding="utf-8") == target.read_text(
            encoding="utf-8")

    def test_it_lands_beside_the_manifest(self, tmp_path):
        """SAME DIRECTORY, and the runner has been wrong about this before: the
        manifest once went to `msmoe_run_auto` while every artifact landed in
        `msmoe_run_7B`, and a manifest in a different directory from the run it
        describes is worse than no manifest. The recipe inherits that lesson
        rather than re-learning it."""
        rec, _ = loaded(tmp_path)
        run = tmp_path / "runs" / "0.5B"
        recipe_file = mf.write_recipe(run, rec.source_text, ".yaml")
        manifest_file = mf.write(run, mf.Manifest(name="x"))
        assert recipe_file.parent == manifest_file.parent

    def test_a_json_recipe_keeps_its_suffix(self, tmp_path):
        """JSON is valid YAML 1.2, which is exactly the kind of technically-true
        that gets somebody to open the wrong parser. The suffix is not ours to
        change."""
        assert mf.write_recipe(tmp_path, '{"name": "x"}', ".json").name == \
            mf.RECIPE_STEM + ".json"

    def test_a_suffixless_source_gets_no_invented_one(self, tmp_path):
        assert mf.write_recipe(tmp_path, "name: x", "").name == mf.RECIPE_STEM

    def test_the_directory_is_created(self, tmp_path):
        """Atomic, like the manifest, which means mkdir is the writer's job -
        the run directory may not exist when the first flush happens."""
        deep = tmp_path / "a" / "b" / "c"
        assert mf.write_recipe(deep, "name: x", ".yaml").is_file()


class TestFindingIt:
    def test_it_is_found_whatever_suffix_it_kept(self, tmp_path):
        for suffix in (".yaml", ".json", ".yml"):
            d = tmp_path / suffix.strip(".")
            mf.write_recipe(d, "x", suffix)
            found = mf.find_recipe(d)
            assert found is not None and found.suffix == suffix

    def test_no_recipe_is_None_and_not_an_empty_path(self, tmp_path):
        """An older run has nothing here, and that is a real answer. It must
        stay distinguishable from a recipe that was written and came back
        blank - those are different problems and only one is yours."""
        (tmp_path / "bare").mkdir()
        assert mf.find_recipe(tmp_path / "bare") is None

    def test_a_blank_recipe_is_found_rather_than_treated_as_missing(self,
                                                                   tmp_path):
        mf.write_recipe(tmp_path, "", ".yaml")
        found = mf.find_recipe(tmp_path)
        assert found is not None and found.read_text(encoding="utf-8") == ""

    def test_a_directory_that_does_not_exist_is_None(self, tmp_path):
        assert mf.find_recipe(tmp_path / "nope") is None


class TestTheRunnerActuallyDoesIt:
    """THE WIRING HALF. Everything above proves the writer works; none of it
    proves anything calls it. A helper nobody invokes is the defect this
    codebase keeps finding, so these drive a real Runner."""

    def _runner(self, tmp_path, recipe):
        return Runner(recipe, None, Translation(), Events(enabled=False),
                      cwd=tmp_path, dryrun=True)

    def test_the_first_flush_preserves_it(self, tmp_path):
        rec, target = loaded(tmp_path)
        runner = self._runner(tmp_path, rec)
        assert mf.find_recipe(runner.run_dir) is None, "before any flush"
        runner._set(st.PREFLIGHT, mf.RUNNING)
        found = mf.find_recipe(runner.run_dir)
        assert found is not None, "the runner never preserved the recipe"
        assert found.read_text(encoding="utf-8") == target.read_text(
            encoding="utf-8")

    def test_it_is_written_once_not_on_every_stage(self, tmp_path,
                                                  monkeypatch):
        """_flush fires on every state change - a nine-hour gauntlet flushes
        dozens of times, and rewriting the recipe at each one would be pure
        churn on a file that cannot have changed."""
        rec, _ = loaded(tmp_path)
        runner = self._runner(tmp_path, rec)
        calls = []
        real = mf.write_recipe
        monkeypatch.setattr(mf, "write_recipe",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        for stage in (st.PREFLIGHT, st.DATA_CORPUS, st.STITCH, st.ROUTER):
            runner._set(stage, mf.RUNNING)
            runner._set(stage, mf.DONE)
        assert len(calls) == 1, f"written {len(calls)} times"

    def test_a_recipe_with_no_source_is_not_an_error(self, tmp_path):
        """`parse()` stays pure, so an in-memory Recipe has no source. Nothing
        to copy means nothing to copy - not a warning, and not a crash."""
        rec, _ = R.parse({"name": "x", "base": "b",
                          "experts": [{"name": "e",
                                       "source": {"kind": "hf", "id": "z"}}]})
        runner = self._runner(tmp_path, rec)
        runner._set(st.PREFLIGHT, mf.RUNNING)        # must not raise
        assert mf.find_recipe(runner.run_dir) is None

    def test_a_write_failure_warns_and_the_build_continues(self, tmp_path,
                                                           monkeypatch):
        """BEST EFFORT, LIKE THE MANIFEST. A build that died because its own
        bookkeeping could not be written would be a worse trade than a record
        missing a file - and the warning goes on the event stream, where the
        person watching a build will actually see it."""
        rec, _ = loaded(tmp_path)
        seen = []
        ev = Events(enabled=False)
        monkeypatch.setattr(ev, "warning", lambda m: seen.append(m))
        monkeypatch.setattr(mf, "write_recipe",
                            lambda *a, **k: (_ for _ in ()).throw(
                                OSError("disk full")))
        runner = Runner(rec, None, Translation(), ev, cwd=tmp_path,
                        dryrun=True)
        runner._set(st.PREFLIGHT, mf.RUNNING)        # must not raise
        assert any("preserve the recipe" in m for m in seen), seen

    def test_it_lands_in_the_same_directory_as_the_manifest(self, tmp_path):
        rec, _ = loaded(tmp_path)
        runner = self._runner(tmp_path, rec)
        runner._set(st.PREFLIGHT, mf.RUNNING)
        assert (runner.run_dir / mf.MANIFEST_NAME).is_file()
        found = mf.find_recipe(runner.run_dir)
        assert found is not None and found.parent == runner.run_dir
