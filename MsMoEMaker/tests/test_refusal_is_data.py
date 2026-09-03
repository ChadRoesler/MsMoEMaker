"""The resume refusal, as something a machine can act on.

THE BUG THIS FILE IS THE FENCE FOR. `build` refuses to resume into a directory
that belongs to a different build, and the refusal is genuinely good prose: it
names the stages it would inherit, lists every field that moved, and offers
three ways out. All of that went to stderr. The `--json` event stream - the
channel every automated consumer reads, and the one seren-theatre's Backstage
forks the command for - carried a single flat sentence.

So an operator running through Backstage got a wall of text with no structure
and no way to act on it, from a program that had the diff as a list and the
options as three lines it was about to print. Data present, not surfaced.

The tests here hold the two channels together. Not "the event has some fields"
- that would pass while the fields said something else - but the STRINGS the
person read and the STRINGS the machine got, asserted equal, plus the guard
that any flag we offer as a button is a flag argparse actually takes. That
last one is not hypothetical: Theatre once grew an `--allow-refusals`
checkbox for a flag ms-moe-maker never had, and every build it started died at
argument parsing with exit 2.
"""
import json
import shutil
from pathlib import Path

import pytest

from ms_moe_maker.__main__ import main
from ms_moe_maker.cli.build import RESUME_DRIFT, RESUME_OPTIONS
from ms_moe_maker.run.runner import Change

EXAMPLE = Path(__file__).resolve().parent.parent / "recipe.example.yaml"


@pytest.fixture
def drifted(tmp_path, monkeypatch):
    """A run directory holding a finished stage from a DIFFERENT build."""
    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / "recipe.yaml"
    shutil.copy(EXAMPLE, recipe)

    from ms_moe_maker.config.pipeline import build_config
    from ms_moe_maker.config.recipe import load
    from ms_moe_maker.run import manifest as mf

    rec, _ = load(str(recipe))
    run_dir = Path(build_config(rec, dryrun=True).output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    mf.write(run_dir, mf.Manifest(
        recipe_id="abc", build_id="0000deadbeef",
        resolved={"target_steps": 400, "expert_names": ["python"]},
        stages=[mf.Stage(id="preflight", label="preflight", status=mf.DONE)]))
    return recipe, run_dir


def _refuse(recipe, capsys):
    """Drive the real refusal and split the two channels."""
    rc = main(["build", str(recipe), "--dryrun", "--json"])
    out, err = capsys.readouterr()
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    return rc, events, err


def _the_error(events):
    errors = [e for e in events if e.get("event") == "error"]
    assert errors, f"no error event in {events}"
    return errors[-1]


class TestTheRefusalReachesBothChannels:

    def test_it_refuses_and_says_why_on_both(self, drifted, capsys):
        recipe, run_dir = drifted
        rc, events, err = _refuse(recipe, capsys)
        assert rc == 1, "a drifted resume must not proceed"
        assert "REFUSING TO RESUME" in err
        bad = _the_error(events)
        assert bad["refusal"] == RESUME_DRIFT
        assert bad["finished"] == ["preflight"]
        # ABSOLUTE, not the build's relative output_root. The reader is a
        # browser on another machine deciding whether to keep this directory
        # or point roots.output somewhere else.
        assert bad["run_dir"] == str(run_dir.resolve())
        assert Path(bad["run_dir"]).is_absolute()

    def test_the_machine_gets_the_same_lines_the_person_did(
            self, drifted, capsys):
        """Not "some fields" - THE SAME STRINGS.

        A dashboard that shows a different diff from the terminal is worse
        than one that shows none, because somebody will act on it.
        """
        recipe, _ = drifted
        rc, events, err = _refuse(recipe, capsys)
        # SCOPED TO THE BLOCK, because `\u00b7 ` is also how recipe warnings
        # print. A test that swept up every bulleted line would compare the
        # diff against the warnings and fail for the wrong reason - or,
        # worse, pass one day when both happened to be empty.
        lines = [ln.strip() for ln in err.splitlines()]
        block = lines[lines.index("What changed:") + 1:]
        block = block[:block.index("Pick one:")]
        printed = [ln[1:].strip() for ln in block
                   if ln.startswith("\u00b7")]
        assert printed, err
        assert _the_error(events)["changed"] == printed

    def test_the_fields_are_a_table_not_a_sentence_to_re_parse(
            self, drifted, capsys):
        """Every field diff arrives split, so nobody has to cut on " -> ".

        A resolved value can be a string CONTAINING an arrow. Splitting the
        prose is a bug waiting for one unusual config.
        """
        recipe, _ = drifted
        rc, events, err = _refuse(recipe, capsys)
        bad = _the_error(events)
        assert bad["fields"], "the diff arrived as prose only"
        for row in bad["fields"]:
            assert set(row) == {"field", "text", "kind", "was", "now"}
            assert row["text"] in bad["changed"], (
                "a field row that is not one of the printed lines is a "
                "second description of the same fact")
        named = [r for r in bad["fields"] if r["field"]]
        assert any(r["field"] == "target_steps" for r in named), named

    def test_the_options_are_offered_identically_to_both(
            self, drifted, capsys):
        recipe, _ = drifted
        rc, events, err = _refuse(recipe, capsys)
        assert _the_error(events)["options"] == RESUME_OPTIONS
        for opt in RESUME_OPTIONS:
            assert opt["do"] in err, (
                f"{opt['do']!r} is offered on the event stream but never "
                f"printed - the two lists have drifted")


class TestAnOfferedFlagIsARealFlag:
    """The `--allow-refusals` fence. An option carrying a `flag` is one a UI
    may append to argv, so argparse has to accept it."""

    def test_every_flag_we_advertise_parses(self, drifted):
        recipe, _ = drifted
        for opt in RESUME_OPTIONS:
            if not opt["flag"]:
                continue
            # --plan resolves everything and runs nothing, so this exercises
            # the real parser without a GPU. --offline keeps it off the network.
            assert main(["build", str(recipe), opt["flag"],
                         "--plan", "--offline"]) == 0, opt

    def test_the_destructive_one_is_marked(self):
        """--force discards finished stages. A button that offers it without
        saying so is how somebody loses half an hour of abliteration."""
        force = [o for o in RESUME_OPTIONS if o["id"] == "force"]
        assert force and force[0]["discards"] is True
        assert not any(o["discards"] for o in RESUME_OPTIONS
                       if o["id"] != "force")


class TestChangeIsOneObjectNotTwo:

    def test_it_reads_as_the_line_and_carries_the_parts(self):
        c = Change("target_steps", 400, 1200)
        assert str(c) == "target_steps: 400 -> 1200"
        assert c == "target_steps: 400 -> 1200"      # plain-str callers
        assert c.as_dict() == {"field": "target_steps",
                               "text": "target_steps: 400 -> 1200",
                               "kind": "moved",
                               "was": "400", "now": "1200"}

    def test_a_field_the_old_manifest_never_recorded_is_not_a_moved_knob(
            self):
        """Resuming into a directory built before a field existed reports
        that field, correctly - unknown is not unchanged. But it is not the
        same news as somebody changing a number, and on a 92-field config
        there are seventy of the first for one of the second."""
        from ms_moe_maker.config.pipeline import ABSENT
        assert Change("warmup_steps", ABSENT, 60).kind == "first-recorded"
        assert Change("old_knob", 3, ABSENT).kind == "no-longer-recorded"
        assert Change("target_steps", 400, 1200).kind == "moved"
        assert Change(None, text="unreadable").kind == "note"

    def test_the_kinds_reach_the_event(self, drifted, capsys):
        """A seeded manifest with two fields against a full resolved config
        is exactly the ragged case: a couple moved, the rest never written
        down. Both kinds have to arrive labelled."""
        recipe, _ = drifted
        rc, events, err = _refuse(recipe, capsys)
        kinds = {r["kind"] for r in _the_error(events)["fields"]}
        assert "first-recorded" in kinds, kinds
        assert "moved" in kinds, kinds

    def test_a_sentence_carries_no_field(self):
        """Some entries are ABOUT the comparison - "the manifest is
        unreadable" - not a field that moved. A renderer keys off that."""
        c = Change(None, text="the previous manifest is unreadable")
        assert str(c) == "the previous manifest is unreadable"
        assert c.as_dict()["field"] is None
        assert c.as_dict()["was"] is None

    def test_json_serialises_it_as_the_line(self):
        """json.dumps(default=str) is what events.emit uses."""
        assert json.loads(json.dumps({"c": Change("a", 1, 2)},
                                     default=str))["c"] == "a: 1 -> 2"
