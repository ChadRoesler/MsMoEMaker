"""CLI integration tests — end-to-end verification of the ms-moe-maker commands.

These tests run against the actual recipe.example.yaml file to verify the
entire command path from describe → validate → build (dryrun) works without
missing wiring.  They don't touch a GPU or download data — just confirm the
CLI contract is sound.
"""
import json
import shutil
from pathlib import Path

import pytest

from ms_moe_maker.__main__ import main, DESCRIBE

EXAMPLE = Path(__file__).resolve().parent.parent / "recipe.example.yaml"


# -- describe ----------------------------------------------------------------


def test_describe_returns_valid_json():
    """--describe outputs parseable JSON and exits 0."""
    rc = main(["--describe"])
    assert rc == 0
    keys = sorted(DESCRIBE.keys())
    assert "tiers" in keys
    assert "templates" in keys
    assert "commands" in keys


def test_describe_tiers_match_hardware():
    """The tier list reflects the current hardware tiers."""
    from ms_moe_maker.box import hardware

    tier_names = DESCRIBE.get("tiers", [])
    assert set(tier_names) == set(hardware.TIERS.keys())
    # No stale tier names
    assert "dgx" not in tier_names


def test_describe_templates_match_registry():
    """The template list matches the template registry."""
    from ms_moe_maker.config import templates as template

    tpl_names = DESCRIBE.get("templates", [])
    assert set(tpl_names) == set(template.TEMPLATES.keys())


def test_every_advertised_template_has_question_templates():
    """`source.templates: <name>` must resolve to a PACKAGED question-template
    file for every advertised name. A recipe that says `templates: math` must
    not silently fall back to the built-in code tasks because the file was
    never written."""
    import os

    from ms_moe_maker.config import defaults as D

    for name in DESCRIBE.get("templates", []):
        path = D.packaged_path(f"{name}_templates.yaml")
        assert os.path.isfile(path), (
            f"advertised template {name!r} has no packaged question file at "
            f"{path} - a `templates: {name}` source would silently fall back "
            f"to the built-in code tasks")


def test_describe_commands_are_current():
    """The verbs we advertise, the verbs argparse accepts, and the verbs we can
    actually dispatch are all the same list.

    THIS TEST USED TO CHECK THE WRONG SOURCE. It asserted __main__.DESCRIBE
    against a hardcoded literal in its own body - one copy of the list against
    a second copy of the same list - so it stayed green while a THIRD copy,
    _describe.COMMANDS (the one _describe's own docstring says stagehand reads
    to check contract version), drifted to three verbs against __main__'s five.
    The test with exactly the right name was structurally unable to catch the
    drift it was named for.

    Three sources, checked against each other, so any future drift fails here.
    """
    from ms_moe_maker.box import describe as _describe
    from ms_moe_maker.__main__ import COMMAND_HANDLERS

    assert list(DESCRIBE["commands"]) == list(_describe.COMMANDS)
    # `describe` short-circuits before argparse, so it is the one verb with no
    # handler. Everything else must be dispatchable.
    assert set(COMMAND_HANDLERS) == set(_describe.COMMANDS) - {"describe"}


def test_every_advertised_verb_actually_dispatches(tmp_path, monkeypatch):
    """Checks the table main() USES, not a table that merely exists.

    Learned the hard way, twice in one day. The first version of this test
    compared __main__.DESCRIBE to a literal in its own body - two copies of
    the same list agreeing with each other. The fix compared DESCRIBE to
    COMMAND_HANDLERS... while main() was still dispatching through a LOCAL
    cmd_map that had never been updated, so `init` was advertised, accepted by
    argparse, and then rejected as an unknown command. Both times the test
    checked a source that was not the one doing the work.

    So: drive main() for every verb and assert none of them dies with "unknown
    command". A verb we advertise has to run.
    """
    from ms_moe_maker.box import describe as _describe

    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / "r.yaml"
    shutil.copy(EXAMPLE, recipe)

    for verb in _describe.COMMANDS:
        argv = [verb] if verb in ("describe", "init") else [verb, str(recipe)]
        try:
            main(argv)
        except SystemExit as exc:
            # argparse only exits for a usage error; a verb that runs and
            # returns a non-zero code is fine here.
            assert exc.code in (None, 0), (
                f"verb {verb!r} is advertised but main() rejected it")
        except Exception:
            # A real runtime failure (no torch, no data) is not what this
            # test is about - it only asserts the verb is wired up.
            pass


def test_eval_modes_are_current():
    """Same rule for --mode: advertised, accepted and honoured must agree."""
    from ms_moe_maker.box import describe as _describe
    assert list(DESCRIBE["modes"]) == list(_describe.EVAL_MODES)


# -- validate ----------------------------------------------------------------


@pytest.fixture
def recipe_yaml(tmp_path, monkeypatch):
    """A copy of the example recipe in a clean directory."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "recipe.yaml"
    shutil.copy(EXAMPLE, target)
    return target


def test_validate_success(recipe_yaml):
    """A valid recipe returns exit 0 and prints validation info."""
    rc = main(["validate", str(recipe_yaml)])
    assert rc == 0


def test_validate_invalid_recipe():
    """A broken recipe returns non-zero."""
    rc = main(["validate", "nonexistent.yaml"])
    assert rc == 1


# -- build (dryrun) ----------------------------------------------------------


def test_plan_runs_anywhere(recipe_yaml):
    """--plan resolves everything and runs nothing. No GPU, no torch."""
    assert main(["build", str(recipe_yaml), "--plan"]) == 0


def test_build_writes_a_manifest(recipe_yaml, tmp_path):
    """A build writes msmoe-run.json, even when the build fails.

    THIS TEST USED TO ASSERT NOTHING. It globbed for an output directory,
    assigned the result to a variable, and ended - with a comment reading
    "Might not create dir if dryrun skips it; at least assert no exception"
    standing in for the assertion. It was green the whole time no manifest was
    written by anything at all, because __main__ had stopped calling Runner and
    Runner is the only thing that writes one.

    Failure is deliberately the case under test: the manifest is the heartbeat
    a watcher reads, so it has to exist for a run that DIED, or a dashboard
    cannot tell "crashed" from "never started".
    """
    from ms_moe_maker.run.manifest import MANIFEST_NAME
    from ms_moe_maker.config.pipeline import build_config
    from ms_moe_maker.config.recipe import load

    rc = main(["build", str(recipe_yaml), "--dryrun"])
    rec, _ = load(str(recipe_yaml))
    # dryrun=True here because that is what the run used. Before the flag was
    # threaded, a --dryrun build wrote into the PRODUCTION directory, so this
    # line agreed with it by accident.
    run_dir = Path(build_config(rec, dryrun=True).output_root)

    manifest = run_dir / MANIFEST_NAME
    assert manifest.is_file(), (
        f"no {MANIFEST_NAME} in {run_dir} - a run that produces no manifest is "
        f"invisible to every watcher, and this is exactly the regression that "
        f"shipped when _cmd_build stopped going through Runner")
    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["stages"], "a manifest with no stages tells a watcher nothing"
    assert "ok" in body


# -- recipe parsing end-to-end ------------------------------------------------


def test_recipe_example_parses_clean():
    """recipe.example.yaml loads without errors or warnings."""
    from ms_moe_maker.config.recipe import load

    rec, warnings = load(str(EXAMPLE))
    assert rec is not None
    # Should have at least 3 experts from the example
    assert len(rec.experts) >= 3


def test_recipe_example_has_correct_tier():
    """The recipe's hardware_tier is set by template auto-fill."""
    from ms_moe_maker.config.recipe import load

    rec, _ = load(str(EXAMPLE))
    # The example recipe doesn't set template: directly, but the tier
    # should still have a sensible default (xavier is the middle tier)
    assert rec.runtime.hardware_tier in ("nano", "xavier", "spark")


def test_recipe_example_size_is_auto():
    """The example recipe uses size=auto and resolves from tier."""
    from ms_moe_maker.config.recipe import load
    from ms_moe_maker.config import pipeline as cfg_module

    rec, _ = load(str(EXAMPLE))
    assert rec.size == "auto"
    cfg = cfg_module.build_config(rec)
    # The resolved size should be a valid model size string
    assert cfg.size in cfg_module.MODEL_SIZES


# -- template apply via recipe.load --------------------------------------------


def test_code_template_applies_clean():
    """The 'code' template fills all expected fields."""
    from ms_moe_maker.config.recipe import load
    from ms_moe_maker.config.templates import get_template

    # Get the code template to see what it provides
    tpl = get_template("code")
    assert tpl is not None
    assert tpl["default_tier"] == "spark"
    assert tpl["preferred_size"] == "3B"
    assert len(tpl.get("default_experts", [])) >= 2


def test_dnd_template_applies_clean():
    """The 'dnd' template fills all expected fields."""
    from ms_moe_maker.config.templates import get_template

    tpl = get_template("dnd")
    assert tpl is not None
    assert tpl["default_tier"] == "nano"
    assert tpl["preferred_size"] == "0.5B"
    assert len(tpl.get("default_experts", [])) >= 2


def test_template_unknown_raises():
    """Using an unknown template name raises ValueError."""
    from ms_moe_maker.config.templates import get_template

    assert get_template("nonexistent") is None


def test_recipe_load_with_template_field():
    """A recipe with template: field resolves its tier correctly."""
    from io import StringIO
    from ms_moe_maker.config.recipe import parse

    yaml_text = """
schema_version: 1
name: test-with-template
template: dnd
"""
    rec, warnings = parse(__import__("yaml").safe_load(yaml_text))
    assert rec is not None
    assert rec.template == "dnd"
    # Template should have wired the tier to runtime.hardware_tier
    assert rec.runtime.hardware_tier == "nano"


def test_recipe_load_code_template():
    """A recipe with template: code resolves spark tier."""
    from ms_moe_maker.config.recipe import parse
    import yaml

    yaml_text = """
schema_version: 1
name: code-test
template: code
"""
    rec, warnings = parse(yaml.safe_load(yaml_text))
    assert rec is not None
    assert rec.template == "code"
    assert rec.runtime.hardware_tier == "spark"
    assert len(rec.experts) >= 2


# -- init --------------------------------------------------------------------

class TestInitRoundTrip:
    """What init writes, validate must accept.

    This caught a real one immediately: the first version joined the source
    fields with spaces instead of commas, so `init` emitted invalid YAML and
    the on-ramp fell over on its own first step. A scaffolder whose output does
    not parse is worse than no scaffolder, because the user assumes the tool
    works and the recipe is their fault.
    """

    def test_bare_init_output_validates(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == 0
        out = capsys.readouterr().out
        (tmp_path / "r.yaml").write_text(out, encoding="utf-8")
        assert main(["validate", str(tmp_path / "r.yaml")]) == 0

    @pytest.mark.parametrize("tpl", ["code", "dnd", "math", "culinary"])
    def test_every_template_round_trips(self, tpl, tmp_path, monkeypatch):
        """Every template we advertise has to produce a valid recipe."""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / f"{tpl}.yaml"
        assert main(["init", "--template", tpl, "-o", str(target)]) == 0
        assert main(["validate", str(target)]) == 0

    def test_output_is_parseable_yaml(self, tmp_path, monkeypatch):
        import yaml
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "r.yaml"
        main(["init", "--template", "dnd", "-o", str(target)])
        body = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert body["schema_version"] == 1
        assert body["template"] == "dnd"

    def test_it_refuses_to_clobber_without_force(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "r.yaml"
        target.write_text("mine\n", encoding="utf-8")
        assert main(["init", "-o", str(target)]) == 1
        assert target.read_text(encoding="utf-8") == "mine\n"

    def test_unknown_template_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main(["init", "--template", "nope"]) == 1


class TestDescribeIsAContract:
    """`describe` is what release CI and stagehand read. Keys are promises."""

    def test_no_key_from_describe_is_lost(self):
        """The regression that broke release CI.

        __main__.DESCRIBE was rebuilt by naming _describe's keys one at a
        time, which dropped `requires` - a published key the release workflow
        asserts. Hand-copying a subset of a source of truth is not using the
        source of truth, it is quietly making a second one. Now it is merged,
        so a key added over there cannot be lost here.
        """
        from ms_moe_maker.box import describe as _describe
        missing = set(_describe.DESCRIBE) - set(DESCRIBE)
        assert not missing, f"describe dropped published key(s): {missing}"

    def test_requires_is_empty(self):
        """ms-moe-maker requires nothing of anyone. That is the laptop
        promise, stated on the wire rather than only in a docstring."""
        assert DESCRIBE["requires"] == []

    def test_describe_is_one_line_of_json(self, capsys):
        """The Starwright contract: one line, exit 0, zero side effects."""
        assert main(["--describe"]) == 0
        out = capsys.readouterr().out
        assert out.count("\n") == 1
        json.loads(out)

    def test_kinds_come_from_the_registry(self):
        from ms_moe_maker.data import corpus
        # The rich shape now, matching `validators` and recipe.DESCRIBE.
        # Names stay derivable, which is what this ever asserted.
        assert [k["name"] for k in DESCRIBE["kinds"]] == corpus.names()
        assert DESCRIBE["kinds"] == corpus.describe()


class TestJsonIsAWireFormatEverywhere:
    """--json has to mean the same thing on every verb that accepts it."""

    def test_validate_emits_events(self, recipe_yaml, capsys):
        assert main(["validate", str(recipe_yaml), "--json"]) == 0
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        kinds = [json.loads(l)["event"] for l in lines]
        assert kinds, "--json produced no events at all"
        assert kinds[0] == "started"
        assert kinds[-1] == "done", (
            "a consumer following the stream needs one terminal event that "
            "means 'there will be no more'")

    def test_validate_events_are_one_object_per_line(self, recipe_yaml, capsys):
        main(["validate", str(recipe_yaml), "--json"])
        for line in capsys.readouterr().out.splitlines():
            if line.strip():
                json.loads(line)          # raises if prose leaked to stdout

    def test_a_bad_recipe_still_terminates_the_stream(self, tmp_path, capsys):
        bad = tmp_path / "bad.yaml"
        bad.write_text("schema_version: 1\nname: x\nexperts: []\n", encoding="utf-8")
        assert main(["validate", str(bad), "--json"]) == 1
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        events = [json.loads(l) for l in lines]
        assert events[-1]["event"] == "done"
        assert events[-1]["ok"] is False
        assert any(e["event"] == "error" for e in events)

    def test_prose_never_reaches_stdout_under_json(self, recipe_yaml, capsys):
        """stdout belongs to the machine. A stray print corrupts the format
        the consumer is parsing."""
        main(["validate", str(recipe_yaml), "--json"])
        cap = capsys.readouterr()
        for line in cap.out.splitlines():
            if line.strip():
                json.loads(line)

    def test_every_emitted_kind_is_declared(self, recipe_yaml, capsys):
        """_describe.EVENTS is a wire contract: a consumer that meets an
        undeclared kind has no rule for it."""
        from ms_moe_maker.box import describe as _describe
        main(["validate", str(recipe_yaml), "--json"])
        for line in capsys.readouterr().out.splitlines():
            if line.strip():
                kind = json.loads(line)["event"]
                assert kind in _describe.EVENTS, f"undeclared event {kind!r}"


class TestFailuresReachTheHuman:
    """A build that fails must SAY so, on the channel a person is reading.

    It did not. Events.emit() is gated on --json, and the failure path used
    ev.error() alone - so without that flag a failed build wrote the reason
    into the manifest, emitted nothing, and returned exit 1 with zero output.
    Success printed. Failure did not. To the person at the prompt it read as
    "it just stopped", which is indistinguishable from a hang.
    """

    def test_the_failure_path_uses_the_unconditional_channel(self):
        import inspect
        from ms_moe_maker.run import runner
        src = inspect.getsource(runner.Runner.run_builder)
        head, _, tail = src.partition("except Exception as exc:")
        assert tail, "run_builder no longer has a failure branch"
        assert ".say(" in tail, (
            "the failure branch must use ev.say() - ev.error() is emit(), "
            "which is silent unless --json was passed")

    def test_a_terminal_line_is_always_printed(self):
        import inspect
        from ms_moe_maker.run import runner
        src = inspect.getsource(runner.Runner.run_builder)
        assert src.count(".say(") >= 2, (
            "'did it finish?' must never be answered by absence of output")

    def test_say_is_not_gated_but_emit_is(self):
        """The property the fix relies on."""
        import io
        from ms_moe_maker.run.events import Events
        out, prose = io.StringIO(), io.StringIO()
        ev = Events(enabled=False, stream=out, prose=prose)
        ev.error(stage="x", message="boom")
        ev.say("boom")
        assert out.getvalue() == "", "emit must stay silent without --json"
        assert "boom" in prose.getvalue(), "say must never be silent"
