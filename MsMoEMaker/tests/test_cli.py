"""Tests for the ms-moe-maker CLI — build / smoke / eval / validate / describe.

After the builder became the default, the old CLI tests (which expected a
fraunkenstein_universal.py pipeline and the --json / --python flags) are
obsolete.  This file covers the *current* behaviour.
"""
import json
import shutil
from pathlib import Path

import pytest

from ms_moe_maker.__main__ import main, DESCRIBE

EXAMPLE = Path(__file__).resolve().parent.parent / "recipe.example.yaml"


@pytest.fixture
def lonely_recipe(tmp_path, monkeypatch):
    """A recipe in a directory with no pipeline anywhere above it."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "recipe.yaml"
    shutil.copy(EXAMPLE, target)
    return target


# -- describe ----------------------------------------------------------------

def test_describe_needs_nothing():
    """Zero-side-effects, no pipeline, no GPU."""
    assert main(["describe"]) == 0


def test_describe_via_flag_short_circuits_before_argparse():
    """It has to answer on a half-installed tool, so nothing may run first."""
    assert main(["--describe"]) == 0


def test_describe_has_known_keys():
    d = DESCRIBE
    for key in ("name", "version", "kinds", "gates", "templates", "tiers", "commands"):
        assert key in d, f"missing key {key}"


# -- validate ----------------------------------------------------------------

def test_validate_works_with_no_pipeline_in_sight(lonely_recipe):
    """The laptop promise — validate on a laptop with no GPU."""
    code = main(["validate", str(lonely_recipe)])
    assert code == 0, "a structurally valid recipe must validate on its own"


def test_validate_invalid_recipe_returns_nonzero(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experts: [unclosed\n", encoding="utf-8")
    assert main(["validate", str(bad)]) != 0


# -- build -------------------------------------------------------------------

def test_plan_succeeds_without_pipeline(lonely_recipe):
    """--plan resolves and reports; it needs no pipeline and no torch."""
    code = main(["build", str(lonely_recipe), "--plan"])
    assert code == 0


def test_build_requires_torch_in_live_mode(lonely_recipe):
    """A real build without torch fails, and reports the failure.

    --dryrun is a CHEAP build, not a skipped one, so it fails here too. That
    distinction is the entire reason --plan exists: the old --dryrun returned 0
    without running anything, which made "the smallest rung works" and "nothing
    was attempted" indistinguishable.

    Asserts non-zero rather than a specific code, because on the legacy
    --pipeline path the code is the forked child's and is not ours to promise.
    """
    assert main(["build", str(lonely_recipe)]) != 0
    assert main(["build", str(lonely_recipe), "--dryrun"]) != 0


def test_missing_pipeline_flag_is_error(lonely_recipe):
    """An explicit --pipeline that does not exist is an error."""
    with pytest.raises(SystemExit):
        main(["build", str(lonely_recipe), "--pipeline", "/nope/nope.py"])


# -- runner tests that still apply -------------------------------------------
# These test Runner internals directly; the CLI wrapper changed but the
# underlying Runner class is unchanged.

def test_python_defaults_to_our_own_interpreter(lonely_recipe, tmp_path):
    import sys
    from ms_moe_maker.runner import Runner
    from ms_moe_maker.levers import Translation
    from ms_moe_maker.events import Events
    from tests.test_runner import FakeRecipe

    r = Runner(FakeRecipe(["python"]), tmp_path / "p.py", Translation(),
               Events(enabled=False))
    assert r.python == sys.executable


def test_an_explicit_interpreter_is_used_instead(tmp_path):
    from ms_moe_maker.runner import Runner
    from ms_moe_maker.levers import Translation
    from ms_moe_maker.events import Events
    from tests.test_runner import FakeRecipe

    r = Runner(FakeRecipe(["python"]), tmp_path / "p.py", Translation(),
               Events(enabled=False), python="/opt/train/bin/python")
    assert r.python == "/opt/train/bin/python"


def test_a_missing_module_in_the_child_is_reported_as_an_env_answer(
        tmp_path, capsys):
    from ms_moe_maker.runner import Runner
    from ms_moe_maker.levers import Translation
    from ms_moe_maker.events import Events
    from tests.test_runner import FakeRecipe

    script = tmp_path / "pipeline.py"
    script.write_text(
        "import torch  # noqa: F401\n", encoding="utf-8")
    r = Runner(FakeRecipe(["python"]), script, Translation(),
               Events(enabled=False), cwd=tmp_path)
    assert r.run() == 1
