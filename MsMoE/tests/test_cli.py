"""The CLI's promises, including the one it was quietly breaking.

THE LAPTOP PROMISE. ms-moe's README says `ms-moe validate` runs on a laptop
with no GPU, so you can check a recipe and see what it will cost before going
anywhere near a machine that can run it. That is the entire argument for
ms-moe being a small separate package instead of part of the pipeline.

It was false. `validate` demanded fraunkenstein_universal.py and exited 1 with
"could not find fraunkenstein_universal.py" before parsing anything - so a
stranger with a recipe and no checkout got no validation at all, which is
exactly the person the promise was written for. Found by running the CI smoke
steps by hand rather than trusting that a workflow which parses is a workflow
that passes.

The fix keeps the distinction honest: recipe SHAPE is checkable alone, refusals
are not, and the difference is reported rather than blurred. "Valid" and
"valid, and nothing checked whether the pipeline can honour it" are different
answers and a consumer has to be able to tell them apart.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ms_moe.__main__ import main

EXAMPLE = Path(__file__).resolve().parent.parent / "recipe.example.yaml"


@pytest.fixture
def lonely_recipe(tmp_path, monkeypatch):
    """A recipe in a directory with no pipeline anywhere above it."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "recipe.yaml"
    shutil.copy(EXAMPLE, target)
    return target


def _events(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [json.loads(l) for l in out.splitlines() if l.strip()]


# -- describe ----------------------------------------------------------------

def test_describe_needs_nothing(capsys):
    assert main(["describe"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["name"] == "ms-moe"
    assert d["requires"] == []


def test_describe_via_flag_short_circuits_before_argparse(capsys):
    """It has to answer on a half-installed tool, so nothing may run first."""
    assert main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "ms-moe"


# -- the laptop promise ------------------------------------------------------

def test_validate_works_with_no_pipeline_in_sight(lonely_recipe, capsys):
    """The bug this file was written for."""
    code = main(["validate", str(lonely_recipe), "--json"])
    assert code == 0, "a structurally valid recipe must validate on its own"
    kinds = [e["event"] for e in _events(capsys)]
    assert kinds[-1] == "done", kinds


def test_validate_says_out_loud_that_refusals_were_not_checked(
        lonely_recipe, capsys):
    """Silence here would be the worst outcome: a clean bill of health from a
    check that never ran."""
    main(["validate", str(lonely_recipe), "--json"])
    events = _events(capsys)
    done = events[-1]
    assert done["refusals_checked"] is False, (
        "a consumer must be able to tell 'no refusals' from 'refusals were "
        "never computed' - it cannot infer that from an empty list")
    assert done["pipeline"] is None
    warnings = [e["message"] for e in events if e["event"] == "warning"]
    assert any("only the recipe" in w.lower() for w in warnings), warnings


def test_validate_with_a_pipeline_does_check_refusals(lonely_recipe, tmp_path,
                                                      capsys):
    pipeline = tmp_path / "fraunkenstein_universal.py"
    pipeline.write_text(
        'MODEL_SIZES = {"0.5B": ("a", "b")}\n'
        "MAX_SEQ_LENGTH = 2048\nPER_DEVICE_BATCH = 4\nGRAD_ACCUM = 2\n"
        "NUM_EXPERTS_PER_TOK = 2\nNORM_TOPK_PROB = True\n"
        "SHARED_EXPERT_WIDTH = 1\n"
        'CODE_LANGUAGES = ["Python"]\n', encoding="utf-8")
    main(["validate", str(lonely_recipe), "--pipeline", str(pipeline), "--json"])
    done = _events(capsys)[-1]
    assert done["refusals_checked"] is True
    assert done["pipeline"] == str(pipeline)
    assert done["refusals"] > 0, (
        "this stub pipeline disagrees with the example recipe about experts "
        "and base, so refusals are the correct answer")


# -- build still requires it -------------------------------------------------

def test_build_still_demands_a_pipeline(lonely_recipe):
    """Optional for validate, REQUIRED for build. There is nothing to fork
    without it, and pretending otherwise would fail later and less clearly."""
    with pytest.raises(SystemExit) as exc:
        main(["build", str(lonely_recipe)])
    assert "fraunkenstein_universal.py" in str(exc.value)


def test_an_explicit_missing_pipeline_is_always_an_error(lonely_recipe):
    """Being told where it is and being wrong is different from not saying."""
    with pytest.raises(SystemExit) as exc:
        main(["validate", str(lonely_recipe), "--pipeline", "/nope/nope.py"])
    assert "does not exist" in str(exc.value)


# -- parse failures ----------------------------------------------------------

def test_a_broken_recipe_is_a_clean_exit_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experts: [unclosed\n", encoding="utf-8")
    assert main(["validate", str(bad), "--json"]) == 2
    kinds = [e["event"] for e in _events(capsys)]
    assert "error" in kinds


def test_stdout_stays_pure_json_lines_under_json(lonely_recipe, capsys):
    """Prose goes to stderr. A consumer piping stdout to jq must never have to
    filter a banner out of it."""
    main(["validate", str(lonely_recipe), "--json"])
    captured = capsys.readouterr()
    for line in captured.out.splitlines():
        if line.strip():
            json.loads(line)        # raises if prose leaked
    assert captured.err, "the human half should still have been written"
