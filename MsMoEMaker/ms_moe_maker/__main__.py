"""The ms-moe-maker CLI - the command stagehand forks and a person types.

Three verbs:

    ms-moe-maker describe                     one line of JSON, exit 0, no side effects
    ms-moe-maker validate recipe.yaml         parse + check, touching nothing
    ms-moe-maker build recipe.yaml            translate, fork the pipeline, report

`ms-moe-maker build recipe.yaml` is deliberately the literal string in the README and
the literal string seren-theatre[stagehand] forks. Not a Python API call, not
an internal entry point with different defaults - the same command. If the two
ever diverge, the hand-run path is the one that rots, because it is the one
with no automated users; making them identical removes the possibility.

--describe is scanned before argparse for the same reason every Seren installer
does it: it has to answer on a broken install, so nothing may run first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ._describe import DESCRIBE
from .events import Events


def _force_utf8_stdio() -> None:
    """UTF-8 regardless of console codepage. Windows defaults to legacy, and
    the pipeline prints emoji milestones - a UnicodeEncodeError mid-build would
    kill a run over a decorative character."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _find_pipeline(explicit: str | None, recipe_path: Path,
                   required: bool) -> Path | None:
    """Locate fraunkenstein_universal.py. None if absent and not required.

    Looked up rather than assumed, and reported when missing, because "wrapped
    a script that isn't there" should be one clear error and not a traceback
    from subprocess. Order: --pipeline, beside the recipe, cwd, then upward.

    `required` is the difference between the two verbs, and it is not a
    convenience:

      build    - REQUIRED. There is nothing to fork without it.
      validate - OPTIONAL. The README promises `ms-moe-maker validate` runs on a
                 laptop with no GPU so you can check a recipe BEFORE going
                 near a machine that can run it. Demanding the pipeline made
                 that promise false: a stranger with a recipe and no checkout
                 got "could not find fraunkenstein_universal.py" and no
                 validation at all. Recipe SHAPE is checkable on its own; only
                 the refusal analysis needs a pipeline to compare against.

    An explicit --pipeline that does not exist is always an error, for either
    verb. Being told where it is and being wrong is different from not saying.
    """
    from .levers import DEFAULT_PIPELINE

    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"--pipeline {p} does not exist")
        return p.resolve()

    for candidate in (recipe_path.parent / DEFAULT_PIPELINE,
                      Path.cwd() / DEFAULT_PIPELINE):
        if candidate.is_file():
            return candidate.resolve()
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / DEFAULT_PIPELINE
        if candidate.is_file():
            return candidate.resolve()
    if required:
        raise SystemExit(
            f"could not find {DEFAULT_PIPELINE}. Pass --pipeline PATH, or run "
            f"from the directory that holds it.")
    return None


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Before argparse: zero side effects, works half-installed.
    if "--describe" in argv or (argv and argv[0] == "describe"):
        print(json.dumps(DESCRIBE))
        return 0

    ap = argparse.ArgumentParser(
        prog="ms-moe-maker",
        description="Build a mixture of experts from a recipe.")
    ap.add_argument("command", choices=["build", "validate", "describe"])
    ap.add_argument("recipe", nargs="?", help="path to the recipe .yaml")
    ap.add_argument("--pipeline", default=None,
                    help="path to fraunkenstein_universal.py (default: found "
                         "beside the recipe, then upward from cwd)")
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events on stdout, prose on stderr")
    ap.add_argument("--python", default=None,
                    help="interpreter to run the pipeline with (default: the "
                         "one running ms-moe-maker). Use this when the trainer lives "
                         "in a different venv - which is the normal case, "
                         "since ms-moe-maker is deliberately small and torch is not.")
    ap.add_argument("--dryrun", action="store_true",
                    help="FRAUNK_DRYRUN=1 - the whole pipeline, small")
    ap.add_argument("--force", action="store_true",
                    help="FRAUNK_FORCE=1 - redo stages whose artifacts exist")
    ap.add_argument("--allow-refusals", action="store_true",
                    help="run even though some recipe fields cannot be "
                         "honoured. They are recorded in the manifest either "
                         "way; this only removes the stop.")
    a = ap.parse_args(argv)

    ev = Events(enabled=a.json)

    if a.command == "describe":
        print(json.dumps(DESCRIBE))
        return 0

    if not a.recipe:
        ap.error("a recipe path is required")

    from .recipe import load, resolve, validate

    recipe_path = Path(a.recipe).resolve()
    try:
        rec, parse_warns = load(str(recipe_path))
    except Exception as exc:  # noqa: BLE001 - the message IS the product
        ev.error("parse", str(exc))
        ev.say(f"FAILED to parse {recipe_path}: {exc}")
        return 2

    errs, warns = validate(rec)
    warns = parse_warns + warns
    for w in warns:
        ev.warning(w)
        ev.say(f"   WARN  {w}")
    for e in errs:
        ev.error("validate", e)
        ev.say(f"   ERROR {e}")
    if errs:
        ev.done(ok=False, stage="validate")
        return 1

    # RESOLVED BEFORE THE PIPELINE LOOKUP, on purpose. Both are explicit
    # arguments, and an explicit argument that is wrong should say so no matter
    # what else is also missing - same rule as --pipeline. Reporting "could not
    # find fraunkenstein_universal.py" to someone who mistyped --python sends
    # them to fix the wrong thing.
    #
    # MSMOE_PYTHON as well as --python: the interpreter is a property of the
    # BOX, not of the run, so it belongs somewhere you set once. Same shape as
    # the family's SEREN_<X>_* levers - a flag for the one-off, an env var for
    # the machine.
    interpreter = a.python or os.environ.get("MSMOE_PYTHON") or None
    if interpreter:
        ipath = Path(interpreter)
        if not ipath.is_file():
            raise SystemExit(f"--python {ipath} does not exist")
        # abspath, NEVER resolve(). A venv's bin/python is a SYMLINK to the
        # base interpreter, and resolving it throws the venv away: you asked
        # for /lab/bin/python and got /usr/bin/python3.12, whose sys.prefix is
        # /usr and whose site-packages has none of your training deps. The
        # failure then reads as "No module named 'torch'" from an interpreter
        # you never named, which is about as misleading as it gets.
        #
        # Measured: running the symlink gives sys.prefix=/tmp/venvtest;
        # running its target gives sys.prefix=/usr. Same file, different venv.
        # abspath normalises the path without following the link.
        interpreter = os.path.abspath(str(ipath))

    pipeline = _find_pipeline(a.pipeline, recipe_path,
                              required=(a.command == "build"))

    from .levers import Translation, translate

    if pipeline is None:
        # Recipe-only validation. Say so LOUDLY rather than reporting a clean
        # bill of health: "valid" and "valid, and nothing checked whether the
        # pipeline can honour it" are different answers, and quietly giving
        # the first when you mean the second is how a document that lies gets
        # blessed on its way out the door.
        tr = Translation()
        no_pipeline_note = (
            "no pipeline found, so ONLY the recipe's own shape was checked. "
            "Refusals could not be computed - run this again beside "
            "fraunkenstein_universal.py, or pass --pipeline PATH, to find out "
            "whether a build would actually honour these fields.")
        ev.warning(no_pipeline_note)
    else:
        no_pipeline_note = ""
        tr = translate(rec, pipeline, force=a.force)

    if tr.refusals:
        ev.refused(tr.refusals)
        ev.say("")
        ev.say(f"   {len(tr.refusals)} recipe field(s) cannot be honoured by "
               f"{pipeline.name}:")
        for r in tr.refusals:
            ev.say(f"     · {r}")
        ev.say("")
        ev.say("   These are not warnings. A recipe is a document you hand to "
               "someone so they get YOUR run, and a field that is silently "
               "ignored makes it a document that lies. Fix the recipe, carve "
               "the stage out, or pass --allow-refusals to proceed knowing "
               "the build will not match the file.")

    if a.command == "validate":
        eff = resolve(rec)
        ev.emit("resolved", **eff)
        ev.say("")
        ev.say(f"Ms.MoE recipe  {rec.name}  [{eff['recipe_id']}]")
        ev.say(f"   pipeline {pipeline if pipeline else '(none found)'}")
        ev.say(f"   honoured {len(tr.agreed)} field(s), "
               f"{len(tr.env)} env lever(s) set")
        ev.say(f"   refused  {len(tr.refusals)} field(s)")
        if no_pipeline_note:
            ev.say("")
            ev.say(f"   NOTE  {no_pipeline_note}")
        ok = not tr.refusals
        ev.done(ok=ok, refusals=len(tr.refusals), agreed=len(tr.agreed),
                env=tr.env, pipeline=str(pipeline) if pipeline else None,
                # The consumer needs to be able to tell "no refusals" from
                # "refusals were never computed". Same key set either way, one
                # honest flag - the alternative is a caller inferring depth of
                # analysis from an empty list, which it cannot do.
                refusals_checked=pipeline is not None)
        return 0 if ok else 1

    # build
    if tr.refusals and not a.allow_refusals:
        ev.done(ok=False, stage="translate", refusals=len(tr.refusals))
        ev.say("   REFUSED - nothing was run.")
        return 3

    from .runner import Runner

    if interpreter:
        ev.say(f"   pipeline interpreter: {interpreter}")
    runner = Runner(rec, pipeline, tr, ev, cwd=pipeline.parent,
                    dryrun=a.dryrun, python=interpreter)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
