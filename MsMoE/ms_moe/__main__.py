"""The ms-moe CLI - the command stagehand forks and a person types.

Three verbs:

    ms-moe describe                     one line of JSON, exit 0, no side effects
    ms-moe validate recipe.yaml         parse + check, touching nothing
    ms-moe build recipe.yaml            translate, fork the pipeline, report

`ms-moe build recipe.yaml` is deliberately the literal string in the README and
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


def _find_pipeline(explicit: str | None, recipe_path: Path) -> Path:
    """Locate fraunkenstein_universal.py.

    Looked up rather than assumed, and reported when missing, because "wrapped
    a script that isn't there" should be one clear error and not a traceback
    from subprocess. Order: --pipeline, beside the recipe, cwd, then upward.
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
    raise SystemExit(
        f"could not find {DEFAULT_PIPELINE}. Pass --pipeline PATH, or run "
        f"from the directory that holds it.")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Before argparse: zero side effects, works half-installed.
    if "--describe" in argv or (argv and argv[0] == "describe"):
        print(json.dumps(DESCRIBE))
        return 0

    ap = argparse.ArgumentParser(
        prog="ms-moe",
        description="Build a mixture of experts from a recipe.")
    ap.add_argument("command", choices=["build", "validate", "describe"])
    ap.add_argument("recipe", nargs="?", help="path to the recipe .yaml")
    ap.add_argument("--pipeline", default=None,
                    help="path to fraunkenstein_universal.py (default: found "
                         "beside the recipe, then upward from cwd)")
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events on stdout, prose on stderr")
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

    pipeline = _find_pipeline(a.pipeline, recipe_path)

    from .levers import translate

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
        ev.say(f"   pipeline {pipeline}")
        ev.say(f"   honoured {len(tr.agreed)} field(s), "
               f"{len(tr.env)} env lever(s) set")
        ev.say(f"   refused  {len(tr.refusals)} field(s)")
        ok = not tr.refusals
        ev.done(ok=ok, refusals=len(tr.refusals), agreed=len(tr.agreed),
                env=tr.env)
        return 0 if ok else 1

    # build
    if tr.refusals and not a.allow_refusals:
        ev.done(ok=False, stage="translate", refusals=len(tr.refusals))
        ev.say("   REFUSED - nothing was run.")
        return 3

    from .runner import Runner

    runner = Runner(rec, pipeline, tr, ev, cwd=pipeline.parent,
                    dryrun=a.dryrun)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
