"""`ms-moe-maker validate` - structure only: no GPU, no network."""
from __future__ import annotations

from ..events import Events
from ._common import _corpus_paths, _load_recipe


def _cmd_validate(args):
    """Validate recipe structure only — no pipeline, no GPU, no network.

    --json WORKS HERE TOO, and that is the point of the flag. It used to be
    wired into `build` alone, so `ms-moe-maker validate r.yaml --json` was
    accepted by argparse and then printed prose - a machine consumer got an
    empty event stream and no way to tell "valid" from "the flag did nothing".
    A wire format that only some verbs speak is not a wire format.

    Validate has no stages, so the stream is short by nature: started, a
    warning per warning, an error per error, and a terminal done. Terminal is
    the part that matters - a consumer following the stream needs one event
    that means "there will be no more".
    """
    events = Events(enabled=bool(args.json))
    say = events.say if args.json else print

    rec, errs, warns = _load_recipe(args.recipe, quiet=bool(args.json), defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        events.emit("started", recipe=str(args.recipe))
        for e in (errs or []):
            events.error(stage="validate", message=e)
        if errs is None:
            events.error(stage="validate", message=f"could not parse {args.recipe}")
        events.done(ok=False, errors=len(errs or []) or 1, warnings=0)
        return 1

    events.emit("started", recipe=str(args.recipe), recipe_id=rec.recipe_id(),
                name=rec.name, size=rec.size,
                experts=[e.name for e in rec.experts])

    say(f"\n  Recipe: {rec.name or '(auto-filled)'}  [{rec.recipe_id()}]")
    # The recipe id is what you wrote; the build id is what this box will make
    # of it. Printing only the first is how "but it works on mine" happens.
    try:
        from ..config import build_config as _bc, build_id as _bid
        say(f"  Build:  {_bid(_bc(rec, dryrun=False))}")
    except Exception:
        pass
    say(f"  Base:   {rec.base or '(auto-filled from tier)'}")
    say(f"  Size:   {rec.size}")
    say(f"  Experts: {[e.name for e in rec.experts]}")
    say(f"  Template: {rec.template or '(none)'}")

    # WHERE EVERY NON-RECIPE VALUE CAME FROM. Defaults live in a file on this
    # box on purpose - so a machine can be set up once for someone else - and
    # the cost of that is a recipe that no longer fully describes its own
    # build. Provenance is what keeps that honest, and validate is the command
    # people run when something is surprising.
    prov = getattr(rec, "defaults_provenance", None) or {}
    if prov:
        say(f"\n  DEFAULTS ({len(prov)} from outside the recipe):")
        for _k, _v in sorted(prov.items()):
            say(f"    {_k:28} <- {_v}")
        # ON THE WIRE NOW. `defaults` is declared in _describe.EVENTS, and the
        # rule there is explicit: a consumer that does not know a kind ignores
        # it, so adding one is additive. It was held back until there was
        # something to say - a vocabulary is easier to add to than to take back.
        events.emit("defaults", provenance=prov,
                    files=dict(getattr(rec, "defaults_digests", None) or {}))

    if warns:
        say(f"\n  WARNINGS ({len(warns)}):")
        for w in warns:
            say(f"    · {w}")
            events.warning(w)
    else:
        say("\n  No warnings.")

    # Refusals are a legitimate answer, not a failure: they are fields the
    # recipe asked for that this build cannot honour. Named on the wire so a
    # consumer can show them without parsing prose.
    from ..levers import translate
    refusals = translate(rec).refusals
    if refusals:
        events.refused(refusals)
        say(f"\n  REFUSED ({len(refusals)}):")
        for r in refusals:
            say(f"    ✗ {r}")

    # CORPUS HEALTH, FOR WHATEVER IS ALREADY ON DISK.
    #
    # It belongs here because it is pure stdlib and honours the laptop promise
    # - no torch, no GPU, no network - and because after a build `validate`
    # becomes a re-check you can run in a second. Before a build there is
    # nothing to read, and it says that rather than printing nothing, since a
    # check that vanishes reads like a check that passed.
    findings = _validate_corpora(rec, say, events)

    # `errs` is always empty here - _load_recipe returns rec=None whenever it
    # is not - so printing "0 errors" was theatre. Say what is true.
    say(f"\n  Valid. {len(warns)} warning(s), "
        f"{findings} corpus finding(s).")
    events.done(ok=True, warnings=len(warns), refusals=len(refusals),
                corpus_findings=findings)
    return 0


def _validate_corpora(rec, say, events) -> int:
    """Report on every corpus that exists. Never builds one."""
    from .. import corpus as corpus_mod
    from .. import corpushealth as ch

    paths = _corpus_paths(rec)
    if not any(paths.values()):
        say("\n  Corpora: none on disk yet - run `build` first, then "
            "`validate` re-checks them.")
        return 0

    say("\n  CORPUS HEALTH")
    total = 0
    for e in rec.experts:
        path = paths.get(e.name) or ""
        if not path:
            say(f"  {e.name}: not collected yet")
            continue
        kind = corpus_mod.get(getattr(e.source, "kind", "")) if e.source else None
        generated = bool(getattr(kind, "generated", False))
        h = ch.inspect(path, generated=generated)
        say(ch.format_health(h))
        for f in h.findings:
            events.warning(f"corpus/{e.name}: {f}")
        total += len(h.findings)
    return total
