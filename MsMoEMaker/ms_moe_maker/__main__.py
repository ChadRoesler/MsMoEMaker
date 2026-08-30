"""Ms.MoE Maker — build targeted Mixtures of Experts from a recipe.

    ms-moe-maker validate recipe.yaml     structure only; no GPU, no network
    ms-moe-maker build    recipe.yaml     full pipeline → GGUF
    ms-moe-maker smoke    recipe.yaml     does the artifact generate at all?
    ms-moe-maker eval     recipe.yaml     routing + quality; dead-expert flag
    ms-moe-maker describe                 one line of JSON, zero side effects

Add --json to any build to get JSON Lines events on stdout and prose on
stderr. That is the stream seren-theatre's stagehand consumes, and it is
declared in _describe.EVENTS as a wire contract: unknown event kinds must be
ignored, so adding one is not breaking, removing or renaming one is.

Each stage is overridable in the recipe (custom scripts).  We provide the
floor; if you want to do your own thing you can replace the stage call with
your own script and pass the path in the recipe's `smoke:` or `eval:` dict.

"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .box import describe as _d
from .box import hardware


def _box_defaults():
    """The layers this install would apply, and what each one sets.

    Reading two small yamls is a READ, not a side effect, so --describe keeps
    its promise. `keys` is provenance: dotted key -> the file that last set it,
    which is the difference between showing a number and explaining it.
    """
    try:
        from .config import defaults as _defaults
        resolved, prov, warns = _defaults.resolve()
        digests = _defaults.file_digests()
        layers = [{"label": label, "path": path,
                   "present": path in digests,
                   "sha256": digests.get(path, "")}
                  for label, path in _defaults.layer_paths()]
        return {"layers": layers, "keys": prov,
                "blocks": sorted(resolved), "warnings": list(warns)}
    except Exception:
        return {"layers": [], "keys": {}, "blocks": [], "warnings": []}


def _box_registry_errors():
    """Entry-point load failures from the pluggable registries.

    A third-party corpus kind or validator that fails to import is exactly the
    thing a person needs told, and it is invisible in the lists above: a kind
    that did not load simply is not there, which looks identical to one that
    was never installed. Carried in --describe so a consumer forking the
    documented command learns about it too, instead of only the callers that
    reach into the package.

    (The reasoning table reports its own problems under `reasoning.warnings`,
    which predates this key and stays where it is.)
    """
    errs = []
    for mod, label in (("data.corpus", "corpus kind"), ("config.validators", "validator")):
        try:
            m = __import__(f"ms_moe_maker.{mod}", fromlist=[mod])
            errs.extend(f"{label} registry: {e}" for e in m.load_errors())
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{label} registry unavailable: {exc}")
    return errs


def _box_validators():
    """The validator registry this install offers.

    Same defensiveness as _box_tiers and _box_reasoning: --describe promises
    exit 0 and one line of JSON, so a broken third-party validator entry point
    degrades to an empty list rather than taking the contract down.
    """
    try:
        from .config import validators as _v
        return _v.describe()
    except Exception:
        return []


def _box_reasoning():
    """Tag styles and model families this install knows about.

    The shaping moved into reasoning.describe(), so this is now a caller like
    any other. It used to be hand-rolled here, which made --describe and
    Backstage's craft form two independent renderings of one registry - the
    same two-implementations-of-one-format bargain seren-theatre's manifest
    contract test exists to police, except nothing was policing this one.
    """
    try:
        from .config import reasoning as _rz
        return {**_rz.describe(), "warnings": _rz.load_errors()}
    except Exception:
        return {"styles": [], "families": [], "warnings": []}


def _box_tiers():
    """Tier names this install offers, floor + whatever the box adds.

    Defensive to a fault because it runs at import time and feeds --describe,
    which promises exit 0 and one line of JSON. A broken defaults file must
    degrade to the floor, never take the contract down with it.
    """
    try:
        from .config import defaults as _defaults
        box, _, _ = _defaults.resolve()
        table, _ = hardware.merge_tiers(box.get("tiers"))
        return sorted(table)
    except Exception:
        return sorted(hardware.TIERS)

# ── allocator policy, set BEFORE anything can initialise CUDA ─────────────
#
# On unified memory the caching allocator's RESERVATION is host RAM. Torch's
# default segment policy strands blocks it cannot reuse, and a MoE decode loop
# - many small gathers of varying token counts, per expert, per layer, per
# step - fragments it badly. Measured on a GB10: 106.6 GB reserved to hold
# 6.4 GB of live tensors, a 16x balloon, 100 GB of it doing nothing. On a
# discrete GPU that is invisible; here it is subtracted from the OS, and the
# OOM killer starts reaping 640 kB daemons because they are the only thing it
# can see.
#
# expandable_segments lets a segment grow instead of stranding blocks. It has
# to be in the environment before the first CUDA allocation, which is why it
# lives at import time in the entry point rather than in eval.py.
#
# An existing value always wins: this is a default, not a mandate.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _version() -> str:
    """The installed version, not a literal. This used to be a hardcoded "1.0"
    sitting next to a setuptools-scm dynamic version in pyproject.toml, which
    is two answers to one question."""
    try:
        from ._version import version  # written by setuptools-scm
        return str(version)
    except Exception:
        try:
            from importlib.metadata import version as _v
            return _v("ms-moe-maker")
        except Exception:
            return "0.0.0.dev0"


# ── helpers ──────────────────────────────────────────────────────────────────

def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


class _Tee:
    """Write to two streams at once, flushing both."""
    def __init__(self, primary, secondary):
        self._a, self._b = primary, secondary

    def write(self, text):
        self._a.write(text)
        try:
            self._b.write(text)
            self._b.flush()
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        for s in (self._a, self._b):
            try:
                s.flush()
            except (OSError, ValueError):
                pass


def _box_corpus_kinds():
    """The corpus registry as ROWS, for --describe and Backstage's craft form.

    Matches `validators` beside it in the payload and `recipe.DESCRIBE`'s key
    of the same name, which was already the rich shape. This key used to be
    bare names, so one package shipped two payloads that both called something
    `kinds` and meant different things - and a consumer building a form off the
    CLI's copy got words with no summary and no `requires` to render.

    Names stay one comprehension away: [k["name"] for k in kinds].
    """
    try:
        from .data import corpus
        return corpus.describe()
    except Exception:
        return []


# ONE SOURCE OF TRUTH, AND MERGED RATHER THAN RETYPED.
#
# There used to be three lists of verbs that disagreed: this dict said five,
# `_describe.COMMANDS` said three, and the module docstring said "Three verbs".
# _describe is canonical - it is stdlib-only by design so it can answer on a
# half-installed tool, and stagehand reads it to check which contract version
# the thing it forked speaks.
#
# The first fix rebuilt this dict by naming _describe's keys ONE BY ONE, and
# promptly dropped `requires` - a published contract key that release CI
# asserts. Hand-copying a subset of a source of truth is not using the source
# of truth, it is making a second one that starts out correct.
#
# So: start from _describe.DESCRIBE whole, then add the keys that only the
# installed CLI can answer. A key added over there can never be lost here.
DESCRIBE = {
    **_d.DESCRIBE,
    "version": _version(),
    "kinds": _box_corpus_kinds(),
    "gates": ["auto", "manual", "skip"],
    "templates": ["code", "dnd", "math", "culinary"],
    # THE TIERS THIS INSTALL ACTUALLY HAS, not the ones the source ships. A box
    # can add or redefine a tier from its defaults file, and --describe is how
    # Starwright and Theatre find out what a machine offers - reporting the
    # built-in floor there would describe a different install than the one
    # answering. Reading two small yamls is a read, not a side effect.
    "tiers": _box_tiers(),
    # WHAT THIS BOX PRESETS, so a front-end can show a machine's configuration
    # without re-implementing the merge. Same defensiveness as _box_tiers: a
    # broken file degrades to "nothing configured", never to a broken contract.
    "defaults": _box_defaults(),
    "reasoning": _box_reasoning(),
    # THE VALIDATOR REGISTRY, WHICH SAID IT WAS HERE AND WAS NOT.
    # validators.describe()'s own docstring reads "for --describe and
    # Backstage's craft form", and it reached the craft form only because
    # seren-theatre imported the module directly rather than asking this
    # command. A consumer that has to reach into the package to learn what the
    # box offers is a consumer that breaks every time the package moves - and
    # `--describe` exists precisely so nobody has to.
    "validators": _box_validators(),
    "registry_errors": _box_registry_errors(),
    "modes": list(_d.EVAL_MODES),
}


# ── the verbs ────────────────────────────────────────────────────────────────────────────
#
# The command implementations live in ms_moe_maker/cli/, one module per verb;
# the entry point keeps argparse and this dispatch table. _print_eval_report
# and _defaults_template_body are re-exported because tests (and any external
# caller) historically imported them from __main__.
from .cli import (_cmd_build, _cmd_corpus, _cmd_eval, _cmd_export, _cmd_init,
                  _cmd_smoke, _cmd_validate, _print_eval_report,
                  _defaults_template_body)


# The dispatch table, at module scope so it is one definition rather than a
# local that only main() can see. `describe` is handled before argparse (zero
# side effects, per the Starwright contract) so it has no entry here - which is
# the one asymmetry, and it is asserted in the tests rather than assumed.
COMMAND_HANDLERS = {
    "init": _cmd_init,
    "build": _cmd_build,
    "corpus": _cmd_corpus,
    "smoke": _cmd_smoke,
    "eval": _cmd_eval,
    "validate": _cmd_validate,
    "export": _cmd_export,
}


def main(argv=None):
    _force_utf8_stdio()
    
    # --describe scanned before argparse for zero-side-effects.
    if "--describe" in (argv or sys.argv):
        print(json.dumps(DESCRIBE))
        return 0

    # Load `.env` (HF_TOKEN, HF_HOME, MSMOE_*) before anything touches the
    # network or the config. The shell still wins over the file, so an
    # explicit export or `huggingface-cli login` always takes precedence.
    from .box.dotenv import load_dotenv
    load_dotenv()
    
    ap = argparse.ArgumentParser(
        prog="ms-moe-maker",
        description="Ms.MoE Maker — build targeted Mixtures of Experts from a recipe.",
    )
    ap.add_argument("command", nargs="?", default="build",
                    choices=list(_d.COMMANDS),
                    help="command to run (default: build)")
    ap.add_argument("recipe", nargs="?", help="path to recipe .yaml or .json")
    # TWO MEANINGS, TWO FLAGS. --dryrun used to do both jobs badly: _cmd_build
    # printed a plan and returned 0 before ever calling the pipeline, while
    # run_pipeline's own signature documents dryrun as "run on the smallest
    # rung for structural testing" and config reads MSMOE_DRYRUN the same way.
    # So the flag that was supposed to select a cheap REAL build instead
    # guaranteed no build happened, and the smallest-rung path was unreachable
    # from the CLI entirely.
    #
    #   --plan    resolve everything, run nothing. No torch, no GPU, exit 0.
    #   --dryrun  a real build on the smallest rung. Needs torch, like a build.
    ap.add_argument("--plan", action="store_true",
                    help="resolve config and stages, run nothing (no GPU)")
    # THE FILE THAT MAKES A RECIPE PORTABLE AGAIN. Defaults normally come from
    # this box (~/.msmoe/defaults.yaml), which is the point - but it also means
    # the same recipe can build differently in two places. Point at an explicit
    # file to reproduce someone else's run, or to pin a build in CI.
    ap.add_argument("--defaults-template", action="store_true",
                    dest="defaults_template",
                    help="init: write a commented defaults file for this box "
                         "(default: ~/.msmoe/defaults.yaml)")
    ap.add_argument("--defaults", metavar="PATH", default=None,
                    help="defaults file to layer under the recipe "
                         "(default: packaged, then ~/.msmoe/defaults.yaml)")
    ap.add_argument("--offline", action="store_true",
                    help="skip reachability checks (no network calls)")
    ap.add_argument("--dryrun", action="store_true",
                    help="real build on the smallest rung (still needs torch)")
    ap.add_argument("--force", action="store_true",
                    help="redo existing artifacts")
    # THE FLAG THAT WAS MISSING. _describe.EVENTS declares a JSON Lines wire
    # vocabulary and calls renaming an event a breaking change - but there was
    # no --json argument at all, so argparse rejected it with exit 2. Since
    # seren-theatre's stagehand forks the literal documented command
    # `ms-moe-maker build <recipe> --json`, every Theatre-driven build died at
    # argument parsing. Events was even imported here and never used.
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events on stdout, prose on stderr")
    ap.add_argument("--pipeline", dest="pipeline",
                    help="fork this legacy pipeline script instead of the "
                         "in-package builder")
    ap.add_argument("--python", dest="python", default=None,
                    help="interpreter that runs the pipeline (default: ours). "
                         "The trainer lives in whatever venv has torch; this "
                         "CLI does not have to.")
    
    # Smoke args
    # Default 0, not the real default: it is the only way to tell "the user
    # did not say" from "the user asked for the same number", and the recipe
    # has to win when they did not say.
    ap.add_argument("--tokens", type=int, default=0,
                    help="smoke test token count (default: the recipe's, or 48)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="smoke timeout in seconds (default: the recipe's, or 300)")
    
    # init args
    ap.add_argument("--template", dest="template", default="",
                    help="start from a template (see `describe`)")
    ap.add_argument("-o", "--output", dest="output", default="-",
                    help="write here instead of stdout")

    # Eval args
    # SLICED BY QUESTION, NOT BY MODEL. This used to be expert|moe|all, which
    # asks "which model do I run" - but the two things worth knowing are
    # separate questions: does the router prefer each expert on its own ground
    # (routing, the dead-expert claim), and does the thing answer well
    # (quality, which needs an answer key only the corpus author has).
    # Default is empty so the recipe's eval.mode wins unless overridden here.
    ap.add_argument("--prune", action="store_true",
                    help="corpus: WRITE a pruned copy next to the original "
                         "(never in place). Without it, only propose.")
    ap.add_argument("--per-repo-cap", dest="per_repo_cap", type=int, default=0,
                    help="corpus: max docs per repo when pruning (default 20)")
    ap.add_argument("--mode", dest="eval_mode",
                    choices=list(_d.EVAL_MODES), default="",
                    help="eval mode (default: the recipe's eval.mode, or all)")
    
    a = ap.parse_args(argv)
    
    if a.command == "describe":
        print(json.dumps(DESCRIBE))
        return 0
    
    # init invents a recipe rather than reading one, so it is the other verb
    # that does not want a path.
    if not a.recipe and a.command not in ("describe", "init"):
        ap.error(f"'{a.command}' requires a recipe path")
    
    handler = COMMAND_HANDLERS.get(a.command)
    if handler is None:
        ap.error(f"unknown command: {a.command}")
    
    return handler(a)


if __name__ == "__main__":
    raise SystemExit(main())
