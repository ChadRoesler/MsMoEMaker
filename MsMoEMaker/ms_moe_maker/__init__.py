"""Ms.MoE - Multi-Specified Mixture of Experts.

Five deliberate experts instead of a hundred lottery tickets. The design thesis
is the INVERSE of a frontier MoE: hand-assign the domains so every expert has a
guaranteed constituency, which eliminates dead and collapsed experts by
construction rather than fighting them with a load-balancing auxiliary loss.
Because each expert does exactly one thing, you can retrain ONE and re-splice
without touching the others - which is what makes it maintainable by one
person.

The real product is the factory, not the model. Swap the expert list and
someone else gets their own Ms.MoE, shaped like THEIR stack.

This package deliberately depends on nothing of Seren's. seren-theatre can
watch a run, and seren-theatre[stagehand] can start one, but the arrow only
points that way - and even then the two never speak, they share a directory.
Opt in, never opt out.
"""
from __future__ import annotations

try:
    from ._version import version as __version__
except Exception:  # noqa: BLE001 - source checkout without a build
    __version__ = "0.0.0+unknown"

from .box.describe import DESCRIBE, NAME  # noqa: F401  (stdlib-only, safe here)

# ── historical module names, now one level down ─────────────────────────────
#
# The flat modules moved into phase subpackages (see cli/__init__ or any
# subpackage docstring for the map). The dotted paths are canonical; these
# aliases keep the historical names importable for one-liners and callers
# that predate the move, e.g. `from ms_moe_maker import manifest`.
#
# LAZY ON PURPOSE: several aliased modules import torch (moe/_moe_stitch does
# it at module level), and `import ms_moe_maker` must stay instant and
# stdlib-light - it is the describe promise on a half-installed tool. A name
# resolves on first access, and is then registered in sys.modules so
# `import ms_moe_maker.<name>` works from that point on too.
#
# Names that BECAME packages (config, data, eval, abliterate) are not here:
# they already resolve, and each of those packages forwards its historical
# API in its own __getattr__ (config -> pipeline, data -> synth, eval ->
# harness, abliterate -> stage).
_COMPAT = {
    "recipe": "config.recipe",
    "defaults": "config.defaults",
    "reasoning": "config.reasoning",
    "template": "config.templates",
    "validators": "config.validators",
    "levers": "config.levers",
    "corpus": "data.corpus",
    "corpushealth": "data.health",
    "finetune": "train.finetune",
    "router": "train.router",
    "experts": "train.experts",
    "stitch": "moe.stitch",
    "_moe_stitch": "moe._moe_stitch",
    "export": "moe.export",
    "evalrecord": "eval.record",
    "builder": "run.builder",
    "runner": "run.runner",
    "stages": "run.stages",
    "manifest": "run.manifest",
    "events": "run.events",
    "preflight": "run.preflight",
    "heretic": "abliterate.heretic",
    "hardware": "box.hardware",
    "dotenv": "box.dotenv",
    "_describe": "box.describe",
}


def __getattr__(name):
    if name in _COMPAT:
        import importlib
        import sys
        mod = importlib.import_module(f"{__name__}.{_COMPAT[name]}")
        sys.modules.setdefault(f"{__name__}.{name}", mod)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DESCRIBE", "NAME", "__version__"]
