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

__all__ = ["DESCRIBE", "NAME", "__version__"]
