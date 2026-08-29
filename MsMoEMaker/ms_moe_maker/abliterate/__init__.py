"""# The decensor pass: the stage wrapper and the vendored Heretic core."""
from __future__ import annotations

from . import heretic
from . import stage

__all__ = ['heretic', 'stage']


def __getattr__(name):
    """Historical API: abliterate.py's names now live on .stage
    (abliterate_base). Forward so `from ms_moe_maker import abliterate` +
    `abliterate.abliterate_base(...)` keeps working."""
    return getattr(stage, name)
