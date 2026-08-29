"""# How good is it: measurement and the eval record."""
from __future__ import annotations

from . import harness
from . import record

__all__ = ['harness', 'record']


def __getattr__(name):
    """Historical API: eval.py's names now live on .harness (run_eval,
    EvalReport, detect_dead_experts, ...). Forward so `from ms_moe_maker import
    eval` + `eval.run_eval(...)` keeps working."""
    return getattr(harness, name)
