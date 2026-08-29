"""# What the experts learn from: corpus collection and synthetic generation."""
from __future__ import annotations

from . import corpus
from . import health
from . import synth

__all__ = ['corpus', 'health', 'synth']


def __getattr__(name):
    """Historical API: data.py's names now live on .synth (collect_corpus,
    generate_*_traces, STACK_REPO, ...). Forward so `from ms_moe_maker import
    data` + `data.collect_corpus(...)` keeps working."""
    return getattr(synth, name)
