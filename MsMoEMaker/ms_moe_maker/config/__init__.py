"""# The recipe -> build bridge: what you asked for, resolved into what a build is."""
from __future__ import annotations

from . import defaults
from . import knobs
from . import levers
from . import pipeline
from . import reasoning
from . import recipe
from . import templates
from . import validators

__all__ = ['defaults', 'knobs', 'levers', 'pipeline', 'reasoning', 'recipe',
           'templates', 'validators']


def __getattr__(name):
    """Historical API: config.py's names now live on .pipeline (build_config,
    resolve_roots, PipelineConfig, ...). Forward so `from ms_moe_maker import
    config` + `config.build_config(...)` keeps working."""
    return getattr(pipeline, name)
