"""Command implementations - one module per verb.

__main__ keeps the argparse wiring and the dispatch table; the verbs
themselves live here so the entry point stays an entry point.
"""
from __future__ import annotations

from .build import _cmd_build
from .corpus import _cmd_corpus
from .eval import _cmd_eval, _print_eval_report
from .export import _cmd_export
from .init import _cmd_init, _defaults_template_body
from .smoke import _cmd_smoke
from .validate import _cmd_validate

__all__ = [
    "_cmd_build",
    "_cmd_corpus",
    "_cmd_eval",
    "_cmd_export",
    "_cmd_init",
    "_cmd_smoke",
    "_cmd_validate",
    # Re-exported for tests and any caller that historically imported the
    # command-layer helpers from ms_moe_maker.__main__.
    "_print_eval_report",
    "_defaults_template_body",
]
