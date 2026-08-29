"""How specialists become one model: stitching and export.

Deliberately lazy. `_moe_stitch` is vendored code that imports torch at MODULE
level, and `preflight` imports `export` on torch-less laptops (`validate` and
`build --plan` must answer without the training stack). An eager `__init__`
would drag torch into every one of those answers, so the siblings load on
first attribute access instead.
"""
from __future__ import annotations

__all__ = ["_moe_stitch", "export", "stitch"]


def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
