"""`ms-moe-maker export` - GGUF export + smoke for an already-built MoE.

The README promised this verb twice; argparse refused it. It re-runs the
export stage against the router's output directory, so a build that skipped
GGUF (no llama.cpp at the time) can be finished later, or a smoke re-run after
llama-cli lands on the box.
"""
from __future__ import annotations

import sys

from ._common import _load_recipe


def _cmd_export(args):
    rec, errs, warns = _load_recipe(
        args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        return 1

    import os

    from ..config.pipeline import build_config
    from ..train.router import router_dir
    from ..moe.export import export_gguf

    config = build_config(rec, force=args.force)
    final_dir = router_dir(config)
    if not os.path.isdir(final_dir):
        print(f"ERROR: no trained MoE at {final_dir}.", file=sys.stderr)
        print(f"  Run 'ms-moe-maker build {args.recipe}' first.",
              file=sys.stderr)
        return 1

    print(f"\nExporting GGUF from {final_dir}")
    try:
        gguf = export_gguf(config, final_dir)
    except Exception as exc:
        print(f"\nExport FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    if gguf is None:
        print("\nGGUF export skipped (no llama.cpp converter).")
        return 1
    print(f"\n[ok] GGUF at {gguf}")
    return 0
