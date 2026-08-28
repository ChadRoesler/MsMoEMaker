"""`ms-moe-maker smoke` - does the GGUF generate at all, outside Python?"""
from __future__ import annotations

import sys

from ._common import _build_output_dir, _find_gguf, _load_recipe


def _cmd_smoke(args):
    """Smoke-test the GGUF model — proves it generates outside Python."""
    from ..export import smoke_gguf
    
    rec, errs, warns = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        return 1
    
    from ..config import build_config
    config = build_config(rec)

    output_dir = _build_output_dir(rec)
    if not output_dir:
        print("ERROR: could not resolve output directory from recipe.")
        return 1
    
    gguf_path = _find_gguf(output_dir)
    if not gguf_path:
        print(f"ERROR: no GGUF file found in {output_dir}.")
        print(f"  Run 'ms-moe-maker build {args.recipe}' first.")
        return 1
    
    print(f"\n{'=' * 60}")
    print(f"Smoking GGUF: {gguf_path}")
    print(f"{'=' * 60}")
    
    if args.dryrun:
        print(f"[DRY-RUN] Would smoke-test: {gguf_path}")
        return 0
    
    try:
        # THE RECIPE IS THE FLOOR, THE FLAGS OVERRIDE IT. rec.smoke was
        # documented, parsed, and then ignored here - exactly the gap eval had,
        # where a block the README tells people to write reached nothing.
        #
        # llama_cpp_dir is passed because smoke previously searched PATH alone
        # while the build searched the llama.cpp tree, so an ordinary checkout
        # worked during a build and not for the command whose whole job is to
        # run it.
        ok = smoke_gguf(
            gguf_path,
            tokens=args.tokens or rec.smoke.tokens,
            timeout=args.timeout or rec.smoke.timeout,
            prompt=rec.smoke.prompt,
            llama_cpp_dir=config.llama_cpp_dir,
        )
    except Exception as exc:
        print(f"\nSmoke test FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    
    if ok:
        print("\nSmoke test PASSED — model generates text outside Python.")
        return 0
    else:
        print("\nSmoke test FAILED — model did not produce valid output.")
        return 2
