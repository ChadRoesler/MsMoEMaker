"""Shared helpers for the CLI commands.

Home of the small, pure pieces more than one verb needs: loading a
recipe, resolving its output dir, finding a GGUF, and the corpus paths
query. They moved out of __main__.py when the command implementations
moved into cli/ - shared state belongs with its consumers, not with the
entry point.
"""
from __future__ import annotations

import os
import sys
from typing import Dict


def _corpus_kinds():
    """The registered corpus KIND NAMES, for prose.

    `init` joins these into a comment line at the top of a starter recipe, so
    this one stays a list of words. The rich rows that a form is built from are
    __main__._box_corpus_kinds() - two callers, two shapes, and the split is
    kept (here vs there) because a `.join()` over dicts is a TypeError at the
    worst possible moment: while somebody is generating their first recipe.
    """
    from ..data import corpus
    return corpus.names()


def _load_recipe(path, quiet: bool = False, defaults_path=None):
    """Load and validate a recipe file. Returns (Recipe, errs, warns).

    `quiet` exists because under --json stdout belongs to the event stream and
    nothing else may write to it. Prose goes to stderr or nowhere; a stray
    print here would corrupt the very format a consumer is parsing.
    """
    from ..config.recipe import load, validate as validate_recipe

    def out(msg):
        if quiet:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    try:
        rec, parse_warns = load(path, defaults_path=defaults_path)
    except Exception as exc:
        out(f"FAILED to parse {path}: {exc}")
        return None, None, None

    errs, warns = validate_recipe(rec)
    warns = parse_warns + warns

    if errs:
        out(f"\nRecipe has {len(errs)} error(s):")
        for e in errs:
            out(f"  ✗ {e}")
        return None, errs, warns

    return rec, errs, warns


def _build_output_dir(rec) -> str:
    """Find the output directory from the recipe or config."""
    from ..config.pipeline import build_config
    
    try:
        config = build_config(rec)
        return config.output_root
    except Exception:
        return ""


def _find_gguf(output_dir: str):
    """Find the GGUF file in the output directory."""
    from pathlib import Path
    for f in Path(output_dir).rglob("*.gguf"):
        if f.is_file():
            return str(f)
    return ""


def _corpus_paths(rec) -> Dict[str, str]:
    """Where this recipe's corpora would live, without building anything."""
    from ..config import pipeline as cfg_module
    roots = cfg_module.resolve_run_roots(rec)
    data_root = roots["data"]
    out: Dict[str, str] = {}
    for e in rec.experts:
        safe = cfg_module.safe_name(e.name)
        for cand in (f"{data_root}/{safe}_code.jsonl", f"{data_root}/{safe}.jsonl"):
            if os.path.isfile(cand):
                out[e.name] = cand
                break
        else:
            out[e.name] = ""
    return out
