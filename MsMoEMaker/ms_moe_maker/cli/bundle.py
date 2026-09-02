"""`ms-moe-maker bundle` - freeze a recipe and everything needed to hand it over.

The verb exists because of one silent failure. A recipe is mostly sentinels
meaning "you decide", so giving somebody `recipe.yaml` gives them your
intentions and their defaults - same file, different model, no error anywhere.
This resolves the recipe HERE and writes the answers back into it, so there is
nothing left for the far box to decide.

WHAT IT WILL NOT DO IS LIE ABOUT THE PART IT CANNOT FIX. Sixteen fingerprint
fields have no recipe key at all - three come from environment variables,
twelve are literals in build_config, one is a CLI flag - and they are printed,
by name, every time. They also go into the manifest so the far side can diff
them. See knobs.UNPINNABLE.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ._common import _load_recipe


def _cmd_bundle(args):
    rec, errs, warns = _load_recipe(args.recipe,
                                    defaults_path=getattr(args, "defaults", None))
    if rec is None:
        return 1

    import yaml

    from ..bundle import pack
    from ..bundle import stamp as stamper
    from ..config.knobs import UNPINNABLE
    from ..config.pipeline import build_config, build_fingerprint, build_id

    config = build_config(rec, force=getattr(args, "force", False))
    fingerprint = build_fingerprint(config)
    ident = build_id(config)

    # THE RAW DICT, NOT THE PARSED RECIPE. Only the raw text still knows the
    # difference between a key the author typed and one a default supplied,
    # and marking that difference is half the value of the output.
    with open(args.recipe, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        print(f"{args.recipe} does not contain a recipe mapping.",
              file=sys.stderr)
        return 1

    stamped, marks = stamper.stamp(raw, config)
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        from .._version import __version__ as tool_version
    except Exception:                                   # noqa: BLE001
        tool_version = "unknown"

    header = [
        f"# Stamped by ms-moe-maker {tool_version} on {when}.",
        "#",
        "# Every line marked `# default` was filled in from this box\'s",
        "# resolved config. The lines WITHOUT that marker are the ones",
        "# somebody actually chose - those are the knobs to look at first.",
        "#",
        f"# This recipe resolves to build_id {ident}. If it does not resolve",
        "# to that on your box, `bundle.json` says which fields differ.",
        "",
    ]
    text = stamper.render(stamped, marks, header=header)

    # VERIFY BEFORE WRITING, NOT AFTER. A stamp that missed a field produces a
    # recipe that builds something else, and the person who would find that out
    # is the one it was given to. So the exporter loads its own output back,
    # resolves it, and compares - and a bundle that does not round-trip is not
    # written at all.
    drift = stamper.verify(text, fingerprint,
                           defaults_path=getattr(args, "defaults", None))
    if drift:
        print("\nREFUSING TO WRITE: the stamped recipe does not rebuild to the "
              "same fingerprint.", file=sys.stderr)
        print("This is a bug in the stamper, not in your recipe. These fields "
              "moved:", file=sys.stderr)
        for name, how in sorted(drift.items()):
            print(f"    {name}: {how['theirs']!r} -> {how['ours']!r}",
                  file=sys.stderr)
        print("\n  Most likely one of them is missing a `recipe=` path in "
              "config/knobs.py.", file=sys.stderr)
        return 2

    notes = ""
    if getattr(args, "notes", None):
        try:
            notes = Path(args.notes).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"could not read --notes {args.notes}: {exc}", file=sys.stderr)
            return 1

    # Corpora, if asked for. OFF BY DEFAULT and the size is printed either way:
    # a synth corpus runs to gigabytes, and handing somebody a surprise 3 GB
    # download is a worse gift than a small one plus a sentence.
    data_dirs = []
    data_root = Path(config.data_root)
    for name in (config.expert_names or []):
        for candidate in (data_root / name, data_root / f"{name}_synth"):
            if candidate.is_dir():
                data_dirs.append((name, candidate))
                break
    if not getattr(args, "with_data", False):
        skipped = sum(f.stat().st_size
                      for _, d in data_dirs for f in d.rglob("*") if f.is_file())
        if data_dirs:
            print(f"\n  {len(data_dirs)} corpus director(ies) NOT included "
                  f"({skipped / 1e6:.1f} MB). Add --with-data to bundle them.")
        data_dirs = []

    meta = {
        "name": rec.name or Path(args.recipe).stem,
        "created": time.time(),
        "created_human": when,
        "tool": "ms-moe-maker",
        "tool_version": tool_version,
        "build_id": ident,
        "source_recipe": Path(args.recipe).name,
        "stamped_keys": sorted(f"{b}.{k}" for b, k in marks),
        "authored_keys": sorted(
            f"{b}.{k}" for b in raw if isinstance(raw.get(b), dict)
            for k in raw[b]),
        # THE WHOLE RESOLVED FINGERPRINT. This is what makes import a DIFF
        # rather than an act of faith - including for the sixteen fields the
        # recipe has no way to carry.
        "fingerprint": fingerprint,
        "unpinnable": stamper.unpinnable_snapshot(config),
        "unpinnable_why": dict(sorted(UNPINNABLE.items())),
    }

    out = Path(getattr(args, "out", None)
               or f"{meta['name']}.zip")
    pack.write(out, recipe_text=text, meta=meta, data_dirs=data_dirs,
               notes=notes)

    print(f"\n  bundle -> {out}")
    print(f"  build_id {ident} · {len(marks)} defaults stamped · "
          f"{len(meta['fingerprint'])} fields recorded")
    print(f"\n  [!] {len(UNPINNABLE)} fields cannot be written into a recipe "
          f"and will follow the OTHER box:")
    for name in sorted(UNPINNABLE):
        print(f"        {name:30} = {meta['unpinnable'][name]!r}")
    print("      They are recorded in bundle.json, so an import can say which "
          "of them\n      differ rather than leaving it to be discovered.")
    return 0
