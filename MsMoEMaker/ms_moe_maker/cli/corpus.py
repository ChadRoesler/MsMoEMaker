"""`ms-moe-maker corpus` - inspect on-disk corpora; --prune proposes a cleaner one."""
from __future__ import annotations

from ._common import _corpus_paths, _load_recipe


def _cmd_corpus(args):
    """Inspect the corpora on disk. With --prune, PROPOSE a cleaner one.

    PROPOSE, NEVER COMMIT. Without --prune this only measures. With --prune it
    writes a NEW file next to the original and leaves the original untouched,
    because a machine deciding unattended that some of your data should not
    exist is the same shape as a consolidator writing to long-term without a
    human - and this project built a gate to stop exactly that. You read the
    proposal, you point the recipe at the pruned file if you agree, and if you
    disagree nothing has happened.

    Note what prune CANNOT fix: repo dominance on a corpus collected before
    provenance stamping. Those rows are `{"text": ...}` with no `repo` field,
    so the rule that matters most cannot run and says so rather than pruning
    on the two weaker signals and reporting success.
    """
    from ..data import corpus as corpus_mod
    from ..data import health as ch

    rec, errs, _ = _load_recipe(args.recipe, defaults_path=getattr(args, 'defaults', None))
    if rec is None:
        for e in (errs or [f"could not parse {args.recipe}"]):
            print(f"  ✗ {e}")
        return 1

    paths = _corpus_paths(rec)
    if not any(paths.values()):
        print("No corpora on disk for this recipe. Run `build` first.")
        return 3

    findings = 0
    for e in rec.experts:
        path = paths.get(e.name) or ""
        if not path:
            print(f"\n  {e.name}: not collected yet")
            continue
        kind = corpus_mod.get(getattr(e.source, "kind", "")) if e.source else None
        generated = bool(getattr(kind, "generated", False))
        h = ch.inspect(path, generated=generated)
        print()
        print(ch.format_health(h))
        findings += len(h.findings)

        cap = int(getattr(args, "per_repo_cap", 0) or 20)
        if args.prune:
            out_path = path.replace(".jsonl", ".pruned.jsonl")
            pr = ch.write_pruned(path, out_path, per_repo_cap=cap)
            print(f"      wrote {out_path}: kept {pr.keep:,}, dropped {pr.drop:,}")
        else:
            pr = ch.propose_prune(path, per_repo_cap=cap)
            print(f"      --prune would keep {pr.keep:,} and drop {pr.drop:,}")
        for reason, n in pr.reasons.most_common():
            print(f"        {n:>7,}  {reason}")
        for u in pr.unmeasured:
            print(f"        [?] {u}")
        if not args.prune and pr.drop:
            print(f"      (nothing written - re-run with --prune to produce "
                  f"{path.replace('.jsonl', '.pruned.jsonl')})")

    return 0 if not findings else 0
