# Recipe Deep Dive: Corpus Strategy

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/TROUBLESHOOTING.md`

## Core ideas

- `min_samples` is a floor, not a target.
- `router_mix_total` affects router step budget and can raise practical corpus needs.
- `per_repo_cap` protects against single-repo dominance.

## Practical checklist

- Start with realistic `max_samples` for your run budget.
- Keep `per_repo_cap` conservative for diversity.
- Use `corpus` command to inspect quality and propose pruning when needed.

## Failure pattern to watch

- Corpus passes, specialists train, router underperforms due to insufficient mix feed.

Mitigation:

- Revisit corpus volume and router mix sizing together, not in isolation.
