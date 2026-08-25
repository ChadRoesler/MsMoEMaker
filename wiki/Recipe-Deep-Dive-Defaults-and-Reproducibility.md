# Recipe Deep Dive: Defaults + Reproducibility

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/ARCHITECTURE.md`
- `MsMoEMaker/docs/SOURCE_OF_TRUTH.md`

## Why this matters

Recipe content alone may not fully determine build behavior if machine defaults
differ. Reproducibility failures usually come from unresolved defaults drift,
not recipe syntax drift.

## Layering model

Effective configuration is layered:

1. floor defaults
2. packaged defaults
3. user defaults (`~/.msmoe/defaults.yaml`)
4. explicit defaults (`--defaults PATH`)
5. recipe values (final override)

### Practical implication

Two builders can run the same recipe and get different builds if layers 3/4
are different.

## Decision framework: when to pin defaults

Use explicit defaults (`--defaults`) when any of these are true:

- run must be reproducible across machines
- run is release-bound
- run is used for benchmark comparisons
- run resumes after environment changes

For exploratory local runs, user defaults may be acceptable.

## Reproducibility runbook

1. Freeze source inputs:
   - commit recipe
   - capture defaults file used for the run
2. Execute with explicit defaults:
   - `ms-moe-maker build recipe.yaml --defaults /path/to/defaults.yaml --json`
3. Record run metadata:
   - commit/tag
   - `recipe_id`
   - `build_id`
   - output root
4. Archive artifacts and metadata together.

## Interpreting `recipe_id` vs `build_id`

- `recipe_id`: identity of recipe content
- `build_id`: identity of resolved runtime config

Use `recipe_id` for recipe review and `build_id` for build parity checks.

## Resume refusal playbook

If build resume is refused due to mismatch:

1. Decide intent:
   - continue exact old build, or
   - intentionally produce a new build
2. For exact continuation:
   - restore original defaults context
   - rerun with original `--defaults` path
3. For intentional change:
   - use `--force` and treat as new build lineage

## Anti-patterns

- Assuming recipe-only versioning is enough for production reproducibility
- Relying on mutable user defaults for release-grade runs
- Mixing artifact outputs from runs with different `build_id`s

## Suggested release checklist

- [ ] Recipe committed
- [ ] Defaults snapshot retained
- [ ] Build invoked with explicit `--defaults`
- [ ] `recipe_id` + `build_id` captured
- [ ] Smoke + eval baseline archived
