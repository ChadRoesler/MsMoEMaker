# Recipe Deep Dive: Defaults + Reproducibility

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/ARCHITECTURE.md`
- `MsMoEMaker/docs/SOURCE_OF_TRUTH.md`

## Why this matters

Recipe content alone may not fully determine build behavior if machine defaults differ.

## Layering model (high level)

- Floor defaults
- Packaged defaults
- User defaults (`~/.msmoe/defaults.yaml`)
- Explicit defaults (`--defaults PATH`)
- Recipe (wins)

## Practical guidance

- Use `--defaults PATH` for cross-machine reproducibility.
- Treat `recipe_id` as recipe identity and `build_id` as resolved-build identity.
- If resume refuses due to build mismatch, either force rebuild or recreate original defaults context.

## Suggested runbook

1. Commit recipe
2. Commit/defaults snapshot used for production run
3. Record tag + build metadata in release notes
