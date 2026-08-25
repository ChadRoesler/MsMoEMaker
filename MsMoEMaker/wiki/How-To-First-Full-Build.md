# How-To: First Full Build

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/ARCHITECTURE.md`

## Goal

Run a full build from recipe validation through `export.gguf`.

## Steps

1. Install build environment:
   - `pip install "ms-moe-maker[train]"`
2. Create or prepare recipe:
   - `ms-moe-maker init --template code > recipe.yaml`
3. Validate and inspect plan:
   - `ms-moe-maker validate recipe.yaml --json`
   - `ms-moe-maker build recipe.yaml --plan`
4. Run build:
   - `ms-moe-maker build recipe.yaml --json`
5. Smoke test:
   - `ms-moe-maker smoke recipe.yaml`

## Success criteria

- Terminal `done` event
- GGUF export stage completes
- Smoke confirms generation

## Reproducibility tip

Pin defaults file when moving between machines:

- `ms-moe-maker build recipe.yaml --defaults /path/to/defaults.yaml`
