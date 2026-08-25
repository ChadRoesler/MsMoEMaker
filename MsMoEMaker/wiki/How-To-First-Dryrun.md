# How-To: First Dryrun

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/README.md`

## Goal

Run a real smallest-rung build path quickly to validate setup.

## Steps

1. Install base + train deps on the build machine:
   - `pip install "ms-moe-maker[train]"`
2. Generate starter recipe:
   - `ms-moe-maker init > recipe.yaml`
3. Validate:
   - `ms-moe-maker validate recipe.yaml`
4. Plan-only preview:
   - `ms-moe-maker build recipe.yaml --plan`
5. Dryrun:
   - `ms-moe-maker build recipe.yaml --dryrun`

## Success criteria

- Build enters staged pipeline and emits stage progress.
- Output artifacts are produced in output root for dryrun path.

## Common failures

- Missing `torch`/train deps -> install `[train]`
- Path/write failures -> verify roots + disk
- See canonical troubleshooting: `MsMoEMaker/docs/TROUBLESHOOTING.md`
