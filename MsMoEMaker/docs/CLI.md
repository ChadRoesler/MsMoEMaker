# CLI Reference

Canonical command and flag reference for `ms-moe-maker`.

## Commands

- `init` - generate a starter recipe or defaults template
- `build` - resolve config and run the pipeline
- `corpus` - inspect corpus health and optionally write pruned copies
- `smoke` - run GGUF smoke test
- `eval` - run routing/quality/expert evaluation
- `validate` - validate recipe and environment-facing constraints
- `describe` - emit one-line JSON capabilities (`--describe` also works globally)

## Copy-paste examples (every command)

```bash
# describe
ms-moe-maker describe
ms-moe-maker --describe

# init
ms-moe-maker init > recipe.yaml
ms-moe-maker init --template dnd --output recipe.dnd.yaml
ms-moe-maker init --defaults-template

# validate
ms-moe-maker validate recipe.yaml
ms-moe-maker validate recipe.yaml --json
ms-moe-maker validate recipe.yaml --defaults /path/to/defaults.yaml --offline

# build
ms-moe-maker build recipe.yaml --plan
ms-moe-maker build recipe.yaml
ms-moe-maker build recipe.yaml --dryrun
ms-moe-maker build recipe.yaml --json --defaults /path/to/defaults.yaml

# corpus
ms-moe-maker corpus recipe.yaml
ms-moe-maker corpus recipe.yaml --prune
ms-moe-maker corpus recipe.yaml --prune --per-repo-cap 15

# smoke
ms-moe-maker smoke recipe.yaml
ms-moe-maker smoke recipe.yaml --tokens 96 --timeout 600

# eval
ms-moe-maker eval recipe.yaml --mode routing
ms-moe-maker eval recipe.yaml --mode quality
ms-moe-maker eval recipe.yaml --mode experts
ms-moe-maker eval recipe.yaml --mode all
```

## Global arguments

- `command` (default: `build`)
- `recipe` path (`.yaml` or `.json`) except for `init`/`describe`
- `--json` JSON Lines events on stdout, prose on stderr
- `--defaults PATH` apply explicit defaults layer under recipe
- `--offline` skip reachability checks
- `--force` redo existing artifacts

## Build-related flags

- `--plan` resolve config + stage plan, run nothing
- `--dryrun` real smallest-rung build path
- `--pipeline PATH` fork legacy pipeline script
- `--python PATH` interpreter used for pipeline process

## Init flags

- `--template NAME` generate from template
- `--defaults-template` write commented defaults template
- `--output/-o PATH` output path (`-` for stdout)

## Corpus flags

- `--prune` write a proposed pruned corpus copy next to original
- `--per-repo-cap N` cap docs per repo in prune mode

## Smoke flags

- `--tokens N` override smoke token count
- `--timeout N` override smoke timeout seconds

## Eval flags

- `--mode routing|quality|experts|all`

## Contract notes

- Canonical command and mode vocabulary is defined in `ms_moe_maker/_describe.py`.
- Event stream kinds under `--json`: `started`, `stage`, `progress`, `refused`, `warning`, `error`, `defaults`, `done`.
- Unknown event kinds should be ignored by consumers (additive compatibility model).
