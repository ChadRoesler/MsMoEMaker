# CLI Reference

Validated against commit/tag: main (unreleased)

The command surface. Prose goes to stderr; `--json` puts JSON Lines events on
stdout for machine consumers. Commands: `init`, `describe`, `validate`, `build`,
`smoke`, `eval`, `corpus`, `export`, `bundle`.

## Commands

### `init`

Generate a starter recipe.

```bash
ms-moe-maker init > recipe.yaml
ms-moe-maker init --template code > recipe.yaml    # code | dnd | math | culinary
ms-moe-maker init -o recipe.yaml                   # write to a file instead of stdout
ms-moe-maker init --defaults-template              # write ~/.msmoe/defaults.yaml
```

### `describe`

Print the box: tiers, templates, corpus kinds, eval modes, overrun policies,
the knob glossary, defaults, and stage vocabulary. Zero side effects, JSON on
stdout.

This is the contract a front-end builds from, which is why vocabularies live
here rather than being hardcoded anywhere: `eval_modes` and
`overrun_policies` are legal values for `eval.mode` and
`budget.teacher_overrun`, `events` is the event vocabulary emitted under
`--json`, and `stages` publishes the pipeline stage rows in execution order.

Each `stages` row includes:

- `id`
- `label`
- `artifact` (or `null`)
- `optional` (whether the stage may be absent on a valid build)
- `parameter` (for templated rows like `finetune.{expert}`)

Adding values/rows is additive — consumers should ignore unknown ones.
Removing or renaming existing ones is breaking for integrations.

```bash
ms-moe-maker describe
```

### `validate`

Check a recipe's structure and the box's ability to honour it. No GPU, no
network.

```bash
ms-moe-maker validate recipe.yaml
```

### `build`

Resolve and run the pipeline.

```bash
ms-moe-maker build recipe.yaml             # full build
ms-moe-maker build recipe.yaml --plan      # resolve everything, run nothing (no GPU)
ms-moe-maker build recipe.yaml --dryrun    # real build on the smallest rung (needs torch)
ms-moe-maker build recipe.yaml --json      # JSON Lines events on stdout
ms-moe-maker build recipe.yaml --force     # redo stages whose artifacts exist
ms-moe-maker build recipe.yaml --defaults /path/defaults.yaml
```

### `smoke`

Prove the exported GGUF generates. Needs no ML stack.

```bash
ms-moe-maker smoke recipe.yaml
ms-moe-maker smoke recipe.yaml --tokens 64 --timeout 600
```

### `eval`

Measure the result. Exit codes: `0` no dead experts and everything measured,
`2` dead expert(s), `3` something could not be measured.

```bash
ms-moe-maker eval recipe.yaml --mode routing   # dead-expert check
ms-moe-maker eval recipe.yaml --mode quality   # generation vs held-out refs
ms-moe-maker eval recipe.yaml --mode experts   # did the specialists diverge
ms-moe-maker eval recipe.yaml                  # the recipe's eval.mode, or all
```

### `corpus`

Inspect the corpora on disk. With `--prune`, write a dominance-cleaned copy
(never in place).

```bash
ms-moe-maker corpus recipe.yaml
ms-moe-maker corpus recipe.yaml --prune --per-repo-cap 10
```

### `export`

GGUF export + smoke for an already-built MoE. Re-runs the export stage against
the router's output directory, so a build that skipped GGUF — no llama.cpp on
the box at the time — can be finished later, or the smoke re-run once
`llama-cli` lands.

```bash
ms-moe-maker export recipe.yaml
```

Fails with a pointer to `build` if there is no trained MoE to export.

### `bundle`

Freeze a recipe and everything needed to hand it to someone else.

A recipe is mostly sentinels meaning "you decide", so handing somebody
`recipe.yaml` hands them your intentions and *their* defaults — same file,
different model, no error anywhere. This resolves the recipe on THIS box and
writes the answers back into it, so there is nothing left for the far side to
decide.

```bash
ms-moe-maker bundle recipe.yaml
```

It will not pretend about the part it cannot fix. Sixteen fingerprint fields
have no recipe key at all — three come from environment variables, twelve are
literals in `build_config`, one is a CLI flag — and they are printed by name
every time, as well as written into the manifest so the far side can diff them.
See `knobs.UNPINNABLE`.

## Global flags

| Flag | Meaning |
|---|---|
| `--plan` | resolve config + stages, run nothing |
| `--dryrun` | real build on the smallest rung (needs torch) |
| `--json` | JSON Lines events on stdout, prose on stderr |
| `--force` | redo stages whose artifacts already exist |
| `--defaults PATH` | layer an explicit defaults file under the recipe |
| `--offline` | skip reachability checks (no network calls) |
| `--pipeline PATH` | fork a legacy pipeline script instead of the in-package builder |
| `--python PATH` | interpreter that runs the pipeline (for a separate torch venv) |

## Environment

| Variable | Overrides |
|---|---|
| `HF_TOKEN` | Hugging Face token (also read from a `.env` next to the recipe) |
| `MSMOE_TIER` | hardware tier (nano / xavier / spark) |
| `MSMOE_BASE_MODEL` | base model id |
| `MSMOE_LORA_R` | LoRA rank |
| `MSMOE_LLAMA_CPP` | path to a llama.cpp checkout |
| `HF_HOME` | Hugging Face cache location |

See [Architecture](Architecture) for the stage list and the contracts.
