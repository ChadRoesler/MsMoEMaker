# Architecture

Validated against commit/tag: main (unreleased)

The pipeline that turns a recipe into a stitched Mixture-of-Experts GGUF.

## Stage list

Stages run in order; each is an independent module with a disk-artifact resume
check. The ids are a public contract — the JSON event stream and the manifest
key off them.

1. `preflight` — check the box: torch, disk, base model, llama.cpp.
2. `abliterate.base` — *optional*. Decensor the base with the vendored Heretic
   core, in its own subprocess.
3. `data.corpus` — collect each expert's corpus (stack / hf / gh / local).
4. `data.synth` — generate synthetic corpora (tools + reasoning experts).
5. `finetune.{expert}` — LoRA-tune each specialist on its corpus.
6. `gate.experts` — advisory: did the specialists diverge, and can a router tell
   them apart?
7. `stitch` — splice the specialists into a `Qwen2Moe` skeleton (streaming, one
   shard in memory at a time).
8. `router` — train only the router gate weights on a stratified expert mix.
9. `export.gguf` — convert to GGUF via llama.cpp and smoke-test it.

## Modules

| Module | Role |
|---|---|
| `recipe.py` | recipe dataclasses, parse, validate |
| `config.py` | resolve recipe + defaults + env into a frozen `PipelineConfig` |
| `builder.py` | orchestrates the stages |
| `data.py` | corpus collection + synth trace generation |
| `finetune.py` | LoRA specialist training |
| `stitch.py` / `_moe_stitch.py` | streaming MoE stitcher |
| `router.py` | router-only training |
| `export.py` | GGUF export + smoke |
| `eval.py` | routing / quality measurement |
| `manifest.py` | run manifest (`msmoe-run.json`) |
| `stages.py` | the stage vocabulary + plan |
| `heretic/` | vendored Heretic ablation core (AGPL-3.0) |
| `dotenv.py` | `.env` loader (HF_TOKEN, etc.) |

## Contracts

- **Stage ids** are stable; renaming one is a breaking change.
- **The manifest** (`msmoe-run.json` in the output root) is the interface
  between the pipeline and viewers. Additive fields don't bump schema.
- **Run recipe copy** is preserved beside the manifest as
  `msmoe-recipe.<source-suffix>` (for example `.yaml` or `.json`).
- **`--json`** emits one JSON object per line on stdout; prose goes to stderr.
- **`recipe_id`** identifies the recipe as written; **`build_id`** identifies the
  resolved build (recipe + defaults + box).

## Stage vocabulary publishing (`describe.stages`)

The stage contract is also published by `describe` so integrations can render
pipeline shape without hardcoding internal lists.

`describe.stages` returns ordered rows with:

- `id`
- `label`
- `artifact`
- `optional`
- `parameter` (for templated rows)

For example, `finetune.{expert}` is published once as a template row, and
consumers expand it using `parameter: "expert"`.

## Stage artifact paths in manifests

Builder stages operate with absolute paths internally. Runner manifest writes
store stage artifacts relative to the run directory, which keeps manifests
portable when runs are moved, copied, or mounted under different prefixes.

Treat `manifest.stage(<id>).artifact` as a run-relative path when present.

## Design invariants

- **The source kind decides the collector, not the expert name.** A `hf` expert
  named `powershell` downloads from Hugging Face; it is never stack-scanned.
- **Hand-assigned experts, no dead ones by construction.** A dead expert is one
  the router never routes to — measured as routing enrichment, not text quality.
- **A silent wrong-but-plausible artifact is worse than a loud failure.** The
  stitch is verified bit-for-bit against its sources before the router trains.
- **`abliterate.base` is a subprocess**, so Heretic's torch global state (grad
  mode, seeds, VRAM) can never leak into the finetune stages.
