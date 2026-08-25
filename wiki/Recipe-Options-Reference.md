# Recipe Options Reference

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/ARCHITECTURE.md`
- `MsMoEMaker/README.md`

This page is a deep explanation of what each recipe block controls,
when to use it, and how options interact.

## Mental model

A recipe answers four questions:

1. What are we building? (`name`, `size`, `base`, experts)
2. How much work should run? (`budget`, `corpus`, `router`)
3. How should the box behave? (`runtime`, `roots`, defaults layering)
4. How do we evaluate outcomes? (`eval`, `smoke`, gates)

Treat the recipe as a build intent document, not just a settings dump.

## Top-level fields

## `schema_version`

- Purpose: recipe format version.
- Typical value: `1`.
- Guidance: keep explicit; useful for tooling and future compatibility.

## `name`

- Purpose: run/model identifier used in outputs and reporting.
- Guidance: use stable, descriptive names (`team-domain-size-purpose`).

## `template`

- Purpose: fast-start preset loader.
- Use when: bootstrapping quickly.
- Guidance: treat as a starting point, then explicitly tune critical blocks.

## `experts`

- Purpose: specialist definitions that become routed experts.
- Each expert usually includes:
  - `name`
  - `source` (`kind` + source-specific fields)
- Guidance:
  - choose experts with meaningful domain contrast
  - avoid near-duplicate experts unless testing specific hypotheses

## `tools_expert`

- Purpose: inject or configure the synthetic tools expert path.
- Values:
  - `true` for default injected tools expert
  - mapping for explicit customization (`name`, `teacher`, etc.)
- Guidance: use when tool-calling behavior is a core target.

## `size`, `base`, `base_kind`

- `size`: requested model rung or auto/defaulted size behavior.
- `base`: concrete base model id/path override.
- `base_kind`: reasoning/nonreasoning hint when auto-detection is ambiguous.

Guidance:

- keep `size` and hardware tier realistic together
- override `base` only when you need explicit checkpoint control
- set `base_kind` explicitly when family detection is uncertain

## `budget` block (specialist training workload)

Primary options usually include:

- `target_steps`
- `max_seq_length`
- `per_device_batch`
- `grad_accum`
- LoRA controls (`lora_r`, `lora_alpha`, `lora_dropout`)
- warmup controls

What it changes:

- wall-clock runtime
- memory pressure
- specialist adaptation strength

Decision order:

1. Set `target_steps` to fit run budget.
2. Fit memory with `max_seq_length`, batch, and accumulation.
3. Refine adapter behavior only after baseline stability.

Common anti-pattern:

- Over-tuning LoRA rank while under-feeding corpus/steps.

## `corpus` block (data volume and diversity)

Primary options include:

- `min_samples`
- `max_samples`
- `router_mix_total`
- `per_repo_cap`
- source-specific controls (`max_shards`, etc.)

What it changes:

- specialist data sufficiency
- router feed feasibility
- diversity vs dominance risk

Important interaction:

- `router_mix_total` and corpus volume must be planned together.

Decision order:

1. Set diversity guardrail (`per_repo_cap`).
2. Set feasible volume envelope (`max_samples`).
3. Ensure router mix goals are actually supportable.

## `router` block (gating behavior)

Primary options include:

- `epochs`
- `batch`
- `accum`
- `lr`
- `aux_loss_coef`
- synth mix controls (for generated experts)

What it changes:

- routing discrimination quality
- stability of expert preference behavior

Tuning rule:

- tune one variable at a time and keep per-run evidence.

## `eval` block (quality/routing measurement)

Primary options include:

- `mode` (`routing`, `quality`, `experts`, `all`)
- `held_out_fraction`
- `num_samples`
- `dead_threshold`
- `script` for custom evaluation integration

What it changes:

- confidence in build outcomes
- ability to compare runs over time

Guidance:

- always capture at least routing baseline + one quality pass for full builds
- keep evaluation settings stable for comparisons

## `gates` block (stop/continue controls)

Primary options include:

- `experts`
- `base_evals`
- `main_evals`

Purpose:

- determines where pipeline blocks for safety signals.

Guidance:

- stricter gates for production/release lanes
- lighter gates for exploratory loops with clear risk acceptance

## `runtime` block (box behavior)

Primary options include:

- `hardware_tier`
- `precision`
- `load_in_4bit`
- `alloc_conf`
- `llama_cpp`

What it changes:

- memory/runtime characteristics
- export/smoke tool path behavior

Guidance:

- keep this explicit for shared build environments
- pin `llama_cpp` when portability matters

## `roots` block (artifact locations)

Primary options include:

- `roots.data`
- `roots.output`

Guidance:

- keep outputs isolated per run lineage
- avoid accidental overlap between rung sizes or experiments

## `smoke` block

Primary options include:

- `tokens`
- `timeout`
- `prompt`
- `script`

Purpose:

- fast "is this artifact alive" verification.

Guidance:

- longer timeouts for slower boxes
- keep prompts stable for repeatability checks

## Source kinds (`experts[].source.kind`)

Common kinds include:

- `stack`
- `hf`
- `gh`
- `local`
- `synth`

Guidance:

- select source kind based on domain fit and data quality signal
- generated (`synth`) experts should be treated as a separate data path with
  explicit quality checks

## Defaults layering and recipe overrides

Effective values are layered beneath recipe values.
Recipe values are final overrides.

Use explicit defaults when run parity across machines matters:

- `ms-moe-maker build recipe.yaml --defaults /path/to/defaults.yaml`

## Recommended profiles

## Exploratory profile

- smaller corpus caps
- reduced steps
- lighter gate strictness
- fast iteration + high change velocity

## Quality profile

- stronger corpus diversity controls
- larger router/supporting data budget
- consistent eval settings for comparison

## Release profile

- explicit defaults path
- locked recipe + metadata capture
- mandatory smoke + eval baseline capture
- archived `recipe_id` + `build_id`

## Change management checklist

When modifying recipe options, record:

- what changed
- why it changed
- expected impact
- observed impact after run
- keep/revert decision

This keeps tuning evidence-driven and reproducible.
