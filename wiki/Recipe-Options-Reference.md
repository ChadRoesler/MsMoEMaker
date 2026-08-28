# Recipe Options Reference

Validated against commit/tag: main (unreleased)

Canonical references:

- [CLI Reference](CLI-Reference)
- [Architecture](Architecture)
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
  Full treatment in the `reasoning` section below.

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
- `teacher_max_new` — max tokens the generic synth/domain teacher emits per
  trace. Default `512`. `-1` (or absent) = use the default.
- `reasoning_teacher_max_new` — max tokens the reasoning teacher emits per
  trace. Default `1024`, because a think block + answer needs more headroom than
  a tool call or plain domain text. `-1` (or absent) = use the default.

What it changes:

- wall-clock runtime
- memory pressure
- specialist adaptation strength

The two `*_teacher_max_new` knobs shape the SYNTHETIC data, not the adapter:
raise `reasoning_teacher_max_new` if the teacher's think/answer is truncated
mid-script, and `teacher_max_new` if plain domain traces are cut short.

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

- advisory stop/continue controls. They warn and auto-skip, never refuse; only
  `preflight` hard-stops a build.

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

## `abliterate` block (base decensoring)

- Purpose: decensor the base model (vendored Heretic core) before any specialist
  trains from it.
- Forms: `abliterate: true` for defaults, or a mapping for explicit control.

Options include:

- `n_trials` — Optuna trials; `-1` means the default (200).
- `seed` — reproducible study seed (unset = random).
- `quantization` — `none` or `bnb_4bit`.
- `trial_index` — which Pareto-front trial to export (unset = first).
- `checkpoint_action` — `continue` (resume a crashed study) or `restart`.
- `export` — `merge` (dense safetensors) or `adapter` (LoRA only).

Notes:

- The stage is `abliterate.base`, between `preflight` and `data.corpus`.
- It runs in its own process, so its global state and VRAM never leak into the
  finetune stages.
- It is a base-level decensor: every specialist inherits it through the stitch.

## `reasoning` (two knobs, two different questions)

These get confused constantly, so: one asks whether the base **already**
reasons, the other **puts reasoning in**. They are independent and you can use
either, both, or neither.

### `base_kind` - is the base already a reasoner?

```yaml
base_kind: auto            # auto | reasoning | nonreasoning
```

- `auto` sniffs the model id against the known families.
- Set it explicitly when the id is not a recognisable reasoning name.
- It changes how the pipeline formats prompts and how eval reads the output. It
  does **not** make a non-reasoning base reason.

### `reasoning: true` on a source - bake reasoning in

```yaml
experts:
  - name: python
    source: { kind: stack, language: Python, reasoning: true }
```

The R1-distill recipe: a reasoning teacher writes trace-plus-answer pairs on
that expert's domain and the specialist is fine-tuned on them. Works on **any**
base, including a small non-reasoning Qwen.

- Default teacher: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, or the `-1.5B` on
  a dryrun.
- Override per source with `teacher:`.
- Because a `reasoning: true` expert puts think blocks into a build whose base
  does not reason, the run resolves a tag style anyway, falling back to plain
  XML exactly the way the generator does.

### `templates:` on a source - choose the questions the teacher answers

Every generated expert that is **not** the tools expert is seeded from question
templates: the `reasoning: true` path asks them of the reasoning teacher, and a
plain `kind: synth` expert asks them of the generic teacher for plain domain
text (no think block, no tool calls).

```yaml
experts:
  - name: shell
    source:
      kind: synth
      templates: dnd        # packaged dnd_templates.yaml
```

`templates:` accepts:

- a bare name (`code` | `dnd` | `math` | `culinary` | `generic`) → the packaged
  `{name}_templates.yaml`
- a filesystem path, or any name ending in `.yaml`/`.yml` → used as-is
- absent/empty → `generic_templates.yaml`

The file is a `Prompts:` list; `{domain}` is substituted with the expert's
display name:

```yaml
Prompts:
  - "Explain the idiomatic way to handle a {domain} task."
  - "Compare two {domain} approaches and pick one."
```

A missing or empty file falls back to the built-in code tasks with a warning.
`templates:` shapes the reasoning and plain-domain generators; the tools expert
(MCP tool-call traces) does not use it.

### The tag table is a file, on purpose

Model families map to tag styles through a layered `reasoning.yaml` rather than
a table baked into a release. The reason is the failure mode: **a wrong tag
style is a silent wrong answer, not a crash.** The splitter finds no
delimiters, reports "did not reason", and scores the whole think block as if it
were the answer.

So the day a family ships a new delimiter, you drop a file instead of waiting
for a version:

```yaml
# ~/.msmoe/reasoning.yaml  (or point $MSMOE_REASONING at one)
Families:
  - Key: acme
    FamilyName: Acme Thinkers
    Models: [Acme-R2, acme-thinker]     # write what is on the model card
    PreferredStyle: xml
```

Layers merge by name; model names match loosely (case, spaces, dots and hyphens
ignored on both sides) and the longest match wins, so the answer never depends
on file order.

See the layering rules and the reproducibility consequences in
[Recipe Deep Dive: Defaults + Reproducibility](Recipe-Deep-Dive-Defaults-and-Reproducibility),
and the operator-side symptom in
[Troubleshooting Signatures](Troubleshooting-Signatures).

Notes:

- The resolved delimiters are stamped into the run's config, so eval splits on
  exactly what the generator wrote - and editing your table changes `build_id`.
- `ms-moe-maker describe` reports the merged table under `.reasoning`, warnings
  included. That is the install answering, not the source shipping.

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
