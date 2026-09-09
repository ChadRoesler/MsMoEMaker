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
### Teacher generation budgets

Three synth loops generate the corpora, and they no longer share one ceiling.
They have opposite appetites: an MCP tool call is a few hundred tokens of JSON
answering a prompt that carries the whole tool surface, a lore passage is a page
of prose, and a reasoning trace is a think block *plus* an answer.

- `teacher_max_new` — the FALLBACK, used by any loop that has not been given its
  own. Default `512`.
- `agent_teacher_max_new` — the MCP tool-call loop. `-1` = use
  `teacher_max_new`. Can stay tight; the answer is one JSON object.
- `domain_teacher_max_new` — the plain-text loop. `-1` = use `teacher_max_new`.
- `reasoning_teacher_max_new` — the reasoning loop. Default `1024`, because a
  think block plus an answer needs more headroom than either of the others.

All four take `0` to mean **unbounded**: as far as the engine window allows. On
the vLLM path that window is `runtime.vllm_max_len` *minus the prompt*, which is
worth reading twice — with `vllm_max_len` at its default `4096`, asking for
unbounded gives you LESS room than naming `4096` outright. Preflight warns when
you ask for unbounded without having set the window yourself.

### When a generation does not fit

A generation cut at its cap is **discarded, not trimmed**. That matters more
than it sounds: a low ceiling does not shorten the corpus, it keeps whichever
material happened to fit and deletes the rest. A real gauntlet at `512` dropped
490 of 740 domain generations — three times the compute for a corpus made of
atypically short material, which the eval then reported as `NO ROUTER SIGNAL`.

Raising the ceiling is the answer if you own the VRAM. These two knobs are the
answer if you do not:

- `teacher_tell_budget` — put the room available into the teacher's own prompt,
  in words. Default `false`. Lowers the overrun rate; bounds nothing, because no
  model obeys a length instruction exactly. No-op for an unbounded budget.
- `teacher_overrun` — what to do when it overruns anyway. Default `discard`,
  which is what every earlier release did.
  - `discard` — throw the generation away.
  - `answer-only` — salvage a generation that already ended its *reasoning* and
    had its answer cut. Never invents a boundary the teacher did not choose.
  - `close-and-answer` — also salvage one still mid-thought: trim to the last
    sentence, write the terminator, spend the reserve on the answer. Yield goes
    to ~100% at any budget, which is what makes a small box able to build at
    all. Those rows carry a thought the teacher was not finished with; the
    answer is generated conditioned on it, so it is a shorter thought rather
    than a wrong one, but it is a choice and `answer-only` exists for anyone who
    does not want it.
- `teacher_answer_reserve` — tokens held back to buy that ending. `-1` = a
  quarter of *that loop's* budget, floored at 64, never more than half.

Only the reasoning loop has a delimiter to force. The agent and domain loops
have nothing to close, so `close-and-answer` behaves as `answer-only` there —
keep writing, accept it if it ends — and a continuation that also runs out is
still discarded. One attempt, never a loop. Preflight says this out loud rather
than letting the recipe imply a promise the loop cannot make.

The happy path costs nothing: a generation that finishes on its own is one
call, and only the overruns pay for a second, batched one.


What it changes:

- wall-clock runtime
- memory pressure
- specialist adaptation strength

The `*_teacher_max_new` knobs shape the SYNTHETIC data, not the adapter. Raise
the one belonging to the loop that is actually being cut — the progress line
names it, and so does the NOTE that fires when more than a quarter of a batch
hits the cap. Raising `teacher_max_new` when the domain loop is starving now
moves only the loops that have not been given their own ceiling.

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
- `max_new_tokens` — tokens generated per sample. `-1` = you decide: `256`, or
  `1024` when the run writes thinking traces. `0` = unbounded, as far as the
  model window allows.
- `script` for custom evaluation integration

Read `max_new_tokens` before you trust a quality column. A real run capped at
`1024` reported sixteen of sixteen generations cut off for five separate
experts — every score in those rows was computed on an amputated answer, and
`reasoned` (which needs a close tag it never reached) was a lower bound being
read as a measurement. The report says so in its caveats; the fix is to raise
this and ask again. `0` makes that caveat impossible rather than smaller, and
it is expensive: generation time is linear in this number and it is paid once
per sample per expert per surface, so preflight counts the generations for you
before you spend them.

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
- `use_vllm` — serve the teacher through vLLM instead of plain transformers.
  Much faster generation, and a second serving stack to install and keep happy.
  It also moves the teacher batch from `96` to `512` and is part of the build
  fingerprint, so preflight FAILS rather than falling back silently: a corpus
  generated without it would not be the corpus that `build_id` describes.
- `vllm_max_len` — the vLLM context window (`max_model_len`). Default `4096`.
  This is what reserves the KV cache, and it is **the real ceiling any unbounded
  teacher budget runs into**, so a recipe asking for unbounded generation
  without raising this has asked for less room than it thinks.

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
