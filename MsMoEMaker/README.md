# Ms.MoE Maker

**Multi-Specified Mixture of Experts.** Five deliberate experts instead of a
hundred lottery tickets.

The design thesis is the inverse of a frontier MoE. Instead of training many
experts and hoping specialisation emerges — then fighting dead and collapsed
experts with a load-balancing auxiliary loss — you *hand-assign* the domains.
Every expert has a guaranteed constituency, so none of them can go dead,
because none of them was speculative.

The corollary is what makes it maintainable by one person: because each expert
does exactly one thing, you can retrain **one** and re-splice without touching
the others.

> Not a coding model. A coding model shaped like *your* stack.

The real product is the factory, not the model. Swap the expert list and
someone else gets their own Ms.MoE.

---

## Install

```bash
pip install ms-moe-maker            # recipe-side: describe, validate, plan
pip install ms-moe-maker[train]     # on the box that actually builds
```

The base install depends only on `pyyaml`, and that is the point:
`ms-moe-maker validate` and `build --plan` run on a laptop with no GPU, so you
can check a recipe and read what it will cost before going near a machine that
can run it. Every torch import in the package is inside a function, so the base
install imports cleanly.

`[train]` adds torch, transformers, datasets, safetensors, accelerate, peft and
trl — everything the build stages actually touch. The stitcher is vendored, so
there is nothing else to check out alongside it.

### Who needs which

The machine that *trains* does not have to be the machine you *use*. Rent a GPU
for the expensive part, or have someone hand you the specialists, and the rest
still works:

| You have | Install | What runs |
|---|---|---|
| A recipe | `ms-moe-maker` | `init`, `validate`, `describe`, `build --plan` |
| A finished MoE or GGUF | `ms-moe-maker` | ...and **`export`** and **`smoke`** |
| Specialists someone trained | `ms-moe-maker[train]` | ...and `stitch`, `router`, `eval` |
| A box with a GPU | `ms-moe-maker[train]` | the whole build |

The second row is the one worth knowing about. `export` and `smoke` shell out to
llama.cpp and import nothing but the standard library — so a person handed a
finished MoE can convert it to GGUF and **prove it generates**, with no ML stack
installed at all. Release CI asserts that, so it stays true.

## Use

```bash
# Start from nothing
ms-moe-maker init > recipe.yaml
ms-moe-maker init --template dnd > recipe.yaml

# Discover what's available (zero side-effects, returns JSON)
ms-moe-maker describe

# Validate recipe structure — no pipeline, no GPU, no network
ms-moe-maker validate recipe.yaml

# Resolve config + stage plan and run NOTHING. Also no GPU.
ms-moe-maker build recipe.yaml --plan

# Run the full pipeline (needs torch, GPU, training venv)
ms-moe-maker build recipe.yaml

# JSON Lines events on stdout, prose on stderr
ms-moe-maker build recipe.yaml --json

# A real build on the smallest rung — cheap, but still a build
ms-moe-maker build recipe.yaml --dryrun

# Smoke-test the GGUF — checks it generates real tokens
ms-moe-maker smoke recipe.yaml

# Does the router prefer each expert on its own ground? (dead-expert check)
ms-moe-maker eval recipe.yaml --mode routing

# Does it answer better than one expert alone?
ms-moe-maker eval recipe.yaml --mode quality
```

`--plan` and `--dryrun` differ on purpose. `--plan` resolves everything and runs
nothing, so it works on a laptop. `--dryrun` is a *real* build on the smallest
rung — cheap, but it needs torch like any other build.

## The recipe

A build, as a document. You can hand it to someone who doesn't have your box
and they get your run.

### Minimal recipe

```yaml
schema_version: 1
name: my-moe

experts:
  - name: python
    source: { kind: stack, language: Python }
  - name: csharp
    source: { kind: stack, language: C# }

size: auto
```

That's it. The rest auto-fills from your **hardware tier**:

| Tier    | VRAM | Default size | LoRA r | Quant |
|---------|------|-------------|--------|-------|
| nano    | 3 GB | 3B          | 32     | Q4_K_M|
| xavier  | 9 GB | 9B          | 64     | Q5_K_M|
| spark   | 36 GB| 32B         | 128    | Q8_0  |

If you omit `size`, the tier's default is used. If you omit the tier, the
middle tier (`xavier`) is the default.

### A small run that is still a real one

The corpus volume defaults to production size. To watch the whole flow finish
this evening — every stage, real artifacts, a loadable GGUF — turn the volume
down rather than reaching for `--dryrun`:

```yaml
corpus:
  min_samples: 300          # fail the stage below this, not train on scraps
  max_samples: 3000         # cap per expert
  router_mix_total: 800     # rows in the router's stratified mix

budget:
  target_steps: 150
  max_seq_length: 1024
```

`--dryrun` is a different thing: it also relabels the run and writes to a
separate directory, because its job is structural testing, not a small build.
`ms-moe-maker build recipe.yaml --plan` prints the volume back at you, so you
can tell which of the two you are about to start.

See `recipe.flow-0.5B.yaml` for a complete worked example.

### Using a template

Templates fill in name, base model, expert list, budget, and MoE config so you
don't have to:

```yaml
template: dnd

experts:
  - name: monster_manual
    source: { kind: hf, repo: PleiaSys/DnD-MonsterManual, text_field: text }
  - name: players_handbook
    source: { kind: hf, repo: PleiaSys/DnD-PlayersHandbook, text_field: text }
  - name: dm_guide
    source: { kind: hf, repo: PleiaSys/DnD-DMG, text_field: text }
```

Available templates: `code`, `dnd`, `math`, `culinary`.

### Source kinds

| Kind    | Source | Use case |
|---------|--------|----------|
| `stack` | BigQuery code stack-v3 by language | Code specialists |
| `hf`    | HuggingFace dataset (repo + text_field) | DnD, math, culinary, etc. |
| `gh`    | Files from a public GitHub repo (repo + glob) | A project's docs or source |
| `local` | Directory of .txt/.jsonl/.md files | Custom corpora |
| `synth` | Generate traces from a teacher model | Agentcore / reasoning |

Kinds are a **registry**, not a fixed list — another package can publish its own
via the `ms_moe_maker.corpus_kinds` entry point without sending a PR here.

`gh` fetches one tarball from codeload rather than cloning, so there is no git
binary needed and no history downloaded. Globs are matched against paths
**relative to the repo root**, and `**/` means zero-or-more directories the way
a shell means it:

```yaml
  - name: llama_docs
    source: { kind: gh, repo: ggml-org/llama.cpp, glob: "docs/**/*.md" }
  - name: my_wiki
    source: { kind: gh, repo: me/notes, ref: main, subdir: wiki, glob: "**/*.md" }
```

Public repos only, deliberately: a recipe is a document people *share*, which
makes it the wrong object to put a credential in.

See `recipe.example.yaml` for the fully annotated version.

## Preflight

Every build starts by asking the cheap questions, so the expensive part never
starts on a box that cannot finish it:

- is torch / transformers / safetensors installed?
- is the base model reachable (or is it gated, or a typo)?
- are the roots writable, with enough room for the specialists + the stitched
  MoE + a GGUF?
- do the `local` corpus paths exist?
- is llama.cpp present? — a **warning**, not a failure. Without it you still
  get the HF checkpoint; you just do not get a GGUF.

`ms-moe-maker build recipe.yaml --plan` runs the same checks and stops there.

Failures carry their remedy, because the person reading one is usually about to
lose an evening.

## The pipeline

A recipe flows through six stages:

1. **data.corpus** — Collect expert corpora (stack scan, HF download, local files)
2. **data.synth** — Generate synthetic traces (if `kind: synth` experts)
3. **finetune.*{expert}*** — LoRA specialist training (one stage per expert)
4. **stitch** — Assemble the MoE skeleton from specialist checkpoints
5. **router** — Train the router gate weights (stratified expert mix)
6. **export.gguf** — Export GGUF and smoke-test it

The pipeline is fully modular. Each stage is an independent Python module. The
orchestrator (`builder.py`) runs them in order, reports progress via a callback,
and resumes from where it left off on re-run.

## Evaluation

After a build, you can check whether your experts actually diverged:

```bash
ms-moe-maker eval recipe.yaml
```

Two different questions, separately runnable:

```bash
ms-moe-maker eval recipe.yaml --mode routing   # the dead-expert check
ms-moe-maker eval recipe.yaml --mode quality   # generation vs held-out refs
ms-moe-maker eval recipe.yaml                  # both (default)
```

**Routing** is the one Ms.MoE uniquely claims, and it is why hand-assigned
experts work at all. A dead expert is not one that writes badly — it is one
**the router never routes to**. So the measurement is routing, not text
quality: held-out text from each expert's own domain goes through the MoE, the
gate decisions are captured, and each expert gets an *enrichment* score — how
much more it is used on its own domain than on average. Above ~1.2x means the
router can tell that domain apart. Around 1.0x means it cannot, and that expert
is dead however well it generates.

The report also names which expert is eating a weak one's traffic, because an
expert can clear the enrichment bar and still be **outranked on its own
domain** by a neighbour. That is a different failure, and a column-only read
misses it.

**Quality** is real generation against held-out references. It needs an answer
key, and whoever wrote the corpus is the only one who has it — which is exactly
why this half is meant to be overridden.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | No dead experts, and every check was actually measured |
| 2 | Dead expert(s) found |
| 3 | Nothing failed — but something could not be measured |

3 exists on purpose. "We could not measure it" must never share an exit code
with "it passed."

### Overriding it

We provide the floor. Both halves are yours to replace, from the recipe:

```yaml
eval:
  script: my_eval.py        # replaces ours entirely
  mode: routing             # routing | quality | all
  held_out_fraction: 0.1
  num_samples: 20
  dead_threshold: 1.2       # minimum enrichment before "dead"

smoke:
  tokens: 48
  timeout: 300
  prompt: "Write a function that works."
```

A custom script is called as

```
my_eval.py --data-root R --output-root O --held-out F --num-samples N
```

which you can implement in any language you like.

## Environment variables

| Variable | Overrides |
|----------|-----------|
| `MSMOE_TIER` | Hardware tier (nano/xavier/spark) |
| `MSMOE_LORA_R` | LoRA rank (integer) |
| `HF_HOME` | HuggingFace cache location |
| `MSMOE_DRYRUN=1` | Smallest rung (same as `--dryrun`) |
| `MSMOE_BASE_MODEL` | Hard-code the base model instead of auto |
| `MSMOE_LLAMA_CPP` | Path to llama.cpp |

The `MSMOE_*` names are inherited from the script this tool was carved out of
and are still read for compatibility. New levers get `MSMOE_*`.

## Supported base models

The fine-tune stage is generic — `AutoModelForCausalLM` will train a specialist
from almost anything. The **stitch** stage is not: it builds a `Qwen2MoeConfig`,
so today the base has to be a Qwen model.

`validate` refuses an unsupported base up front, deliberately. Without that
check a Llama base collects its corpora, trains every specialist over several
hours, and then dies at stage 4 — the most expensive possible place to find
out.

## Licence

GPL-3.0-only.
