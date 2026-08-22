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
| xavier  | 9 GB | 7B          | 64     | Q5_K_M|
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

One thing to know before it surprises you: `min_samples` is a floor, not a
target, and it **rises to meet `router_mix_total`** — the router's mix is drawn
from the training split, so each expert needs enough collected documents to
fill its share of the mix. The run says so when it happens. See
[`corpus:`](#corpus--how-much-text-and-how-varied) below.

`--dryrun` is a different thing: it also relabels the run and writes to a
separate directory, because its job is structural testing, not a small build.
`ms-moe-maker build recipe.yaml --plan` prints the volume back at you, so you
can tell which of the two you are about to start.

See `recipe.flow-0.5B.yaml` for a complete worked example.

### Every knob, and what actually moves

Everything below is optional. A recipe with nothing but `experts:` builds.
This section is for when you want to turn something and would like to know
what it does before you spend four hours finding out.

Where a number here is called **measured**, it came off a real run on a DGX
Spark at 0.5B with three code experts, top-2 of 3 — not off a napkin. Where
it isn't, it's a default someone picked, and you should feel free to disagree
with it.

**`-1` means "you decide"** on every numeric knob. That is not the same as
`0`. A recipe that omits a block behaves exactly like one that sets every
field in it to `-1`, which is what lets us change a default without breaking
your file.

#### `budget:` — how hard each specialist trains

| Knob | Default | What it does |
|---|---|---|
| `target_steps` | 1200 | Optimiser steps per expert. The single biggest lever on wall-clock: total ≈ `target_steps × experts × seconds-per-step`. |
| `max_seq_length` | 2048 | Tokens per training row. Halving it roughly halves memory and time, and truncates long files. |
| `per_device_batch` | 4 | Rows per forward pass. Raise until you OOM, then back off one. |
| `grad_accum` | 2 | Batches per optimiser step. `per_device_batch × grad_accum` is your effective batch. |
| `lora_r` | tier | Adapter rank — how much the expert is *allowed* to differ from the base. nano 32 / xavier 64 / spark 128, clamped to 1–256. |
| `lora_alpha` | 32 | Scaling on the adapter. Leave it unless you know why. |
| `lora_dropout` | 0.0 | Regularisation. Non-zero costs a little speed. |
| `warmup_ratio` | 0.05 | Fraction of steps spent ramping the LR. |
| `warmup_floor` | 10 | Never warm up for fewer steps than this, however short the run. |
| `collect_headroom` | 1.5 | How much more text to gather than the step budget strictly needs, so packing doesn't starve. |
| `doc_ceiling` | 2000 | Upper bound on documents pulled per expert when deriving the token target. |

Reach for `target_steps` and `corpus.max_samples` before you reach for
`lora_r`. Measured: at 0.5B the rank was already 128 while each expert saw
1.23M tokens — one sixteenth of the rung that worked. A large adapter over a
small corpus is what a 0.05-nat expert looks like.

Specialists are the cheap part to get right and the expensive part to run.
The gate below is the opposite.

#### `router:` — how hard the *gate* trains

This is the block that decides whether your MoE routes at all. The stitch
seeds the gate with small noise (`router_init: random`), so the router starts
uniform-ish and has to learn everything it knows from this budget — and the
budget's step count is the lever that actually moves enrichment.

| Knob | Default | What it does |
|---|---|---|
| `epochs` | 1.0 | Passes over the router mix. **The cheapest way to buy router steps.** |
| `batch` | 8 | Per-device batch for the gate. **Must be > 1** so the load-balancing loss sees a mixed batch — at batch 1 every batch is one domain and the aux loss can't balance anything. |
| `accum` | 1 | Gradient accumulation for the gate. |
| `lr` | 1e-4 | Gate learning rate. |
| `aux_loss_coef` | 0.02 | Load-balancing pressure. Mixtral's value, kept because lowering it bought nothing measurable and spent margin against collapse. |
| `agent_mix_fraction` | 0.15 | Share of the mix drawn from generated (synth) experts. The rest split what's left, evenly. |

The number that actually governs the gate is **steps**, and steps are not a
knob — they're arithmetic:

```
router steps = router_mix_total × epochs ÷ (batch × accum)
```

Reps and sets: a mix row is a rep, an epoch is a set. `epochs: 2` and
`router_mix_total: 8000` buy the same steps; the first is free, the second
costs a corpus.

**Measured dose-response** — same stitch, same experts, only the router's step
budget changed:

| Router steps | Enrichment | Reading |
|---|---|---|
| 150 | 1.06× | Barely off its initialisation |
| 500 | 1.16× | Real preference, weak |
| 1000 | 1.23× | Usable |
| 2000 | 1.34× | Usable, and still climbing |

The fit across those four points is `enrichment − 1 ≈ 0.0054 × steps^0.55`,
which is a rule of thumb worth carrying: **doubling the router's steps buys
about 46% more excess-over-1.0.** It does not double anything. Input-dependence
(JS divergence between routing distributions) moves the same way, about ×1.5
per doubling.

The ceiling for top-2 of 3 experts is **2.0×** — if every token routed to its
own expert plus one other, own-source share is 2/3 and the mean other share is
1/3. So 1.34× is roughly a third of the way to a gate that is perfectly
opinionated, and no amount of steps takes you past 2.0 at this topology.

#### `moe:` — the shape of the stitched model

| Knob | Default | What it does |
|---|---|---|
| `experts_per_tok` | 2 | Top-k. **Do not set this to 1** — see below. |
| `norm_topk_prob` | true | Renormalise the top-k gate weights to sum to 1. |
| `router_init` | random | `random` \| `zero`. `random` seeds small noise (as Switch and Mixtral do) so the gate isn't perfectly symmetric. `zero` exists only for the stitch's bit-equality check. |
| `router_init_std` | 0.02 | Noise scale when `router_init: random`. |
| `shared_expert_width` | 1 | Width multiplier for the always-on shared expert. |
| `shared_expert_gate_fill` | 0.02 | Initial gate value for the shared expert. |
| `dense_layers` | auto | `auto`, or an explicit list of layer indices to leave dense instead of MoE-ifying. |

**`experts_per_tok: 1` is refused at validate, on purpose.** With `k=1` and
`norm_topk_prob: true`, the single gate weight is divided by itself — it is
always exactly 1.0, the language-model loss has no gradient path to the gate,
and the router learns nothing while looking like it trained. With
`norm_topk_prob: false` it's worse in a quieter way: the gate probability
becomes a free scalar gain on a frozen expert, so the gate can lower the loss
by adjusting *volume* rather than by choosing correctly. Neither is a router.

Setting `experts_per_tok` equal to your expert count is legal — it's a dense
ensemble — but it makes the dead-expert measurement impossible rather than
merely hard, because every expert is selected on every token by arithmetic.

`router_init: zero` is available for verifying the stitch (it makes the
untrained MoE reproduce one expert exactly), but it is the wrong starting
point for training: a perfectly symmetric gate can only be broken by the
load-balancing loss, and three router trainings on one zero-init skeleton each
collapsed onto a single expert with a *different* winner each time. That is
why `random` is the default.

#### `corpus:` — how much text, and how varied

| Knob | Default | What it does |
|---|---|---|
| `min_samples` | 2000 (500 dry) | Floor per expert. Below it, the stage fails instead of training on scraps. |
| `max_samples` | 100k (10k dry) | Cap per expert. |
| `router_mix_total` | 16000 (4000 dry) | Rows in the router's stratified mix. Also the numerator in the steps formula above; at the default batch 8 × accum 1 this is 2,000 router steps. |
| `per_repo_cap` | 20 | Max files from **one repository, per language**. |
| `max_shards` | 80 | How many corpus shards the scan may pull before giving up. ~0.57 GB each. |

Two things here talk to each other, and you should know it before it surprises
you:

- **`min_samples` rises to meet `router_mix_total`.** The mix is drawn from
  the `.train` split only — held-out has to stay held out — so each expert
  needs roughly `router_mix_total ÷ experts ÷ 0.9` collected documents before
  the gate can be fed. The floor you set is a *minimum*; if the mix needs
  more, the floor goes up and the run tells you so:
  `[cfg] corpus floor raised to 1,556 docs/expert …`. It never goes down.
- **`max_samples` below what the mix needs is refused at validate**, because
  otherwise the corpus stage passes, every specialist trains, and the router
  comes up short of quota hours later — which reads as a gate that wouldn't
  learn when the truth is a gate that wasn't fed.

**`per_repo_cap` is not a tuning knob, it's a correctness one.** Measured: a
C# bucket filled 78% of its token quota out of a single enterprise codebase.
The resulting expert was fluent, passed every downstream check, and had
learned one company's house style rather than the language. Lower is more
diverse and needs more shards; if you raise it, you are trading variety for a
shorter scan and you should mean it.

#### `eval:` — how the result gets measured

| Knob | Default | What it does |
|---|---|---|
| `mode` | all | `routing` \| `quality` \| `experts` \| `all`. |
| `held_out_fraction` | 0.1 | Share of each corpus reserved from training and used to test. |
| `num_samples` | 20 | Generations per expert for the quality half. |
| `dead_threshold` | 1.2 | Enrichment below this marks an expert as not meaningfully preferred. |
| `script` | — | Replaces our eval entirely. Called with `--data-root --output-root --held-out --num-samples`. |

`dead_threshold: 1.2` is deliberately above what a 150-step router produces.
That is the point: an under-trained gate should be reported as undiscriminating,
not quietly passed.

#### `gates:` — where the build stops and asks

| Knob | Default | What it does |
|---|---|---|
| `experts` | auto | `auto` \| `cheap` \| `skip`. The pre-stitch expert audit. `cheap` keeps the free weight-divergence half and drops the loss matrix, which is the only half that can tell you whether the router has a gradient at all. |
| `base_evals` | auto | `auto` \| `manual` \| `skip`. The cheap checks before the build proper. |
| `main_evals` | auto | `auto` \| `manual` \| `skip`. Whether the expensive suite runs unattended. |

`main_evals: auto` removes the last human checkpoint. It will happily run the
full suite against a NaN'd model that generates at full speed and emits one
token forever.

#### `runtime:` — the box, not the model

| Knob | Default | What it does |
|---|---|---|
| `hardware_tier` | xavier | `nano` \| `xavier` \| `spark`. Picks the default size, LoRA rank and quant. |
| `precision` | float16 | Compute dtype. |
| `load_in_4bit` | false | Quantise the base for training. Buys memory, costs fidelity. |
| `direct_load` | false | Skip the staging copy when loading checkpoints. |
| `alloc_conf` | — | Passed straight to `PYTORCH_CUDA_ALLOC_CONF`. `expandable_segments:True` is the one that matters on unified memory. |
| `llama_cpp` | (search) | Path to your llama.cpp build. **Put it here, not in an env var** — this is the one path most likely to differ per box, so a recipe that can't carry it is a recipe that exports nothing on your friend's machine. |

#### `smoke:` and `roots:` — the small print

| Knob | Default | What it does |
|---|---|---|
| `smoke.tokens` | 48 | Tokens to generate when checking the GGUF is alive. |
| `smoke.timeout` | 300 | Seconds before the smoke test gives up. |
| `smoke.prompt` | "Write a function that works." | What to ask it. |
| `smoke.script` | — | Replace the smoke test entirely. |
| `roots.data` | `{size}/corpus` | Where corpora land. |
| `roots.output` | `{size}/train` | Where checkpoints and the export land. |

Keep the `{size}` in `roots.output`. Without it every rung of the ladder writes
to the same directory and the 3B run quietly eats the 0.5B one.

#### If you only turn three knobs

1. **`budget.target_steps`** — how good each specialist gets, and most of your
   wall-clock.
2. **`router.epochs`** — how opinionated the gate gets, at close to zero cost.
   If your eval says *undiscriminating*, this is the knob, not the corpus.
3. **`corpus.per_repo_cap`** — whether your expert learned a language or one
   codebase.

Everything else is refinement.

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

### The tools (MCP) expert

A tool-calling specialist is the largest domain contrast a Ms.MoE can have —
chat-formatted JSON-RPC against raw source — and it is the one expert whose
corpus has to be *generated* rather than scraped. So it gets a dedicated knob
instead of a pile of synth plumbing:

```yaml
tools_expert: true
```

That injects a default tools expert (named `agentcore`, `kind: synth`, a
sensible default teacher) into your expert list — a recipe with two code
experts becomes a three-expert MoE with no other changes. To customise it, give
a mapping instead of `true`:

```yaml
tools_expert:
  name: my_mcp        # what the specialist (and its directory) is called
  teacher: Qwen/Qwen2.5-7B-Instruct   # the model that generates the traces
```

Anything you set wins over the default; `kind` is always `synth`. If an expert
of that name already exists in your `experts:` list, it is used as the tools
expert rather than duplicated.

## Where llama.cpp lives

The GGUF export shells out to `convert_hf_to_gguf.py`, which lives in a
llama.cpp checkout rather than on PyPI. That path is the most box-specific
thing in a build, so a recipe can carry it:

```yaml
runtime:
  llama_cpp: /mnt/nvme/llama.cpp
```

Resolution order is **recipe → `MSMOE_LLAMA_CPP` → a short search** of
`./llama.cpp`, `../llama.cpp`, `~/llama.cpp` and `/opt/llama.cpp`. The search
looks for the converter itself, not just a directory with the right name.

Not finding it is a **warning**, never a failure: you still get the HF
checkpoint, which is a real result. `export` and `smoke` are also the only two
verbs that need no ML stack at all — see the install table above.

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

Three questions, separately runnable:

```bash
ms-moe-maker eval recipe.yaml --mode routing   # the dead-expert check
ms-moe-maker eval recipe.yaml --mode quality   # generation vs held-out refs
ms-moe-maker eval recipe.yaml --mode experts   # did the specialists diverge at all?
ms-moe-maker eval recipe.yaml                  # all of the above (default)
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
  mode: routing             # routing | quality | experts | all
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
