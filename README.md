# Ms.MoE

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
pip install ms-moe-maker
```

That gets you the CLI and the contract — about a megabyte, no torch. The heavy
machinery lives in the pipeline this forks, in whatever venv you train in. That
split is the point: `ms-moe-maker validate` runs on a laptop, so you can check a
recipe and see what it will cost before going near a machine that can run it.

## Use

```bash
ms-moe-maker describe                  # one line of JSON, exit 0, no side effects
ms-moe-maker validate recipe.yaml      # parse, check, translate — touches nothing
ms-moe-maker build recipe.yaml         # run it
ms-moe-maker build recipe.yaml --json  # JSON Lines on stdout, prose on stderr
```

`ms-moe-maker build recipe.yaml` is the literal command. It's what's in this README,
it's what a person types, and it's exactly what `seren-theatre[stagehand]`
forks — no separate API path with different defaults. If those two ever
diverged, the hand-run path would rot, because it's the one with no automated
users. Making them identical removes the possibility.

## The recipe

A build, as a document. The point is that you can hand it to someone who
doesn't have your box and they get your run — that's the difference between
"it works, look" and a result.

```yaml
schema_version: 1
name: msmoe-coder-5x-dryrun
size: 0.5B
base: huihui-ai/Qwen2.5-Coder-0.5B-Instruct-abliterated

experts:
  - name: powershell
    source: { kind: hf, repo: SaeedRahmani/codeparrot_github_code_powershell, text_field: code }
  - name: python
    source: { kind: stack, language: Python }
  # ...

budget:
  target_steps: 150        # 1200 for a real rung; 150 is the shakedown
  max_seq_length: 2048
  per_device_batch: 4
  grad_accum: 2
```

See `recipe.example.yaml` for the annotated version — every field carries the
measurement that chose it.

Budgets are in **tokens**, derived from steps. Capping documents instead looked
like balance and wasn't: at 10,000 documents each, PowerShell received 4.3× the
gradient updates Shell did, and a different LR curve besides.

## Refusals — read this bit

Right now `ms-moe-maker build` drives an existing pipeline script by setting
environment variables and forking it. That script exposes sixteen levers. A
recipe declares far more than sixteen things.

So the naive wrapper would accept `per_device_batch: 8`, run the build at 4,
and report success — leaving you with a document that *looks* authoritative and
silently isn't. That's the worst possible place to put that trap, in the one
file whose entire selling point is reproducing someone else's run.

**So a recipe field is honoured, or the build refuses. Never ignored.**

The check isn't "is there a lever" — it's "will the run actually do what the
document says". Ms.MoE reads the pipeline's own constants statically (via
`ast`, never importing — importing it would cost you a CUDA context) and
compares each field against the value that will really be used. Agreement is
silence. Only disagreement refuses.

```
$ ms-moe-maker build recipe.yaml
   2 recipe field(s) cannot be honoured by fraunkenstein_universal.py:
     · budget.per_device_batch=8 cannot be applied: the pipeline uses 4 from
       PER_DEVICE_BATCH and exposes no environment lever for it.
     · gates.main_evals='manual' cannot be honoured: the pipeline runs end to
       end with no stage boundary a gate could pause at.
   REFUSED - nothing was run.
```

`--allow-refusals` proceeds anyway. The refusals are recorded in the run
manifest either way, because the person who needs to know a lever was ignored
is the one reading the dashboard six hours later, not the one who saw the
terminal at kickoff.

**The refusal list is the roadmap.** Each entry is a field somebody wanted to
set and couldn't — which is exactly the priority order for pulling that part of
the script into a real stage. When the list is empty, the decomposition is
finished, and nobody had to guess when.

## The run manifest

A build writes `msmoemaker-run.json` into its run directory: what the run is, the
ordered stage list, each stage's status and artifact, and any refusals.

That file is the **only** interface between this package and any viewer.
Nothing imports anything. `seren-theatre` reads the manifest when it's there
and falls back to reading the directory when it isn't — so an instrumented run
is exact and an uninstrumented directory still works. Neither package is
required, and neither knows the other exists.

## Events

Under `--json`: one JSON object per line on stdout, prose on stderr, never
interleaved.

| event | when |
|---|---|
| `started` | the build begins; carries the resolved env and run dir |
| `stage` | a stage changes status |
| `progress` | something worth knowing inside a stage |
| `refused` | recipe fields that couldn't be honoured |
| `warning` / `error` | trouble |
| `done` | terminal, with `ok` |

Every line is flushed. A consumer following a six-hour build through a pipe
would otherwise see nothing until the buffer filled — and that looks exactly
like a hang.

## Status

The stage machinery, contract and CLI are real. The pipeline itself is still
the original 2483-line script, driven from the outside — **wrap-then-carve**.
The contract is the product; the internals move behind it without anything
downstream noticing.

## Licence

GPL-3.0-only.
