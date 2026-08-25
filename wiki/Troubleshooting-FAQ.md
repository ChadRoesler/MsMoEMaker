# Troubleshooting FAQ

Validated against commit/tag: main (unreleased)

Canonical source of signatures and exact remediation:

- [Troubleshooting Signatures](Troubleshooting-Signatures)

Use this page as an operator decision index.

## Fast triage flow

1. Did command parsing fail before execution?
   - check command/flag compatibility via `describe`
2. Did validate fail?
   - resolve recipe/environment constraints first
3. Did build fail after plan succeeded?
   - inspect runtime deps, roots, storage, and platform conditions
4. Did run complete but smoke/eval fail?
   - inspect export/runtime path and post-build quality signals

## FAQ by symptom

### "Module not found" during build

Likely cause:

- training dependencies not installed in active environment

Action:

- install `ms-moe-maker[train]`
- rerun validate, then build

### Gated model or private dataset 404/401

Likely cause:

- the model or dataset is gated/private and no Hugging Face token is present

Action:

- put `HF_TOKEN=...` in a `.env` file next to the recipe (auto-loaded), or
- run `huggingface-cli login` once, or
- `export HF_TOKEN=...` for the current shell

The shell always wins over `.env`.

### Resume refused due to build mismatch

Likely cause:

- defaults context or key settings changed between runs

Action:

- continue exact lineage with original defaults context, or
- intentionally fork lineage with `--force`

### Smoke hangs or times out

Likely cause:

- runtime path/tooling mismatch or timeout too low for environment

Action:

- verify `runtime.llama_cpp`
- increase timeout for slower boxes

### Routing eval is flat/weak

Likely cause:

- corpus quality and router budget are misaligned

Action:

- verify corpus strategy first
- then run controlled router tuning loop

### Behavior differs across machines

Likely cause:

- defaults layering differs

Action:

- run with explicit `--defaults`
- compare `recipe_id` and `build_id` semantics in run metadata

## Escalation checklist

Before filing an issue or asking for review, capture:

- command invoked
- recipe and defaults context
- commit/tag
- terminal error signature
- whether failure reproduces on rerun

Then cross-check canonical signatures in
[Troubleshooting Signatures](Troubleshooting-Signatures).
