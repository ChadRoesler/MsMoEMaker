# Tuning Playbook: Router

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/ARCHITECTURE.md`
- `MsMoEMaker/docs/CLI.md`
- `MsMoEMaker/docs/TROUBLESHOOTING.md`

## Goal

Improve routing discrimination while maintaining stable, repeatable build
behavior.

## Before tuning

Confirm prerequisites first:

- corpus quality has been reviewed
- baseline build completed successfully
- routing eval baseline exists

Router tuning on weak corpora often produces noisy conclusions.

## Tuning strategy

Use controlled sweeps, one lever at a time.

Primary levers:

- `router.epochs`
- `router_mix_total`
- `router.lr`
- `router.aux_loss_coef`

Recommended order:

1. `router_mix_total`
2. `router.epochs`
3. `router.lr`
4. `router.aux_loss_coef`

## Iteration runbook

1. Capture baseline:
   - `ms-moe-maker eval recipe.yaml --mode routing`
2. Apply exactly one router change.
3. Rebuild required stages.
4. Re-run routing eval.
5. Record delta and decide keep/revert.

## Evidence template per iteration

Record for each run:

- parameter changed
- old value -> new value
- routing enrichment result
- notable warnings/errors
- keep/revert decision

## Decision rules

- Keep changes with consistent positive deltas across repeated runs.
- Revert changes that improve one run but regress repeated runs.
- If all deltas are flat, revisit corpus strategy before more router changes.

## Failure modes

### Unstable behavior across repeats

- lower step aggressiveness
- reduce simultaneous changes
- verify same defaults context across runs

### No measurable routing gain

- increase data quality/contrast first
- then revisit router budget

## Guardrails

- Change one variable per iteration.
- Keep logs and run metadata together.
- Avoid interpreting single-run variance as a tuning win.
