# Tuning Playbook: Router

Validated against commit/tag: `<fill-me>`

Canonical references:

- `MsMoEMaker/docs/ARCHITECTURE.md`
- `MsMoEMaker/docs/CLI.md`

## Goal

Improve routing discrimination without destabilizing build behavior.

## Baseline loop

1. Run baseline eval:
   - `ms-moe-maker eval recipe.yaml --mode routing`
2. Adjust one knob at a time:
   - `router.epochs`
   - `router_mix_total`
   - `router.lr`
   - `router.aux_loss_coef`
3. Re-run build/eval and record enrichment deltas.

## Guardrails

- Change one variable per iteration.
- Keep logs + run metadata together.
- Do not treat one run as conclusion; compare short controlled sweeps.
