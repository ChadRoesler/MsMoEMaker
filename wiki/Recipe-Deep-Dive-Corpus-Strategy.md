# Recipe Deep Dive: Corpus Strategy

Validated against commit/tag: main (unreleased)

Canonical references:

- [CLI Reference](CLI-Reference)
- [Troubleshooting Signatures](Troubleshooting-Signatures)

## Objective

Build corpora that are both large enough for router training and diverse enough
to avoid expert collapse into single-source style learning.

## Core dynamics

- `min_samples` is a floor, not a target — it rises to meet `router_mix_total`,
  and `--plan` prints the raised floor before the run starts.
- `router_mix_total` drives router feed requirements and step budget.
- `per_repo_cap` limits single-source dominance.
- `max_samples` caps cost but can starve diversity if set too low.

## Planning framework

### Step 1: define run intent

- exploratory run: smaller caps, faster iteration
- quality run: larger corpora, stronger diversity controls
- release run: explicit corpus strategy and recorded diagnostics

### Step 2: set corpus envelope

- choose `max_samples` by time/storage budget
- set conservative `per_repo_cap`
- ensure practical room for router mix goals

### Step 3: validate and inspect

- run `ms-moe-maker build recipe.yaml --plan`
- run `ms-moe-maker corpus recipe.yaml`

### Step 4: prune or adjust

- if dominance appears, use prune mode and lower per-repo cap
- if router underfeeds, increase corpus envelope or reduce mix ambition

## Iteration loop

1. Run baseline build/eval.
2. Inspect corpus and routing outcomes.
3. Change one corpus lever at a time.
4. Rebuild and compare deltas.

Keep notes per iteration to avoid repeating ineffective settings.

## Failure patterns and responses

### Pattern: corpus looks healthy, router remains weak

Likely cause: corpus quantity/diversity not aligned with router mix needs.

Response:

- adjust corpus and router parameters together
- avoid isolated router-only tuning when inputs are weak

### Pattern: experts overfit one repository style

Likely cause: `per_repo_cap` too high.

Response:

- lower `per_repo_cap`
- increase source variety and rerun corpus checks

## Anti-patterns

- maximizing samples without checking diversity
- tuning router before validating corpus quality
- treating one successful build as corpus-proof for all future runs

## Operator checklist

- [ ] corpus constraints align with run intent
- [ ] per-repo dominance reviewed
- [ ] router mix feasibility checked
- [ ] corpus diagnostics archived with run metadata
