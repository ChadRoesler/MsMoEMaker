# How-To: First Full Build

Validated against commit/tag: main (unreleased)

Canonical references:

- [CLI Reference](CLI-Reference)
- [Architecture](Architecture)
- [Troubleshooting Signatures](Troubleshooting-Signatures)

## Goal

Run a full production-path build from validation through `export.gguf` and
confirm the resulting artifact is usable.

## Preconditions

Before starting a full build, ensure:

- Dryrun path has succeeded at least once.
- Training dependencies are installed on the build machine.
- Output/data roots are intentional and have enough disk.
- You know whether this run should be reproducible across machines.

## Full build runbook

1. Install build dependencies:
   - `pip install "ms-moe-maker[train]"`
2. Create or prepare recipe:
   - `ms-moe-maker init --template code > recipe.yaml`
3. Validate and inspect resolved plan:
   - `ms-moe-maker validate recipe.yaml`
   - `ms-moe-maker build recipe.yaml --plan`
4. Execute full build:
   - `ms-moe-maker build recipe.yaml --json`
5. Verify generated artifact:
   - `ms-moe-maker smoke recipe.yaml`
6. Capture evaluation baselines:
   - `ms-moe-maker eval recipe.yaml --mode routing`
   - `ms-moe-maker eval recipe.yaml --mode quality`

## Success criteria

A full build is considered successful when all are true:

- terminal completion (`done`) is reached
- export stage completes (`export.gguf`)
- smoke validates generation
- initial routing/quality baselines are captured

## Reproducibility practices

For cross-machine parity:

- use explicit defaults:
  - `ms-moe-maker build recipe.yaml --defaults /path/to/defaults.yaml`
- store recipe + defaults snapshot together
- record commit/tag and run metadata in release notes

## Common risk points

- Defaults drift between machines
- Insufficient corpus volume for router mix goals
- Long builds without incremental evaluation checkpoints

Use deep dives for mitigation strategies:

- [Recipe Deep Dive: Defaults + Reproducibility](Recipe-Deep-Dive-Defaults-and-Reproducibility)
- [Recipe Deep Dive: Corpus Strategy](Recipe-Deep-Dive-Corpus-Strategy)
- [Tuning Playbook: Router](Tuning-Playbook-Router)
