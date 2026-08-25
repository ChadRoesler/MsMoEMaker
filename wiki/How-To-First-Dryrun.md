# How-To: First Dryrun

Validated against commit/tag: main (unreleased)

Canonical references:

- [CLI Reference](CLI-Reference)
- [Troubleshooting Signatures](Troubleshooting-Signatures)
- `MsMoEMaker/README.md`

## Goal

Run the smallest-rung real build path to verify your environment, defaults, and
stage flow before spending time on full-size runs.

## When to use dryrun

Use dryrun when you want to prove pipeline integrity:

- command wiring works
- dependencies are usable
- roots and artifacts are writable
- stage flow reaches terminal completion

Dryrun is not a no-op planner. It executes a real reduced build path.

## Preflight checklist (before commands)

- You are in the intended virtual environment.
- Build machine has training deps available.
- You know where outputs should land.
- You have enough free disk for temporary artifacts.

## Runbook

1. Install build dependencies:
   - `pip install "ms-moe-maker[train]"`
2. Generate starter recipe:
   - `ms-moe-maker init > recipe.yaml`
3. Validate structural and environment constraints:
   - `ms-moe-maker validate recipe.yaml`
4. Inspect plan and resolved defaults:
   - `ms-moe-maker build recipe.yaml --plan`
5. Execute dryrun:
   - `ms-moe-maker build recipe.yaml --dryrun --json`
6. Verify generated artifact behavior:
   - `ms-moe-maker smoke recipe.yaml`

## What to record from this run

Capture these items for repeatability:

- commit/tag used
- recipe path and hash
- defaults path (if explicit)
- final `done` outcome and any warnings
- output root containing artifacts

## Success criteria

A dryrun is considered successful when all are true:

- stage flow starts and advances through expected stages
- terminal completion is reached (`done` event)
- output artifacts are present in the output root
- smoke confirms generation

## Failure triage map

If validate fails:

- fix recipe/schema or dependency issues before retrying

If plan succeeds but dryrun fails:

- environment/runtime issue is likely
- inspect roots, disk, and dependency availability

If dryrun completes but smoke fails:

- focus on export/smoke runtime path and timeout conditions

Canonical failure signatures live in [Troubleshooting Signatures](Troubleshooting-Signatures).

## Next step

After this run is stable, continue with
[How-To: First Full Build](How-To-First-Full-Build).
