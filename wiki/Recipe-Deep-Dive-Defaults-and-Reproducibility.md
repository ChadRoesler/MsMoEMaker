# Recipe Deep Dive: Defaults + Reproducibility

Validated against commit/tag: main (unreleased)

Canonical references:

- [CLI Reference](CLI-Reference)
- [Architecture](Architecture)
- [Recipe Options Reference](Recipe-Options-Reference)

## Why this matters

Recipe content alone may not fully determine build behavior if machine defaults
differ. Reproducibility failures usually come from unresolved defaults drift,
not recipe syntax drift.

## Layering model

Effective configuration is layered:

1. floor defaults
2. packaged defaults
3. user defaults (`~/.msmoe/defaults.yaml`)
4. explicit defaults (`--defaults PATH`)
5. recipe values (final override)

### Practical implication

Two builders can run the same recipe and get different builds if layers 3/4
are different.

## The other layered table: reasoning tag styles

`defaults.yaml` is not the only file that layers. The reasoning tag table does
too, deliberately on the same rules, because two files that layer differently
is two things to explain and two things to get wrong:

1. floor - one style, in code. A panic minimum, so a missing or broken file can
   never take a build down.
2. packaged - `ms_moe_maker/reasoning.yaml`, the real table.
3. user - `~/.msmoe/reasoning.yaml`, or `$MSMOE_REASONING`.
4. explicit - a path, for CI and for reproducing someone else's run.

Layers merge **by name**, never wholesale: adding one family does not cost you
the shipped ones.

### Why a tag table belongs on a reproducibility page

Because nobody thinks of it as config, and it changes results without changing
anything you would look at. Two builders on the same recipe and the same
`defaults.yaml` can still split thinking traces on different delimiters if one
of them has a `~/.msmoe/reasoning.yaml` and the other does not - and the
failure is not a crash, it is a set of quality numbers that quietly include the
reasoning trace. See
[Troubleshooting Signatures](Troubleshooting-Signatures) for what that looks
like from the operator's chair.

The pipeline closes the hole by **stamping the resolved delimiters into the
run's config** (`reasoning_type`, `reasoning_open`, `reasoning_close`,
`reasoning_interwoven`) instead of looking the table up again later. Two
consequences worth knowing:

- **Eval splits on exactly the tags the generator wrote.** Editing the table
  while a build is running cannot make the scorer measure a different artifact
  than the one on disk.
- **Editing your reasoning table changes `build_id`.** `build_fingerprint` is
  fail-closed - every resolved field *minus* an explicit exclusion list - so the
  stamped tags are in the hash by construction, not because someone remembered
  to add them. A corrected table means a rebuild, not a re-score, and a resume
  will refuse and tell you which field moved.

To see what a box actually merged:

```bash
ms-moe-maker describe        # .reasoning holds the merged styles + families
```

That output is the install answering, not the source shipping - a user file on
that box is already folded in, warnings included.

## Decision framework: when to pin defaults

Use explicit defaults (`--defaults`) when any of these are true:

- run must be reproducible across machines
- run is release-bound
- run is used for benchmark comparisons
- run resumes after environment changes

For exploratory local runs, user defaults may be acceptable.

## Reproducibility runbook

1. Freeze source inputs:
   - commit recipe
   - capture defaults file used for the run
2. Execute with explicit defaults:
   - `ms-moe-maker build recipe.yaml --defaults /path/to/defaults.yaml --json`
3. Record run metadata:
   - commit/tag
   - `recipe_id`
   - `build_id`
   - output root
4. Archive artifacts and metadata together.

## Interpreting `recipe_id` vs `build_id`

- `recipe_id`: identity of recipe content
- `build_id`: identity of resolved runtime config

Use `recipe_id` for recipe review and `build_id` for build parity checks.

## Resume refusal playbook

If build resume is refused due to mismatch:

1. Decide intent:
   - continue exact old build, or
   - intentionally produce a new build
2. For exact continuation:
   - restore original defaults context
   - rerun with original `--defaults` path
3. For intentional change:
   - use `--force` and treat as new build lineage

## Anti-patterns

- Assuming recipe-only versioning is enough for production reproducibility
- Relying on mutable user defaults for release-grade runs
- Mixing artifact outputs from runs with different `build_id`s

## Suggested release checklist

- [ ] Recipe committed
- [ ] Defaults snapshot retained
- [ ] Reasoning table accounted for - either no `~/.msmoe/reasoning.yaml` on
      the box, or it is captured alongside the defaults snapshot
- [ ] Build invoked with explicit `--defaults`
- [ ] `recipe_id` + `build_id` captured
- [ ] Smoke + eval baseline archived
