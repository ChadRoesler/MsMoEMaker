# Documentation Backlog

## Quick wins (1-2 days)

1. Keep README fast path stable (no deep theory growth in top sections). *(ongoing)*
2. Expand `docs/TROUBLESHOOTING.md` with real error excerpts from recent failures. *(done)*
3. Add one copy-paste example for each CLI command in `docs/CLI.md`. *(done)*
4. Add a short "docs changed" checklist to PR template (if templates are enabled). *(done)*

## Near-term (1-2 weeks)

1. Add architecture sequence diagrams (stage order + event flow).
2. Add reproducibility runbook (recipe/defaults/build_id interactions).
3. Add platform notes (Windows/Linux caveats, including known test portability edge cases).
4. Seed wiki cookbook pages from recurring operator scenarios.

## Ongoing maintenance

1. Monthly docs drift pass (README vs CLI vs code contracts).
2. Monthly wiki validation pass against latest tagged release.
3. Keep troubleshooting entries tied to real incident signatures.

## Prioritization principle

- If mismatch can break a run: fix repo canonical docs first.
- If mismatch is explanatory only: wiki can lead, then back-link to canonical docs.
