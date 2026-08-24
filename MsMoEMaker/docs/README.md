# Documentation Map

This directory is the versioned documentation for behavior that must stay in lockstep with code and releases.

## Scope boundaries

- `README.md` (repo root): fast path only (install, first success, key concepts).
- `docs/*`: canonical references for commands, architecture, troubleshooting, and contributor process.
- GitHub Wiki: higher-churn guides, cookbooks, and operational notes that can evolve faster than release-tagged docs.

If a behavior can break a run (CLI flags, defaults layering, stage semantics, manifest/event contracts), it belongs in this repo docs set first.

## Canonical docs in this folder

- `CLI.md` - commands, flags, examples, and expected outcomes.
- `ARCHITECTURE.md` - stage flow, event stream, manifest contract, and boundaries.
- `TROUBLESHOOTING.md` - common failure signatures and direct fixes.
- `SOURCE_OF_TRUTH.md` - topic ownership map (README vs docs vs wiki).
- `WIKI_BOOTSTRAP.md` - starter information architecture for the GitHub Wiki.
- `DOC_GAPS.md` - audience-based gap inventory.
- `DOC_BACKLOG.md` - prioritized implementation backlog.
- `MAINTENANCE_CADENCE.md` - review/ownership schedule.

Related root docs:

- `../CONTRIBUTING.md` - local dev workflow and doc update rules.
- `../SECURITY.md` - vulnerability reporting policy.

## Update triggers

Update docs when any of these changes:

1. A command/flag or mode changes.
2. Config/default behavior changes.
3. Build stages/events/manifest contracts change.
4. CI/release behavior that users rely on changes.
5. User-facing error text or recommended remediation changes.
