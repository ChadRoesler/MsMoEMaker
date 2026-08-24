# Documentation Maintenance Cadence

## Ownership

- Primary owner: repository maintainer (see `.github/CODEOWNERS`).
- Contributors updating behavior are responsible for matching doc updates in the same PR.

## Cadence

### Per PR (required)

- If command/flag behavior changed: update `docs/CLI.md`.
- If stage/event/manifest behavior changed: update `docs/ARCHITECTURE.md`.
- If failure/remediation changed: update `docs/TROUBLESHOOTING.md`.
- If onboarding path changed: update `README.md` quick-start sections.

### Monthly (recommended)

- README drift check against current CLI behavior.
- Wiki drift check against canonical repo docs.
- Refresh top 5 troubleshooting signatures from recent issues.

### Per release tag

- Validate all wiki contract claims against the release tag.
- Update wiki page banners with `Validated against commit/tag`.

## Acceptance criteria for docs PRs

1. Links resolve (CI docs-check passes).
2. Canonical ownership is respected (`docs/SOURCE_OF_TRUTH.md`).
3. User-facing behavior statements match current CLI.
4. Examples are runnable or clearly marked illustrative.

## Labels (suggested)

- `docs-gap` - missing documentation
- `docs-drift` - inaccurate/outdated documentation
- `wiki-sync` - wiki and repo divergence
