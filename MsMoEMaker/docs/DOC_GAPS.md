# Documentation Gaps (Current State)

This is the blunt inventory of what was missing or weak, grouped by audience.

## New users (first success path)

### Missing / weak

- README was carrying too much deep theory before quick operational answers.
- No single CLI reference page for command/flag lookup.
- No dedicated troubleshooting page for common failure signatures.

### Fixed in this pass

- Added a docs map and source-of-truth policy.
- Added `docs/CLI.md`.
- Added `docs/TROUBLESHOOTING.md`.

## Operators (repeatability + run hygiene)

### Missing / weak

- No single architecture page describing stage boundaries and contracts.
- No single page explaining where behavior contracts live.
- Wiki boundaries were undefined, creating drift risk.

### Fixed in this pass

- Added `docs/ARCHITECTURE.md`.
- Added `docs/SOURCE_OF_TRUTH.md` with repo-vs-wiki rules.
- Added `docs/WIKI_BOOTSTRAP.md` to launch the wiki with anti-drift controls.

## Contributors (maintainers + PR authors)

### Missing / weak

- No contributor guide with local workflow and doc update expectations.
- No security reporting policy file.

### Fixed in this pass

- Added `CONTRIBUTING.md`.
- Added `SECURITY.md`.
- Added `.github/CODEOWNERS` with docs ownership defaults.

## Remaining debt (intentional follow-up)

Completed from this follow-up pass:

1. Markdown lint + link checking in CI for docs PRs (release workflow `docs-check`).
2. Docs checklist in PR template (`.github/pull_request_template.md`).
3. Monthly wiki drift review issue automation (`.github/workflows/docs-drift-review.yml`).

Current remaining debt:

- Keep troubleshooting examples tied to real, recent failure signatures.
- Add architecture sequence diagrams for stage + event flow.
- Publish initial wiki pages from `docs/WIKI_BOOTSTRAP.md`.