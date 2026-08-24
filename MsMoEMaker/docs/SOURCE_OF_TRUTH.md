# Source of Truth Map

Use this file to prevent duplication and drift across README, `/docs`, and wiki.

## Canonical location by topic

| Topic | Canonical location | Why |
|---|---|---|
| Install shapes and dependency split (`base` vs `[train]`) | `README.md` + `pyproject.toml` | First-run critical and versioned |
| CLI commands, flags, JSON behavior | `docs/CLI.md` | Detailed, versioned reference |
| Defaults layering/provenance behavior | `docs/CLI.md` + code docs in `defaults.py` | Must match runtime exactly |
| Build stages and pipeline contracts | `docs/ARCHITECTURE.md` | Shared contract for contributors/operators |
| Event and manifest compatibility expectations | `docs/ARCHITECTURE.md` | Integration-facing contract |
| Failure signatures and fixes | `docs/TROUBLESHOOTING.md` | Fast support path |
| Contributor workflow and doc policy | `CONTRIBUTING.md` | Repo process policy |
| Long-form how-to playbooks and scenario guides | GitHub Wiki | High churn / lower contract risk |
| FAQ and glossary | GitHub Wiki | Evolves frequently |

## Repo vs wiki rules

- Repo docs win on any conflict for behavior and command semantics.
- Wiki pages should link to canonical repo docs for behavior claims.
- Wiki pages must include a “Validated against commit/tag” note.

## Doc PR minimum bar

For code PRs that change behavior:

1. Update canonical doc(s) listed above.
2. Add/refresh at least one example or failure case if user-visible behavior changed.
3. Ensure README links still point to valid targets.
