# Contributing

## Local setup

```bash
cd MsMoEMaker
python -m venv .venv
# activate venv
pip install -e ".[dev]"
```

For build-path development:

```bash
pip install -e ".[train]"
```

## Test workflow

- Run full tests: `python -m pytest -q`
- Keep command/mode/event contract changes synchronized with docs and tests.

## Documentation policy

When a PR changes user-visible behavior, update docs in the same PR:

- command or flag changes -> `docs/CLI.md`
- stage/event/manifest behavior changes -> `docs/ARCHITECTURE.md`
- failure mode/remediation changes -> `docs/TROUBLESHOOTING.md`
- README only for quick-start and high-level usage path

Refer to `docs/SOURCE_OF_TRUTH.md` for ownership boundaries.

## Commit/PR expectations

- Keep docs and behavior in sync.
- Prefer additive compatibility for event/contract vocabularies.
- Call out any breaking contract changes explicitly in PR notes.
