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

When a PR changes user-visible behavior, update the wiki in the same PR:

- command or flag changes -> `wiki/CLI-Reference.md`
- stage/event/manifest behavior changes -> `wiki/Architecture.md`
- failure mode/remediation changes -> `wiki/Troubleshooting-Signatures.md`
- README only for quick-start and high-level usage path

The wiki reference pages are the source of truth for behavior; the README is
quick-start only.

## Commit/PR expectations

- Keep docs and behavior in sync.
- Prefer additive compatibility for event/contract vocabularies.
- Call out any breaking contract changes explicitly in PR notes.
