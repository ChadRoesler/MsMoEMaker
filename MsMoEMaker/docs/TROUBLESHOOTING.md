# Troubleshooting

Use this page for fast diagnosis of common failures.

## 1) Command rejected or missing behavior

### Signature

```text
usage: ms-moe-maker ...
ms-moe-maker: error: unrecognized arguments: --json
```

or

```text
ms-moe-maker: error: invalid choice: 'export'
```

### Checks

- Run `ms-moe-maker describe` and confirm expected commands/modes.
- Verify install shape: `pip show ms-moe-maker`.
- Confirm you are in the intended virtual environment.

### Fix

- Reinstall package in clean venv.
- If behavior changed, compare against `docs/CLI.md` and project version.

## 2) Build cannot start due to missing training deps

### Signature

```text
ModuleNotFoundError: No module named 'torch'
```

(or `transformers`, `datasets`, etc. during build path)

### Fix

- Install training extra on build machine:
  - `pip install "ms-moe-maker[train]"`
- Re-run: `ms-moe-maker validate recipe.yaml` then `ms-moe-maker build recipe.yaml`.

## 3) `--plan` works but build fails later

### Signature

- Plan succeeds on laptop; runtime build fails on GPU machine.

### Why

- `--plan` is resolution/validation path; full build still needs runtime deps + hardware/runtime prerequisites.

### Fix

- Validate env on target build machine with `validate`, then run build.
- Check roots write permissions and available storage.

## 4) Resume refusal due to build mismatch

### Signature

```text
REFUSING TO RESUME: this run directory was built by a different build.
What changed:
  · target_steps: 400 -> 1200
```

### Fix options

- Use `--force` to rebuild all with new settings.
- Reuse the original defaults file via `--defaults`.
- Write outputs to a different root and keep both runs.

## 5) GGUF smoke path fails

### Signature

```text
smoke ... did not finish in <N>s
```

or converter/script resolution errors in the smoke/export path.

### Fix

- Ensure llama.cpp checkout path is set in recipe `runtime.llama_cpp`.
- Confirm converter script exists in resolved llama.cpp path.
- Increase timeout for slow boxes: `ms-moe-maker smoke recipe.yaml --timeout 600`.

## 6) Windows-specific test failure (`os.geteuid`)

### Signature

```text
AttributeError: module 'os' has no attribute 'geteuid'
```

(Observed in `tests/test_preflight.py` on Windows.)

### Status

- Known portability issue in tests, not core runtime logic.

### Mitigation

- Guard the test for platforms where `os.geteuid` exists.
