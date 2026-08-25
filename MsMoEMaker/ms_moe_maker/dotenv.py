"""Load a `.env` file into os.environ, without adding a dependency.

huggingface_hub and datasets already read HF_TOKEN (and HF_HOME,
HF_HUB_DISABLE_XET, HF_ENDPOINT, ...) from os.environ natively — the "auto
consume" half is already done by those libraries. This is the "carry it in a
file" half: a tiny, stdlib-only parser so a box can hold HF_TOKEN in `.env`
next to the recipe instead of a shell export or `huggingface-cli login`.

Precedence: the SHELL wins. A variable already in os.environ is never
overwritten, so an explicit `export HF_TOKEN=...` (or `huggingface-cli login`)
always beats the file. Values are raw except for matching single/double quotes,
which are stripped.

This loads EVERY `KEY=VALUE`, not just HF_* — so MSMOE_TIER and the other
MSMOE_* levers work here too.
"""
from __future__ import annotations

import os
from pathlib import Path


def _parse(line: str):
    """One `.env` line -> (key, value), or None to skip."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(path=None) -> dict:
    """Load KEY=VALUE pairs from `path` (default `.env` in cwd) into os.environ.

    Never overrides a variable already set in the environment. Returns the dict
    of variables that were actually applied.
    """
    p = Path(path) if path else Path(".env")
    if not p.is_file():
        return {}

    applied = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        parsed = _parse(raw)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue  # the shell wins over the file
        os.environ[key] = value
        applied[key] = value
    return applied
