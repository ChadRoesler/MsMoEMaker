"""`abliterate.base` stage — decensor the base with the vendored Heretic core.

The Heretic core runs in its OWN process (`python -m
ms_moe_maker.heretic.abliterate`), so its process-global state — torch grad
mode, seeds, logging verbosity, and its CUDA context — dies with the child
instead of leaking into the finetune stages. The stage writes a settings JSON,
spawns the child, and the builder repoints `config.base` at the result so every
specialist LoRA-trains from the decensored checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def abliterate_dir(config) -> str:
    return f"{config.output_root}/abliterated_base"


def abliterate_is_done(config) -> bool:
    if config.force:
        return False
    return os.path.exists(os.path.join(abliterate_dir(config), "config.json"))


def abliterate_base(config) -> str:
    """Abliterate `config.base` in a subprocess; return the decensored dir."""
    out_dir = abliterate_dir(config)
    if abliterate_is_done(config):
        print(f"[skip] abliterated base already present at {out_dir}")
        return out_dir

    print(f"Abliterating base {config.base} (Heretic core, subprocess)...")

    # Only the knobs this stage owns. The child reconstructs a Heretic Settings
    # from this JSON and leaves every other field at its default, so the parent
    # never imports the (heavy, stateful) Heretic modules at all.
    checkpoint_dir = os.path.join(config.output_root, "abliterate_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    settings_path = os.path.join(checkpoint_dir, "settings.json")
    payload = {
        "model": config.base,
        "save_directory": out_dir,
        "export_strategy": config.abliterate_export,
        "checkpoint_action": config.abliterate_checkpoint_action,
        "trial_index": config.abliterate_trial_index,
        "n_trials": config.abliterate_n_trials,
        "seed": config.abliterate_seed,
        "quantization": config.abliterate_quantization,
        "study_checkpoint_dir": checkpoint_dir,
    }
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    completed = subprocess.run(
        [sys.executable, "-m", "ms_moe_maker.heretic.abliterate",
         "--settings", settings_path],
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"abliteration subprocess failed with exit code "
            f"{completed.returncode}; settings at {settings_path}")
    return out_dir
