"""`abliterate.base` stage — decensor the base with the vendored Heretic core.

The heavy import lives inside the function, so `ms-moe-maker validate` and
`build --plan` stay torch-free (the base install only needs pyyaml). The stage
produces OUTPUT_ROOT/abliterated_base, and the builder repoints `config.base`
at it so every specialist LoRA-trains from the decensored checkpoint.
"""
from __future__ import annotations

import os


def abliterate_dir(config) -> str:
    return f"{config.output_root}/abliterated_base"


def abliterate_is_done(config) -> bool:
    if config.force:
        return False
    return os.path.exists(os.path.join(abliterate_dir(config), "config.json"))


def abliterate_base(config) -> str:
    """Run the Heretic core on `config.base`, return the decensored dir."""
    # Lazy import: torch/transformers/optuna/peft must never be pulled in by
    # `validate` or `build --plan`.
    from .heretic.abliterate import run_abliteration
    from .heretic.config import ExportStrategy, QuantizationMethod, Settings

    out_dir = abliterate_dir(config)
    if abliterate_is_done(config):
        print(f"[skip] abliterated base already present at {out_dir}")
        return out_dir

    print(f"Abliterating base {config.base} (Heretic core)...")
    settings = Settings(
        model=config.base,
        save_directory=out_dir,
        export_strategy=ExportStrategy(config.abliterate_export),
        checkpoint_action=config.abliterate_checkpoint_action,
        trial_index=config.abliterate_trial_index,
        n_trials=config.abliterate_n_trials,
        seed=config.abliterate_seed,
        quantization=QuantizationMethod(config.abliterate_quantization),
        study_checkpoint_dir=os.path.join(config.output_root, "abliterate_checkpoints"),
    )
    run_abliteration(settings)
    return out_dir
