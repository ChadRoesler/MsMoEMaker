"""Hardware tier definitions — the ONE place tier defaults live.

Tiers are ordered from lowest to highest capability. When a recipe omits
`size`, the builder picks the tier's `default_size`; the tier also supplies the
default LoRA rank and GGUF quant. Nothing else is per-tier: `max_seq_length`,
`target_steps`, `per_device_batch`, `grad_accum` and `lora_alpha` come from the
recipe's `budget:` block with flat defaults, and keeping second copies of those
here is how this table and config.py drifted apart — xavier defaulted to 9B
here and 7B in config, and 9B is not even a model size the base resolver knows.

  nano   — Jetson Orin Nano class, ~3 GB.  Q4 quant, 0.5B–3B.
  xavier — ~9 GB class, the middle and the default when a recipe omits the
           tier.  Q5 quant, up to 7B.
  spark  — NVIDIA DGX Spark class, ~36 GB.  Q8 quant, up to 32B.

config.py imports these definitions. The `_TIER_HINTS` / `_TIER_RANK` tables it
used to keep were the second copy, and they drifted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TierSpec:
    """One hardware tier's capabilities and the defaults it owns."""
    max_vram_gb: int
    """Maximum VRAM this tier is designed for."""
    recommended_sizes: List[str]
    """Model sizes this tier is a good home for, smallest to largest."""
    default_size: str
    """Size chosen when the recipe omits 'size'."""
    default_lora_r: int
    """Default LoRA rank for fine-tuning."""
    default_quant: str
    """Default GGUF quantisation format."""
    supports_mgpu: bool
    """Can distribute training across GPUs?"""
    supports_8bit: bool
    """Can use an 8-bit optimiser (AdamW 8-bit)?"""
    supports_fp8: bool
    """Can use FP8 quantisation?"""


# ── Tier definitions ──────────────────────────────────────────────────────────

TIERS: Dict[str, TierSpec] = {
    "nano": TierSpec(
        max_vram_gb=3,
        recommended_sizes=["0.5B", "1.5B", "3B"],
        default_size="3B",
        default_lora_r=32,
        default_quant="Q4_K_M",
        supports_mgpu=False,
        supports_8bit=True,
        supports_fp8=False,
    ),
    "xavier": TierSpec(
        max_vram_gb=9,
        recommended_sizes=["0.5B", "1.5B", "3B", "7B"],
        default_size="7B",
        default_lora_r=64,
        default_quant="Q5_K_M",
        supports_mgpu=False,
        supports_8bit=True,
        supports_fp8=False,
    ),
    "spark": TierSpec(
        max_vram_gb=36,
        recommended_sizes=["0.5B", "1.5B", "3B", "7B", "14B", "32B"],
        default_size="32B",
        default_lora_r=128,
        default_quant="Q8_0",
        supports_mgpu=False,
        supports_8bit=True,
        supports_fp8=True,
    ),
}


def resolve_tier(recipe_tier: Optional[str] = None,
                 detected_vram_gb: Optional[int] = None) -> str:
    """Decide which tier to use.

    Priority: explicit recipe tier > auto-detect from VRAM > 'xavier' (middle).
    """
    if recipe_tier and recipe_tier in TIERS:
        return recipe_tier

    # Auto-detect from available VRAM
    if detected_vram_gb is not None:
        if detected_vram_gb <= 3:
            return "nano"
        elif detected_vram_gb <= 9:
            return "xavier"
        else:
            return "spark"

    # Default to the middle tier
    return "xavier"


def get_tier(tier_name: str) -> TierSpec:
    """Get the spec for a tier name."""
    if tier_name not in TIERS:
        raise ValueError(
            f"unknown tier {tier_name!r}. Known: {', '.join(sorted(TIERS))}")
    return TIERS[tier_name]


def tier_for_size(size: str) -> str:
    """Find the smallest tier that is a good home for a model size."""
    for name, spec in TIERS.items():
        if size in spec.recommended_sizes:
            return name
    return "spark"  # fallback: the largest tier supports everything
