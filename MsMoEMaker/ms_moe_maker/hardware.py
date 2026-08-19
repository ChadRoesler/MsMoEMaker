"""Hardware tier definitions — auto-select model size, LoRA params, quant format.

Tiers are ordered from lowest to highest hardware capability.  When a recipe
omits size / budget / quant, the builder picks from the tier that matches the
target device.  The user can still override any value.

  nano   — Jetson Orin Nano / similar 8 GB class devices.
           Lowest common denominator.  Q4 GGUF, single-expert builds, 0.5 B.

  spark  — NVIDIA DGX Spark / 128 GB unified VRAM.
           Mid-tier.  LoRA r=64, up to 3 B experts, Q8 quant.

  dgx    — Multi-GPU DGX / A100/H100 class.
           Full-bore.  LoRA r=128+, up to 7 B, multi-GPU, FP8 capable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TierSpec:
    """One hardware tier's capabilities and auto-defaults."""
    max_vram_gb: int
    """Maximum VRAM this tier is designed for."""
    recommended_sizes: List[str]
    """Recommended model sizes, ordered from smallest to largest."""
    default_size: str
    """Size chosen when recipe omits 'size'."""
    default_lora_r: int
    """Default LoRA rank for fine-tuning."""
    default_lora_alpha: int
    """Default LoRA alpha."""
    default_quant: str
    """Default GGUF quantisation format."""
    default_max_seq: int
    """Default sequence length."""
    default_target_steps: int
    """Default training steps per expert."""
    default_per_device_batch: int
    """Default per-device batch size."""
    default_grad_accum: int
    """Default gradient accumulation."""
    supports_mgpu: bool
    """Can distribute training across GPUs?"""
    supports_8bit: bool
    """Can use 8-bit optimiser (AdamW 8-bit)?"""
    supports_fp8: bool
    """Can use FP8 quantisation?"""


# ── Tier definitions ──────────────────────────────────────────────────────────

TIERS: Dict[str, TierSpec] = {
    "nano": TierSpec(
        max_vram_gb=3,
        recommended_sizes=["0.5B", "1.5B", "3B"],
        default_size="3B",
        default_lora_r=32,
        default_lora_alpha=16,
        default_quant="Q4_K_M",
        default_max_seq=2048,
        default_target_steps=500,
        default_per_device_batch=1,
        default_grad_accum=4,
        supports_mgpu=False,
        supports_8bit=True,
        supports_fp8=False,
    ),
    "xavier": TierSpec(
        max_vram_gb=9,
        recommended_sizes=["0.5B", "1.5B", "3B", "9B"],
        default_size="9B",
        default_lora_r=64,
        default_lora_alpha=32,
        default_quant="Q5_K_M",
        default_max_seq=4096,
        default_target_steps=800,
        default_per_device_batch=2,
        default_grad_accum=4,
        supports_mgpu=False,
        supports_8bit=True,
        supports_fp8=False,
    ),
    "spark": TierSpec(
        max_vram_gb=36,
        recommended_sizes=["0.5B", "1.5B", "3B", "9B", "36B"],
        default_size="36B",
        default_lora_r=128,
        default_lora_alpha=64,
        default_quant="Q8_0",
        default_max_seq=4096,
        default_target_steps=1200,
        default_per_device_batch=4,
        default_grad_accum=2,
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

    # Default to middle tier
    return "xavier"


def get_tier(tier_name: str) -> TierSpec:
    """Get the spec for a tier name."""
    if tier_name not in TIERS:
        raise ValueError(
            f"unknown tier {tier_name!r}. Known: {', '.join(sorted(TIERS))}")
    return TIERS[tier_name]


def tier_for_size(size: str) -> str:
    """Find the tier that best supports a given model size."""
    for name, spec in TIERS.items():
        if size in spec.recommended_sizes:
            return name
    return "spark"  # fallback: largest tier: dgx supports everything
