"""Recipe → PipelineConfig bridge with auto-fill from hardware tier.

Takes a Recipe dataclass (possibly minimal — just experts) and fills in
sensible defaults from the target hardware tier.  The result is a single
frozen config that every pipeline module can consume.

Env vars from MSMOE_* and MSMOE_* override recipe values, so a human or
seren-theatre can still tweak behaviour without editing files.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── what we can actually stitch ────────────────────────────────────────────────
#
# stitch.py builds a Qwen2MoeConfig and router.py loads a Qwen2MoeForCausalLM.
# The FINE-TUNE half is generic - AutoModelForCausalLM will happily train a
# Llama or a Mistral specialist - which is exactly what makes this dangerous:
# a non-Qwen base gets all the way through corpus collection and EVERY
# specialist, hours of GPU, and then dies at stage 4 when a Qwen MoE skeleton
# cannot be built out of Llama checkpoints.
#
# So the supported set is declared here and checked by `validate`, which runs
# on a laptop with no GPU. Refusing in two seconds beats refusing after four
# hours, and "we cannot do this yet, here is what we can do" is the kind
# version of the same answer.
#
# Adding an architecture means adding its MoE config/model classes to stitch
# and router; this list is the promise, not the wish.
SUPPORTED_MOE_ARCHS: Dict[str, str] = {
    "qwen2": "Qwen2MoeConfig / Qwen2MoeForCausalLM",
}

# Substrings that identify a base model as belonging to a supported family.
# Matched case-insensitively against the model id, because that is all we have
# before anything is downloaded - `validate` deliberately touches no network.
SUPPORTED_BASE_HINTS: Dict[str, str] = {
    "qwen": "qwen2",
}


# ── pipeline constants ─────────────────────────────────────────────────────────

MODEL_SIZES: Dict[str, Tuple[str, str]] = {
    "0.5B": (
        "Qwen/Qwen2.5-Coder-0.5B",
        "huihui-ai/Qwen2.5-Coder-0.5B-Instruct-abliterated",
    ),
    "1.5B": (
        "Qwen/Qwen2.5-Coder-1.5B",
        "huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated",
    ),
    "3B": (
        "Qwen/Qwen2.5-Coder-3B",
        "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
    ),
    "7B": (
        "Qwen/Qwen2.5-Coder-7B",
        "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated",
    ),
    "14B": (
        "Qwen/Qwen2.5-Coder-14B",
        "huihui-ai/Qwen2.5-Coder-14B-Instruct-abliterated",
    ),
    "32B": (
        "Qwen/Qwen2.5-Coder-32B",
        "huihui-ai/Qwen2.5-Coder-32B-Instruct-abliterated",
    ),
}

CODE_LANGUAGES: List[str] = ["Python", "C#", "PowerShell", "Shell"]

LANGUAGE_SOURCES: Dict[str, Dict[str, str]] = {
    "PowerShell": {
        "repo": "SaeedRahmani/codeparrot_github_code_powershell",
        "split": "train",
        "text_field": "code",
    },
}

DISPLAY_LANG: Dict[str, str] = {
    "python": "Python",
    "csharp": "C#",
    "powershell": "PowerShell",
    "shell": "Bash",
    "monster_manual": "Monster Manual",
    "players_handbook": "Player's Handbook",
    "dm_guide": "DMG",
    "arithmetic": "Arithmetic",
    "algebra": "Algebra",
    "geometry": "Geometry",
    "word_problems": "Word Problems",
    "ingredients": "Ingredients",
    "techniques": "Techniques",
    "cuisines": "Cuisines",
}


# ── hardware tier → model hints ───────────────────────────────────────────────

_TIER_HINTS: Dict[str, Dict[str, str]] = {
    "nano": {
        "model_prefix": "Qwen/Qwen2.5",
        "preferred_size": "3B",
        "quant": "Q4_K_M",
    },
    "xavier": {
        "model_prefix": "Qwen/Qwen2.5",
        "preferred_size": "7B",
        "quant": "Q5_K_M",
    },
    "spark": {
        "model_prefix": "Qwen/Qwen2.5",
        "preferred_size": "32B",
        "quant": "Q8_0",
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    """All values needed by the pipeline stages, resolved once at the top.

    Every field has a clear source: recipe value, env override, or default
    below.  Functions downstream never look at env vars or recipes — they
    read this object.
    """
    # Identity
    name: str
    size: str
    base: str
    base_safe: str
    dryrun: bool = False
    force: bool = False

    # Data roots
    data_root: str = ""
    output_root: str = ""

    # Corpus collection
    num_code_samples: int = 100_000
    collect_token_target: float = 0.0
    chars_per_token_est: float = 3.2
    min_samples_per_expert: int = 2_000
    max_shards: int = 80
    shard_cache: str = "shard_cache"

    # Agent data
    num_agent_samples: int = 15_000
    teacher_model: str = ""
    teacher_max_memory: str = "110GiB"
    teacher_batch: int = 96
    use_vllm: bool = False
    vllm_batch: int = 512
    vllm_gpu_util: float = 0.88
    vllm_max_len: int = 4096
    vllm_quantization: Optional[str] = None
    teacher_max_new: int = 224

    # LoRA / fine-tuning
    max_seq_length: int = 2048
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    load_in_4bit: bool = False
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    packing_strategy: str = "wrapped"
    target_modules: List[str] = field(default_factory=lambda: [
        "gate_proj", "up_proj", "down_proj"])
    attn_impl: str = "sdpa"
    per_device_batch: int = 4
    grad_accum: int = 2
    lr_lora: float = 2e-4
    specialist_save_steps: int = 200
    use_unsloth: bool = False

    # Budget
    target_steps: int = 1200
    expert_token_budget: int = 0
    warmup_steps: int = 60

    # Router
    router_mix_total: int = 12_000
    agent_mix_fraction: float = 0.15
    router_batch: int = 1
    router_accum: int = 8
    lr_router: float = 1e-4
    router_aux_loss_coef: float = 0.001

    # MoE
    experts_per_tok: int = 2
    norm_topk_prob: bool = True
    shared_expert_width: int = 1
    shared_expert_gate_fill: float = 0.02
    mlp_only_layers: List[int] = field(default_factory=list)

    # GGUF / inference
    llama_cpp_dir: str = "llama.cpp"
    gguf_smoke_prompt: str = ""
    gguf_smoke_tokens: int = 48
    gguf_smoke_timeout: int = 300
    gguf_degenerate_run: int = 32

    # HF cache
    hf_home: str = ""

    # Prompts
    code_prompt_templates: List[str] = field(default_factory=lambda: [
        "Write {lang}:",
        "Write a {lang} script for this.",
        "In {lang}, implement the following.",
        "Give me some {lang} code.",
        "{lang}, please:",
        "Can you write this in {lang}?",
    ])
    code_prompt_unnamed: List[str] = field(default_factory=lambda: [
        "Write code:",
        "Implement this.",
        "Here is some code:",
    ])
    code_prompt_unnamed_fraction: float = 0.25

    # Expert order
    expert_names: List[str] = field(default_factory=list)

    # Hardware tier
    tier: str = "spark"


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes")


def resolve_roots(size: str, dryrun: bool) -> Dict[str, str]:
    """Where a run's data and outputs live, relative to cwd.

    Named after this tool, not after the project it was carved out of. A user
    who pip-installs ms-moe-maker and runs it in their own directory should
    not find a folder called `fraunkenstein_agent_3B` in it.
    """
    if dryrun:
        return {"data": "msmoe_data", "output": f"msmoe_dryrun_{size}"}
    return {"data": "msmoe_data", "output": f"msmoe_run_{size}"}


def _resolve_base(recipe, size: str, tier_name: str) -> Tuple[str, str]:
    """Resolve (safe_base_id, abliterated_id) from recipe + tier.

    Priority: explicit recipe.base > env override > tier hint > model size.
    """
    # 1. Explicit recipe base
    if recipe.base and recipe.base.strip():
        base = recipe.base.strip()
        safe, ablated = MODEL_SIZES.get(size, ("", ""))
        if "abliterated" in base or "Instruct" in base:
            base_model = base
        else:
            base_model = ablated if ablated else base
        return base_model, safe

    # 2. Env override
    env_base = os.environ.get("MSMOE_BASE_MODEL", "").strip()
    if env_base:
        safe, ablated = MODEL_SIZES.get(size, ("", ""))
        if "abliterated" in env_base or "Instruct" in env_base:
            return env_base, safe
        return ablated if ablated else env_base, safe

    # 3. Tier hint → auto-select best model
    hint = _TIER_HINTS.get(tier_name, _TIER_HINTS["spark"])
    prefix = hint["model_prefix"]
    base = f"{prefix}-{size}"
    safe, ablated = MODEL_SIZES.get(size, (base, ""))
    if ablated and ("abliterated" in ablated or "Instruct" in ablated):
        base_model = ablated
    else:
        base_model = base
    return base_model, safe


def _resolve_size(recipe, tier_name: str) -> str:
    """Resolve model size.  Priority: recipe.size > tier default > '3B'."""
    recipe_size = recipe.size if recipe.size else "auto"
    if recipe_size != "auto" and recipe_size in MODEL_SIZES:
        return recipe_size

    hint = _TIER_HINTS.get(tier_name, _TIER_HINTS["spark"])
    return hint["preferred_size"]


def _auto_name(recipe, expert_names: List[str]) -> str:
    """Auto-generate a name from experts + timestamp."""
    if recipe.name and recipe.name.strip():
        return recipe.name.strip()
    parts = [n.replace("_", "-") for n in expert_names]
    ts = time.strftime("%Y%m%d", time.gmtime())
    return f"moe-{'-'.join(parts)}-{ts}"


def resolve_dryrun(dryrun: Optional[bool] = None) -> bool:
    """Is this a smallest-rung run?

    THE FLAG HAS TO REACH HERE. build_config used to read MSMOE_DRYRUN and
    nothing else, so `--dryrun` was inert: __main__ passed it to run_pipeline,
    which set `config.dryrun = True` AFTER build_config had already resolved
    every budget from the environment - and no stage module reads
    config.dryrun at all.

    The result was the worst possible shape for a flag. Asking for a cheap
    structural test gave you 10x the corpus (100,000 samples instead of
    10,000), 3x the router mix, the production minimum-samples floor, and it
    wrote all of it into the PRODUCTION run directory rather than the dryrun
    one - so it also collided with, and could be mistaken for, a real run.

    `None` means "ask the environment", which keeps MSMOE_DRYRUN=1 working
    for the legacy subprocess path and for anyone who scripted it.
    """
    if dryrun is None:
        return _env_bool("MSMOE_DRYRUN", False)
    return bool(dryrun)


def resolve_tier(recipe) -> str:
    """The hardware tier this recipe will actually run at."""
    tier_name = "spark"
    rt = getattr(recipe, "runtime", None)
    t = getattr(rt, "hardware_tier", "") if rt is not None else ""
    if t and t in _TIER_HINTS:
        tier_name = t
    if os.environ.get("MSMOE_TIER", "") in _TIER_HINTS:
        tier_name = os.environ["MSMOE_TIER"]
    return tier_name


def resolve_run_roots(recipe, dryrun: Optional[bool] = None) -> Dict[str, str]:
    """Where THIS recipe's run lives. The one answer, for everybody.

    Extracted because Runner and build_config were each working it out and
    getting different answers: Runner called resolve_roots(recipe.size, ...)
    with the RAW size, which is "auto" in any recipe that lets the tier pick,
    while build_config resolved "auto" to a concrete size first. The manifest
    went to msmoe_run_auto and every artifact to msmoe_run_7B.

    Runner does not need a whole PipelineConfig to know where to write - it
    needs the paths - so this is the smallest thing that can be shared. Two
    callers deriving one fact independently is how they drift.
    """
    size = _resolve_size(recipe, resolve_tier(recipe))
    return resolve_roots(size, resolve_dryrun(dryrun))


def build_config(recipe, force: bool = False,
                 dryrun: Optional[bool] = None) -> PipelineConfig:
    """Create a PipelineConfig from a Recipe + environment.

    Fills missing fields from hardware tier defaults.
    """
    dryrun = resolve_dryrun(dryrun)

    # Detect tier from recipe or environment
    tier_name = "spark"  # default middle tier
    if hasattr(recipe, 'runtime') and hasattr(recipe.runtime, 'hardware_tier'):
        t = recipe.runtime.hardware_tier
        if t and t in _TIER_HINTS:
            tier_name = t
    if os.environ.get("MSMOE_TIER", "") in _TIER_HINTS:
        tier_name = os.environ["MSMOE_TIER"]

    # Size — auto-resolve from tier or recipe
    size = _resolve_size(recipe, tier_name)

    # Base model — resolve from recipe/tier
    base_model, base_safe = _resolve_base(recipe, size, tier_name)

    # Name — auto from experts
    expert_names = [e.name for e in recipe.experts]
    name = _auto_name(recipe, expert_names)

    # Compute budgets from steps
    b = recipe.budget
    tokens_per_step = b.max_seq_length * b.per_device_batch * b.grad_accum
    target_steps = recipe.budget.target_steps
    expert_token_budget = target_steps * tokens_per_step
    warmup_steps = max(10, round(b.warmup_ratio * target_steps))
    collect_token_target = expert_token_budget * recipe.budget.collect_headroom

    # Agent samples
    agent_override = os.environ.get("MSMOE_AGENT_SAMPLES", "")
    if agent_override:
        num_agent_samples = int(agent_override)
    elif dryrun:
        num_agent_samples = 2_000
    else:
        num_agent_samples = 15_000

    # RECIPE FIRST, THEN THE RUN'S DEFAULT. -1 means "you decide", so a
    # recipe that says nothing behaves exactly as before and a recipe that
    # wants a small REAL run can say so without pretending to be a dryrun.
    _corpus = getattr(recipe, "corpus", None)

    def _corpus_knob(attr: str, dry: int, prod: int) -> int:
        asked = getattr(_corpus, attr, -1) if _corpus is not None else -1
        if asked is not None and asked >= 0:
            return int(asked)
        return dry if dryrun else prod

    num_code_samples = _corpus_knob("max_samples", 10_000, 100_000)

    # Hardware-appropriate defaults
    tier_spec = _TIER_HINTS.get(tier_name, _TIER_HINTS["spark"])

    # LoRA params — scale by tier
    lora_r_env = os.environ.get("MSMOE_LORA_R", "").strip()
    lora_r = int(lora_r_env) if lora_r_env else recipe.budget.target_steps
    lora_r = min(lora_r, 256)  # cap
    if not os.environ.get("MSMOE_LORA_R", ""):
        if tier_name == "nano":
            lora_r = 32
        elif tier_name == "xavier":
            lora_r = 64
        elif tier_name == "spark":
            lora_r = 128
        else:
            lora_r = 64  # fallback to middle

    # Quantization hint from tier
    quant = tier_spec.get("quant", "Q4_K_M")
    load_4bit = quant in ("Q4_K_M", "Q4_0", "Q4_1") if tier_name == "nano" else False

    # Unsoth / vLLM
    use_unsloth = _env_bool("MSMOE_UNSLOTH", False)
    use_vllm = _env_bool("MSMOE_VLLM", False)

    # Optimiser
    optim = "adamw_8bit" if use_unsloth else "adamw_torch"

    # Gradient checkpointing
    grad_ckpt = _env_bool("MSMOE_GRAD_CKPT", False)

    # Dense layers from env or recipe
    env_dense = os.environ.get("MSMOE_DENSE_LAYERS", "").strip()
    if env_dense:
        mlp_only_layers = [int(x) for x in env_dense.split(",") if x.strip()]
    else:
        mlp_only_layers = (
            recipe.moe.dense_layers
            if recipe.moe.dense_layers and recipe.moe.dense_layers != "auto"
            else []
        )

    # Llama.cpp dir
    llama_cpp_dir = os.environ.get("MSMOE_LLAMA_CPP", "llama.cpp")

    # Smoke timeout
    smoke_timeout_str = os.environ.get("MSMOE_SMOKE_TIMEOUT", "")
    smoke_timeout = int(smoke_timeout_str) if smoke_timeout_str else 300

    # HF cache
    hf_home_override = os.environ.get("HF_HOME", "")
    if not hf_home_override:
        hf_home_override = os.path.join(
            os.path.dirname(
                recipe.roots.output.format(size=size)
                if "{size}" in recipe.roots.output
                else recipe.roots.output
            ),
            "hf_cache",
        )

    # Compute roots
    roots = resolve_roots(size, dryrun)

    return PipelineConfig(
        name=name,
        size=size,
        base=base_model,
        base_safe=base_safe,
        dryrun=dryrun,
        force=force,
        data_root=roots["data"],
        output_root=roots["output"],
        # Corpus
        num_code_samples=num_code_samples,
        collect_token_target=collect_token_target,
        chars_per_token_est=3.2,
        min_samples_per_expert=_corpus_knob("min_samples", 500, 2_000),
        max_shards=80,
        shard_cache="shard_cache",
        # Agent data
        num_agent_samples=num_agent_samples,
        teacher_model=(
            "Qwen/Qwen2.5-7B-Instruct" if dryrun
            else "Qwen/Qwen2.5-32B-Instruct"
        ),
        teacher_max_memory="110GiB",
        teacher_batch=(96 if not use_vllm else 512),
        use_vllm=use_vllm,
        vllm_batch=512,
        vllm_gpu_util=0.88,
        vllm_max_len=4096,
        vllm_quantization=None,
        teacher_max_new=224,
        # LoRA
        max_seq_length=b.max_seq_length,
        lora_r=lora_r,
        lora_alpha=32,
        lora_dropout=0.0,
        load_in_4bit=load_4bit,
        optim=optim,
        gradient_checkpointing=grad_ckpt,
        packing_strategy="wrapped",
        target_modules=["gate_proj", "up_proj", "down_proj"],
        attn_impl="sdpa",
        per_device_batch=b.per_device_batch,
        grad_accum=b.grad_accum,
        lr_lora=2e-4,
        specialist_save_steps=200,
        use_unsloth=use_unsloth,
        # Budget
        target_steps=target_steps,
        expert_token_budget=expert_token_budget,
        warmup_steps=warmup_steps,
        # Router
        router_mix_total=_corpus_knob("router_mix_total", 4_000, 12_000),
        agent_mix_fraction=0.15,
        router_batch=1,
        router_accum=8,
        lr_router=1e-4,
        router_aux_loss_coef=0.001,
        # MoE
        experts_per_tok=recipe.moe.experts_per_tok,
        norm_topk_prob=recipe.moe.norm_topk_prob,
        shared_expert_width=recipe.moe.shared_expert_width,
        shared_expert_gate_fill=0.02,
        mlp_only_layers=mlp_only_layers,
        # GGUF
        llama_cpp_dir=llama_cpp_dir,
        gguf_smoke_prompt=f"Write a {CODE_LANGUAGES[0].lower()} function that works.",
        gguf_smoke_tokens=48,
        gguf_smoke_timeout=smoke_timeout,
        gguf_degenerate_run=32,
        # HF
        hf_home=hf_home_override,
        # Prompts
        expert_names=expert_names,
        # Tier
        tier=tier_name,
    )


def safe_name(language: str) -> str:
    """Pipeline's language → safe directory name convention."""
    return {"c#": "csharp", "c++": "cpp"}.get(
        language.lower(), language.lower().replace(" ", "_"),
    )


def name_to_display(name: str) -> str:
    """Expert name → display name for prompts."""
    return DISPLAY_LANG.get(name, name.replace("_", " ").title())
