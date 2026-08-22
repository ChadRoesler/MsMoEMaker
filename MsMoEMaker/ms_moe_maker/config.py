"""Recipe → PipelineConfig bridge with auto-fill from hardware tier.

Takes a Recipe dataclass (possibly minimal — just experts) and fills in
sensible defaults from the target hardware tier.  The result is a single
frozen config that every pipeline module can consume.

Env vars from MSMOE_* and MSMOE_* override recipe values, so a human or
seren-theatre can still tweak behaviour without editing files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from . import hardware
from . import reasoning as _reasoning


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

# ── reasoning tag styles & families ───────────────────────────────────────────
#
# THE TABLE MOVED TO A FILE. It used to be two dicts here with a comment saying
# they mirrored `reasoning.yaml` - which is two sources of truth for one fact,
# declared as intent, and they had already drifted: the yaml knew DeepSeek V4
# and Llama 3.3, the Python knew QwQ, and neither knew what the other did.
#
# `reasoning.py` owns it now, loads it from a layered yaml, and keeps a
# one-entry floor so a missing file cannot take a build down. Nothing here
# holds a copy. See that module for why this is a file at all: a wrong tag
# style is a silent wrong answer, not a crash.
from .reasoning import ReasoningStyle, ReasoningFamily  # noqa: F401  (re-export)


def reasoning_table(recipe=None):
    """(styles, families) for this box. Warnings are surfaced by recipe.load."""
    styles, families, _ = _reasoning.load()
    return styles, families


def reasoning_type_of(recipe) -> str:
    """The reasoning STYLE key a base model uses, or '' for a plain base.

    base_kind=nonreasoning -> ''; reasoning -> the family's style, else 'xml';
    auto -> sniff the id against the families table.
    """
    styles, families = reasoning_table(recipe)
    return _reasoning.style_for_base(
        getattr(recipe, "base", "") or "",
        getattr(recipe, "base_kind", "auto"),
        styles, families)


def reasoning_style_of(recipe) -> Optional[ReasoningStyle]:
    """The ReasoningStyle, or None for a non-reasoning base."""
    styles, _ = reasoning_table(recipe)
    return styles.get(reasoning_type_of(recipe))


def is_reasoning_base(recipe) -> bool:
    """Does this recipe's base model want reasoning handling?"""
    return reasoning_type_of(recipe) != ""


def reasoning_style_of_config(config) -> Optional[ReasoningStyle]:
    """The style a RUN writes and reads, rebuilt from the resolved config.

    The delimiters are stamped INTO PipelineConfig at build time rather than
    looked up later, so a run keeps reading traces with the same tags it wrote
    them with even if the box's table is edited mid-build - and so the tags
    land in build_fingerprint, where changing your reasoning table correctly
    counts as changing the build.
    """
    if not getattr(config, "reasoning_open", "") or \
            not getattr(config, "reasoning_close", ""):
        return None
    return ReasoningStyle(
        name=getattr(config, "reasoning_type", "") or "reasoning",
        open=config.reasoning_open, close=config.reasoning_close,
        interwoven=bool(getattr(config, "reasoning_interwoven", False)))


def teacher_for(recipe, config, expert_name: str) -> str:
    """The teacher model for one GENERATED expert (synth / tools / reasoning).

    Priority: the source's `teacher` field > the reasoning default (when the
    expert is `reasoning: true`) > the generic synth teacher. `source.teacher`
    was declared and validated for the whole life of the synth pipeline and
    then never read — this is the threading that makes it real.
    """
    for e in getattr(recipe, "experts", None) or []:
        if getattr(e, "name", "") == expert_name:
            src = getattr(e, "source", None)
            t = getattr(src, "teacher", None) if src else None
            if t:
                return t
            if getattr(src, "reasoning", False):
                return config.reasoning_teacher or config.teacher_model
            break
    return config.teacher_model


# ── what a BUILD is, as opposed to what a RECIPE is ───────────────────────────
#
# `recipe_id` hashes the recipe. It excludes `runtime`, so it has never
# identified a build - two boxes could always produce different artifacts from
# the same id. The defaults layer made that gap load-bearing: the values that
# decide what gets trained now legitimately live in a file the recipe never
# mentions, so "same recipe" stopped implying "same build" in the normal case
# rather than the exotic one.
#
# `build_id` closes it. It hashes the RESOLVED config - everything that
# actually determines an artifact, after every layer has had its say.
#
# FAIL CLOSED. The fingerprint is every PipelineConfig field MINUS an explicit
# exclusion list, not a hand-picked list of included ones. A field added next
# year is covered by default; forgetting to add it to a list is the failure
# mode that makes a fingerprint quietly stop fingerprinting, and this codebase
# has met that bug under several other names.
_FINGERPRINT_EXCLUDE: frozenset = frozenset({
    # Identity and location, not content.
    "name",          # _auto_name embeds a timestamp
    "data_root", "output_root", "shard_cache", "hf_home", "llama_cpp_dir",
    # Flags about REDOING work, not about what the work produces.
    "force", "floor_raised",
    # Throughput. Changing how fast the teacher runs does not change what a
    # specialist learns; changing WHAT it generates does, so use_vllm,
    # vllm_quantization, vllm_max_len and teacher_max_new stay IN.
    "teacher_max_memory", "teacher_batch", "vllm_batch", "vllm_gpu_util",
    # The smoke test inspects the artifact; it does not build it.
    "gguf_smoke_prompt", "gguf_smoke_tokens", "gguf_smoke_timeout",
    "gguf_degenerate_run",
})


def build_fingerprint(config) -> Dict[str, Any]:
    """Every resolved value that decides what this build produces.

    Sorted and JSON-safe, because it is hashed and also stored in the manifest
    so a resumed run can say WHICH field moved rather than only that something
    did. "The fingerprint changed" is a fact; "target_steps 400 -> 1200" is an
    answer.
    """
    out: Dict[str, Any] = {}
    for f in fields(config):
        if f.name in _FINGERPRINT_EXCLUDE:
            continue
        v = getattr(config, f.name)
        if isinstance(v, (list, tuple)):
            v = list(v)
        elif isinstance(v, dict):
            v = {str(k): v[k] for k in sorted(v)}
        out[f.name] = v
    return dict(sorted(out.items()))


def build_id(config) -> str:
    """A short stable digest of build_fingerprint(config)."""
    blob = json.dumps(build_fingerprint(config), sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def fingerprint_diff(old: Dict[str, Any],
                     new: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """(field, was, now) for every value that moved. Sorted, so it reads the
    same twice."""
    keys = sorted(set(old) | set(new))
    return [(k, old.get(k, "(absent)"), new.get(k, "(absent)"))
            for k in keys if old.get(k, "(absent)") != new.get(k, "(absent)")]


# ── the router's appetite, expressed in documents ──────────────────────────────
#
# TWO NUMBERS FOR ONE FACT, AND THEY DID NOT TALK.
#
# `corpus.min_samples` is a floor on COLLECTED documents per expert, checked at
# the end of the corpus stage.  `corpus.router_mix_total` is a row count the
# ROUTER asks for hours later, split across experts, and drawn from the `.train`
# split ONLY - held-out has to stay held out or eval's routing probe narrows
# without saying so (see router.train_router for the run where that bit us).
#
# Nothing connected them.  So a recipe could collect exactly its floor, pass
# the corpus stage green, and then have the router come up SHORT of quota on
# every expert - which reads as "the gate did not learn" when the truth is
# "the gate was not fed".  That is the exact class of bug this codebase keeps
# finding: a check that reports less than it knows.
#
# One fact, one place.  Derive what the mix will ask for and let it RAISE the
# floor.  It never lowers it: a recipe that asked for more still gets more.
TRAIN_SPLIT_SHARE = 0.9   # complement of eval.held_out_fraction's default
ROUTER_DOC_MARGIN = 1.05  # rounding, dedupe, and the odd unparseable row


def router_doc_need(recipe, mix_total: int, agent_fraction: float = 0.15) -> int:
    """Documents per expert the ROUTER's mix will ask for.

    Mirrors the quota arithmetic in `router.train_router` deliberately rather
    than importing it, because that module pulls torch and this one has to stay
    laptop-safe - `validate` runs with no GPU and no transformers installed.
    If the quota math there changes, change it here; the two are pinned
    together by a test.

    Returns 0 when there is nothing to derive (no experts, no mix, or a build
    whose experts are all generated), which callers treat as "no opinion".
    """
    try:
        mix_total = int(mix_total)
    except (TypeError, ValueError):
        return 0
    if mix_total <= 0:
        return 0

    names = [getattr(e, "name", "") for e in (getattr(recipe, "experts", None) or [])]
    names = [n for n in names if n]
    if not names:
        return 0

    # The tools expert takes its slice off the top, then the rest is split
    # evenly across the remaining (collected) experts - mirroring
    # router.train_router's quota arithmetic.
    tools_name = tools_expert_name_of(recipe)
    rest = float(mix_total)
    if tools_name and tools_name in names:
        try:
            rest -= int(mix_total * float(agent_fraction))
        except (TypeError, ValueError):
            rest -= int(mix_total * 0.15)

    # The tools expert is GENERATED, not collected, so the corpus floor does
    # not govern it - only the collected experts matter here.
    collected = [n for n in names if n != tools_name]
    if not collected or rest <= 0:
        return 0
    biggest = rest / len(collected)

    # The mix is drawn from `.train`, which is the complement of whatever eval
    # holds out - so ask for the docs that make the TRAIN half big enough.
    share = 1.0 - held_out_fraction(recipe)

    return int(math.ceil(biggest / share * ROUTER_DOC_MARGIN))


def tools_expert_name_of(recipe) -> str:
    """The name of THE tools/MCP expert, or '' when there is none.

    The `tools_expert` flag sets it during parse. This fallback covers a Recipe
    built by hand rather than parsed: a synth expert literally named 'agentcore'
    is still recognised, preserving the pre-flag convention.
    """
    name = getattr(recipe, "tools_expert_name", "")
    if name:
        return name
    for e in getattr(recipe, "experts", None) or []:
        if getattr(e, "name", "") == "agentcore" and \
                getattr(getattr(e, "source", None), "kind", "") == "synth":
            return "agentcore"
    return ""


def held_out_fraction(recipe) -> float:
    """The eval held-out fraction, resolved and clamped in ONE place.

    router_doc_need's floor math and the router's `.train` split both derive
    from this, and they have to agree or the raised floor is a lie - a recipe
    whose held-out fraction differs from the split the router actually trains
    on is the exact "two numbers for one fact" failure this module exists to
    kill. A fraction of 0.95+ leaves no train split worth training a router
    on, so it falls back to the default, mirroring the guard that used to live
    inline in router_doc_need.
    """
    held_out = getattr(getattr(recipe, "eval", None), "held_out_fraction", 0.1)
    try:
        held_out = float(held_out)
    except (TypeError, ValueError):
        return 0.1
    if 0.0 <= held_out < 0.95:
        return held_out
    return 0.1


# ── pipeline constants ─────────────────────────────────────────────────────────

MODEL_SIZES: Dict[str, Tuple[str, str]] = {
    "0.5B": (
        "Qwen/Qwen2.5-Coder-0.5B",
        "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3",
    ),
    "1.5B": (
        "Qwen/Qwen2.5-Coder-1.5B",
        "huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated",
    ),
    "3B": (
        "Qwen/Qwen2.5-Coder-3B",
        "huihui-ai/Qwen2.5-3B-Instruct-abliterated-SFT",
    ),
    "7B": (
        "Qwen/Qwen2.5-Coder-7B",
        "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2",
    ),
    "14B": (
        "Qwen/Qwen2.5-Coder-14B",
        "huihui-ai/Qwen2.5-Coder-14B-Instruct-abliterated",
    ),
    "32B": (
        "Qwen/Qwen2.5-Coder-32B",
        "huihui-ai/Qwen2.5-32B-Instruct-abliterated",
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


# ── hardware tiers ────────────────────────────────────────────────────────────
# The tier table lives in hardware.py and is imported, not re-declared. A copy
# here (`_TIER_HINTS` + `_TIER_RANK`) drifted from it: xavier's default size was
# 9B there and 7B here, and 9B is not even a MODEL_SIZES key.
#
# And hardware.TIERS is now the FLOOR, not the last word. A box can redefine a
# tier - or add one - from its defaults file, because "what is a spark" is a
# fact about a machine. Everything below asks the RECIPE for the table rather
# than reading the module global, so a build sees the box it is running on.


def tier_table(recipe=None) -> Dict[str, "hardware.TierSpec"]:
    """The tier table this recipe builds against: floor + the box's overrides.

    A recipe parsed from a dict (not loaded from a path) carries no box, which
    is deliberate - parse() is pure, so it resolves against the floor alone and
    a unit test does not inherit whoever's laptop it runs on.
    """
    return _tier_table_and_warnings(recipe)[0]


def _tier_table_and_warnings(recipe=None):
    box = getattr(recipe, "resolved_defaults", None) or {}
    return hardware.merge_tiers(box.get("tiers"))


def model_sizes(recipe=None) -> Dict[str, Tuple[str, str]]:
    """size -> (safe base id, abliterated/instruct id), floor + the box.

    A box with a local mirror, or a house-preferred checkpoint for one size,
    says so once here instead of in every recipe's `base:`.
    """
    table = dict(MODEL_SIZES)
    box = getattr(recipe, "resolved_defaults", None) or {}
    over = box.get("models")
    if not isinstance(over, dict):
        return table
    for size, spec in over.items():
        if isinstance(spec, dict):
            safe, abl = table.get(size, ("", ""))
            table[size] = (str(spec.get("safe") or safe),
                           str(spec.get("abliterated") or abl))
        elif isinstance(spec, str):
            # A bare string is the kind reading: "this size is this model."
            table[size] = (spec, spec)
    return table


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
    # Whether the base is a reasoning model (thinking trace before answers).
    # Resolved once from base_kind (see is_reasoning_base); downstream prompt /
    # eval handling keys off this rather than sniffing the id again.
    reasoning: bool = False
    # Which reasoning CONVENTION this run uses ('' when nothing reasons). A key
    # into the reasoning table; the DELIMITERS themselves are stamped below.
    reasoning_type: str = ""
    # THE TAGS, RESOLVED AND CARRIED. The table is a file now, so looking the
    # style up again at eval time could read a table that was edited while the
    # build ran - a scorer splitting on different tags than the generator wrote
    # is measuring a different artifact than the one on disk. Stamping them
    # also puts them in build_fingerprint, where "I changed my reasoning table"
    # correctly counts as changing the build.
    reasoning_open: str = ""
    reasoning_close: str = ""
    reasoning_interwoven: bool = False
    # Expert names whose source carries `reasoning: true` — these get reasoning
    # traces GENERATED for them (force reasoning into a non-reasoning base)
    # instead of a scraped corpus.
    reasoning_experts: List[str] = field(default_factory=list)
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
    # Default teacher for a `reasoning: true` expert. A small R1-distill base,
    # because that is the proof a small non-reasoning base can be trained into
    # a reasoner, and it runs on the same hardware tiers.
    reasoning_teacher: str = ""
    teacher_max_memory: str = "110GiB"
    teacher_batch: int = 96
    use_vllm: bool = False
    vllm_batch: int = 512
    vllm_gpu_util: float = 0.88
    vllm_max_len: int = 4096
    vllm_quantization: Optional[str] = None
    teacher_max_new: int = 224
    # Reasoning traces need headroom for think + answer, not just a tool call.
    reasoning_teacher_max_new: int = 1024

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
    #
    # THE LEVER. The router's own step count is what decides whether a Ms.MoE
    # actually routes: measured on a 0.5B rung, 150 steps -> 1.06x enrichment,
    # 500 -> 1.16x, 1000 -> 1.23x, 2000 -> 1.34x, still climbing. Corpus
    # quality, domain contrast, expert strength and aux_loss_coef all measured
    # ~no effect. Steps = mix_total / (batch x accum) x epochs, so the defaults
    # below buy 2,000 steps. batch is 8 not 1 because the load-balancing loss
    # must see a MIXED batch to mean anything - at batch 1 every batch is one
    # domain. aux stays at Mixtral's 0.02; lowering it buys nothing and spends
    # margin against the collapse mode.
    router_mix_total: int = 16_000
    per_repo_cap: int = 20
    router_init: str = "random"
    router_init_std: float = 0.02
    seed: int = 42
    agent_mix_fraction: float = 0.15
    router_batch: int = 8
    router_accum: int = 1
    lr_router: float = 1e-4
    router_aux_loss_coef: float = 0.02
    router_epochs: float = 1.0

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

    # The name of the tools/MCP expert, or '' when there is none. Downstream
    # stages ask "is this the tools expert" via this name instead of the
    # literal 'agentcore'.
    tools_expert_name: str = ""

    # Hardware tier
    tier: str = "spark"

    # Eval / reporting
    #
    # The held-out fraction the split machinery actually uses, resolved once
    # so the gate's held-out split, the router's `.train` split and eval's own
    # re-split all derive from the same number (see held_out_fraction).
    eval_held_out_fraction: float = 0.1

    # True when build_config raised the corpus floor above the recipe's
    # explicit floor to feed the router mix. Recorded, never printed here:
    # build_config runs under --json too, where stdout belongs to the event
    # stream and nothing else may write to it. The CLI routes it to the prose
    # channel (see _cmd_build).
    floor_raised: bool = False


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


def _resolve_base(recipe, size: str) -> Tuple[str, str]:
    """Resolve (safe_base_id, abliterated_id) from recipe + size.

    Priority: explicit recipe.base > env override > the size's own default.
    The tier no longer contributes here: every tier maps to the same model
    family, and MODEL_SIZES is the real answer for a given size.
    """
    # 1. Explicit recipe base
    if recipe.base and recipe.base.strip():
        base = recipe.base.strip()
        safe, ablated = model_sizes(recipe).get(size, ("", ""))
        if "abliterated" in base or "Instruct" in base:
            base_model = base
        else:
            base_model = ablated if ablated else base
        return base_model, safe

    # 2. Env override
    env_base = os.environ.get("MSMOE_BASE_MODEL", "").strip()
    if env_base:
        safe, ablated = model_sizes(recipe).get(size, ("", ""))
        if "abliterated" in env_base or "Instruct" in env_base:
            return env_base, safe
        return ablated if ablated else env_base, safe

    # 3. The size's own default (the abliterated instruct coder, when it exists)
    #
    # THE SUBSTRING SNIFF USED TO GATE THIS: `if ablated and ("abliterated" in
    # ablated or "Instruct" in ablated)`. Every entry in the built-in table
    # passes that test, so it never did anything there - but the table is no
    # longer only ours. A box that points a size at `/mnt/models/local-0.5B`
    # is making an explicit statement, and a name-sniff quietly discarding it
    # in favour of a HuggingFace id is the tool deciding it knows better than
    # the person who configured the machine. An explicit answer beats a guess
    # about a filename.
    fallback = f"Qwen/Qwen2.5-{size}"
    safe, ablated = model_sizes(recipe).get(size, (fallback, ""))
    return (ablated or safe), safe


def _resolve_size(recipe, tier_name: str) -> str:
    """Resolve model size.  Priority: recipe.size > the tier's default."""
    recipe_size = recipe.size if recipe.size else "auto"
    if recipe_size != "auto" and recipe_size in model_sizes(recipe):
        return recipe_size
    return tier_table(recipe)[tier_name].default_size


def _auto_name(recipe, expert_names: List[str]) -> str:
    """Auto-generate a name from experts + timestamp."""
    if recipe.name and recipe.name.strip():
        return recipe.name.strip()
    parts = [n.replace("_", "-") for n in expert_names]
    ts = time.strftime("%Y%m%d", time.gmtime())
    return f"moe-{'-'.join(parts)}-{ts}"


def _resolve_llama_cpp(recipe) -> str:
    """Where llama.cpp is, asked in the order a person would ask it.

    RECIPE > ENV > THE OBVIOUS PLACES. It used to be env-only, which made the
    single most box-specific path in the whole build the one thing a recipe
    could not carry: hand your recipe to someone else and their export stage
    silently skips, or - before the argv[0] fix - dies naming your model id.

    The search exists because "I cloned llama.cpp right here" is what almost
    everyone does, and making them export a variable to say so is a papercut
    with no upside. Absence is still only a WARNING at preflight: no converter
    means no GGUF, and the HF checkpoint is a real result on its own.
    """
    rt = getattr(recipe, "runtime", None)
    asked = (getattr(rt, "llama_cpp", "") or "").strip() if rt else ""
    if asked:
        return os.path.expanduser(asked)

    env = os.environ.get("MSMOE_LLAMA_CPP", "").strip()
    if env:
        return os.path.expanduser(env)

    # Cheap, ordered, and it stops at the first one that looks real - a
    # directory holding the converter, not merely a directory called llama.cpp.
    for cand in ("llama.cpp",
                 os.path.join("..", "llama.cpp"),
                 os.path.expanduser("~/llama.cpp"),
                 "/opt/llama.cpp"):
        if os.path.isfile(os.path.join(cand, "convert_hf_to_gguf.py")):
            return cand
    return "llama.cpp"


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
    """The hardware tier this recipe will actually run at.

    Priority: recipe runtime tier > MSMOE_TIER env > the middle tier (xavier).
    Validated against hardware.TIERS, so a typo can't silently select a bogus
    tier. The old copy here defaulted to 'spark' with a comment calling it the
    "middle tier" - wrong twice - and never actually used that default, because
    the recipe dataclass already defaults to 'xavier'.
    """
    table = tier_table(recipe)
    tier_name = hardware.resolve_tier()  # the middle tier: 'xavier'
    rt = getattr(recipe, "runtime", None)
    t = getattr(rt, "hardware_tier", "") if rt is not None else ""
    if t in table:
        tier_name = t
    env = os.environ.get("MSMOE_TIER", "")
    if env in table:
        tier_name = env
    if tier_name not in table:  # the floor's middle tier was renamed away
        tier_name = sorted(table)[0]
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

    tier_name = resolve_tier(recipe)

    # Size — auto-resolve from tier or recipe
    size = _resolve_size(recipe, tier_name)

    # Base model — resolve from recipe/size
    base_model, base_safe = _resolve_base(recipe, size)

    # Name — auto from experts
    expert_names = [e.name for e in recipe.experts]
    name = _auto_name(recipe, expert_names)

    # The tools/MCP expert's name, resolved once (flag, or legacy 'agentcore').
    tools_expert_name = tools_expert_name_of(recipe)

    # THE RUN'S STYLE, resolved once. `reasoning_type` is about the BASE; a
    # `reasoning: true` expert puts think blocks in a build whose base does not
    # reason, so the tags have to exist in that case too. Falls back to plain
    # xml exactly the way the generator does - one answer, one place.
    def _resolve_run_style(rec, experts):
        styles, families = reasoning_table(rec)
        key = _reasoning.style_for_base(
            getattr(rec, "base", "") or "", getattr(rec, "base_kind", "auto"),
            styles, families)
        if not key and experts:
            key = "xml"
        return key, styles.get(key)

    # Experts whose source asks to have reasoning baked in (reasoning: true).
    reasoning_experts = [
        e.name for e in recipe.experts
        if getattr(getattr(e, "source", None), "reasoning", False)
    ]
    _run_style = _resolve_run_style(recipe, reasoning_experts)

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

    def _knob(value, default):
        """A recipe value, or the default when the recipe declined to choose.

        The sentinel is `-1` (or anything below zero) rather than None because
        these come out of a dataclass with typed defaults, so a recipe that
        omits the block must be indistinguishable from one that never heard
        of it.
        """
        try:
            if value is None or value < 0:
                return default
            return type(default)(value)
        except (TypeError, ValueError):
            return default

    def _corpus_knob(attr: str, dry: int, prod: int) -> int:
        asked = getattr(_corpus, attr, -1) if _corpus is not None else -1
        if asked is not None and asked >= 0:
            return int(asked)
        return dry if dryrun else prod

    num_code_samples = _corpus_knob("max_samples", 10_000, 100_000)
    per_repo_cap = _corpus_knob("per_repo_cap", 20, 20)

    # THE COLLECTOR NOW KNOWS WHAT THE ROUTER WILL ASK FOR. See router_doc_need
    # at the top of this module for why these two numbers had to stop being
    # independent. The floor is the LARGER of what the recipe asked for and
    # what the mix needs, so nobody's explicit setting gets quietly lowered.
    _mix_total = _corpus_knob("router_mix_total", 4_000, 16_000)
    _agent_frac = _knob(recipe.router.agent_mix_fraction, 0.15)
    _router_docs = router_doc_need(recipe, _mix_total, _agent_frac)
    _asked_floor = _corpus_knob("min_samples", 500, 2_000)
    min_samples = max(_asked_floor, _router_docs)
    # NOT PRINTED HERE. build_config runs under --json too, where stdout
    # belongs to the event stream and nothing else may write to it. Record the
    # fact on the config object and let the CLI route it to the prose channel
    # (see _cmd_build), which is stderr when --json is on.
    floor_raised = bool(_router_docs) and _router_docs > _asked_floor

    # Hardware-appropriate defaults, read from hardware.py rather than a copy.
    tier_spec = tier_table(recipe)[tier_name]

    # LoRA params — recipe first, then env, then the hardware tier.
    #
    # THE OLD LINE ASSIGNED A STEP COUNT TO A RANK:
    #     lora_r = int(lora_r_env) if lora_r_env else recipe.budget.target_steps
    # then overwrote it from the tier whenever the env var was unset, so it
    # was dead in both paths and read as though target_steps drove the rank.
    # A knob that appears to do something and does not is worse than a missing
    # one - the missing one at least makes you ask.
    lora_r_env = os.environ.get("MSMOE_LORA_R", "").strip()
    if recipe.budget.lora_r and recipe.budget.lora_r > 0:
        lora_r = int(recipe.budget.lora_r)
    elif lora_r_env:
        lora_r = int(lora_r_env)
    else:
        lora_r = tier_spec.default_lora_r
    lora_r = max(1, min(lora_r, 256))
    lora_alpha = _knob(recipe.budget.lora_alpha, 32)
    lora_dropout = _knob(recipe.budget.lora_dropout, 0.0)

    # Quantization hint from tier
    quant = tier_spec.default_quant
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
    llama_cpp_dir = _resolve_llama_cpp(recipe)

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

    # Resolve the held-out fraction once, so the gate's held-out split and the
    # router's `.train` split derive from the same number as eval's.
    eval_held_out_fraction = held_out_fraction(recipe)

    return PipelineConfig(
        name=name,
        size=size,
        base=base_model,
        base_safe=base_safe,
        reasoning=is_reasoning_base(recipe),
        reasoning_type=_run_style[0],
        reasoning_open=_run_style[1].open if _run_style[1] else "",
        reasoning_close=_run_style[1].close if _run_style[1] else "",
        reasoning_interwoven=bool(_run_style[1].interwoven if _run_style[1] else False),
        reasoning_experts=reasoning_experts,
        dryrun=dryrun,
        force=force,
        data_root=roots["data"],
        output_root=roots["output"],
        # Corpus
        num_code_samples=num_code_samples,
        collect_token_target=collect_token_target,
        chars_per_token_est=3.2,
        min_samples_per_expert=min_samples,
        max_shards=_corpus_knob("max_shards", 80, 80),
        shard_cache="shard_cache",
        # Agent data
        num_agent_samples=num_agent_samples,
        teacher_model=(
            "Qwen/Qwen2.5-7B-Instruct" if dryrun
            else "Qwen/Qwen2.5-32B-Instruct"
        ),
        reasoning_teacher=(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" if dryrun
            else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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
        per_repo_cap=per_repo_cap,
        router_mix_total=_mix_total,
        agent_mix_fraction=_knob(recipe.router.agent_mix_fraction, 0.15),
        # RECIPE FIRST, DEFAULT SECOND. `-1` means "you pick", which is how a
        # recipe that says nothing about the router keeps the old behaviour
        # exactly. See recipe.Router for why these stopped being hardcoded.
        router_batch=_knob(recipe.router.batch, 8),
        router_accum=_knob(recipe.router.accum, 1),
        router_epochs=_knob(recipe.router.epochs, 1.0),
        lr_router=_knob(recipe.router.lr, 1e-4),
        router_aux_loss_coef=_knob(recipe.router.aux_loss_coef, 0.02),
        # MoE
        experts_per_tok=recipe.moe.experts_per_tok,
        norm_topk_prob=recipe.moe.norm_topk_prob,
        router_init=recipe.moe.router_init,
        router_init_std=recipe.moe.router_init_std,
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
        tools_expert_name=tools_expert_name,
        # Eval / reporting
        eval_held_out_fraction=eval_held_out_fraction,
        floor_raised=floor_raised,
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
