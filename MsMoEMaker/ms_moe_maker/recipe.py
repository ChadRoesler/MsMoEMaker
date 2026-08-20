"""Recipe dataclasses — expert definition, budget, MoE config, and parsing.

A recipe is a YAML or JSON document that describes one build:
  - Which base model to start from
  - Which experts to specialise
  - How many steps, what budget
  - MoE routing architecture

The recipe is the contract between the human and the pipeline:
every field is either honoured, or the build refuses to proceed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from . import corpus


SCHEMA_VERSION = 1


@dataclass
class Source:
    """Where one expert's training text comes from.

    THE KINDS ARE A REGISTRY, NOT A LIST — see corpus.py.
    Built-ins:

      hf     — any HuggingFace dataset (repo + text_field).
      gh     — files out of a public GitHub repo (repo + glob).
      stack  — scan the code corpus for a language name.
      synth  — generate with a teacher model + rejection sampling.
      local  — text already on this box.

    Fields are the union across kinds. Each kind declares which it needs;
    the rest are silently ignored by kinds that don't care.
    """
    kind: str
    # kind=hf
    repo: Optional[str] = None
    split: str = "train"
    text_field: str = "code"
    # kind=stack
    language: Optional[str] = None
    max_shards: int = 80
    # kind=synth
    teacher: Optional[str] = None
    generator: Optional[str] = None
    examples: int = 15_000
    # kind=gh
    ref: Optional[str] = None       # branch or tag; default branch if unset
    subdir: Optional[str] = None    # narrow to one directory in the repo
    # kind=local
    path: Optional[str] = None
    glob: str = "**/*.txt"


@dataclass
class Expert:
    """One specialist."""
    name: str
    source: Optional[Source] = None
    tokens: Optional[int] = None  # per-expert budget override


@dataclass
class Budget:
    """Training budget per expert, expressed in steps."""
    target_steps: int = 1200
    max_seq_length: int = 2048
    per_device_batch: int = 4
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    warmup_floor: int = 10
    collect_headroom: float = 1.5
    doc_ceiling: int = 2000


@dataclass
class MoE:
    """MoE routing architecture."""
    experts_per_tok: int = 2
    norm_topk_prob: bool = True
    # zero | random. Zero makes the untrained MoE reproduce one expert exactly
    # and lets verify_stitch assert bit-equality; it is also a perfectly
    # symmetric starting point that a top-k router can only leave via the
    # load-balancing loss. Three router trainings on one zero-init skeleton all
    # collapsed onto a single expert, with a different winner each time.
    # `random` seeds small noise, as Switch and Mixtral do.
    router_init: str = "zero"
    router_init_std: float = 0.02
    shared_expert_width: int = 1
    shared_expert_gate_fill: float = 0.02
    # ANNOTATED, AND THAT IS THE FIX. This was `dense_layers = "auto"` with no
    # type annotation, which in a dataclass makes it a plain CLASS ATTRIBUTE
    # rather than a field. Three consequences, all silent:
    #
    #   * dataclasses.fields(MoE) did not contain it, so _build refused it with
    #     "moe.dense_layers is not a known key - IGNORED" - on the tool's own
    #     dnd template, which sets it.
    #   * a user's `moe: {dense_layers: [0,1,2]}` was therefore DISCARDED, and
    #     levers.translate read the class default forever, so the env lever it
    #     sets could never differ from "auto".
    #   * asdict() skipped it, so it was absent from recipe_id() - two recipes
    #     differing only in dense_layers hashed identically, which is the kind
    #     of thing that makes a resume pick up someone else's run.
    #
    # "auto" or an explicit list of layer indices to leave dense.
    dense_layers: Any = "auto"


@dataclass
class Gates:
    """Eval gate config."""
    base_evals: str = "auto"
    main_evals: str = "auto"
    # The pre-stitch expert check. `auto` runs weight divergence AND the
    # cross-domain loss matrix; `cheap` runs only the weight comparisons (CPU,
    # seconds); `skip` runs neither.
    #
    # Three settings rather than a bool because the two halves have wildly
    # different costs. The weight side is free at any model size. The loss side
    # loads every specialist and scores it on every domain - trivial at 0.5B,
    # a real bill at 32B - and it is also the ONLY one that can answer whether
    # the router has a gradient at all. Someone on a big run should be able to
    # keep the free half without buying the expensive one.
    experts: str = "auto"


@dataclass
class Runtime:
    """Runtime flags (precision, GPU config, hardware tier)."""
    precision: str = "float16"
    load_in_4bit: bool = False
    direct_load: bool = False
    alloc_conf: str = ""
    hardware_tier: str = "xavier"  # nano | xavier | spark
    # WHERE llama.cpp LIVES. Previously env-only (MSMOE_LLAMA_CPP), which
    # makes the one path most likely to differ per box the one thing a recipe
    # could not carry - so a recipe you hand someone else silently exports
    # nothing on their machine. Empty means: look in the usual places.
    llama_cpp: str = ""


@dataclass
class Corpus:
    """How much text to gather per expert.

    THE KNOBS THAT WERE NOT THERE. These three were hardcoded in config, with
    the ONLY lever being --dryrun, which also relabels the run as a structural
    test and moves it to a different output directory. So there was no way to
    ask for what a first end-to-end run actually wants: a REAL build, all
    stages, real artifacts, just small enough to watch it finish.

    That is the opposite of the rule the rest of this tool follows - accept a
    minimum, fill sensible defaults, and let anyone who wants to twiddle knobs
    twiddle them. The defaults below are exactly the previous hardcoded values,
    so nothing changes for a recipe that does not mention them.

    `-1` on any field means "use the default for this run", which is how a
    dryrun still gets its smaller floor without the recipe having to know.
    """
    min_samples: int = -1        # floor per expert before the stage fails
    max_samples: int = -1        # cap per expert
    router_mix_total: int = -1   # rows in the router's stratified mix
    # Max files taken from ONE repository, per language. The token quota can
    # otherwise be satisfied entirely from a single large codebase - measured
    # at 78% of a C# corpus from one enterprise application, which trained a
    # house-style expert that passed every downstream check. Lower is more
    # diverse and needs more shards.
    per_repo_cap: int = -1


@dataclass
class Router:
    """How the router gate is trained. THE KNOB THAT MATTERED WAS NOT THERE.

    Same omission as Corpus above, and the docstring three classes up already
    states the rule this broke: accept a minimum, fill sensible defaults, let
    anyone who wants to twiddle knobs twiddle them. Every one of these was
    hardcoded in config.build_config.

    It surfaced on the first real 0.5B run. The stitch zeroes the gate by
    construction, so the router starts at exactly uniform; the run then trained
    it for one epoch over 800 rows at batch 1 x accum 8 - a hundred optimizer
    steps - at 1e-4. It came out at 1.02x enrichment, which is a gate that has
    barely left its initialisation. The single most useful experiment at that
    point is "same stitch, more router training", and there was no way to ask
    for it from a recipe.

    `-1` means "use the default", so a recipe that does not mention this block
    behaves exactly as before.

    Note `corpus.router_mix_total` is the OTHER half of this - it sets how many
    rows the mix has, and lives with the corpus knobs because that is what it
    is. Steps = mix_total / (batch * accum) * epochs.
    """
    lr: float = -1.0             # learning rate for the gate
    batch: int = -1              # per-device batch
    accum: int = -1              # gradient accumulation
    epochs: float = -1.0         # passes over the router mix
    aux_loss_coef: float = -1.0  # load-balancing loss weight


@dataclass
class EvalSpec:
    """The `eval:` block. We provide the floor; this is the door out of it.

    Documented in the README and read by eval.run_eval since the beginning -
    but there was no field here and `eval` was not in _KNOWN_TOP, so a user who
    wrote exactly what the README told them to got "eval is not a known
    top-level key - IGNORED" and their script never ran. The consumer existed;
    the schema did not.

    `script` replaces our eval entirely and is called as

        <script> --data-root R --output-root O --held-out F --num-samples N

    which is a contract a stranger can implement in any language they like.
    """
    script: str = ""
    mode: str = "all"              # routing | quality | all
    held_out_fraction: float = 0.1
    num_samples: int = 20
    dead_threshold: float = 1.2    # minimum router enrichment before "dead"


@dataclass
class SmokeSpec:
    """The `smoke:` block. Does the exported artifact generate at all?"""
    script: str = ""
    tokens: int = 48
    timeout: int = 300
    prompt: str = "Write a function that works."


@dataclass
class Roots:
    """Output directory templates."""
    data: str = "{size}/corpus"
    output: str = "{size}/train"


@dataclass
class Recipe:
    """Complete recipe.  name/base are optional — auto-filled from
    template/tier when not provided."""
    name: str = ""
    base: str = ""
    experts: List[Expert] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    size: str = "auto"
    budget: Budget = field(default_factory=Budget)
    moe: MoE = field(default_factory=MoE)
    gates: Gates = field(default_factory=Gates)
    runtime: Runtime = field(default_factory=Runtime)
    roots: Roots = field(default_factory=Roots)
    corpus: Corpus = field(default_factory=Corpus)
    router: Router = field(default_factory=Router)
    eval: EvalSpec = field(default_factory=EvalSpec)
    smoke: SmokeSpec = field(default_factory=SmokeSpec)
    template: str = ""  # optional: "code" | "dnd" | "math" | "culinary"

    @property
    def tokens_per_step(self) -> int:
        b = self.budget
        return b.max_seq_length * b.per_device_batch * b.grad_accum

    @property
    def tokens_per_expert(self) -> int:
        return self.budget.target_steps * self.tokens_per_step

    @property
    def warmup_steps(self) -> int:
        return max(self.budget.warmup_floor,
                   round(self.budget.warmup_ratio * self.budget.target_steps))

    @property
    def collect_tokens(self) -> int:
        return int(self.tokens_per_expert * self.budget.collect_headroom)

    def recipe_id(self) -> str:
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("runtime", "name")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── parsing ────────────────────────────────────────────────────────────────────

_KNOWN_TOP = {
    "name", "base", "experts", "schema_version", "size", "budget",
    "moe", "gates", "runtime", "roots", "corpus", "router", "eval", "smoke",
    "template",
}


def _build(cls, data: Dict[str, Any], path: str,
           warnings: List[str]):
    """Construct a dataclass, reporting unknown keys."""
    fields = {f for f in cls.__dataclass_fields__}
    unknown = sorted(set(data) - fields)
    for u in unknown:
        warnings.append(f"{path}.{u} is not a known key - IGNORED "
                        f"(known: {', '.join(sorted(fields))})")
    return cls(**{k: v for k, v in data.items() if k in fields})


def _apply_template(data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a named template into the recipe dict.

    Template fields fill in wherever the recipe dict is empty / missing.
    The recipe's own values always win.
    """
    from .template import apply_template

    tpl_name = data.get("template")
    if not tpl_name:
        return data

    try:
        return apply_template(data, tpl_name)
    except ValueError as exc:
        # Unknown template — leave recipe as-is but warn
        data.setdefault("_template_warnings", []).append(str(exc))
        return data


def parse(data: Dict[str, Any]) -> Tuple[Recipe, List[str]]:
    warnings: List[str] = []

    if not isinstance(data, dict):
        raise ValueError("a recipe must be a mapping at the top level")

    # Apply template first (fills name, base, experts, budget, etc.)
    data = _apply_template(data)

    # Keys starting with "_" are internal plumbing that apply_template writes
    # for config to read (_base_hint). Warning about them tells the user their
    # brand-new templated recipe is malformed, which it is not, and which they
    # cannot act on.
    for u in sorted(k for k in set(data) - _KNOWN_TOP
                    if not str(k).startswith("_")):
        warnings.append(f"{u} is not a known top-level key - IGNORED")

    raw_experts = data.get("experts") or []
    if not isinstance(raw_experts, list):
        raise ValueError("experts must be a list")
    experts: List[Expert] = []
    for i, e in enumerate(raw_experts):
        if not isinstance(e, dict):
            raise ValueError(f"experts[{i}] must be a mapping")
        src = e.get("source")
        if not isinstance(src, dict):
            raise ValueError(f"experts[{i}].source must be a mapping with a "
                             f"'kind' of hf | stack | synth | local")
        expert = _build(Expert, {k: v for k, v in e.items() if k != "source"},
                        f"experts[{i}]", warnings)
        expert.source = _build(Source, src, f"experts[{i}].source", warnings)
        experts.append(expert)

    rec = Recipe(
        name=data.get("name") or "",
        base=data.get("base") or "",
        experts=experts,
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        size=data.get("size", "auto"),
        budget=_build(Budget, data.get("budget") or {}, "budget", warnings),
        moe=_build(MoE, data.get("moe") or {}, "moe", warnings),
        gates=_build(Gates, data.get("gates") or {}, "gates", warnings),
        runtime=_build(Runtime, data.get("runtime") or {}, "runtime", warnings),
        roots=_build(Roots, data.get("roots") or {}, "roots", warnings),
        corpus=_build(Corpus, data.get("corpus") or {}, "corpus", warnings),
        router=_build(Router, data.get("router") or {}, "router", warnings),
        eval=_build(EvalSpec, data.get("eval") or {}, "eval", warnings),
        smoke=_build(SmokeSpec, data.get("smoke") or {}, "smoke", warnings),
        template=data.get("template", ""),
    )
    # Wire template tier → runtime hardware_tier
    t = data.get("default_tier") or data.get("tier")
    if t and t in ("nano", "xavier", "spark"):
        rec.runtime.hardware_tier = t
    return rec, warnings


def load(path: str) -> Tuple[Recipe, List[str]]:
    text = open(path, encoding="utf-8").read()
    if path.endswith((".json",)):
        return parse(json.loads(text))
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "reading a .yaml recipe needs pyyaml (pip install pyyaml), or "
            "write the recipe as .json - the schema is identical either way.")
    return parse(yaml.safe_load(text) or {})


# ── validation ────────────────────────────────────────────────────────────────

def validate(rec: Recipe) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors mean do not build."""
    errs: List[str] = []
    warns: List[str] = []

    if rec.schema_version != SCHEMA_VERSION:
        warns.append(f"schema_version {rec.schema_version} != {SCHEMA_VERSION} "
                     f"- fields may be read differently than you intend")

    # -- base model architecture --------------------------------------------
    #
    # The single most expensive failure this tool can have is a base model it
    # can fine-tune but cannot stitch. Catch it here, where it costs two
    # seconds on a laptop, not at stage 4 after every specialist has trained.
    if rec.base:
        from .config import SUPPORTED_BASE_HINTS, SUPPORTED_MOE_ARCHS
        low = rec.base.lower()
        if not any(hint in low for hint in SUPPORTED_BASE_HINTS):
            errs.append(
                f"base {rec.base!r} is not a supported MoE architecture. "
                f"The specialists would train fine and the build would then "
                f"fail at the stitch stage, after every expert had trained. "
                f"Supported today: "
                f"{', '.join(sorted(SUPPORTED_MOE_ARCHS.values()))}. "
                f"Leave `base` empty to get a supported default for your size.")

    # -- routing measurability ----------------------------------------------
    #
    # top-k == expert count is legal and runs fine - it is just a dense
    # ensemble of every expert - but it makes the ROUTER a no-op, and it makes
    # the dead-expert measurement impossible rather than merely difficult:
    # every expert is selected on every token, so the shares are 1/E by
    # arithmetic and read exactly like a router that ignores its input.
    #
    # A warning, not an error, because someone may genuinely want the dense
    # ensemble. But they should know they have given up the thing this project
    # is for.
    if rec.experts and rec.moe.experts_per_tok >= len(rec.experts):
        warns.append(
            f"moe.experts_per_tok={rec.moe.experts_per_tok} equals the expert "
            f"count ({len(rec.experts)}), so every expert is selected on every "
            f"token and the router cannot discriminate. `eval --mode routing` "
            f"will report UNMEASURABLE. Use experts_per_tok=1 AND "
            f"norm_topk_prob=false for a 2-expert MoE, or add a third expert "
            f"and keep top-2.")

    # -- experts ------------------------------------------------------------
    if not rec.experts:
        errs.append("experts must list at least one expert")
    if len(rec.experts) < 2:
        errs.append("at least 2 experts are needed; a 1-expert MoE is a dense "
                    "model with extra steps")
    names = [e.name for e in rec.experts]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        errs.append(f"duplicate expert names: {sorted(dupes)} - expert index "
                    f"follows this list and is stamped into config.json; two "
                    f"with one name makes every downstream analysis ambiguous")
    for e in rec.experts:
        if not e.name or not e.name.replace("_", "").isalnum():
            errs.append(f"expert name {e.name!r} must be alphanumeric/underscore "
                        f"- it becomes a directory and a filename")
        if e.name != e.name.lower():
            warns.append(f"expert name {e.name!r} is not lowercase; the "
                         f"pipeline lowercases paths and 'C#' becomes 'csharp'")
        s = e.source
        if s:
            kind_errs, kind_warns = corpus.check(s.kind, s)
            errs.extend(f"{e.name}: {m}" for m in kind_errs)
            warns.extend(f"{e.name}: {m}" for m in kind_warns)

            if s.kind == "stack" and s.language:
                warns.append(f"{e.name}: source.kind=stack language {s.language!r} "
                             f"must be spelled EXACTLY as the corpus spells it")
            if s.kind == "synth":
                if s.teacher:
                    small = any(t in s.teacher for t in
                                ("0.5B", "1.5B", "3B", "0.6B", "1B", "2B"))
                    if small:
                        warns.append(
                            f"{e.name}: teacher {s.teacher!r} looks small. "
                            f"A 0.5B teacher is too weak to emit schema-valid "
                            f"tool calls; 7B is the smallest that reliably clears it.")

    # -- budget -------------------------------------------------------------
    b = rec.budget
    if b.target_steps <= 0:
        errs.append("budget.target_steps must be > 0")
    if not 0 <= b.warmup_ratio <= 0.5:
        errs.append(f"budget.warmup_ratio {b.warmup_ratio} is outside 0..0.5")
    if b.target_steps and rec.warmup_steps / b.target_steps > 0.2:
        warns.append(
            f"warmup is {100*rec.warmup_steps/b.target_steps:.0f}% of the "
            f"schedule ({rec.warmup_steps}/{b.target_steps}). Above ~20% the "
            f"loss curve flattens for schedule reasons and reads as "
            f"data-starvation. Raise target_steps or lower warmup_ratio.")
    if b.per_device_batch * b.grad_accum != 8:
        warns.append(
            f"effective batch is {b.per_device_batch * b.grad_accum}, not 8. "
            f"Fine - but the PRODUCT is the thing that must stay constant if "
            f"you are comparing against earlier rungs.")
    if b.doc_ceiling < 1000:
        warns.append(f"budget.doc_ceiling {b.doc_ceiling} is low; it is a "
                     f"CEILING, not a target - the token budget is what "
                     f"decides how much each expert actually gets")

    # -- moe ----------------------------------------------------------------
    m = rec.moe
    if not 1 <= m.experts_per_tok <= max(len(rec.experts), 1):
        errs.append(f"moe.experts_per_tok {m.experts_per_tok} must be between "
                    f"1 and the number of experts ({len(rec.experts)})")
    if isinstance(m.dense_layers, str):
        if m.dense_layers != "auto":
            errs.append(f"moe.dense_layers must be 'auto' or a list of ints, "
                        f"got {m.dense_layers!r}")
    elif isinstance(m.dense_layers, list):
        if any(not isinstance(x, int) or x < 0 for x in m.dense_layers):
            errs.append("moe.dense_layers must be non-negative ints")
        if len(set(m.dense_layers)) != len(m.dense_layers):
            errs.append("moe.dense_layers has duplicates")
        if m.dense_layers:
            warns.append(
                f"{len(m.dense_layers)} dense layers set explicitly. This is "
                f"architecture and is fixed at STITCH time - re-running with a "
                f"different value does NOTHING once a skeleton exists.")
    else:
        errs.append("moe.dense_layers must be 'auto' or a list")
    if m.shared_expert_width == 0:
        errs.append("moe.shared_expert_width=0 produces a zero-element GGUF "
                    "tensor whose element count overflows llama.cpp's loader")
    # TOP-1 WITH norm_topk_prob=TRUE SEVERS THE ROUTER FROM THE LOSS.
    #
    # This is not a tuning preference, it is arithmetic, and it cost a full
    # diagnostic arc on the first real 0.5B build. Qwen2MoeSparseMoeBlock does
    #
    #     routing_weights, selected = topk(routing_weights, top_k, dim=-1)
    #     if norm_topk_prob:
    #         routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    #
    # and at top_k=1 that sum IS routing_weights. It divides by itself. Every
    # token's weight becomes the constant 1.0, d(w/w)/dw = 0, and the gate
    # receives NO GRADIENT FROM THE LM LOSS AT ALL. The only signal left is the
    # load-balancing aux loss, which pushes toward uniform - so training the
    # router harder makes it MORE uniform, which is exactly what we measured:
    # enrichment went 1.02x -> 1.00x after tripling the steps, on a model whose
    # experts had a measured 0.49-nat cross-domain loss gap sitting unused.
    #
    # At top-1 the correct formulation is Switch Transformer's: scale the
    # expert output by the gate probability p, which is what carries the
    # gradient. That is norm_topk_prob=false.
    #
    # An ERROR, not a warning, because there is no configuration in which this
    # combination does what the user wants. It does not produce a worse router;
    # it produces a router that cannot train.
    if m.experts_per_tok == 1 and m.norm_topk_prob:
        errs.append(
            "moe.experts_per_tok=1 with moe.norm_topk_prob=true severs the "
            "router from the loss: normalising a single top-k weight divides "
            "it by itself, so every routing weight is the constant 1.0 and the "
            "gate gets zero gradient from the LM loss. The router can only "
            "move toward uniform, driven by the aux loss. Set "
            "moe.norm_topk_prob=false (this is Switch Transformer's top-1 "
            "formulation), or use experts_per_tok>=2 with 3+ experts.")

    # The scaling warning applies to top-k >= 2, where normalisation is a real
    # choice. At top-1 it is the ONLY correct setting, so warning about it
    # there pushed users straight into the error above.
    if m.router_init not in ("zero", "random"):
        errs.append(f"moe.router_init must be zero | random, got "
                    f"{m.router_init!r}")
    if m.router_init == "random" and not (0 < m.router_init_std <= 0.5):
        errs.append(f"moe.router_init_std {m.router_init_std} must be in "
                    f"(0, 0.5] - larger and the untrained MoE routes on noise")

    if not m.norm_topk_prob and m.experts_per_tok >= 2:
        warns.append("moe.norm_topk_prob=false scales the stitched model to "
                     "~0.40x at init, so the router trains on the wrong problem")

    # -- gates --------------------------------------------------------------
    for g, v in (("base_evals", rec.gates.base_evals),
                 ("main_evals", rec.gates.main_evals)):
        if v not in ("auto", "manual", "skip"):
            errs.append(f"gates.{g} must be auto | manual | skip, got {v!r}")
    if rec.gates.experts not in ("auto", "cheap", "skip"):
        errs.append(f"gates.experts must be auto | cheap | skip, got "
                    f"{rec.gates.experts!r}")
    if rec.gates.main_evals == "auto":
        warns.append("gates.main_evals=auto removes the third switch. The "
                     "expensive suite will run unattended on whatever the "
                     "build produced, including a NaN'd model that generates "
                     "at full speed and emits one token forever.")

    # -- roots --------------------------------------------------------------
    if rec.roots.data == rec.roots.output:
        errs.append("roots.data and roots.output must differ")
    if "{size}" not in rec.roots.output:
        warns.append("roots.output has no {size} - every rung of the ladder "
                     "will write to the same directory and _done() will skip "
                     "training on the wrong-sized specialists it finds there")

    return errs, warns


def resolve(rec: Recipe) -> Dict[str, Any]:
    """The EFFECTIVE build — derived values made explicit."""
    return {
        "recipe_id": rec.recipe_id(),
        "name": rec.name,
        "base": rec.base,
        "size": rec.size,
        "experts": [e.name for e in rec.experts],
        "sources": {e.name: e.source.kind for e in rec.experts if e.source},
        "target_steps": rec.budget.target_steps,
        "tokens_per_step": rec.tokens_per_step,
        "tokens_per_expert": rec.tokens_per_expert,
        "collect_tokens": rec.collect_tokens,
        "warmup_steps": rec.warmup_steps,
        "effective_batch": rec.budget.per_device_batch * rec.budget.grad_accum,
        "experts_per_tok": rec.moe.experts_per_tok,
        "dense_layers": rec.moe.dense_layers,
        "gates": asdict(rec.gates),
        "data_root": rec.roots.data,
        "output_root": rec.roots.output.replace("{size}", rec.size),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

DESCRIBE = {
    "name": "ms-moe-maker-recipe",
    "schema_version": SCHEMA_VERSION,
    "kinds": corpus.describe(),
    "gates": ["auto", "manual", "skip"],
    "templates": ["code", "dnd", "math", "culinary"],
    "description": "Recipe schema for Ms.MoE. Validate before you burn a GPU.",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="msmoe_recipe",
        description="Ms.MoE recipe — validate, resolve, and describe.")
    ap.add_argument("recipe", nargs="?", help="path to recipe .yaml or .json")
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--resolve", action="store_true",
                    help="print the EFFECTIVE build as JSON")
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events instead of prose")
    ap.add_argument("--describe", action="store_true",
                    help="one line of JSON, exit 0, zero side effects")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors")
    a = ap.parse_args()

    if a.describe:
        print(json.dumps(DESCRIBE))
        return 0
    if not a.recipe:
        ap.error("a recipe path is required (or use --describe)")

    def emit(kind: str, **kw) -> None:
        if a.json:
            print(json.dumps({"event": kind, **kw}), flush=True)

    try:
        rec, parse_warns = load(a.recipe)
    except Exception as exc:  # noqa: BLE001 - the message IS the product
        emit("error", stage="parse", message=str(exc))
        if not a.json:
            print(f"FAILED to parse {a.recipe}: {exc}", file=sys.stderr)
        return 2

    errs, warns = validate(rec)
    warns = parse_warns + warns
    eff = resolve(rec)

    if a.json:
        emit("parsed", recipe_id=eff["recipe_id"], experts=eff["experts"])
        for w in warns:
            emit("warning", message=w)
        for e in errs:
            emit("error", stage="validate", message=e)
        emit("resolved", **eff)
        emit("done", ok=not errs and not (a.strict and warns))
    else:
        print(f"\nMs.MoE recipe  {rec.name or 'unnamed'}  [{eff['recipe_id']}]")
        print(f"   base       {rec.base or '(auto)' }")
        print(f"   size       {rec.size}")
        print(f"   experts    {eff['experts']}")
        print(f"   targets    {eff['target_steps']} steps, "
              f"{eff['tokens_per_expert']} tokens/expert")
        print(f"   batch      {eff['effective_batch']} "
              f"(per_device x accum = {rec.budget.per_device_batch} x "
              f"{rec.budget.grad_accum})")
        print(f"   MoE        experts_per_tok={eff['experts_per_tok']}")
        print(f"   roots      data={rec.roots.data} output={rec.roots.output}")
        if warns:
            print(f"\n   WARNINGS ({len(warns)}):")
            for w in warns:
                print(f"     · {w}")
        if errs:
            print(f"\n   ERRORS ({len(errs)}):")
            for e in errs:
                print(f"     ✗ {e}")
        if eff.get("_template_warnings"):
            for tw in eff["_template_warnings"]:
                print(f"\n   NOTE  {tw}")

    return 0 if not errs and not (a.strict and warns) else 1


if __name__ == "__main__":
    raise SystemExit(main())
