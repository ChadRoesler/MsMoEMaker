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

from ..data import corpus


SCHEMA_VERSION = 1

# The tools (MCP) expert. `tools_expert: true` injects one with these defaults;
# a mapping overrides them. The NAME is the one handle downstream uses to ask
# "is this the tools expert" rather than the literal "agentcore" — which is how
# a differently-named tools expert used to silently become a code expert in the
# router's formatting and quota logic.
# These two now come from `defaults.FLOOR` / `defaults.yaml` so a box can be
# configured once for someone else instead of every recipe carrying the values.
# They are re-exported here because the legacy "an expert literally named
# agentcore IS the tools expert" convention has to keep working.
from . import defaults as _defaults

DEFAULT_TOOLS_EXPERT_NAME = _defaults.FLOOR["tools_expert"]["name"]
DEFAULT_TOOLS_EXPERT_TEACHER = _defaults.FLOOR["tools_expert"]["teacher"]


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
    # Per-source cap on shards the stack scan may pull for THIS expert. The
    # run-wide ceiling is corpus.max_shards; this narrows (or widens, below the
    # ceiling) the window for one source.
    max_shards: int = 80
    # kind=synth
    teacher: Optional[str] = None
    # How many traces to GENERATE for this expert. Overrides corpus.synth_samples
    # (and its resolved default) for this one source. Unset (-1) means "the run
    # default".
    examples: int = -1
    # Question templates for synth: a path to a YAML of prompts, or a bare
    # name (code | dnd | math | culinary | generic) resolving to the packaged
    # *_templates.yaml. Empty = generic_templates.yaml. The prompts are DOMAIN
    # questions; the reasoning instruction is the system prompt, injected by
    # `reasoning: true`.
    templates: str = ""
    # Force reasoning into this expert: generate reasoning traces
    # (<think>…answer) with a reasoning teacher instead of scraping a corpus.
    # The R1-distill recipe, applied to ONE specialist. Works on any base.
    reasoning: bool = False
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
    # Max new tokens the REASONING teacher may emit per trace. -1 = default
    # (1024): reasoning traces need headroom for think + answer, unlike a tool
    # call. Raise it if the teacher's answers are truncated mid-script.
    reasoning_teacher_max_new: int = -1
    # Max new tokens the generic synth/domain teacher may emit. -1 = default
    # (512): plain domain text (no think block) needs less headroom than a
    # reasoning trace, but more than a tool call.
    teacher_max_new: int = -1

    # THE ADAPTER'S SHAPE, which was reachable only from an env var.
    #
    # -1 means "use the hardware tier's default" (nano 32, xavier 64, spark
    # 128), which is what every run so far has silently been getting. The
    # config code that looked like it derived rank from the recipe read
    #     lora_r = ... if env else recipe.budget.target_steps
    # - assigning a STEP COUNT to a LoRA rank, then overwriting it with the
    # tier value two lines later. Harmless in effect and actively misleading
    # to read: setting target_steps looks like it should move the rank, and it
    # never did.
    #
    # Measured context for anyone turning these: at 0.5B the rank was already
    # 128 against the Lab's 64 while each expert saw 1.23M tokens - one
    # SIXTEENTH of the proven rung's 19.7M. A large adapter over a small
    # corpus is what a 0.05-nat expert looks like, so reach for target_steps
    # and max_samples before reaching for rank.
    lora_r: int = -1
    lora_alpha: int = -1
    lora_dropout: float = -1.0


@dataclass
class MoE:
    """MoE routing architecture."""
    experts_per_tok: int = 2
    norm_topk_prob: bool = True
    # zero | random. `random` is the default: it seeds small noise, as Switch
    # and Mixtral do, because a perfectly symmetric zero-init gate collapses
    # onto a single expert (three trainings on one zero-init skeleton did, with
    # a different winner each time). `zero` still exists for verify_stitch's
    # bit-equality check - "is the skeleton well-formed" - not for training.
    router_init: str = "random"
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
    # None = "derive from the hardware tier" (the nano tier quantises to 4-bit).
    # An explicit True or False wins in BOTH directions: a nano box can run one
    # build in float16, and a big box can opt into 4-bit without editing its
    # tier. Bool-or-None, because False is a real answer that must not read as
    # "you did not say".
    load_in_4bit: Optional[bool] = None
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
    # Traces to GENERATE per generated expert. Distinct from max_samples
    # because that one is a CEILING on documents collected and this is a
    # TARGET for documents written - a teacher runs until it hits this
    # number, so it is time, not disk. Unset means the run default,
    # capped by max_samples when the recipe stated one.
    #
    # NOT `agent_samples`. It governs BOTH generated corpora - the tools
    # expert's traces AND generate_reasoning_traces, which reads the same
    # number (`n = config.num_agent_samples`). Naming a knob after one of
    # the two things it controls is how `data.code` happened; this one is
    # still free to name, so it gets named for the whole job.
    synth_samples: int = -1
    router_mix_total: int = -1   # rows in the router's stratified mix
    # Max files taken from ONE repository, per language. The token quota can
    # otherwise be satisfied entirely from a single large codebase - measured
    # at 78% of a C# corpus from one enterprise application, which trained a
    # house-style expert that passed every downstream check. Lower is more
    # diverse and needs more shards.
    per_repo_cap: int = -1
    # How many shards the scan may pull before giving up. Each is ~0.57 GB.
    # This was hardcoded at 80, so a recipe setting it was accepted by the
    # parser and silently ignored - which is worse than rejecting it, because
    # the run then fails for a reason the setting was meant to prevent.
    max_shards: int = -1


@dataclass
class Router:
    """How the router gate is trained. THE KNOB THAT MATTERED WAS NOT THERE.

    Same omission as Corpus above, and the docstring three classes up already
    states the rule this broke: accept a minimum, fill sensible defaults, let
    anyone who wants to twiddle knobs twiddle them. Every one of these was
    hardcoded in config.build_config.

    The router's own step count is the lever that decides whether a Ms.MoE
    actually routes. Measured on a 0.5B rung (same stitch, same experts): 150
    steps -> 1.06x enrichment, 500 -> 1.16x, 1000 -> 1.23x, 2000 -> 1.34x, and
    the curve was steepening, not flattening. Corpus quality, domain contrast,
    expert strength and the aux coefficient all measured ~no effect. The
    defaults buy 2,000 steps: router_mix_total 16,000 over batch 8 x accum 1.
    `batch` is 8 not 1 because the load-balancing loss must see a MIXED batch
    to mean anything - at batch 1 every batch is a single domain.

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
    # Share of the router's mix drawn from GENERATED (synth) experts. The
    # remaining experts split what is left, evenly. Hardcoded at 0.15 since
    # the days when the only synth expert was one person's MCP traces - which
    # means a 3-expert build with one synth expert showed it to the router 15%
    # of the time while the other two took 42.5% each, and nobody could change
    # that from a recipe.
    agent_mix_fraction: float = -1.0


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
    """Output directory templates.

    Empty = the tool's historical defaults (msmoe_data / msmoe_run_{size}).
    Set one and `{size}` is substituted with the resolved size, e.g.
    `data: corpora/{size}`. THE DEFAULTS USED TO BE `{size}/corpus` and
    `{size}/train` here while resolve_roots hardcoded msmoe_data - a recipe
    that never mentioned roots got neither, and a recipe that set roots got
    the hardcoded answer anyway. Empty means "you did not say", which is the
    only way a real default and a silent one can coexist.
    """
    data: str = ""
    output: str = ""


@dataclass
class Abliterate:
    """The `abliterate:` block. Decensor the base before training specialists.

    Runs the vendored Heretic core (ms_moe_maker.abliterate.heretic) on the resolved base,
    then points the build at the decensored result. `abliterate: true` is the
    on-ramp; a mapping customises the knobs below.
    """
    enabled: bool = False
    n_trials: int = -1            # -1 = Heretic default (200 Optuna trials)
    seed: Optional[int] = None
    quantization: str = "none"    # none | bnb_4bit
    trial_index: Optional[int] = None     # None = first Pareto-front trial
    checkpoint_action: str = "continue"   # continue | restart
    export: str = "merge"         # merge | adapter


@dataclass
class Recipe:
    """Complete recipe.  name/base are optional — auto-filled from
    template/tier when not provided."""
    name: str = ""
    base: str = ""
    # auto | reasoning | nonreasoning. Whether the base is a reasoning model
    # (thinking-token / step-by-step trace in its output). `auto` sniffs the
    # model id; set it explicitly when the id is not a known reasoning name.
    base_kind: str = "auto"
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
    abliterate: Abliterate = field(default_factory=Abliterate)
    template: str = ""  # optional: "code" | "dnd" | "math" | "culinary"
    # bool | mapping. When truthy, a tools (MCP) expert is added to `experts`:
    # `true` uses the defaults, a mapping customises name/teacher/etc.
    tools_expert: Any = False

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
        """The identity of the RECIPE AS WRITTEN. Not of the build.

        `load()` sets `_recipe_only_id` from a parse of the raw file, before
        any defaults are laid under it - because otherwise this moved with the
        box, and a recipe you email someone would have a different id on their
        machine. Two levels, two names: `recipe_id` is what you wrote,
        `config.build_id` is what will actually be built. Conflating them
        leaves no word for either.
        """
        pinned = getattr(self, "_recipe_only_id", "")
        if pinned:
            return pinned
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("runtime", "name")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── parsing ────────────────────────────────────────────────────────────────────

_KNOWN_TOP = {
    "name", "base", "base_kind", "experts", "schema_version", "size", "budget",
    "moe", "gates", "runtime", "roots", "corpus", "router", "eval", "smoke",
    "template", "tools_expert", "abliterate",
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


def _build_abliterate(raw: Any, warnings: List[str]) -> "Abliterate":
    """`abliterate: true` or a mapping -> an Abliterate. Anything else -> off."""
    if raw is True:
        return Abliterate(enabled=True)
    if isinstance(raw, dict):
        return _build(Abliterate, {"enabled": True, **raw}, "abliterate", warnings)
    return Abliterate(enabled=False)


def _apply_template(data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a named template into the recipe dict.

    Template fields fill in wherever the recipe dict is empty / missing.
    The recipe's own values always win.
    """
    from .templates import apply_template

    tpl_name = data.get("template")
    if not tpl_name:
        return data

    try:
        return apply_template(data, tpl_name)
    except ValueError as exc:
        # Unknown template — leave recipe as-is but warn
        data.setdefault("_template_warnings", []).append(str(exc))
        return data


def parse(data: Dict[str, Any],
          defaults: Optional[Dict[str, Any]] = None) -> Tuple[Recipe, List[str]]:
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

    # -- the tools (MCP) expert --------------------------------------------
    #
    # `tools_expert: true` is the on-ramp: it INJECTS a default tools expert so
    # a recipe with two code experts becomes a three-expert MoE with a
    # tool-calling specialist, no knowledge of synth plumbing required. A
    # mapping customises it. If an expert of that name already exists it is
    # USED (and marked as the tools expert) rather than duplicated.
    # THE DEFAULTS LAYER IS WHERE `true` GETS ITS CONTENT. `tools_expert: true`
    # is the whole point of the flag - one word, and the box's own answer for
    # what a tool-calling specialist should be fills in the rest. A mapping in
    # the recipe still overrides key by key.
    tools_defaults = dict((defaults or {}).get("tools_expert") or {})
    tools_expert = data.get("tools_expert", False)
    tools_name = ""
    if tools_expert:
        spec = dict(tools_defaults)
        if isinstance(tools_expert, dict):
            spec.update({k: v for k, v in tools_expert.items()
                         if not (isinstance(v, (int, float))
                                 and not isinstance(v, bool) and v == -1)})
        tools_name = str(spec.get("name") or DEFAULT_TOOLS_EXPERT_NAME).strip()
        if not any(e.name == tools_name for e in experts):
            src = {"kind": "synth",
                   "teacher": spec.get("teacher") or DEFAULT_TOOLS_EXPERT_TEACHER}
            # `kind` is fixed to synth (a tools expert IS generated); `name` is
            # the expert's name, not a source field. Everything else passes
            # through so _build can warn on typos rather than drop them.
            src.update({k: v for k, v in spec.items() if k not in ("name", "kind")})
            experts.append(Expert(
                name=tools_name,
                source=_build(Source, src, "tools_expert", warnings)))
    elif any(e.name == DEFAULT_TOOLS_EXPERT_NAME
             and getattr(getattr(e, "source", None), "kind", "") == "synth"
             for e in experts):
        # Legacy: an expert literally named "agentcore" with a synth source was
        # the tools expert before the flag existed. Honour it so existing
        # recipes keep their meaning without requiring the flag.
        tools_name = DEFAULT_TOOLS_EXPERT_NAME

    rec = Recipe(
        name=data.get("name") or "",
        base=data.get("base") or "",
        base_kind=data.get("base_kind", "auto"),
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
        abliterate=_build_abliterate(data.get("abliterate"), warnings),
        template=data.get("template", ""),
        tools_expert=tools_expert,
    )
    # The tools expert's NAME, as resolved by the injection above. A plain
    # instance attribute rather than a dataclass field so recipe_id() (asdict)
    # is not polluted with a value derivable from `experts`.
    rec.tools_expert_name = tools_name
    # The template's base-family hint (apply_template writes it; parse exempts
    # underscore keys from typo warnings so it can travel silently). config's
    # _resolve_base turns it into the concrete checkpoint for the run's size.
    rec._base_hint = str(data.get("_base_hint") or "").strip()
    # Wire template tier → runtime hardware_tier
    t = data.get("default_tier") or data.get("tier")
    if t and t in ("nano", "xavier", "spark"):
        rec.runtime.hardware_tier = t
    return rec, warnings


def load(path: str, defaults_path: Optional[str] = None,
         include_user_defaults: bool = True
         ) -> Tuple[Recipe, List[str]]:
    """Read a recipe FILE, with the box's defaults underneath it.

    `parse` stays pure on purpose: a unit test that calls parse() must not
    inherit whatever is in the running user's ~/.msmoe/defaults.yaml, or it is
    not a unit test. Reading a path is the moment a box gets involved, so this
    is where the layers are applied.
    """
    text = open(path, encoding="utf-8").read()
    if path.endswith((".json",)):
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "reading a .yaml recipe needs pyyaml (pip install pyyaml), or "
                "write the recipe as .json - the schema is identical either "
                "way.")
        data = yaml.safe_load(text) or {}

    resolved, prov, dwarns = _defaults.resolve(
        defaults_path, include_user=include_user_defaults)
    merged = _defaults.apply_to(data, resolved)
    rec, warns = parse(merged, defaults=resolved)
    # The recipe's own id, from the recipe alone. See Recipe.recipe_id.
    try:
        _bare, _ = parse(data)
        rec._recipe_only_id = _bare.recipe_id()
    except Exception:
        pass
    # Provenance travels WITH the recipe, so --plan and validate can print
    # where every non-recipe value came from without re-resolving.
    #
    # Content-only blocks are reported only when the recipe ASKED for them. A
    # box that defines what a tools expert should be has not put a tools expert
    # in a recipe that never mentions one, and listing it as an applied default
    # would be reporting more than we know - the failure this codebase keeps
    # finding, pointed at its own output.
    # The BOX travels with the recipe from the moment it was read off a path:
    # config asks the recipe for its resolved defaults rather than re-resolving
    # (and possibly re-deciding) them halfway down the stack.
    rec.resolved_defaults = resolved
    rec.defaults_digests = _defaults.file_digests(
        defaults_path, include_user=include_user_defaults)

    # THE BOX'S OWN TABLE, CHECKED OUT LOUD. merge_tiers refuses a malformed or
    # incomplete tier rather than raising - adding hardware should not be a
    # cliff - but a refusal nobody sees is the same as no check at all.
    # THE REASONING TABLE, CHECKED OUT LOUD TOO. Same rule as the tier table:
    # it refuses malformed entries rather than raising, so the warning is the
    # only thing between a typo in a tag style and a run that scores think
    # blocks as answers.
    from . import reasoning as _rz
    _, _, _rz_warns = _rz.load()
    dwarns = list(dwarns) + _rz_warns

    from ..box import hardware as _hw
    _table, _tier_warns = _hw.merge_tiers(resolved.get("tiers"))
    dwarns = list(dwarns) + _tier_warns
    _named = getattr(getattr(rec, "runtime", None), "hardware_tier", "")
    if _named and _named not in _table:
        dwarns.append(
            f"runtime.hardware_tier {_named!r} is not a tier on this box "
            f"({', '.join(sorted(_table))}) - the middle tier will be used. "
            f"Define it under `tiers:` in a defaults file if this machine "
            f"has one.")
    rec.defaults_provenance = {
        k: v for k, v in prov.items()
        if k.split(".")[0] not in _defaults.CONTENT_ONLY or data.get(k.split(".")[0])
    }
    return rec, list(dwarns) + list(warns)


# ── validation ────────────────────────────────────────────────────────────────

def validate(rec: Recipe) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors mean do not build."""
    errs: List[str] = []
    warns: List[str] = []

    if rec.schema_version != SCHEMA_VERSION:
        warns.append(f"schema_version {rec.schema_version} != {SCHEMA_VERSION} "
                     f"- fields may be read differently than you intend")

    if rec.base_kind not in ("auto", "reasoning", "nonreasoning"):
        errs.append(f"base_kind must be auto | reasoning | nonreasoning, got "
                    f"{rec.base_kind!r}")

    # -- abliteration ---------------------------------------------------------
    if rec.abliterate.enabled:
        if rec.abliterate.quantization not in ("none", "bnb_4bit"):
            errs.append(f"abliterate.quantization must be none | bnb_4bit, got "
                        f"{rec.abliterate.quantization!r}")
        if rec.abliterate.export not in ("merge", "adapter"):
            errs.append(f"abliterate.export must be merge | adapter, got "
                        f"{rec.abliterate.export!r}")
        if rec.abliterate.checkpoint_action not in ("continue", "restart"):
            errs.append(f"abliterate.checkpoint_action must be continue | restart, "
                        f"got {rec.abliterate.checkpoint_action!r}")
        if rec.abliterate.n_trials == 0:
            errs.append("abliterate.n_trials must be -1 (default) or a positive "
                        "integer")

    # -- base model architecture --------------------------------------------
    #
    # The single most expensive failure this tool can have is a base model it
    # can fine-tune but cannot stitch. Catch it here, where it costs two
    # seconds on a laptop, not at stage 4 after every specialist has trained.
    if rec.base:
        from .pipeline import SUPPORTED_BASE_HINTS, SUPPORTED_MOE_ARCHS
        low = rec.base.lower()
        if not any(hint in low for hint in SUPPORTED_BASE_HINTS):
            errs.append(
                f"base {rec.base!r} is not a supported MoE architecture. "
                f"The specialists would train fine and the build would then "
                f"fail at the stitch stage, after every expert had trained. "
                f"Supported today: "
                f"{', '.join(sorted(SUPPORTED_MOE_ARCHS.values()))}. "
                f"Leave `base` empty to get a supported default for your size.")
        # AN EXPLICIT BASE THAT WILL BE SWAPPED, SAID OUT LOUD.
        #
        # _resolve_base treats a base whose id contains neither "abliterated"
        # nor "Instruct" as "you probably meant the instruct variant" and
        # substitutes the table's entry for your size. That may well be what
        # you wanted - but it is a GUESS overriding an EXPLICIT statement, and
        # it happens without a word, so `base: Qwen/Qwen2.5-Coder-0.5B` builds
        # a different model than the one written down. Warn rather than change
        # the behaviour: existing recipes depend on the substitution.
        if not any(h in rec.base for h in ("abliterated", "Instruct",
                                           "instruct")):
            warns.append(
                f"base {rec.base!r} names neither an instruct nor an "
                f"abliterated model, so the build will SUBSTITUTE this size's "
                f"instruct variant instead of using it. If you meant this "
                f"exact checkpoint, name the variant explicitly, or set it "
                f"under `models:` in a defaults file where an explicit answer "
                f"is honoured as written.")

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
            # HARD ERROR, NOT A WARNING. This is not a style note. data/synth.py
            # keys every corpus by safe_name(expert), which lowercases, and
            # run/builder.py looks the corpus up by the RAW name - so `Bestiary`
            # files bestiary_code.jsonl and then dies with "No data path for
            # expert Bestiary" two experts into fine-tuning, with the scrape and
            # the first specialist already paid for. Refuse it at parse time,
            # where it costs nothing.
            errs.append(f"expert name {e.name!r} must be lowercase - the "
                        f"pipeline files this expert's corpus under "
                        f"{e.name.lower()!r} and then looks it up under "
                        f"{e.name!r}, which fails after the scrape")
        s = e.source
        if s:
            kind_errs, kind_warns = corpus.check(s.kind, s)
            errs.extend(f"{e.name}: {m}" for m in kind_errs)
            warns.extend(f"{e.name}: {m}" for m in kind_warns)

            if s.kind == "stack" and s.language:
                warns.append(f"{e.name}: source.kind=stack language {s.language!r} "
                             f"must be spelled EXACTLY as the corpus spells it")
            if getattr(s, "reasoning", False) and s.kind != "synth":
                # reasoning: true GENERATES reasoning traces and ignores the
                # scraped source. Say so rather than silently not scraping.
                warns.append(
                    f"{e.name}: source.reasoning=true generates reasoning "
                    f"traces, so the {s.kind!r} source is NOT collected. This "
                    f"is the R1-distill recipe - a reasoning teacher writes "
                    f"<think>…answer pairs for this specialist.")
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
    # AN IMPOSSIBLE FLOOR, CAUGHT BEFORE THE SCAN RATHER THAN AFTER IT.
    # min_samples is a floor and max_samples is a ceiling; a floor above the
    # ceiling can never be satisfied, and the scan would walk shards until the
    # cap discovering that.
    if (rec.corpus.min_samples > 0 and rec.corpus.max_samples > 0
            and rec.corpus.min_samples > rec.corpus.max_samples):
        errs.append(
            f"corpus.min_samples ({rec.corpus.min_samples}) is above "
            f"corpus.max_samples ({rec.corpus.max_samples}) - the floor is "
            f"higher than the ceiling, so no scan can satisfy it")

    # A MIX THE CEILING CANNOT FILL. Same family as the check above, one step
    # further out: corpus.max_samples caps what each expert may collect, and
    # the router later asks for its share of router_mix_total out of the
    # `.train` split of exactly that. If the cap is below what the mix needs,
    # the corpus stage passes, every specialist trains, and the router comes up
    # short of quota on every expert - hours later, reported as a gate that
    # would not learn. Refuse in two seconds on a laptop instead.
    if rec.corpus.max_samples > 0:
        from .pipeline import router_doc_need
        _mix = (rec.corpus.router_mix_total
                if rec.corpus.router_mix_total > 0 else 16_000)
        _frac = (rec.router.agent_mix_fraction
                 if rec.router.agent_mix_fraction >= 0 else 0.15)
        _need = router_doc_need(rec, _mix, _frac)
        if _need > rec.corpus.max_samples:
            errs.append(
                f"corpus.max_samples ({rec.corpus.max_samples:,}) cannot feed a "
                f"{_mix:,}-row router mix: each expert would need about "
                f"{_need:,} collected docs so its share survives the held-out "
                f"split. Raise corpus.max_samples to {_need:,}+, lower "
                f"corpus.router_mix_total, or buy the same router steps more "
                f"cheaply with router.epochs "
                f"(steps = router_mix_total x epochs / (batch x accum)).")

    if rec.gates.experts not in ("auto", "cheap", "skip"):
        errs.append(f"gates.experts must be auto | cheap | skip, got "
                    f"{rec.gates.experts!r}")
    if rec.gates.main_evals == "auto":
        warns.append("gates.main_evals=auto removes the third switch. The "
                     "expensive suite will run unattended on whatever the "
                     "build produced, including a NaN'd model that generates "
                     "at full speed and emits one token forever.")

    # -- roots --------------------------------------------------------------
    # Both empty = the tool's defaults (msmoe_data / msmoe_run_{size}), which
    # differ by construction - only compare what the recipe actually SAID.
    if rec.roots.data and rec.roots.output and rec.roots.data == rec.roots.output:
        errs.append("roots.data and roots.output must differ")
    if rec.roots.output and "{size}" not in rec.roots.output:
        warns.append("roots.output has no {size} - every rung of the ladder "
                     "will write to the same directory and _done() will skip "
                     "training on the wrong-sized specialists it finds there")

    return errs, warns


def resolve(rec: Recipe) -> Dict[str, Any]:
    """The EFFECTIVE build — derived values made explicit."""
    # Roots come from the real resolver, not rec.roots verbatim: the raw
    # templates still carry "{size}", and rec.size is "auto" in any recipe that
    # lets the tier pick. resolve_run_roots answers both at once.
    from .pipeline import resolve_run_roots  # lazy: keep the base import light
    roots = resolve_run_roots(rec)
    return {
        "recipe_id": rec.recipe_id(),
        "name": rec.name,
        "base": rec.base,
        "size": rec.size,
        "experts": [e.name for e in rec.experts],
        "sources": {e.name: e.source.kind for e in rec.experts if e.source},
        "tools_expert": getattr(rec, "tools_expert_name", ""),
        "target_steps": rec.budget.target_steps,
        "tokens_per_step": rec.tokens_per_step,
        "tokens_per_expert": rec.tokens_per_expert,
        "collect_tokens": rec.collect_tokens,
        "warmup_steps": rec.warmup_steps,
        "effective_batch": rec.budget.per_device_batch * rec.budget.grad_accum,
        "experts_per_tok": rec.moe.experts_per_tok,
        "dense_layers": rec.moe.dense_layers,
        "gates": asdict(rec.gates),
        "data_root": roots["data"],
        "output_root": roots["output"],
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
