#!/usr/bin/env python3
"""
Ms.MoE - the recipe (GPL-3.0)

A recipe is the whole build, as a document. Pass one to the CLI and get a run;
hand one to somebody else and they get YOUR run. That second sentence is the
entire reason this file exists - "it works, look" is a demo, and a recipe
somebody else can execute is a result.

    ms-moe-maker build recipe.yaml          # stagehand runs exactly this
    python3 msmoe_recipe.py --validate recipe.yaml
    python3 msmoe_recipe.py --resolve  recipe.yaml    # the EFFECTIVE build

WHY STDLIB ONLY
---------------
No pydantic, no torch, nothing. Three different things have to read a recipe:
the pipeline (which drags in 6 GB of ML deps), SerenTheatre (which deliberately
drags in none of them), and a human with bare python who wants to know whether
their file is valid before burning a GPU-week. A schema that needs the heaviest
consumer's dependencies installed is a schema the other two cannot use.

WHY EVERY VALIDATION HERE IS A SCAR
-----------------------------------
Nothing in `validate()` is defensive-programming reflex. Each check is a bug
that has actually happened, in this lab, and cost real hours:

  * budget in DOCUMENTS instead of tokens gave one expert 4.3x the gradient
    updates of another while both read "10,000 samples"
  * a fixed warmup was 4% of the real run and 33% of the dry run, so the two
    rungs were not the same experiment
  * dense_layers picked from an assumption ("signal climbs to the top") that
    the discrimination probe later contradicted
  * a 0.5B teacher too weak to emit schema-valid tool calls
  * a flag that silently did nothing, twice, because it was consumed at a
    stage that had already been skipped

So the recipe is not just configuration - it is the place those lessons get
enforced instead of remembered.
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

# ── knobs that are DERIVED, never written by hand ───────────────────────────
# Anything computed here is computed in exactly one place so the number the
# pipeline uses and the number the dashboard displays cannot disagree. That
# failure has its own entry in the port-map fact; it is not hypothetical.


@dataclass
class Source:
    """Where one expert's training text comes from.

    THE KINDS ARE A REGISTRY, NOT A LIST - see corpus.py. They used to be the
    literal tuple ("hf", "stack", "synth") written down in three places: this
    docstring, the validator below, and the --describe payload. Adding one
    meant finding all three, and missing one meant a recipe that validated and
    then failed at build time.

    Run `ms-moe-maker recipe --describe` for the kinds registered on THIS box,
    including any a plugin added. The built-ins:

      hf     - any HuggingFace dataset (repo + text_field). Domain-neutral:
               equally a code corpus, a lore corpus or a pile of lecture notes.
      stack  - the one code-specific kind; scan the general code corpus for a
               language name.
      synth  - generate it with a teacher model + a rejection-sampling
               validator, for a domain no corpus exists to scrape.
      local  - text already on this box. Nothing leaves the machine.

    The fields below are the union across kinds. A kind declares which it
    requires; the rest are simply ignored by kinds that do not care.
    """
    kind: str
    # kind=hf
    repo: Optional[str] = None
    split: str = "train"
    # NOTE the default is "code" for backwards compatibility with every recipe
    # written so far. A lore corpus will want text_field: text, and validate
    # says so rather than letting it silently read an absent column.
    text_field: str = "code"
    # kind=stack
    language: Optional[str] = None
    max_shards: int = 80
    # kind=synth
    teacher: Optional[str] = None
    generator: Optional[str] = None      # named generator in the pipeline
    examples: int = 15_000
    # kind=local
    path: Optional[str] = None
    glob: str = "**/*.txt"


@dataclass
class Expert:
    """One specialist. Five deliberate experts, none of them dead."""
    name: str
    # Defaulted so the two-step construction in parse() works: the nested
    # source is built separately (it needs its own unknown-key reporting) and
    # assigned after. parse() refuses a missing or non-mapping source before
    # it ever gets here, so None never escapes into a build.
    source: Optional[Source] = None
    # Optional per-expert override of the shared token budget. Leave unset.
    # If you find yourself setting this, ask whether you actually want a
    # different TARGET_STEPS - unequal experts are the thing the budget exists
    # to prevent, and an override is you turning that off on purpose.
    tokens: Optional[int] = None


@dataclass
class Budget:
    """How much training each expert gets, in the unit that decides it.

    With packing="wrapped" the trainer concatenates the corpus and slices
    fixed max_seq_length blocks, so step count is a pure function of TOKENS:

        steps = tokens / (max_seq_length * per_device_batch * grad_accum)

    Which is why the budget is expressed as TARGET_STEPS and the token count is
    derived. Steps are what LoRA schedule health actually depends on, and
    holding steps flat across rungs is what makes a 3B result evidence about
    14B rather than a differently-shaped experiment.
    """
    target_steps: int = 1200
    max_seq_length: int = 2048
    per_device_batch: int = 4
    grad_accum: int = 2
    # Fraction, not a fixed count. A fixed 50 was 4% of a 1200-step run and
    # 33% of a 150-step run - the small rung spent a third of itself ramping
    # up, which flattens the loss curve and makes a healthy run look starved.
    warmup_ratio: float = 0.05
    warmup_floor: int = 10
    # Collect this much more than the budget: the chat template wrapped around
    # every sample is real tokens too, and the char-based estimate used while
    # scanning is deliberately optimistic. Over-collecting costs disk;
    # under-collecting costs another shard walk.
    collect_headroom: float = 1.25
    doc_ceiling: int = 100_000           # a ceiling, NOT a target


@dataclass
class MoE:
    """The architecture. Everything here is fixed at STITCH time, not load."""
    experts_per_tok: int = 2
    norm_topk_prob: bool = True
    aux_loss_coef: float = 0.001
    shared_expert_width: int = 1
    # Layers that stay dense (one shared FFN) instead of becoming MoE layers.
    # The ONLY lever that actually shrinks the model, and it is architecture -
    # setting it at load time does nothing at all.
    #
    # "auto" means: refuse to guess, and tell the operator to measure. The
    # 0.5B discrimination probe found routing signal peaking MID-STACK and
    # falling off toward the top, which contradicts the "signal climbs
    # monotonically" assumption a previous 14B run's dense-layer choice was
    # built on. Take dense layers from the BOTTOM of a measured ranking, not
    # from the bottom of the stack.
    dense_layers: Any = "auto"           # "auto" | [] | [0, 2, 10, ...]


@dataclass
class Gates:
    """Where a hand goes on the surface.

    Third instance of the same principle in this stack, after the consolidator's
    draft review and the tool registry's propose_tool: the cheap thing runs, the
    expensive/irreversible thing waits for a person.

    auto   - run it as part of the build
    manual - build stops here and waits. THROW THE THIRD SWITCH.
    skip   - do not run at all (and say so in the report, never silently)
    """
    base_evals: str = "auto"
    main_evals: str = "manual"


@dataclass
class Runtime:
    """Machine-shaped knobs. Nothing here changes what is built, only whether
    it survives being built."""
    dtype: str = "bfloat16"
    # from_pretrained needs ~2x the model on a unified-memory box: it
    # materialises the empty skeleton on device first, then accumulates host
    # staging copies that are never freed. Measured 70.3 GB -> 125 GB, OOM.
    direct_load: bool = True
    # Read BEFORE torch initialises or it does nothing. Measured on a 1.9B
    # five-expert MoE generating 66 tokens: 106.6 GB reserved to hold 6.4 GB
    # of live tensors, MemAvailable floor 16.7 GB, box OOM-killed. With this,
    # 8.3 GB reserved and a 114.2 GB floor. Same model, same weights.
    alloc_conf: str = "expandable_segments:True"
    llama_cpp: str = "llama.cpp"
    # Prove it leaves Python. Three of this project's nastiest bugs were
    # invisible inside transformers and only appeared past that boundary.
    export_gguf: bool = True


@dataclass
class Roots:
    """Two roots, because they have different lifetimes.

    A corpus of PowerShell does not care what size model consumes it - scan
    once, reuse at every rung of the ladder. Model artefacts are size-shaped
    and mutually incompatible, so they fork per rung. Sharing one root means a
    3B specialist lands where the 7B run looks for one, gets skipped as
    'already trained', and reaches the stitcher as a shape mismatch after you
    have paid for the other four.
    """
    data: str = "msmoe_data"
    output: str = "msmoe_{size}"


@dataclass
class Recipe:
    name: str
    base: str
    experts: List[Expert]
    schema_version: int = SCHEMA_VERSION
    size: str = "auto"                   # label for the output root
    budget: Budget = field(default_factory=Budget)
    moe: MoE = field(default_factory=MoE)
    gates: Gates = field(default_factory=Gates)
    runtime: Runtime = field(default_factory=Runtime)
    roots: Roots = field(default_factory=Roots)

    # ── derived ────────────────────────────────────────────────────────────
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
        """Stable short hash of the build-affecting fields.

        Runtime is EXCLUDED on purpose: dtype and allocator flags change
        whether a build survives, not what it produces. Two runs that differ
        only in runtime should compare as the same recipe, or the id is
        useless for exactly the comparison you want to make.
        """
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("runtime", "name")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── parsing ─────────────────────────────────────────────────────────────────

_KNOWN_TOP = {"name", "base", "experts", "schema_version", "size", "budget",
              "moe", "gates", "runtime", "roots"}


def _build(cls, data: Dict[str, Any], path: str,
           warnings: List[str]):
    """Construct a dataclass, reporting unknown keys instead of eating them.

    Lenient parse, LOUD about it. A typo'd key that silently does nothing is
    the exact failure this project keeps re-learning - a flag that did nothing
    looks identical to a flag that did not help. So: unknown keys never abort
    the build, and never pass unremarked either.
    """
    fields = {f for f in cls.__dataclass_fields__}
    unknown = sorted(set(data) - fields)
    for u in unknown:
        warnings.append(f"{path}.{u} is not a known key - IGNORED "
                        f"(known: {', '.join(sorted(fields))})")
    return cls(**{k: v for k, v in data.items() if k in fields})


def parse(data: Dict[str, Any]) -> Tuple[Recipe, List[str]]:
    warnings: List[str] = []
    if not isinstance(data, dict):
        raise ValueError("a recipe must be a mapping at the top level")

    for u in sorted(set(data) - _KNOWN_TOP):
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
                             f"'kind' of hf | stack | synth")
        expert = _build(Expert, {k: v for k, v in e.items() if k != "source"},
                        f"experts[{i}]", warnings)
        expert.source = _build(Source, src, f"experts[{i}].source", warnings)
        experts.append(expert)

    rec = Recipe(
        name=data.get("name") or "unnamed",
        base=data.get("base") or "",
        experts=experts,
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        size=data.get("size", "auto"),
        budget=_build(Budget, data.get("budget") or {}, "budget", warnings),
        moe=_build(MoE, data.get("moe") or {}, "moe", warnings),
        gates=_build(Gates, data.get("gates") or {}, "gates", warnings),
        runtime=_build(Runtime, data.get("runtime") or {}, "runtime", warnings),
        roots=_build(Roots, data.get("roots") or {}, "roots", warnings),
    )
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


# ── validation ──────────────────────────────────────────────────────────────

def validate(rec: Recipe) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors mean do not build."""
    errs: List[str] = []
    warns: List[str] = []

    if rec.schema_version != SCHEMA_VERSION:
        warns.append(f"schema_version {rec.schema_version} != {SCHEMA_VERSION} "
                     f"- fields may be read differently than you intend")
    if not rec.base:
        errs.append("base is required (a HuggingFace id or a local path)")
    elif "/" not in rec.base and not rec.base.startswith("."):
        warns.append(f"base {rec.base!r} has no '/' - if that is a hub id it "
                     f"will 404. Guessing an id cost this project an evening.")

    # -- experts ------------------------------------------------------------
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
        # The registry decides what kinds exist and what each one requires, so
        # a plugin's kind validates here without this file having heard of it.
        # What stays below is the ADVICE - heuristics about values that are
        # legal but probably wrong, which is a different job from schema.
        kind_errs, kind_warns = corpus.check(s.kind, s)
        errs.extend(f"{e.name}: {m}" for m in kind_errs)
        warns.extend(f"{e.name}: {m}" for m in kind_warns)

        if s.kind == "stack" and s.language:
            warns.append(f"{e.name}: source.kind=stack language {s.language!r} "
                         f"must be spelled EXACTLY as the corpus spells it - an "
                         f"inexact match is a silent zero for an "
                         f"unrelated-looking reason")
        if s.kind == "synth":
            if s.teacher:
                small = any(t in s.teacher for t in
                            ("0.5B", "1.5B", "3B", "0.6B", "1B", "2B"))
                if small:
                    warns.append(
                        f"{e.name}: teacher {s.teacher!r} looks small. A 0.5B "
                        f"teacher is too weak to emit schema-valid tool calls "
                        f"and just trips the accept-rate tripwire; 7B is the "
                        f"smallest that reliably clears it.")

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
            f"you are comparing against earlier rungs; batch 1 x accum 8 and "
            f"batch 4 x accum 2 optimise identically and differ 16x in speed.")
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
                f"different value does NOTHING once a skeleton exists. Take "
                f"them from the BOTTOM of a probe_router_discrimination "
                f"ranking, not the bottom of the stack: measured routing "
                f"signal peaks mid-stack and falls off toward the top.")
    else:
        errs.append("moe.dense_layers must be 'auto' or a list")
    if m.shared_expert_width == 0:
        errs.append("moe.shared_expert_width=0 produces a zero-element GGUF "
                    "tensor whose element count overflows llama.cpp's loader")
    if not m.norm_topk_prob:
        warns.append("moe.norm_topk_prob=false scales the stitched model to "
                     "~0.40x at init, so the router trains on the wrong problem")

    # -- gates --------------------------------------------------------------
    for g, v in (("base_evals", rec.gates.base_evals),
                 ("main_evals", rec.gates.main_evals)):
        if v not in ("auto", "manual", "skip"):
            errs.append(f"gates.{g} must be auto | manual | skip, got {v!r}")
    if rec.gates.main_evals == "auto":
        warns.append("gates.main_evals=auto removes the third switch. The "
                     "expensive suite will run unattended on whatever the "
                     "build produced, including a NaN'd model that generates "
                     "at full speed and emits one token forever.")

    # -- roots --------------------------------------------------------------
    if rec.roots.data == rec.roots.output:
        errs.append("roots.data and roots.output must differ - sharing them is "
                    "how a 3B specialist ends up in the 7B run's directory")
    if "{size}" not in rec.roots.output:
        warns.append("roots.output has no {size} - every rung of the ladder "
                     "will write to the same directory and _done() will skip "
                     "training on the wrong-sized specialists it finds there")
    return errs, warns


def resolve(rec: Recipe) -> Dict[str, Any]:
    """The EFFECTIVE build - derived values made explicit.

    This is what should be stamped at the top of a run log. The whole [cfg]
    stamp discipline exists because a value you assumed and a value in force
    look identical until one of them costs you 23 hours.
    """
    return {
        "recipe_id": rec.recipe_id(),
        "name": rec.name,
        "base": rec.base,
        "size": rec.size,
        "experts": [e.name for e in rec.experts],
        "sources": {e.name: e.source.kind for e in rec.experts},
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


# ── CLI ─────────────────────────────────────────────────────────────────────

DESCRIBE = {
    "name": "ms-moe-maker-recipe",
    "schema_version": SCHEMA_VERSION,
    # The live registry, so a plugin's kind is advertised without
    # this literal being edited. It was a frozen list in three
    # places; that is how a kind gets supported but not offered.
    "kinds": corpus.describe(),
    "gates": ["auto", "manual", "skip"],
    "description": "Recipe schema for Ms.MoE. Validate before you burn a "
                   "GPU-week.",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="msmoe_recipe",
        description="Ms.MoE recipe - validate and resolve a build document.")
    ap.add_argument("recipe", nargs="?", help="path to recipe .yaml or .json")
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--resolve", action="store_true",
                    help="print the EFFECTIVE build as JSON")
    ap.add_argument("--json", action="store_true",
                    help="JSON Lines events instead of prose (Starwright "
                         "contract; stagehand reads this)")
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
    except Exception as exc:  # noqa: BLE001 - the message IS the product here
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
        print(f"\nMs.MoE recipe  {rec.name}  [{eff['recipe_id']}]")
        print(f"   base     {rec.base}")
        print(f"   experts  {', '.join(eff['experts'])} "
              f"(top-{rec.moe.experts_per_tok} of {len(rec.experts)})")
        print(f"   budget   {eff['target_steps']} steps x "
              f"{eff['tokens_per_step']:,} tok/step = "
              f"{eff['tokens_per_expert']/1e6:.2f}M tokens per expert")
        print(f"            warmup {eff['warmup_steps']} "
              f"({100*eff['warmup_steps']/max(rec.budget.target_steps,1):.1f}%)"
              f"   collect {eff['collect_tokens']/1e6:.2f}M")
        print(f"   gates    base={rec.gates.base_evals} "
              f"main={rec.gates.main_evals}")
        print(f"   roots    data={eff['data_root']}  "
              f"output={eff['output_root']}")
        if a.resolve:
            print("\n" + json.dumps(eff, indent=2))
        for w in warns:
            print(f"\n   WARN  {w}")
        for e in errs:
            print(f"\n   ERROR {e}")
        ok = not errs and not (a.strict and warns)
        print(f"\n   {'VALID' if ok else 'REJECTED'} - "
              f"{len(errs)} error(s), {len(warns)} warning(s)\n")

    return 0 if (not errs and not (a.strict and warns)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
