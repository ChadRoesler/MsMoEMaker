"""What every field in a run's fingerprint MEANS, in words a person can read.

`build_fingerprint` emits 75 resolved values, and a viewer (seren-theatre's
playbill) renders them off the manifest as a table. `collect_token_target
29491200` is a true row and an unreadable one: nobody typed that number, and
nothing on the page says where it came from.

THIS LIVES IN THE WRITER, DELIBERATELY. A copy of these sentences in the
viewer would go stale, and that exact failure has already happened four times
between these two packages - `mlp_only_layers`, the `WARNED` status, a renamed
module, three manifest fields - each time silently, because a doc-string does
not fail a test. The glossary is stamped INTO the manifest beside `resolved`,
so it travels with the run: a base seren-theatre install has no ms-moe-maker
to ask, and an archived run from a year ago still has to explain itself. If
Theatre cannot hold a copy, Theatre cannot lie.

STDLIB ONLY, AND IT STAYS THAT WAY. No torch, no transformers, no IO, no
reading of anything. `validate`, `describe` and `build --plan` run on a laptop
and this module is imported by all three.

TWO HALVES PER ENTRY, AND THE SECOND IS THE VALUABLE ONE:

  summary       what turning it does, and what it costs. Never "the LoRA
                rank" - defining a term with another term explains nothing to
                the person who needed the tooltip.
  derived_from  the ACTUAL expression, in field names, for a value nobody
                typed. A field nobody typed is the one they stare at. `None`
                means a recipe key or a default supplied it directly.

The expressions are transcribed from `pipeline.build_config`, which is where
these values are really computed. A wrong formula here is worse than none, so
when one changes over there, change it here - `tests/test_knob_glossary.py`
pins the field NAMES in every expression, and every field having an entry at
all, but no test can check that the arithmetic still matches. That one is on
whoever moves the line.

ONE SOURCE FOR THE SENTENCES. Around 38 of these knobs also appear in the
README's knob tables, which would be the same sentences written twice. `readme`
records where, and the test asserts the two agree word for word (the README may
add emphasis and backticks, not words). Edit the summary here and the test
tells you which README cell moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

# Names that legitimately appear in a `derived_from` expression but are NOT
# PipelineConfig fields: recipe keys, and the tier table's own column. Declared
# so the test can insist every other underscored name in an expression is a
# real fingerprint field - which is what catches a rename silently turning a
# formula into fiction.
RECIPE_TERMS: frozenset = frozenset({
    "collect_headroom", "min_samples", "max_samples", "synth_samples",
    "dense_layers", "tools_expert", "reasoning_expert", "held_out_fraction",
    "default_size",
})


@dataclass(frozen=True)
class Knob:
    """One row of the playbill, explained."""

    summary: str
    derived_from: Optional[str] = None
    # "<block>.<row>" in the README's knob tables, when the same sentence is
    # printed there too. Empty when this field has no README row.
    readme: str = ""
    # Where in a RECIPE this value is written, as "<block>.<key>". Usually the
    # same string as `readme` - the README's knob tables are addressed by
    # recipe block and key - so it is only set here when there is no README row
    # or the two genuinely differ. `ms-moe-maker bundle` stamps along these
    # paths; see recipe_path() and UNPINNABLE below.
    recipe: str = ""


KNOBS: Dict[str, Knob] = {

    # ── identity ──────────────────────────────────────────────────────────
    "size": Knob(
        "How big the base model is. The whole cost curve of the build hangs "
        "off it - corpus, VRAM, hours and disk all move together.",
        derived_from="tier (its default_size), unless the recipe names a size"),
    "base": Knob(
        "The checkpoint every specialist is trained from, and the skeleton "
        "the MoE is built on.",
        derived_from="size (the instruct entry in the box's model table), "
                     "unless the recipe or MSMOE_BASE_MODEL names one"),
    "base_safe": Knob(
        "The plain, non-instruct checkpoint for this size. Kept as the "
        "fallback for stages that need a base when none was named.",
        derived_from="size (the plain entry in the box's model table)"),
    "tier": Knob(
        "The box this build is aimed at. It picks the default model size, "
        "adapter rank and export quantisation whenever the recipe does not.",
        readme="runtime.hardware_tier"),
    "dryrun": Knob(
        "True when this is the smallest-rung run: less corpus, a smaller "
        "teacher, and its own output directory so it can never be mistaken "
        "for a real build."),
    "expert_names": Knob(
        "The experts this build assembles, in the order they are spliced "
        "into the model. Reordering them is a different model, not the same "
        "one relabelled.",
        derived_from="experts[].name, in recipe order"),

    # ── abliteration ──────────────────────────────────────────────────────
    "abliterate_enabled": Knob(
        "Strips the base model's refusal direction before any specialist "
        "trains, so the finished MoE inherits it instead of being decensored "
        "afterwards.",
        recipe="abliterate.enabled"),
    "abliterate_n_trials": Knob(
        "How many candidate decensorings the search tries before one is "
        "picked. The whole cost of the stage lives here - a 0.5B run is "
        "20-30 minutes on one card.",
        readme="abliterate.n_trials"),
    "abliterate_seed": Knob(
        "Fixes the search's random draws so the same recipe picks the same "
        "decensoring twice. Unset means a fresh study every run.",
        readme="abliterate.seed"),
    "abliterate_quantization": Knob(
        "Loads the base at 4-bit (bnb_4bit) during the search so it fits in "
        "less VRAM, at some cost to what the search can measure. Needs "
        "bitsandbytes installed.",
        readme="abliterate.quantization"),
    "abliterate_trial_index": Knob(
        "Which of the search's best candidates to keep. Unset takes the "
        "first; an index past the end is clamped rather than raised, because "
        "the study has already been paid for by then.",
        readme="abliterate.trial_index"),
    "abliterate_checkpoint_action": Knob(
        "What happens to a study that was interrupted. continue picks up the "
        "trials already paid for; restart throws them away and searches "
        "again.",
        readme="abliterate.checkpoint_action"),
    "abliterate_export": Knob(
        "What the stage writes. merge saves a whole decensored model the "
        "specialists train from; adapter saves only the difference, which is "
        "smaller but has to be applied by whatever loads it.",
        readme="abliterate.export"),

    # ── reasoning ─────────────────────────────────────────────────────────
    "reasoning": Knob(
        "True when the base model writes out its thinking before it answers. "
        "Prompt building, trace parsing and how much room eval gives a "
        "generation all key off this rather than sniffing the model name "
        "again.",
        derived_from="base, sniffed against the reasoning families table"),
    "reasoning_type": Knob(
        "Which thinking-trace convention this run writes and reads, or empty "
        "when nothing in it thinks. Names the tag pair below.",
        derived_from="base sniffed against the reasoning families table, "
                     "else xml when reasoning_experts is not empty"),
    "reasoning_open": Knob(
        "The exact tag that opens a block of thinking. Stamped onto the run "
        "so a scorer splits traces on the same tags the generator wrote them "
        "with, even if the box's tag table is edited mid-build.",
        derived_from="reasoning_type"),
    "reasoning_close": Knob(
        "The exact tag that closes a block of thinking. Stamped for the same "
        "reason as the opening one: the reader must not be able to disagree "
        "with the writer.",
        derived_from="reasoning_type"),
    "reasoning_interwoven": Knob(
        "True when this convention lets thinking and answer alternate through "
        "the response instead of all the thinking coming first.",
        derived_from="reasoning_type"),
    "reasoning_experts": Knob(
        "Experts whose corpus is generated with thinking baked into every "
        "row, which is how a base that does not think gets trained into one.",
        derived_from="experts[].name where source.reasoning is true"),
    "reasoning_expert_name": Knob(
        "The one injected expert whose corpus is built to span every other "
        "expert's domain, or empty when there is none. Not the same as "
        "reasoning_experts, whose corpora each stay inside one domain.",
        derived_from="reasoning_expert.name, when an expert of that name is "
                     "on the roster"),
    "synth_experts": Knob(
        "Experts whose corpus is written by a teacher model rather than "
        "collected. Their rows are kept as whole transcripts instead of being "
        "wrapped in a write-me-some-code prompt.",
        derived_from="experts[].name where source.kind is synth"),
    "tools_expert_name": Knob(
        "The expert whose corpus is generated tool-calling traces, or empty "
        "when there is none. Downstream stages ask this instead of looking "
        "for the literal name agentcore.",
        derived_from="tools_expert.name, else a synth expert named agentcore"),

    # ── corpus collection ─────────────────────────────────────────────────
    "num_code_samples": Knob(
        "Ceiling on documents kept per expert. It caps collection; it never "
        "becomes a target.",
        derived_from="corpus.max_samples, else 100,000 (10,000 on --dryrun)",
        readme="corpus.max_samples"),
    "min_samples_per_expert": Knob(
        "Documents an expert must have collected before it is allowed to "
        "train. Below it the stage fails rather than train on scraps, and it "
        "rises on its own when the router's mix needs more than you asked "
        "for.",
        derived_from="max(corpus.min_samples, ceil(router_mix_total / "
                     "collected experts / (1 - eval_held_out_fraction) * "
                     "1.05)) - router_mix_total is cut by agent_mix_fraction "
                     "first when there is a tools expert",
        readme="corpus.min_samples"),
    "collect_token_target": Knob(
        "How many tokens of text to gather per expert before collection is "
        "allowed to stop. Deliberately above what training will consume, so "
        "packing rows together never leaves an expert short.",
        derived_from="expert_token_budget * collect_headroom"),
    "chars_per_token_est": Knob(
        "Characters per token, used to guess how much text has been gathered "
        "without tokenising it first. A stopping estimate only - the trainer "
        "counts the real tokens later."),
    "max_shards": Knob(
        "How many corpus shards the scan may pull before giving up, at "
        "roughly 0.57 GB each.",
        readme="corpus.max_shards"),
    "per_repo_cap": Knob(
        "Most files one repository may contribute to one language. Not a "
        "tuning knob: measured, a single enterprise codebase filled 78% of a "
        "C# corpus and the expert learned one company's house style instead "
        "of the language.",
        readme="corpus.per_repo_cap"),

    # ── generated (teacher) data ──────────────────────────────────────────
    "num_agent_samples": Knob(
        "How many traces the teacher writes for each generated expert. The "
        "most expensive documents in the build, because every one of them is "
        "produced rather than downloaded.",
        derived_from="corpus.synth_samples, else min(corpus.max_samples, the "
                     "run default of 15,000 - 2,000 on --dryrun)"),
    "teacher_model": Knob(
        "The model that writes the training traces for generated experts. "
        "Its quality is the ceiling on what those experts can learn.",
        derived_from="dryrun (a smaller teacher on --dryrun)"),
    "reasoning_teacher": Knob(
        "The model that writes traces for an expert that has to think. A "
        "distilled reasoner, because that is the proof a small non-reasoning "
        "base can be trained into one.",
        derived_from="dryrun (a smaller teacher on --dryrun)"),
    "use_vllm": Knob(
        "Serves the teacher through vLLM instead of plain transformers. Much "
        "faster generation, and a second serving stack to install and keep "
        "happy. Also moves the teacher batch from 96 to 512, so it changes "
        "what gets generated and not only how fast.",
        recipe="runtime.use_vllm"),
    "vllm_max_len": Knob(
        "Longest prompt-plus-answer the vLLM teacher will hold in one "
        "request. Too small and long traces are cut off mid-sentence; too "
        "large and it reserves memory it never uses."),
    "vllm_quantization": Knob(
        "Quantisation for the vLLM teacher, for when it will not otherwise "
        "fit. Unset loads it at full bfloat16."),
    "teacher_max_new": Knob(
        "Most tokens the teacher may write for one generated example. Too "
        "low and answers stop mid-script.",
        recipe="budget.teacher_max_new"),
    "reasoning_teacher_max_new": Knob(
        "Most tokens the teacher may write for one reasoning example. Higher "
        "than the plain ceiling because the thinking alone can eat the whole "
        "budget before the answer starts.",
        recipe="budget.reasoning_teacher_max_new"),

    # ── specialist training ───────────────────────────────────────────────
    "max_seq_length": Knob(
        "Tokens per training row. Halving it roughly halves memory and time, "
        "and truncates long files.",
        readme="budget.max_seq_length"),
    "lora_r": Knob(
        "How far a specialist is allowed to move away from the base model. "
        "Bigger means more room to specialise and more weights to train, "
        "store and stitch; over a small corpus it mostly buys memorisation.",
        readme="budget.lora_r"),
    "lora_alpha": Knob(
        "How strongly the trained difference is applied on top of the base. "
        "Leave it unless you know why you are moving it.",
        readme="budget.lora_alpha"),
    "lora_dropout": Knob(
        "Randomly ignores part of the adapter on each step so it generalises "
        "instead of memorising. Non-zero costs a little speed.",
        readme="budget.lora_dropout"),
    "load_in_4bit": Knob(
        "Loads the base model at 4-bit while training. Buys memory, costs "
        "fidelity, and a specialist saved this way is refused by the "
        "stitcher for holding packed bytes instead of real matrices.",
        readme="runtime.load_in_4bit"),
    "optim": Knob(
        "Which optimiser the specialists train with. Follows use_unsloth: "
        "its 8-bit optimiser when that is on, plain AdamW when it is not.",
        derived_from="use_unsloth"),
    "gradient_checkpointing": Knob(
        "Recomputes activations during the backward pass instead of keeping "
        "them around. Buys a lot of memory and costs a good slice of the "
        "speed."),
    "packing_strategy": Knob(
        "How short training rows are packed together to fill a sequence. "
        "wrapped lets a row run on past the boundary, so no place in a batch "
        "is spent on padding."),
    "target_modules": Knob(
        "Which parts of each layer the adapter is attached to. These three "
        "are the feed-forward block - the part the stitcher copies per "
        "expert, so training anything else would not survive the stitch."),
    "attn_impl": Knob(
        "Which attention implementation the specialists train with. sdpa is "
        "the one built into PyTorch, so it needs nothing installed."),
    "per_device_batch": Knob(
        "Rows the GPU processes at once. Raise it until you run out of "
        "memory, then back off one.",
        readme="budget.per_device_batch"),
    "grad_accum": Knob(
        "How many of those batches are added up before the model is actually "
        "updated. per_device_batch × grad_accum is the effective batch.",
        readme="budget.grad_accum"),
    "lr_lora": Knob(
        "How big a correction a specialist makes each step. Too high and the "
        "adapter never settles; too low and the whole step budget barely "
        "moves it."),
    "specialist_save_steps": Knob(
        "How often a specialist writes a checkpoint. More often loses less "
        "to a crash at expert four and spends more time writing; only the two "
        "newest are kept."),
    "use_unsloth": Knob(
        "Trains through Unsloth's kernels instead of plain transformers. "
        "Faster on some boxes and measured 5.5x SLOWER on a GB10, so it is "
        "off unless asked for."),
    "seed": Knob(
        "The one number every random draw in the build comes from - prompt "
        "choice, adapter initialisation, gate noise, teacher sampling. Change "
        "it and the same recipe gives a different run; keep it and two boxes "
        "agree."),

    # ── budget ────────────────────────────────────────────────────────────
    "target_steps": Knob(
        "Optimiser steps each specialist trains for. The single biggest lever "
        "on wall-clock: total is about target_steps × experts × "
        "seconds-per-step.",
        readme="budget.target_steps"),
    "expert_token_budget": Knob(
        "Tokens of training text each specialist is actually allowed to "
        "consume. This is what makes experts comparable - the one with the "
        "biggest corpus does not quietly get the longest run.",
        derived_from="target_steps * max_seq_length * per_device_batch * "
                     "grad_accum"),
    "warmup_steps": Knob(
        "Steps spent easing the learning rate up at the start of a "
        "specialist. A planning figure only: the trainer recomputes it "
        "against the steps the corpus really affords and caps it at half the "
        "run, so warmup can never BE the run.",
        derived_from="max(warmup_floor, round(warmup_ratio * target_steps))"),
    "warmup_ratio": Knob(
        "Fraction of the run spent easing the learning rate up from zero "
        "instead of hitting the model at full strength on step one.",
        readme="budget.warmup_ratio"),
    "warmup_floor": Knob(
        "Never warm up for fewer steps than this, however short the run.",
        readme="budget.warmup_floor"),

    # ── the router (the gate) ─────────────────────────────────────────────
    "router_mix_total": Knob(
        "Rows in the stratified mix the gate trains on. Divided by batch "
        "× accum and multiplied by epochs, this is the router's step "
        "count - the number that decides whether the MoE routes at all.",
        derived_from="corpus.router_mix_total, else 16,000 (4,000 on "
                     "--dryrun)",
        readme="corpus.router_mix_total"),
    "agent_mix_fraction": Knob(
        "Share of the router's mix taken from generated (synth) experts. The "
        "rest split what is left, evenly.",
        readme="router.agent_mix_fraction"),
    "router_batch": Knob(
        "Rows the gate sees at once. Must be more than 1: the load-balancing "
        "loss needs several domains in one batch to mean anything, and at 1 "
        "every batch is a single domain and it can balance nothing.",
        readme="router.batch"),
    "router_accum": Knob(
        "How many of the gate's batches are added up before it is updated. "
        "Raising it buys fewer, steadier updates out of the same mix.",
        readme="router.accum"),
    "router_epochs": Knob(
        "Passes over the router's mix. The cheapest way to buy router steps - "
        "another pass is free, more mix rows cost a corpus.",
        readme="router.epochs"),
    "lr_router": Knob(
        "How big a correction the gate makes each step. Too high and it slams "
        "onto one expert; too low and it never leaves the noise it started "
        "from.",
        readme="router.lr"),
    "router_aux_loss_coef": Knob(
        "How hard the gate is pushed to spread traffic instead of collapsing "
        "onto one expert. Mixtral's value, kept because lowering it bought "
        "nothing measurable and spent margin against collapse.",
        readme="router.aux_loss_coef"),
    "router_init": Knob(
        "How the gate's weights start out. random seeds small noise so no two "
        "experts begin identical; zero exists only for the stitch's "
        "bit-equality check, and three trainings from a zero gate each "
        "collapsed onto a different single expert.",
        readme="moe.router_init"),
    "router_init_std": Knob(
        "How much noise the gate starts with when router_init is random. Too "
        "little and every expert looks the same to it on step one.",
        readme="moe.router_init_std"),

    # ── the stitched MoE ──────────────────────────────────────────────────
    "experts_per_tok": Knob(
        "How many experts each token is sent to. 1 is refused: the single "
        "gate weight is then divided by itself, so nothing ever teaches the "
        "gate to choose.",
        readme="moe.experts_per_tok"),
    "norm_topk_prob": Knob(
        "Rescales the chosen experts' weights to sum to 1, so the gate "
        "decides the blend and not the overall volume.",
        readme="moe.norm_topk_prob"),
    "shared_expert_width": Knob(
        "Width of the always-on expert every token passes through whatever "
        "the gate chooses. At the default it is inert by construction and the "
        "routed experts do all the work.",
        readme="moe.shared_expert_width"),
    "shared_expert_gate_fill": Knob(
        "What the always-on expert's gate holds before training. It must not "
        "be zero - silu(0)/0 is NaN after GGUF export, a break that only "
        "shows up outside Python.",
        readme="moe.shared_expert_gate_fill"),
    "mlp_only_layers": Knob(
        "Layers left as one ordinary feed-forward block instead of being "
        "split into experts, given as a list of layer indices. That block is "
        "the average of every specialist, so it costs no routing and carries "
        "no specialisation.",
        derived_from="moe.dense_layers or MSMOE_DENSE_LAYERS (empty when it "
                     "is auto)",
        readme="moe.dense_layers"),

    # ── prompting ─────────────────────────────────────────────────────────
    "code_prompt_templates": Knob(
        "The user turns a code row can be wrapped in when its language is "
        "named. The trainer draws from its own copy of this list, so editing "
        "it here moves the build id and nothing else."),
    "code_prompt_unnamed": Knob(
        "The user turns used when a row deliberately does not name its "
        "language. Same as the list above: the trainer holds its own copy, so "
        "this records the intent rather than driving it."),
    "code_prompt_unnamed_fraction": Knob(
        "How often a training row is asked for without naming its language, "
        "so the expert learns the domain and not the word for it."),

    # ── evaluation ────────────────────────────────────────────────────────
    "eval_held_out_fraction": Knob(
        "Share of each corpus kept out of training and used to score the "
        "result. Raising it buys a more trustworthy score and takes text away "
        "from the expert; 0.95 and above is ignored, because it leaves "
        "nothing to train on.",
        readme="eval.held_out_fraction"),
}


# ── accessors ─────────────────────────────────────────────────────────────
#
# The emitted CONTRACT is exactly two keys, and the viewer is built against it:
#
#   "knobs": {"lora_r": {"summary": "...", "derived_from": null}, ...}
#
# `readme` is bookkeeping for the duplication test and never goes on the wire.

def entry(name: str) -> Optional[Knob]:
    """The glossary entry for one fingerprint field, or None."""
    return KNOBS.get(name)


def _row(k: Knob) -> Dict[str, Any]:
    return {"summary": k.summary, "derived_from": k.derived_from}


def for_fields(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """The contract shape, for exactly the fields a manifest carries.

    A field with no entry - or an entry with a blank summary - is simply
    absent, so the viewer renders no `?` for it. Partial coverage degrades to
    "no affordance", never to an empty tooltip that looks like a bug in the
    page rather than a gap in the glossary.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        k = KNOBS.get(name)
        if k is None or not k.summary.strip():
            continue
        out[name] = _row(k)
    return dict(sorted(out.items()))


def describe() -> List[Dict[str, Any]]:
    """The glossary as rows, for `--describe`.

    `[{name, summary, derived_from}, ...]` - the same shape `kinds` and
    `validators` already use beside it in that payload, so a front-end reads
    all three the same way. The playbill reads the MANIFEST copy; this one is
    for Backstage and for anyone asking a live install what it knows.
    """
    return [{"name": name, **_row(k)}
            for name, k in sorted(KNOBS.items()) if k.summary.strip()]


def readme_rows() -> Dict[str, str]:
    """`<block>.<row>` in the README's knob tables -> the field it explains."""
    return {k.readme: name for name, k in sorted(KNOBS.items()) if k.readme}


# ── what a bundle cannot pin, and why ────────────────────────────────────────
#
# `ms-moe-maker bundle` writes a recipe with every default STAMPED IN, so a
# recipe handed to someone else builds the same thing on their box instead of
# quietly picking up their defaults. It can only do that for values the recipe
# LANGUAGE CAN EXPRESS.
#
# These cannot be expressed. Every one of them is in `build_fingerprint` - so
# every one of them changes what the build produces - and none of them has a
# recipe key to write it into. Three kinds, and the distinction matters because
# the fixes are different:
#
#   "env"       your SHELL decides part of the model. The worst of the three
#               for handing work to somebody, because nothing on disk records
#               it and the person receiving it has no reason to suspect.
#   "constant"  a literal in build_config. Stable until an upgrade moves it,
#               at which point every exported recipe silently gets the new one.
#   "cli"       correctly not a recipe key - it describes THIS invocation.
#
# THE ANSWER IS NOT TO PRETEND. A bundle records the whole resolved fingerprint
# in `bundle.json`, and import diffs it against what this box resolves - so an
# unpinnable field that differs is reported by NAME rather than discovered as a
# model that came out wrong. Same move as stamping the defaults_files sha256:
# the divergence is not prevented, it is made impossible to miss.
#
# Adding a recipe key for any of these is a real improvement and a separate
# decision. Until then this list is the honest statement of the gap, and
# tests/test_bundle_stamp.py insists every direct fingerprint field is either
# pinnable or named here - so a NEW unpinnable field is a choice somebody makes
# on purpose rather than a hole that opens quietly.
UNPINNABLE: Dict[str, str] = {
    # -- the shell decides ------------------------------------------------
    # use_vllm USED TO LIVE HERE, with the reason "which generator produced
    # the corpus is not recorded anywhere on disk". That was true, and it
    # was an argument for giving it a recipe key rather than for leaving it
    # unrecordable: it is a fingerprinted field that changes the corpus, so
    # a bundle that could not say which generator made the data was a
    # bundle missing something it needed. It is `runtime.use_vllm` now and
    # stamps like any other knob.
    "use_unsloth": "env: MSMOE_UNSLOTH.",
    "gradient_checkpointing": "env: MSMOE_GRAD_CKPT.",
    # -- literals in build_config -----------------------------------------
    "attn_impl": "constant in build_config ('sdpa').",
    "vllm_max_len": "constant in build_config (4096).",
    "vllm_quantization": "constant in build_config (None).",
    "packing_strategy": "constant in build_config ('wrapped').",
    "target_modules": "constant in build_config (the three MLP projections).",
    "lr_lora": "constant in build_config (2e-4). A release that retunes this "
               "changes every exported recipe's meaning without editing one.",
    "specialist_save_steps": "constant in build_config (200).",
    "chars_per_token_est": "constant in build_config (3.2). Feeds the corpus "
                           "size estimate, so it moves how much data is "
                           "collected.",
    "seed": "PipelineConfig default (42); nothing sets it from a recipe.",
    "code_prompt_templates": "PipelineConfig default. See the note in the "
                             "README about this field being unreachable.",
    "code_prompt_unnamed": "PipelineConfig default, same as above.",
    "code_prompt_unnamed_fraction": "PipelineConfig default, same as above.",
    # -- about this invocation, not about the recipe -----------------------
    "dryrun": "cli: --dryrun. It describes THIS run, not the recipe, and "
              "stamping it would hand someone a recipe that can only ever "
              "build the small rung.",
}


def recipe_path(field: str) -> str:
    """Where in a recipe this fingerprint field is written, or "".

    `readme` records the README knob-table cell, which is addressed by recipe
    block and key - so for every direct field that has one it IS the recipe
    path, and tests/test_bundle_stamp.py resolves all of them against the real
    Recipe dataclass rather than trusting that sentence. `recipe` overrides it
    for the handful where the two differ or where there is no README row.
    """
    knob = KNOBS.get(field)
    if knob is None:
        return ""
    return knob.recipe or knob.readme


def pinnable() -> Dict[str, str]:
    """Every fingerprint field a bundle can stamp, to its recipe path.

    DERIVED FIELDS ARE EXCLUDED, and that is the load-bearing half. A value
    computed from other values must be RECOMPUTED on the far box, not frozen:
    stamping `collect_token_target` would leave it fighting the
    `collect_headroom` it is supposed to follow, and the recipe has no key for
    it anyway. Stamp the inputs; let the outputs fall out. That is also why the
    build_id round-trips - the inputs are pinned, so the outputs re-derive to
    the same numbers.
    """
    return {name: recipe_path(name)
            for name, knob in KNOBS.items()
            if knob.derived_from is None and recipe_path(name)}
