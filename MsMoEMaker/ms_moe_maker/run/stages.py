"""The stage vocabulary - what a Ms.MoE build is made of.

These ids are a PUBLIC CONTRACT. seren-theatre paints from them, a future
form-in-the-viewer will offer gates against them, and anything reading the
--json event stream keys off them. Renaming one is a breaking change; adding
one is not.

WHERE THEY CAME FROM. Not invented. `fraunkenstein_universal.py` already has
these boundaries and has had them for a long time - they are the eight places
it calls `_done(path, what)`, the function that decides whether a stage can be
skipped because its artifact is already on disk. That function is the resume
mechanism, which means the pipeline ALREADY agrees these are the points where
work can be considered finished. Naming them here doesn't impose a structure,
it writes down the one that was load-bearing already.

Which is also the safety argument for wrap-then-carve: because the boundaries
were derived from the resume logic rather than from taste, carving the script
into modules later cannot move them without breaking resume, and breaking
resume is loud.

ORDER IS MEANINGFUL. `ORDER` is the sequence the orchestrator runs, so a
viewer can render "3 of 9" without being told, and a reader can tell "not
started yet" from "skipped" by position. The per-expert stages are templated
because the expert list comes from the recipe - five experts means five
finetune stages, and a recipe with three means three.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# -- fixed stages ------------------------------------------------------------

PREFLIGHT = "preflight"
# Decensor the base with the vendored Heretic core, before any specialist
# trains from it. Optional; gated by recipe.abliterate.enabled.
ABLITERATE_BASE = "abliterate.base"
# WAS data.code / data.agent. Renamed while the only consumer was seren-theatre
# and no stranger's recipe existed yet, because this vocabulary is a public
# contract and the old spelling had a DOMAIN ASSUMPTION baked into it.
#
# Nothing between the corpus and the GGUF knows what the text means - tokenise,
# finetune, stitch, router-train, export and smoke are all domain-blind. A
# Ms.MoE built to drill D&D lore, or exam material, or case law, runs the exact
# same eight stages. Shipping ids that call the corpus "code" would have told
# every one of those people they were using someone else's tool, permanently,
# in a wire format that says at the top of this file that renaming is a
# breaking change.
#
# Cheapest it will ever be was the day it was noticed. This is that day.
DATA_CORPUS = "data.corpus"
DATA_SYNTH = "data.synth"
# A GATE, NOT A PROBE, AND THE DIFFERENCE IS WHICH SIDE OF THE WORK IT SITS ON.
# See the note in plan() about why the evals stay out of this list. This one
# is allowed in for the same reason preflight is: it grades the build's INPUTS
# (are these specialists actually different, and is there anything for a router
# to learn from them) before the expensive stages spend a night finding out.
GATE_EXPERTS = "gate.experts"
STITCH = "stitch"
ROUTER = "router"
EXPORT_GGUF = "export.gguf"

# Templated per expert: finetune.python, finetune.powershell, finetune.bestiary
FINETUNE_PREFIX = "finetune."

LABELS: Dict[str, str] = {
    PREFLIGHT: "Preflight - stamp the levers, check the disk",
    ABLITERATE_BASE: "Abliterate the base model",
    DATA_CORPUS: "Collect the expert corpora",
    DATA_SYNTH: "Generate the synthetic corpora",
    GATE_EXPERTS: "Check the experts diverged and can be routed between",
    STITCH: "Stitch the MoE skeleton",
    ROUTER: "Train the router",
    EXPORT_GGUF: "Export GGUF and smoke-test it",
}


def finetune_id(expert: str) -> str:
    return f"{FINETUNE_PREFIX}{expert}"


def finetune_label(expert: str) -> str:
    return f"Fine-tune the {expert} specialist"


def label_for(stage_id: str) -> str:
    if stage_id in LABELS:
        return LABELS[stage_id]
    if stage_id.startswith(FINETUNE_PREFIX):
        return finetune_label(stage_id[len(FINETUNE_PREFIX):])
    return stage_id


def plan(experts: List[str],
         synth: Sequence[str] = (),
         gates: bool = True,
         abliterate: bool = False) -> List[Tuple[str, str]]:
    """The full ordered stage list for a build of these experts.

    Returns [(id, label), ...] in execution order, mirroring the orchestrator
    at the bottom of the pipeline:

        preflight -> gather corpora -> generate synthetic corpora
                  -> finetune each specialist
                  -> stitch skeleton -> train router -> export GGUF

    `gates=False` drops the gate stages, and exists for ONE caller: the runner
    that wraps the legacy `fraunkenstein_universal.py`. That script predates
    the gate and cannot emit it, so planning it there would seed a stage
    nothing can ever close - and "a stage that never reaches a terminal state"
    is precisely the failure the runner's own tests exist to catch. Better to
    say out loud that this pipeline has no gate than to let the manifest carry
    one that hangs forever.

    `synth` is the experts whose source has to be GENERATED rather than
    downloaded, and it is a parameter because it used to be the literal check
    `if "agentcore" in experts`. That worked exactly as long as the only
    generated expert in the world was Chad's MCP-trace one. A recipe with a
    synth expert called anything else got no data.synth stage at all, so the
    longest phase of its build was invisible in the viewer - the failure being
    silence, which is the worst shape for it.

    The recipe already knows this (`source.kind == "synth"`), so the caller
    passes what it knows instead of this module guessing from a name.

    The probes and evals are deliberately NOT here. They are separate commands
    a person runs afterward to answer separate questions, and folding them in
    would make the build the thing that grades itself. Keeping the evidence
    layer outside the pipeline is the same reason SerenProbe is its own
    service.

    GATE_EXPERTS IS THE ONE EXCEPTION, AND THE RULE ABOVE IS WHY IT QUALIFIES.
    The objection to folding evidence in is self-grading: a build that decides
    whether its own output is good has marked its own homework. A gate that
    measures its INPUTS has not.

        preflight     grades the machine    before we use it
        gate.experts  grades the specialists before we stitch them
        eval          grades the MoE        after it exists      <- self-grading

    The first two answer "should we proceed"; only the third answers "was it
    any good". eval stays outside. This does not, because the alternative is
    learning that two experts were interchangeable after paying for a stitch,
    a router train and a GGUF export - which is exactly how it was learned the
    first time.
    """
    out: List[Tuple[str, str]] = [
        (PREFLIGHT, LABELS[PREFLIGHT]),
    ]
    if abliterate:
        out.append((ABLITERATE_BASE, LABELS[ABLITERATE_BASE]))
    out.append((DATA_CORPUS, LABELS[DATA_CORPUS]))
    if synth:
        out.append((DATA_SYNTH, LABELS[DATA_SYNTH]))
    for expert in experts:
        out.append((finetune_id(expert), finetune_label(expert)))
    if gates and len(experts) >= 2:
        out.append((GATE_EXPERTS, LABELS[GATE_EXPERTS]))
    out.extend([
        (STITCH, LABELS[STITCH]),
        (ROUTER, LABELS[ROUTER]),
        (EXPORT_GGUF, LABELS[EXPORT_GGUF]),
    ])
    return out


# The artifact each stage produces, relative to the run (output) root. These
# are the SAME paths `_done()` tests, which is what lets a reader confirm a
# manifest against the disk instead of trusting it - and what lets the runner
# recognise that a stage was skipped rather than run.
#
# THE NAMES ARE OURS NOW. They used to be `qwen_coder_{expert}`,
# `fraunkenstein_moe_untrained` and `fraunkenstein_agent_final`, inherited from
# the script this tool was carved out of. Both halves of that were wrong for a
# published tool, for the same reason `data.code` was renamed to `data.corpus`
# further up this file:
#
#   * `qwen_coder_` bakes in a MODEL FAMILY and a DOMAIN. A recipe that builds
#     a Monster Manual expert on a non-Qwen base wrote its output to a
#     directory calling itself a Qwen coder, which tells that user they are
#     using someone else's tool - permanently, on their own disk.
#   * `fraunkenstein_` is a project name from a repo the installer does not
#     ship and the user has never heard of.
#
# `specialist_` is what the code already calls them everywhere
# (fine_tune_specialist, specialist_dirs), so this is the codebase agreeing
# with itself rather than a new coinage.
#
# Renaming is a breaking change for anything holding old paths - which, this
# early and with no stranger's run on disk, costs one rebuild and buys every
# future user a directory that is about their project instead of ours.
ARTIFACTS: Dict[str, str] = {
    STITCH: "moe_untrained",
    ROUTER: "moe_trained",
}
FINETUNE_ARTIFACT = "specialist_{expert}"


def artifact_for(stage_id: str) -> str | None:
    if stage_id in ARTIFACTS:
        return ARTIFACTS[stage_id]
    if stage_id.startswith(FINETUNE_PREFIX):
        return FINETUNE_ARTIFACT.format(
            expert=stage_id[len(FINETUNE_PREFIX):])
    return None


# ── the substitution the repeated stages take ───────────────────────────────
#
# Published so a consumer can expand `finetune.{expert}` without parsing braces
# out of an id, and built from the parameter name rather than beside it, so the
# two can never disagree about what the placeholder is called.
EXPERT_PARAMETER = "expert"
EXPERT_TEMPLATE = "{" + EXPERT_PARAMETER + "}"


def _row(stage_id: str,
         optional: bool = False,
         parameter: str | None = None) -> Dict[str, Any]:
    """One published row, with label and artifact read through the real lookups.

    Deliberately NOT a literal per row. Every row - the templated one included -
    goes through label_for() and artifact_for(), which are the same two
    functions the runner and the viewer use. A published payload assembled from
    hand-written strings is a second source of truth that starts out correct;
    this one cannot drift from the constants above because it does not hold a
    copy of them.
    """
    return {
        "id": stage_id,
        "label": label_for(stage_id),
        "artifact": artifact_for(stage_id),
        "optional": optional,
        "parameter": parameter,
    }


def describe() -> List[Dict[str, Any]]:
    """This pipeline's stage vocabulary, in execution order.

    WHY IT IS BUILT HERE AND NOT IN box/describe.py. `--describe` is stdlib-only
    so it can answer on a half-installed tool, which rules out importing this
    module over there - but a hand-written copy of this list in the identity
    card would be a second source of truth, the exact mistake `COMMANDS` was
    moved OUT of __main__ to fix. So the payload is built beside the constants
    it describes, and __main__ asks for it the way it already asks
    corpus.describe() and validators.describe(). This module is stdlib-only
    too; only its package __init__ is heavy, which is why the caller degrades.

    THE ORDER IS PART OF THE ANSWER. These rows are the sequence plan() emits,
    with the per-expert block in its real slot, so a front-end can draw the
    whole shape of a build before the first stage opens - and say "3 of 9"
    honestly instead of counting stages as they arrive. plan() remains the code
    that actually runs; test_stage_vocabulary pins this list against what plan()
    returns, so a reordering there cannot leave this quietly wrong. That is a
    wiring check, not two literals agreeing with each other.

    EVERY ROW CARRIES EVERY KEY, so a consumer never branches on which fields a
    row happens to have:

        id         the stage id, or a {parameter} template for repeated ones
        label      the sentence a human reads
        artifact   the directory it leaves in the run root, or None
        optional   whether a build can legitimately skip it
        parameter  the name substituted into id/label/artifact, or None

    `optional` is the key worth having and the one a consumer would otherwise
    guess wrong: a build with one expert emits no gate.experts and a recipe with
    no generated source emits no data.synth. Without this, a viewer drawing a
    plan from the vocabulary reports two stages as missing forever.

    Adding a row is additive under the same rule as EVENTS - a consumer that
    does not know a stage must ignore it. Renaming or removing one is not.
    """
    return [
        _row(PREFLIGHT),
        _row(ABLITERATE_BASE, optional=True),
        _row(DATA_CORPUS),
        _row(DATA_SYNTH, optional=True),
        _row(finetune_id(EXPERT_TEMPLATE), parameter=EXPERT_PARAMETER),
        _row(GATE_EXPERTS, optional=True),
        _row(STITCH),
        _row(ROUTER),
        _row(EXPORT_GGUF),
    ]
