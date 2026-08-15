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

from typing import Dict, List, Sequence, Tuple

# -- fixed stages ------------------------------------------------------------

PREFLIGHT = "preflight"
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
STITCH = "stitch"
ROUTER = "router"
EXPORT_GGUF = "export.gguf"

# Templated per expert: finetune.python, finetune.powershell, finetune.bestiary
FINETUNE_PREFIX = "finetune."

LABELS: Dict[str, str] = {
    PREFLIGHT: "Preflight - stamp the levers, check the disk",
    DATA_CORPUS: "Collect the expert corpora",
    DATA_SYNTH: "Generate the synthetic corpora",
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
         synth: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """The full ordered stage list for a build of these experts.

    Returns [(id, label), ...] in execution order, mirroring the orchestrator
    at the bottom of the pipeline:

        preflight -> gather corpora -> generate synthetic corpora
                  -> finetune each specialist
                  -> stitch skeleton -> train router -> export GGUF

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
    """
    out: List[Tuple[str, str]] = [
        (PREFLIGHT, LABELS[PREFLIGHT]),
        (DATA_CORPUS, LABELS[DATA_CORPUS]),
    ]
    if synth:
        out.append((DATA_SYNTH, LABELS[DATA_SYNTH]))
    for expert in experts:
        out.append((finetune_id(expert), finetune_label(expert)))
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
# `{expert}` is substituted for per-expert stages. The `qwen_coder_` prefix is
# the pipeline's own naming and is quoted here rather than reimplemented,
# because when the carve renames it, THIS is the one place that has to change
# and a test will say so.
ARTIFACTS: Dict[str, str] = {
    STITCH: "fraunkenstein_moe_untrained",
    ROUTER: "fraunkenstein_agent_final",
}
FINETUNE_ARTIFACT = "qwen_coder_{expert}"


def artifact_for(stage_id: str) -> str | None:
    if stage_id in ARTIFACTS:
        return ARTIFACTS[stage_id]
    if stage_id.startswith(FINETUNE_PREFIX):
        return FINETUNE_ARTIFACT.format(
            expert=stage_id[len(FINETUNE_PREFIX):])
    return None
