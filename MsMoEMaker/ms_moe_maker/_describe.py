"""ms-moe-maker's identity card. STDLIB ONLY - nothing imported, nothing read.

Same contract and the same reason as every Seren service's `_describe`, even
though ms-moe-maker is NOT a Seren package and never imports one: `--describe` has to
answer on a half-installed tool, so it cannot need torch, pydantic, or yaml.
The moment you most want something to be able to say its own name is when its
install is broken.

ms-moe-maker is deliberately Seren-agnostic. No seren-* dependency, no assumption
that Lodestar exists, no Seren in the name. It is a pipeline for building a
specified mixture of experts, usable by someone who has never heard of any of
this. seren-theatre[stagehand] depends on ms-moe-maker; ms-moe-maker depends on nothing of
Chad's. Mandate is not ethos - the connection is opt-in from the Seren side,
and the run DIRECTORY is the only thing the two ever share.
"""
from __future__ import annotations

NAME = "ms-moe-maker"
DESCRIPTION = ("Build a mixture of experts from deliberately chosen "
               "specialists. Not a coding model - a coding model shaped like "
               "your stack.")

# The verbs. Named here so a front-end can offer them without parsing --help,
# and so `stagehand` can check the tool it forked speaks the version of the
# contract it expects. THIS IS THE CANONICAL LIST - __main__ derives its
# DESCRIBE from it rather than keeping a second copy, which is how `smoke` and
# `eval` came to exist in the CLI while this tuple still said three.
COMMANDS = ("init", "build", "smoke", "eval", "validate", "describe")

# What `eval` can be asked. Three different questions, deliberately separable:
#   routing - does each expert own its own ground? (the dead-expert claim)
#   quality - does it answer better than one expert alone? (needs an answer key)
#   experts - was there ever anything to route on? (divergence + cross-loss)
#
# `experts` answers the question the other two make you ask. A routing table at
# 1.00x enrichment reads as a router problem, and whether it IS one depends on
# whether the specialists differ and whether routing correctly lowers the loss.
# Those are separate measurements and they were three separate hand-run probes
# before they were a mode.
#
# Adding a mode is additive: a consumer that does not know it will not ask for
# it, and stagehand's existing --mode values keep meaning what they meant.
EVAL_MODES = ("routing", "quality", "experts", "all")

# The event vocabulary emitted under --json. A consumer that does not know an
# event kind must ignore it, so adding one is not a breaking change; removing
# or renaming one is.
EVENTS = ("started", "stage", "progress", "refused", "warning", "error", "done")

DESCRIBE = {
    "name": NAME,
    "kind": "pipeline",
    "description": DESCRIPTION,
    "commands": list(COMMANDS),
    "eval_modes": list(EVAL_MODES),
    "events": list(EVENTS),
    # The manifest schema this build writes into a run directory. A reader
    # (seren-theatre) can check compatibility before it trusts a file.
    "manifest_schema_version": 1,
    "recipe_schema_version": 1,
    # Requires nothing, of anyone. The heavy deps (torch, transformers,
    # datasets) belong to the PIPELINE it forks, not to this CLI - which is
    # what lets `ms-moe-maker validate` run on a laptop with no GPU and no CUDA.
    "requires": [],
}
