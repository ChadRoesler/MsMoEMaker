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

# The three verbs. Named here so a front-end can offer them without parsing
# --help, and so `stagehand` can check the tool it forked speaks the version of
# the contract it expects.
COMMANDS = ("build", "validate", "describe")

# The event vocabulary emitted under --json. A consumer that does not know an
# event kind must ignore it, so adding one is not a breaking change; removing
# or renaming one is.
EVENTS = ("started", "stage", "progress", "refused", "warning", "error", "done")

DESCRIBE = {
    "name": NAME,
    "kind": "pipeline",
    "description": DESCRIPTION,
    "commands": list(COMMANDS),
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
