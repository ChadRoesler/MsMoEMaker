"""Translation / refusal tracking — what the recipe asked for, what was honoured.

WHY THE REFUSAL LIST IS NOT DECORATION. It is the progress metric for the
carve. Every entry is a field a user wrote in a recipe that this build cannot
apply, which means the list going empty is how you know wrap-then-carve is
finished, and nobody had to guess when. That is also why it must never be
emptied by DELETING the mechanism - a refusal list that is empty because
nothing checks is indistinguishable from one that is empty because everything
works, and only one of those is good news.

It is also the honest half of a lenient parser. We accept a minimal recipe and
fill sensible defaults; the price of that kindness is saying out loud which of
the things you did write we are not going to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Translation:
    """Result of mapping a recipe onto pipeline config.

    env        — extra env vars to set for the run.
    refusals   — fields the recipe asked for that cannot be applied.
    agreed     — fields the recipe asked for that the build honours.
    force      — redo existing artifacts?
    """
    env: Dict[str, str] = field(default_factory=dict)
    refusals: List[str] = field(default_factory=list)
    agreed: List[str] = field(default_factory=list)
    force: bool = False

    @property
    def ok(self) -> bool:
        return not self.refusals


def translate(recipe: Any, force: bool = False,
              dryrun: bool = False) -> Translation:
    """Map a Recipe onto env levers, and say what we cannot honour.

    The env half only bites on the legacy subprocess path - the in-package
    builder reads the recipe directly through config.build_config. It is still
    computed and reported either way, because "what would this run set" is a
    question worth being able to answer before the run.
    """
    tr = Translation(force=force)

    def agree(field_name: str, value: Any) -> None:
        tr.agreed.append(f"{field_name}={value}")

    rt = getattr(recipe, "runtime", None)
    bud = getattr(recipe, "budget", None)
    moe = getattr(recipe, "moe", None)

    if rt is not None:
        if getattr(rt, "alloc_conf", ""):
            # Must be set BEFORE torch initialises or it does nothing at all.
            # In a child process the environment is simply correct at exec
            # time; in-process we would be racing our own import graph.
            tr.env["PYTORCH_CUDA_ALLOC_CONF"] = rt.alloc_conf
            agree("runtime.alloc_conf", rt.alloc_conf)
        if getattr(rt, "load_in_4bit", False):
            tr.env["MSMOE_LOAD_IN_4BIT"] = "1"
            agree("runtime.load_in_4bit", True)
        if getattr(rt, "direct_load", False):
            tr.env["MSMOE_DIRECT_LOAD"] = "1"
            agree("runtime.direct_load", True)
        if getattr(rt, "hardware_tier", ""):
            tr.env["MSMOE_TIER"] = rt.hardware_tier
            agree("runtime.hardware_tier", rt.hardware_tier)
        if getattr(rt, "precision", "") not in ("", "float16"):
            tr.refusals.append(
                f"runtime.precision={rt.precision!r} cannot be honoured - the "
                f"stitch and router stages assume float16 weights")

    if bud is not None and getattr(bud, "target_steps", None):
        tr.env["MSMOE_TARGET_STEPS"] = str(bud.target_steps)
        agree("budget.target_steps", bud.target_steps)

    if moe is not None:
        dense = getattr(moe, "dense_layers", "auto")
        if dense != "auto":
            tr.env["MSMOE_DENSE_LAYERS"] = ",".join(str(x) for x in dense)
            agree("moe.dense_layers", dense)

    if dryrun:
        tr.env["MSMOE_DRYRUN"] = "1"

    gates = getattr(recipe, "gates", None)
    if gates is not None and getattr(gates, "main_evals", "") == "manual":
        # The honest refusal, and the original one: the pipeline runs end to
        # end with no stage boundary a gate could pause at. Keeping it named
        # is what makes "the list is empty" mean something later.
        tr.refusals.append(
            "gates.main_evals='manual' cannot be honoured - the build runs end "
            "to end with no boundary a gate could pause at. Run "
            "`ms-moe-maker eval` yourself after the build instead.")

    return tr
