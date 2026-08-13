"""Recipe -> environment, and an honest account of what did NOT translate.

THE PROBLEM THIS SOLVES. `ms-moe build recipe.yaml` currently drives
`fraunkenstein_universal.py` by setting environment variables and forking it.
That script exposes sixteen env levers. A recipe declares far more than sixteen
things. So the naive wrapper - set what you can, run it, hope - would accept a
recipe saying `per_device_batch: 8`, run a build at 4, and report success. The
recipe would be a document that LOOKS authoritative and silently isn't.

That is the exact failure the script's own `[cfg]` banner exists to prevent.
Its comment says it plainly: a flag that silently does nothing looks exactly
like a flag that worked, and it has cost that project four runs and once a
23-hour job. Rebuilding that trap one layer up, in the thing whose entire
selling point is "hand this file to someone and they get your run", would be
the worst possible place to put it.

THE RULE. A recipe field is honoured, or the build REFUSES. Never ignored.

THE NUANCE THAT MAKES IT USABLE. Refusing on every unlevered field would
refuse every recipe, because most fields are things the script hardcodes to
sensible values the recipe agrees with. So the check is not "is there a lever"
but "will the run actually do what the document says". We read the script's own
module-level constants STATICALLY - via ast, never importing, because importing
that module pulls in torch and starts allocating - and compare each recipe
field against the value that will really be used. Agreement is silence. Only
DISAGREEMENT refuses.

Concretely: `per_device_batch: 4` runs clean because the script says 4.
`per_device_batch: 8` refuses, and says why, and names the constant.

WHY REFUSALS ARE A FEATURE. The refusal list is the carve roadmap. Each entry
is a field somebody wanted to set and couldn't, which is exactly the priority
order for pulling that part of the script into a real stage module. When the
list is empty, wrap-then-carve is finished and nobody had to guess when.

The refusals are also written into the run manifest, not just printed, because
the person who needs to know a lever was ignored is the one reading the
dashboard six hours later, not the one who saw the terminal at kickoff.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# The script's default entry point. Overridable, because the whole point of
# the carve is that this eventually stops being a script.
DEFAULT_PIPELINE = "fraunkenstein_universal.py"


@dataclass
class Translation:
    """The result of pointing a recipe at a pipeline."""

    env: Dict[str, str] = field(default_factory=dict)
    refusals: List[str] = field(default_factory=list)
    # Fields we checked and found already in agreement. Reported under --json
    # so a reader can tell "honoured because we set it" from "honoured because
    # it already matched", which are different kinds of guarantee.
    agreed: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals


# -- reading the pipeline's constants without importing it -------------------

_WANTED = {
    "MAX_SEQ_LENGTH", "PER_DEVICE_BATCH", "GRAD_ACCUM", "NUM_CODE_SAMPLES",
    "NUM_AGENT_SAMPLES", "NUM_EXPERTS_PER_TOK", "NORM_TOPK_PROB",
    "SHARED_EXPERT_WIDTH", "CODE_LANGUAGES", "MODEL_SIZES", "MAX_SHARDS",
}


def read_pipeline_constants(path: Path) -> Dict[str, Any]:
    """Statically evaluate the pipeline's simple module-level constants.

    ast, not import. Importing fraunkenstein_universal.py executes 800 lines of
    configuration, prints a banner, sets PYTORCH_CUDA_ALLOC_CONF and imports
    torch - none of which a validation pass has any business doing. It would
    also make `ms-moe build --check` cost a CUDA context.

    Only literal assignments are read. Anything computed (TARGET_STEPS, which
    is an env lookup, or BASE_MODEL, which is a conditional) is deliberately
    not resolvable here and is handled explicitly below where the logic can be
    stated rather than guessed at.
    """
    out: Dict[str, Any] = {}
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"cannot read pipeline constants from {path}: {exc}")

    for node in tree.body:              # module level only, on purpose
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in _WANTED:
                continue
            try:
                out[target.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass                    # computed, not literal - fine, skip
    return out


# -- the translation ---------------------------------------------------------

def translate(recipe: Any, pipeline: Path,
              force: bool = False) -> Translation:
    """Map a validated Recipe onto env levers, refusing what cannot be honoured.

    `recipe` is a ms_moe.recipe.Recipe. Typed loosely so this module has no
    import cycle with the recipe parser and stays testable with a stub.
    """
    consts = read_pipeline_constants(pipeline)
    t = Translation()

    def refuse(field_path: str, wanted: Any, actual: Any, lever: str) -> None:
        t.refusals.append(
            f"{field_path}={wanted!r} cannot be applied: the pipeline uses "
            f"{actual!r} from {lever} and exposes no environment lever for it. "
            f"Either change the recipe to {actual!r}, or carve this stage out "
            f"so the value comes from the recipe.")

    def agree(field_path: str) -> None:
        t.agreed.append(field_path)

    def check(field_path: str, wanted: Any, const_name: str) -> None:
        """Honour-by-agreement, or refuse."""
        if const_name not in consts:
            t.refusals.append(
                f"{field_path}={wanted!r} cannot be verified: {const_name} is "
                f"not a literal constant in {pipeline.name}, so there is no "
                f"way to confirm the run will honour it. Refusing rather than "
                f"assuming.")
            return
        actual = consts[const_name]
        if actual == wanted:
            agree(field_path)
        else:
            refuse(field_path, wanted, actual, const_name)

    # ── HONOURED: there is a real lever ────────────────────────────────────
    # size picks the rung and, through MODEL_SIZES, the base model.
    t.env["FRAUNK_SIZE"] = str(recipe.size)

    # target_steps is the token budget in disguise; the script derives
    # EXPERT_TOKEN_BUDGET from it.
    t.env["FRAUNK_TARGET_STEPS"] = str(recipe.budget.target_steps)

    # dense_layers: "auto" means the script decides (its own MLP_ONLY_LAYERS
    # default). An explicit list is a stitch-time lever the script reads from
    # the environment - but ONLY when no skeleton exists yet, which the script
    # itself warns about loudly. We pass it and let that warning stand.
    dense = getattr(recipe.moe, "dense_layers", "auto")
    if dense not in ("auto", None, ""):
        t.env["FRAUNK_DENSE_LAYERS"] = ",".join(str(x) for x in dense)

    if getattr(recipe.runtime, "direct_load", False):
        t.env["FRAUNK_DIRECT_LOAD"] = "1"

    # Read by torch at import, and the script setdefault()s it - so setting it
    # in the child's environment WINS, which is what we want, and is also the
    # only reason this works at all. Measured: 106.6 GB reserved vs 8.3 GB.
    alloc = getattr(recipe.runtime, "alloc_conf", "")
    if alloc:
        t.env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc)

    if force:
        t.env["FRAUNK_FORCE"] = "1"

    # ── CHECKED: no lever, but agreement is enough ─────────────────────────
    check("budget.max_seq_length", recipe.budget.max_seq_length,
          "MAX_SEQ_LENGTH")
    check("budget.per_device_batch", recipe.budget.per_device_batch,
          "PER_DEVICE_BATCH")
    check("budget.grad_accum", recipe.budget.grad_accum, "GRAD_ACCUM")
    check("moe.experts_per_tok", recipe.moe.experts_per_tok,
          "NUM_EXPERTS_PER_TOK")
    check("moe.norm_topk_prob", recipe.moe.norm_topk_prob, "NORM_TOPK_PROB")
    check("moe.shared_expert_width", recipe.moe.shared_expert_width,
          "SHARED_EXPERT_WIDTH")

    # The expert list. The script hardcodes CODE_LANGUAGES and appends
    # "agentcore"; a recipe naming a different set would silently train the
    # script's set instead. This is the single biggest thing the carve buys,
    # because "swap CODE_LANGUAGES and someone else gets their own Ms.MoE" is
    # the actual product.
    langs = consts.get("CODE_LANGUAGES")
    if langs is None:
        t.refusals.append(
            "experts cannot be verified: CODE_LANGUAGES is not a literal "
            "constant in the pipeline.")
    else:
        # The recipe names experts in lowercase safe-form; the script names
        # languages as the corpus spells them ("C#", "PowerShell").
        script_experts = [_safe(x) for x in langs] + ["agentcore"]
        want = [e.name for e in recipe.experts]
        if sorted(script_experts) == sorted(want):
            agree("experts")
        else:
            t.refusals.append(
                f"experts={want!r} cannot be applied: the pipeline trains "
                f"{script_experts!r} from its CODE_LANGUAGES constant and "
                f"exposes no lever for the expert list. This is the field the "
                f"decomposition should free first - a recipe that cannot "
                f"choose its own experts is not yet a factory.")

    # The base model, resolved the way the script resolves it. It is a
    # conditional on USE_PRE_ABLITERATED, so we cannot literal_eval it; but
    # MODEL_SIZES is a literal and the recipe's `base` should equal one of the
    # two entries for its size.
    sizes = consts.get("MODEL_SIZES") or {}
    pair = sizes.get(recipe.size)
    if pair is None:
        t.refusals.append(
            f"size={recipe.size!r} is not one of the pipeline's known sizes "
            f"{sorted(sizes)!r}.")
    elif recipe.base in tuple(pair):
        agree("base")
    else:
        t.refusals.append(
            f"base={recipe.base!r} cannot be applied: for size {recipe.size} "
            f"the pipeline can only use {tuple(pair)!r}, chosen by its "
            f"USE_PRE_ABLITERATED flag, and exposes no lever for an arbitrary "
            f"base model.")

    # ── GATES: no mechanism at all ─────────────────────────────────────────
    # The script runs start to finish. There is nowhere for a gate to sit yet -
    # that is precisely what "stages the gates can sit between" means and it is
    # the second thing the carve buys. `auto` is honest (nothing is gated);
    # `manual` is a promise we cannot keep, so it refuses.
    for name in ("base_evals", "main_evals"):
        value = getattr(recipe.gates, name, "auto")
        if value == "auto":
            agree(f"gates.{name}")
        else:
            t.refusals.append(
                f"gates.{name}={value!r} cannot be honoured: the pipeline runs "
                f"end to end with no stage boundary a gate could pause at. "
                f"Carve the stages out first, then this becomes real. Set it "
                f"to 'auto' to acknowledge the build is ungated.")

    # ── ROOTS: derived, not settable ───────────────────────────────────────
    # DATA_ROOT and OUTPUT_ROOT are computed from DRYRUN and MODEL_SIZE. The
    # recipe's templated `msmoe_{size}` cannot be expressed, so the runner
    # RESOLVES the real ones the same way the script does and reports them,
    # rather than pretending the recipe chose them.
    if getattr(recipe.roots, "output", None) not in (None, "", "msmoe_{size}"):
        t.refusals.append(
            f"roots.output={recipe.roots.output!r} cannot be applied: the "
            f"pipeline derives OUTPUT_ROOT from FRAUNK_DRYRUN and FRAUNK_SIZE "
            f"and exposes no lever. Leave it templated.")

    return t


def _safe(language: str) -> str:
    """The pipeline's language -> safe-name convention, quoted not guessed.

    'C#' -> 'csharp', 'PowerShell' -> 'powershell'. Kept here rather than
    imported because importing the pipeline is the thing this module exists to
    avoid; a test asserts the two agree on the real corpus names.
    """
    return {"c#": "csharp", "c++": "cpp"}.get(
        language.lower(), language.lower().replace(" ", "_"))


def resolved_roots(size: str, dryrun: bool) -> Dict[str, str]:
    """What the pipeline will ACTUALLY use, computed its way.

    Mirrors:
        DATA_ROOT   = "dryrun_data" if DRYRUN else "fraunkenstein_data"
        OUTPUT_ROOT = f"dryrun_{size}" if DRYRUN else f"fraunkenstein_agent_{size}"

    Duplicated logic is a liability, so a test pins these strings against the
    real source file. When the carve moves them, that test fails first.
    """
    return {
        "data": "dryrun_data" if dryrun else "fraunkenstein_data",
        "output": f"dryrun_{size}" if dryrun else f"fraunkenstein_agent_{size}",
    }
