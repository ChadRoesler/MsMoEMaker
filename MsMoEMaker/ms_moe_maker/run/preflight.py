"""Preflight — find out what will go wrong BEFORE the expensive part.

WHAT PREFLIGHT USED TO BE. A print of the config stamp and free disk, then
`cb.stage(PREFLIGHT, done, "config stamped")`. Nothing was checked. Every
failure that could have been caught in two seconds was instead discovered at
the stage that needed it:

    base model missing / gated / typo'd  -> stage 3, after corpus collection
    torch not installed                  -> stage 3
    llama.cpp absent                     -> stage 6, after ALL the training
    output root not writable             -> stage 3
    local corpus path does not exist     -> stage 1, but as a bare traceback

The llama.cpp one is the whole argument by itself. A user finishes six hours of
fine-tuning, the stitch lands, the router trains, and THEN the run reports that
the converter was never on the box. Everything before it was correct and none
of it was wasted, which is exactly what makes it infuriating: the answer was
knowable before a single token was read.

TWO SEVERITIES, AND THE DIFFERENCE MATTERS.

  FAIL  the build cannot succeed. Stop now, having spent nothing.
  WARN  the build can succeed but will do less than the recipe asks - a
        missing llama.cpp means no GGUF, which is a real result for someone
        who only wants the HF checkpoint.

A warning that stops the build is a failure wearing a friendly word, and a
failure reported as a warning is worse. So they are separate, and only FAIL
stops anything.

EVERY FAILURE CARRIES ITS REMEDY. "base model not found" is a fact; "base model
not found - check the id, or run `huggingface-cli login` if it is gated" is an
answer. The person reading this is usually about to lose an evening, and the
difference between those two strings is whether they lose it.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..eval.record import FAIL, PASS, UNMEASURABLE

WARN = "warn"


@dataclass
class Check:
    """One preflight answer."""
    name: str
    status: str            # pass | warn | fail | unmeasurable
    detail: str = ""
    remedy: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


@dataclass
class Preflight:
    checks: List[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "",
            remedy: str = "") -> None:
        self.checks.append(Check(name, status, detail, remedy))

    @property
    def ok(self) -> bool:
        return not any(c.blocking for c in self.checks)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]


def _check_training_stack(pf: Preflight) -> bool:
    """Can this box train at all? Returns True if torch imported."""
    try:
        import torch
    except ImportError:
        pf.add("training stack", FAIL,
               "torch is not installed",
               "pip install 'ms-moe-maker[train]' on the box that builds. "
               "`validate` and `build --plan` do not need it.")
        return False

    try:
        import transformers  # noqa: F401
    except ImportError:
        pf.add("training stack", FAIL,
               "transformers is not installed",
               "pip install 'ms-moe-maker[train]'")
        return False

    try:
        from safetensors import safe_open  # noqa: F401
    except ImportError:
        # Named separately because it is the one people miss: the stitcher
        # streams tensors out of safetensors files one at a time, and that is
        # what keeps peak memory at one shard instead of the whole model.
        pf.add("training stack", FAIL,
               "safetensors is not installed",
               "pip install safetensors — the stitcher streams from "
               "safetensors, it is not an optional accelerator")
        return False

    cuda = torch.cuda.is_available()
    pf.add("training stack", PASS if cuda else WARN,
           f"torch {torch.__version__}, CUDA "
           f"{'available' if cuda else 'NOT available'}",
           "" if cuda else "training on CPU is possible and very slow; check "
                           "your CUDA install if this box has a GPU")
    return True


def _check_base_model(pf: Preflight, config, offline: bool = False) -> None:
    """Is the base model actually reachable? Stage 3 is too late to ask."""
    base = config.base
    if not base:
        pf.add("base model", FAIL, "no base model resolved",
               "set `base:` in the recipe or leave `size:` so a tier default "
               "can fill it")
        return

    # THE ARCHITECTURE GUARD, ON THE RESOLVED BASE. validate() only checks
    # `recipe.base`; an env override (MSMOE_BASE_MODEL) or a box defaults
    # entry (`models:`) can swap in a non-Qwen checkpoint that validate never
    # sees - and the recipe check's own comment names the failure: every
    # specialist trains, then the stitch dies at stage 4. This is the same
    # check, asked of the id the run will ACTUALLY use, before anything
    # expensive starts.
    from ..config.pipeline import SUPPORTED_BASE_HINTS, SUPPORTED_MOE_ARCHS
    if os.path.isdir(base):
        low = os.path.basename(base).lower()
    else:
        low = base.lower()
    if not any(hint in low for hint in SUPPORTED_BASE_HINTS):
        pf.add("base model", FAIL,
               f"{base} is not a supported MoE architecture",
               f"supported today: {', '.join(sorted(SUPPORTED_MOE_ARCHS.values()))}. "
               f"Point `base:` (or MSMOE_BASE_MODEL / the box's `models:`) at "
               f"a Qwen2 family checkpoint, or leave it empty for a supported "
               f"default.")
        return

    if os.path.isdir(base):
        pf.add("base model", PASS, f"local path {base}")
        return

    if offline:
        pf.add("base model", UNMEASURABLE, f"{base} (offline, not checked)")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        pf.add("base model", UNMEASURABLE,
               f"{base} — huggingface_hub not installed, cannot check",
               "pip install 'ms-moe-maker[train]'")
        return

    try:
        HfApi().model_info(base)
        pf.add("base model", PASS, f"{base} is reachable")
    except Exception as exc:            # noqa: BLE001 - any failure is the same answer
        name = exc.__class__.__name__
        remedy = ("check the repo id for typos; if it is gated, accept the "
                  "licence on the model page and run `huggingface-cli login`")
        if "Connection" in name or "Timeout" in name:
            remedy = ("no network to huggingface.co — pre-cache the model, or "
                      "point `base:` at a local directory")
        pf.add("base model", FAIL, f"{base} — {name}: {str(exc)[:120]}", remedy)


def _check_datasets(pf: Preflight, recipe, offline: bool = False) -> None:
    """Do the corpora exist where the recipe says they do?

    THE SAME CHECK AS THE BASE MODEL, AND FOR THE SAME REASON: to say "you did
    not misspell it" before anything expensive starts. It was missing, and that
    is exactly what bit a real run - preflight confirmed the base model was
    reachable, passed, and then stage 1 went looking for a dataset repo and
    died there instead.

    A dead or renamed dataset id is not hypothetical. The default base model
    this tool shipped with had already been deleted upstream; ids rot, and the
    only question is whether you find out in two seconds or after preflight has
    told you everything is fine.
    """
    if offline:
        pf.add("datasets", UNMEASURABLE, "offline, not checked")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        pf.add("datasets", UNMEASURABLE,
               "huggingface_hub not installed, cannot check")
        return

    api = HfApi()
    wanted: Dict[str, str] = {}          # repo id -> which expert wants it
    for e in recipe.experts:
        src = getattr(e, "source", None)
        kind = getattr(src, "kind", "") if src else ""
        if kind == "hf" and getattr(src, "repo", ""):
            wanted[src.repo] = e.name
        elif kind == "stack":
            from ..data.synth import STACK_REPO
            wanted.setdefault(STACK_REPO, e.name)

    for repo, who in wanted.items():
        try:
            api.dataset_info(repo)
            pf.add(f"dataset/{repo}", PASS, f"reachable (for {who})")
        except Exception as exc:        # noqa: BLE001
            name = exc.__class__.__name__
            remedy = ("check the repo id for typos; if it is gated, accept the "
                      "terms on its page and run `huggingface-cli login`")
            if "Connection" in name or "Timeout" in name:
                remedy = ("no network to huggingface.co - pre-cache the corpus "
                          "or use a `local` source")
            pf.add(f"dataset/{repo}", FAIL,
                   f"{name}: {str(exc)[:140]}", remedy)


def _check_roots(pf: Preflight, config) -> None:
    """Writable, and enough room. Both are cheap to ask and awful to discover.

    THE CORPUS VOLUMES COUNT TOO. This used to check data_root and
    output_root only, while a 45 GB shard scan fills shard_cache and the base
    weights land in hf_home - preflight passed green and stage 1 filled the
    volume, leaving the half-written corpus 4.7's skip path then trusted.
    """
    for label, path in (("data root", config.data_root),
                        ("output root", config.output_root),
                        ("shard cache", getattr(config, "shard_cache", "")),
                        ("hf cache", getattr(config, "hf_home", ""))):
        if not path:
            continue
        # PROBE THE NEAREST EXISTING PARENT, do not create the directory.
        # Preflight runs under `--plan` too, and a command whose whole promise
        # is "resolve everything, run nothing" must not leave run directories
        # scattered around someone's home folder as the price of telling them
        # what would happen.
        probe_dir = os.path.abspath(path)
        while probe_dir and not os.path.isdir(probe_dir):
            parent = os.path.dirname(probe_dir)
            if parent == probe_dir:
                break
            probe_dir = parent

        # ASK, DO NOT POKE. This used to write a probe file and delete it,
        # which is a stronger test and the wrong trade. Some perfectly usable
        # filesystems refuse to create a dotfile or refuse the unlink - network
        # mounts and bridged volumes especially - and the probe then reported
        # "not writable" for a directory that builds fine.
        #
        # The two failure modes are not symmetric. A false NO blocks a build
        # that would have worked, at the very first stage, with a wrong reason.
        # A false YES costs nothing: the build fails later with the real errno
        # from the real write. So prefer the check that cannot produce the
        # expensive mistake, and leave nothing behind either way.
        if not os.access(probe_dir, os.W_OK):
            pf.add(label, FAIL, f"{path} is not writable",
                   "check permissions, or set roots: in the recipe")
            continue

        try:
            free_gb = shutil.disk_usage(probe_dir).free / 2 ** 30
        except OSError:
            pf.add(label, UNMEASURABLE, f"{path} (free space unreadable)")
            continue

        # Rough floor. A specialist per expert plus the stitched MoE plus a
        # GGUF is many times the base model, and running out mid-stitch leaves
        # a half-written checkpoint that _done() may then treat as finished.
        # The shard cache gets the corpus estimate instead: each shard is
        # ~0.57 GB and max_shards caps how many the scan may pull.
        if label == "shard cache":
            need = getattr(config, "max_shards", 0) * 0.57 + 1
        else:
            need = _estimated_gb(config)
        if free_gb < need:
            remedy = ("free space or move the roots to a bigger volume. "
                      "Running out mid-stitch can leave a partial checkpoint "
                      "that looks finished to a resume.")
            if label == "shard cache":
                remedy = ("free space, move the roots, or lower "
                          "corpus.max_shards - each shard is ~0.57 GB and a "
                          "half-downloaded shard corrupts the next scan's "
                          "resume.")
            pf.add(label, FAIL,
                   f"{path}: {free_gb:.0f} GB free, ~{need:.0f} GB needed",
                   remedy)
        else:
            pf.add(label, PASS, f"{path}: {free_gb:.0f} GB free "
                                f"(~{need:.0f} GB estimated)")


def _estimated_gb(config) -> float:
    """A floor, not a forecast. Deliberately crude and deliberately stated."""
    try:
        billions = float(str(config.size).rstrip("Bb") or 1)
    except ValueError:
        billions = 1.0
    n = max(len(config.expert_names or []), 1)
    per_model = billions * 2          # bf16 bytes per param, in GB
    return per_model * (n + 2) + 5    # specialists + stitched + trained + slack


def _check_exporter(pf: Preflight, config) -> None:
    """llama.cpp — a WARNING, because the checkpoint is still a real result.

    TWO BINARIES, TWO ANSWERS. The converter and llama-cli are separate
    things and can be separately missing: a checkout with
    convert_hf_to_gguf.py but no built binaries converts a GGUF and then
    cannot prove it generates. export.py is explicit that "converted" and
    "smoke-passed" are different states, so preflight should not collapse
    them into one green tick.
    """
    from ..moe.export import resolve_llama_binary

    conv = os.path.join(config.llama_cpp_dir, "convert_hf_to_gguf.py")
    cli = resolve_llama_binary(config.llama_cpp_dir, "llama-cli")

    if os.path.exists(conv) and cli:
        pf.add("gguf exporter", PASS, f"{config.llama_cpp_dir} (+ {cli})")
        return
    if os.path.exists(conv) and not cli:
        pf.add("gguf exporter", WARN,
               f"converter found, but no llama-cli under "
               f"{config.llama_cpp_dir}",
               "the GGUF will convert and the smoke test will be skipped - "
               "so it ships UNPROVEN outside Python, which is where this "
               "project's nastiest bugs live. Build it: cmake --build build "
               "--config Release -j")
        return
    pf.add("gguf exporter", WARN,
           f"convert_hf_to_gguf.py not found under {config.llama_cpp_dir}",
           "the build will finish and the export stage will be skipped — you "
           "still get the HF checkpoint. For a GGUF: git clone --depth 1 "
           "https://github.com/ggml-org/llama.cpp and set MSMOE_LLAMA_CPP")


def _needs_a_teacher(config) -> bool:
    """Will this build stand a teacher model up at all?

    Three ways to acquire synth work, and a recipe with none of them never
    constructs a teacher - so demanding its serving stack would be a
    preflight failure about a stage that is not in the plan.
    """
    return bool(getattr(config, "synth_experts", None)
                or getattr(config, "tools_expert_name", "")
                or getattr(config, "reasoning_expert_name", "")
                or getattr(config, "reasoning_experts", None))


def _check_generator(pf: Preflight, config) -> None:
    """Which stack serves the teacher, and is it actually here?

    THE ASYMMETRY THIS FIXES. llama.cpp has had a check since the start and
    it is a WARNING, correctly: a missing exporter costs you the GGUF and
    you still keep the checkpoint, which is a real result. vLLM had NO
    check at all, and `from vllm import LLM` sat bare inside the teacher's
    constructor - so a box without it got a ModuleNotFoundError in the
    synth stage, which on a real gauntlet is roughly fifty minutes past
    preflight, past abliterate and past corpus collection, on a booked GPU.

    FAIL, NOT WARN, and the difference from llama.cpp is the whole point:
    there is no degraded result here. Quietly falling back to transformers
    would produce a corpus generated at batch 96 by a different sampler,
    under a build_id whose fingerprint says vLLM - the artifact and its
    claim would disagree, which is worse than refusing.

    find_spec, not import: preflight is the laptop answer and must not pull
    a serving stack and a CUDA context in to decide whether one exists.

    It reports on the plain path too. "transformers at batch 96" is the
    answer to "why is synth taking three hours", said in the place somebody
    is already looking.
    """
    import importlib.util

    if not _needs_a_teacher(config):
        return

    # TWO BATCH FIELDS, AND ONLY ONE IS READ ON EACH PATH. _HFTeacher takes
    # config.teacher_batch, _VLLMTeacher takes config.vllm_batch - which
    # both happen to say 512 under vLLM today, so reporting the wrong one
    # would print the right number for the wrong reason and diverge the
    # first time somebody tuned one of them.
    if not getattr(config, "use_vllm", False):
        pf.add("teacher", PASS,
               f"transformers, batch {getattr(config, 'teacher_batch', 0)}")
        return

    if importlib.util.find_spec("vllm") is None:
        pf.add("teacher", FAIL,
               "runtime.use_vllm is on and vllm is not installed",
               "pip install vllm on the box that builds, or set "
               "`runtime: {use_vllm: false}` in the recipe. Falling back "
               "silently is not offered: use_vllm moves the teacher batch "
               "from 96 to 512 and is part of the build fingerprint, so a "
               "corpus made without it would not be the corpus this "
               "build_id describes.")
        return
    pf.add("teacher", PASS,
           f"vLLM, batch {getattr(config, 'vllm_batch', 0)}")


def _budget_word(value: int) -> str:
    """A budget as a person reads it. 0 is a word, not a number."""
    return "unbounded" if int(value or 0) == 0 else f"{int(value)}"


def _check_budgets(pf: Preflight, config, recipe=None) -> None:
    """WHICH CEILING EACH GENERATOR WILL ACTUALLY WRITE UNDER.

    This exists because the answer was only discoverable from a running
    build, and only then if you happened to be tailing the log. The domain
    loop hit its cap on 66% of generations in a real gauntlet; the NOTE that
    said so fired seven times, named a knob, and by then the run was hours
    deep and the corpus already biased - because a generation cut at the cap
    is DISCARDED, so a low ceiling does not shorten that corpus, it selects
    the material short enough to fit and throws the rest away.

    Three numbers on one line, before anything is booked. No routing logic
    here on purpose: which loops a run uses is the builder's decision and
    reimplementing it would be a second copy that drifts. Printing all three
    costs one line and cannot be wrong.

    THE TRAP THIS WARNS ABOUT IS SPECIFIC AND EASY TO WALK INTO. On the vLLM
    path, "unbounded" is not "as much as it takes" - it is max_model_len,
    which is `runtime.vllm_max_len`, and that defaults to 4096. So a recipe
    carrying `reasoning_teacher_max_new: 4096` alongside `vllm_max_len: 4096`
    gets LESS room by asking for unbounded than by naming the number, because
    the prompt comes out of the same window. Silently. That is the reverse of
    what the setting reads like, and the only place to catch it is here.
    """
    if not _needs_a_teacher(config):
        return

    from ..box.describe import OVERRUN_CLOSE_AND_ANSWER, OVERRUN_DISCARD
    from ..config import pipeline as cfg_module

    agent = getattr(config, "agent_teacher_max_new", 0)
    domain = getattr(config, "domain_teacher_max_new", 0)
    reason = getattr(config, "reasoning_teacher_max_new", 0)
    pf.add("teacher budgets", PASS,
           f"agent {_budget_word(agent)}, domain {_budget_word(domain)}, "
           f"reasoning {_budget_word(reason)} tokens per generation")

    # ── what happens when one of those is not enough ──────────────────────
    policy = getattr(config, "teacher_overrun", OVERRUN_DISCARD)
    asked_reserve = getattr(config, "teacher_answer_reserve", -1)
    if policy == OVERRUN_DISCARD:
        pf.add("overrun policy", PASS,
               "discard - a generation that hits its cap is thrown away "
               "(budget.teacher_overrun)")
    else:
        # THE RESERVE IS PER LOOP, and printing one number would be a lie.
        # It is a quarter of whichever budget that loop is spending, so the
        # three differ whenever the budgets do - which is now the normal case.
        reserves = ", ".join(
            f"{name} {cfg_module.answer_reserve(budget, asked_reserve)}"
            for name, budget in (("agent", agent), ("domain", domain),
                                 ("reasoning", reason)))
        pf.add("overrun policy", PASS,
               f"{policy} - an overrun is finished with a reserve instead of "
               f"discarded ({reserves} tokens held back)")
        # SAY WHERE THE PROMISE IS WEAKER. Only the reasoning loop has a
        # structural terminator to force; prose and JSON have only EOS, so
        # there the policy raises the yield without bounding anything, and a
        # continuation that also runs out is still discarded. A recipe reading
        # `close-and-answer` should not be left to infer that.
        if policy == OVERRUN_CLOSE_AND_ANSWER:
            pf.add("overrun policy", PASS,
                   "close-and-answer has a gate to close only in the reasoning "
                   "loop; the agent and domain loops have no delimiter, so "
                   "there it behaves as answer-only - keep writing, accept it "
                   "if it ends")

    if not getattr(config, "use_vllm", False):
        return
    window = int(getattr(config, "vllm_max_len", 0) or 0)
    if not window:
        return
    # Only the budgets that ASK for unbounded can be surprised by the window.
    unbounded = [name for name, val in (("agent_teacher_max_new", agent),
                                        ("domain_teacher_max_new", domain),
                                        ("reasoning_teacher_max_new", reason))
                 if int(val or 0) == 0]
    if not unbounded:
        return

    # WHAT UNBOUNDED RESOLVES TO IS ALWAYS WORTH PRINTING. It is the one fact
    # the word hides.
    detail = (f"{', '.join(unbounded)} = unbounded, which on the vLLM path "
              f"means runtime.vllm_max_len ({window}) minus the prompt")

    # THE WARN IS FOR NOT HAVING TOUCHED THE THING THAT DECIDES IT, not for
    # the window being small. Asked of the RECIPE rather than the config
    # because the config cannot tell "set to 4096" from "defaulted to 4096",
    # and those two deserve different answers: somebody who raised this knob
    # has already thought about the ceiling and does not need warning about
    # it, while somebody who never set it is about to discover that asking
    # for unbounded made the budget SMALLER than the number they replaced.
    asked_window = 0
    if recipe is not None:
        asked_window = int(getattr(getattr(recipe, "runtime", None),
                                   "vllm_max_len", 0) or 0)
    if asked_window:
        pf.add("teacher budgets", PASS, detail)
        return
    pf.add("teacher budgets", WARN,
           detail + f" - and vllm_max_len is not set in the recipe, so that "
                    f"{window} is a default nobody chose",
           f"unbounded is only as generous as the window it runs into. At "
           f"{window} it can be SMALLER than the number you replaced, "
           f"silently, because the prompt comes out of the same window. Set "
           f"runtime.vllm_max_len deliberately: it is the real ceiling and it "
           f"is what reserves the KV cache.")


def _check_budget_hint(pf: Preflight, config) -> None:
    """Is anything being told how much room it has, and can it be?

    A no-op is worth a line when the setting is on: `teacher_tell_budget` has
    nothing to say about an unbounded budget, so a recipe that turns it on and
    also asks for unbounded has configured a sentence that never gets written.
    Silent success and silent no-op look identical from the outside.
    """
    if not _needs_a_teacher(config):
        return
    if not getattr(config, "teacher_tell_budget", False):
        return
    named = [name for name, budget in
             (("agent", getattr(config, "agent_teacher_max_new", 0)),
              ("domain", getattr(config, "domain_teacher_max_new", 0)),
              ("reasoning", getattr(config, "reasoning_teacher_max_new", 0)))
             if int(budget or 0) > 0]
    if not named:
        pf.add("budget hint", WARN,
               "budget.teacher_tell_budget is on and every teacher budget is "
               "unbounded, so there is no room to describe and no instruction "
               "is written",
               "harmless, but it is doing nothing - either name a budget or "
               "turn the hint off, so the recipe says what the run does")
        return
    pf.add("budget hint", PASS,
           f"the {', '.join(named)} teacher{'' if len(named) == 1 else 's'} "
           f"{'is' if len(named) == 1 else 'are'} told how much room "
           f"{'it' if len(named) == 1 else 'they'} ha"
           f"{'s' if len(named) == 1 else 've'}, in words")


def _check_eval_budget(pf: Preflight, config, recipe) -> None:
    """What the eval will generate per sample, and what unbounded costs.

    Reported rather than merely validated because the number silently decides
    whether a whole column of the report means anything. A real run capped at
    1024 returned SIXTEEN OF SIXTEEN generations cut off for five separate
    experts - every score in those rows was computed on an amputated answer,
    and `reasoned` (which needs a close tag it never reached) was a lower
    bound being read as a measurement.
    """
    from ..config import pipeline as cfg_module

    asked = getattr(getattr(recipe, "eval", None), "max_new_tokens", -1)
    resolved = cfg_module.eval_max_new_tokens(config, asked)
    if resolved == 0:
        # "EXPENSIVE" IS AN ADJECTIVE; THIS CAN COUNT. Every sample is
        # generated once per expert and once more against the MoE, so the
        # multiplier is knowable from the recipe before anything is booked -
        # and it is the difference between a setting somebody tries and a
        # setting somebody regrets. A 0.5B with a 32k window at 16 samples
        # across 9 experts is 288 generations of up to ~31k tokens each; the
        # same run capped at 1024 took just under an hour.
        n_samples = int(getattr(getattr(recipe, "eval", None),
                                "num_samples", 0) or 0)
        n_experts = len(getattr(recipe, "experts", None) or ())
        # Specialist + MoE = two surfaces per expert.
        generations = n_samples * n_experts * 2
        cost = (f"{generations:,} generations ({n_samples} samples x "
                f"{n_experts} experts x 2 surfaces)"
                if generations else "one generation per sample per expert per "
                                    "surface")
        pf.add("eval budget", WARN,
               "unbounded: every sample generates until it stops on its own "
               "or fills the model window",
               f"this is the honest setting for a `reasoned` number you mean "
               f"to trust, and generation time is linear in it: {cost}, each "
               f"free to run to the model's whole context. On a 32k-window "
               f"base that is tens of thousands of tokens per sample. Name a "
               f"number instead (2048-4096 covers a think block and its "
               f"answer) and read `capped_generations` to find out whether it "
               f"was enough - unbounded is for the run you have already "
               f"decided to spend a night on.")
        return
    pf.add("eval budget", PASS,
           f"{resolved} tokens per sample"
           + (" (automatic: this run writes thinking traces)"
              if int(asked or -1) < 0 else ""))


def _check_reasoning(pf: Preflight, config, recipe) -> None:
    """WHICH TAGS, AND WHERE THEY CAME FROM.

    A wrong tag style is a silent wrong answer - the splitter finds no
    delimiters, eval reports "did not reason", and the think block gets scored
    as though it were the answer. The table that decides it merges three
    layers (the packaged asset, ~/.msmoe/reasoning.yaml, $MSMOE_REASONING),
    any of which can be the one that mattered. So the answer to "where did
    these delimiters come from" belongs in the preflight somebody already ran,
    not at the end of an afternoon in the source.

    TWO STYLES, AND REPORTING ONLY ONE IS WHAT MAKES THIS CONFUSING.
    synth.py keeps them apart deliberately:

        target        what the SPECIALIST learns and eval reads back. Off the
                      run - reasoning_open/close, stamped into the config at
                      build time and part of the fingerprint.
        teacher_style what the TEACHER natively emits. Off the TEACHER's id,
                      with kind forced to "reasoning" - a teacher was picked
                      to reason, so the question is which dialect, never
                      whether.

    They legitimately differ: DeepSeek-R1 proper writes <|reasoning|> while
    the specialist learns <think>. Conflating them "rejected a good teacher's
    output 288 times in a row" per the note in synth. The first version of
    this check made the same mistake in the other direction - it asked about
    the BASE with the base's kind, and printed "base_kind is nonreasoning, so
    nothing reasons" on the same line as <think>…</think>.

    SAME CALLS AS THE BUILD, deliberately. reasoning_style_of_config for the
    target and style_for_base(teacher, "reasoning") for the teacher - the same
    two lines synth runs. A preflight that derived these its own way would be
    a second answer to one question, which is the disease this repo keeps
    finding.
    """
    from ..config import reasoning as _reasoning
    from ..config.pipeline import reasoning_style_of_config

    teacher = (getattr(config, "reasoning_teacher", "")
               or getattr(config, "teacher_model", "") or "")
    target = reasoning_style_of_config(config)
    wants = bool(getattr(config, "reasoning_expert_name", "")
                 or getattr(config, "reasoning_experts", None)
                 or target)
    if not wants:
        return

    # A malformed user table degrades to the packaged one rather than raising,
    # so the only way anyone learns it happened is if something asks.
    for warn in _reasoning.load_errors():
        pf.add("reasoning table", WARN, warn,
               "the table fell back a layer - the tags below may not be the "
               "ones you edited")

    if target is None:
        pf.add("reasoning", WARN,
               "a reasoning expert is planned and no delimiters resolved onto "
               "the config",
               "traces will be written with the fallback <think></think>. If "
               "this base speaks something else, add a family for it to "
               "~/.msmoe/reasoning.yaml - that file is what it is FOR.")
        return

    pf.add("reasoning", PASS,
           f"specialist learns {target.open}…{target.close}"
           f"{f' ({target.name})' if target.name else ''}")

    if not teacher:
        return
    try:
        why = _reasoning.explain_base(teacher, "reasoning")
    except Exception as exc:                              # noqa: BLE001
        pf.add("reasoning teacher", WARN,
               f"could not resolve the teacher's tag style: {exc}")
        return

    styles, _, _ = _reasoning.load()
    spoken = styles.get(why["style"]) if why["style"] else None

    # KEY OFF THE FAMILY, NOT THE STYLE, and the difference is the whole
    # check. explain_base is asked with kind="reasoning" - a teacher was
    # picked to reason, so the question is which dialect - and that kind
    # FALLS BACK to plain xml when nothing matches. So a style always
    # resolves, and testing `spoken is None` made the warning unreachable:
    # an unknown teacher printed "family '' on hint '' · from the built-in
    # floor", which is the exact species of confident-looking nonsense this
    # check was added to delete.
    #
    # A family matched or it did not. That is the fact; the xml is a guess.
    if spoken is None or not why.get("family"):
        # No family matched, so synth falls back to the TARGET style for the
        # teacher too - which is right often enough to be worth doing and
        # wrong quietly enough to be worth saying.
        pf.add("reasoning teacher", WARN,
               f"{teacher} matches no family in the reasoning table",
               f"it will be assumed to speak the target style "
               f"({target.open}…{target.close}). If it does not, every trace "
               f"is rejected or mis-split. Add a family for it to "
               f"~/.msmoe/reasoning.yaml.")
        return

    same = "" if (spoken.open, spoken.close) != (target.open, target.close) \
        else " (same as the target)"
    pf.add("reasoning teacher", PASS,
           f"{teacher} emits {spoken.open}…{spoken.close}{same} · "
           f"family {why['family']!r} on hint {why['hint']!r} · from "
           f"{why['source'] or 'the built-in floor'}")


def _check_trainer(pf: Preflight, config) -> None:
    """unsloth, and what happens when it was asked for and is not here.

    FALL BACK AND SAY SO LOUDLY, AT VALIDATE TIME. That is the shape asked
    for and it is the llama.cpp shape, not the vLLM one: a plain fine-tune
    is a real result, so this warns rather than refuses. What it must not
    do is what it used to - fall back with a print, five hours into a run,
    on a knob that is part of the build fingerprint.

    AND THE FALLBACK IS NOT CLEAN, which is the part worth saying out loud.
    `config.optim` is resolved to "adamw_8bit" from use_unsloth back in
    build_config, and finetune passes `optim=config.optim` to the trainer
    whichever path it took - so the plain path runs with the 8-bit
    optimiser unsloth was going to provide, and needs bitsandbytes to do
    it. A reader deserves to know that before the GPU is booked.

    load_in_4bit is the one hard refusal here. finetune already raises for
    it, deliberately, BEFORE training - but it raises after the recipe has
    been loaded on the build box. Saying it at preflight costs nothing and
    says it on the laptop.
    """
    import importlib.util

    if not (getattr(config, "use_unsloth", False)
            or getattr(config, "load_in_4bit", False)):
        return

    have = importlib.util.find_spec("unsloth") is not None

    if getattr(config, "load_in_4bit", False) and not have:
        pf.add("trainer", FAIL,
               "runtime.load_in_4bit is on and unsloth is not installed",
               "4-bit TRAINING needs unsloth's save_pretrained_merged to "
               "produce a mergeable specialist - the plain path would "
               "train and then fail at the save, after the whole bill is "
               "paid. Set `runtime: {load_in_4bit: false}`, or install "
               "unsloth.")
        return

    if not getattr(config, "use_unsloth", False):
        return
    if have:
        pf.add("trainer", PASS, f"unsloth, optim {config.optim}")
        return

    pf.add("trainer", WARN,
           "MSMOE_UNSLOTH is set and unsloth is not installed - the build "
           "will train on the plain path",
           f"that is a real fine-tune and a real checkpoint, so it is not "
           f"refused. But optim stays {config.optim!r} (resolved from "
           f"use_unsloth) and the plain trainer needs bitsandbytes to "
           f"honour it, and the manifest records use_unsloth=true for a "
           f"run that did not use it. Install unsloth, or unset "
           f"MSMOE_UNSLOTH so the recorded build matches the one that ran.")


def _check_sources(pf: Preflight, recipe) -> None:
    """Anything checkable about the corpora, without fetching them."""
    from ..data import corpus as corpus_mod

    for e in recipe.experts:
        src = getattr(e, "source", None)
        kind = getattr(src, "kind", "") if src else ""
        if not kind:
            pf.add(f"source/{e.name}", FAIL, "no source kind",
                   f"give {e.name} a source: with a kind of "
                   f"{', '.join(corpus_mod.names())}")
            continue
        if corpus_mod.get(kind) is None:
            pf.add(f"source/{e.name}", FAIL, f"unknown kind {kind!r}",
                   f"known kinds: {', '.join(corpus_mod.names())}")
            continue
        if kind == "local":
            path = getattr(src, "path", "") or ""
            if not os.path.isdir(path):
                pf.add(f"source/{e.name}", FAIL,
                       f"local path does not exist: {path}",
                       "point `path:` at a directory that exists on THIS box")
                continue
            pf.add(f"source/{e.name}", PASS, f"local {path}")
        else:
            pf.add(f"source/{e.name}", PASS, f"kind={kind}")


def run(config, recipe, offline: bool = False,
        need_exporter: bool = True) -> Preflight:
    """Every cheap question worth asking before the expensive part starts."""
    pf = Preflight()
    have_torch = _check_training_stack(pf)
    _check_roots(pf, config)
    _check_sources(pf, recipe)
    _check_generator(pf, config)
    _check_budgets(pf, config, recipe)
    _check_budget_hint(pf, config)
    _check_eval_budget(pf, config, recipe)
    _check_trainer(pf, config)
    _check_reasoning(pf, config, recipe)
    if have_torch:
        _check_base_model(pf, config, offline=offline)
    _check_datasets(pf, recipe, offline=offline)
    if need_exporter:
        _check_exporter(pf, config)
    return pf


def render(pf: Preflight) -> List[str]:
    """Human lines. Failures last, so the thing to act on is nearest the prompt."""
    icon = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL", UNMEASURABLE: "?   "}
    lines = [f"   [{icon.get(c.status, '?')}] {c.name:16} {c.detail}"
             for c in pf.checks if c.status in (PASS, UNMEASURABLE)]
    for c in pf.warnings:
        lines.append(f"   [warn] {c.name:16} {c.detail}")
        if c.remedy:
            lines.append(f"          -> {c.remedy}")
    for c in pf.failures:
        lines.append(f"   [FAIL] {c.name:16} {c.detail}")
        if c.remedy:
            lines.append(f"          -> {c.remedy}")
    return lines
