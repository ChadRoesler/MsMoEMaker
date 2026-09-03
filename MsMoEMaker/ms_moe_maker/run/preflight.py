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

    batch = getattr(config, "teacher_batch", 0)
    if not getattr(config, "use_vllm", False):
        pf.add("teacher", PASS, f"transformers, batch {batch}",
               "")
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
    pf.add("teacher", PASS, f"vLLM, batch {batch}")


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
