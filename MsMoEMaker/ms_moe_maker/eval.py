"""Eval pipeline — two questions, answered honestly or not at all.

    ms-moe-maker eval --mode routing   does each expert own its own ground?
    ms-moe-maker eval --mode quality   does it answer better than one expert?
    ms-moe-maker eval --mode all       both (default)

ROUTING is the dead-expert measurement and it is the one Ms.MoE uniquely
claims. A dead expert is not one that scores badly, it is one the ROUTER NEVER
ROUTES TO — which is why hand-assigning domains prevents them, and why this
number is the thesis. Reported as enrichment: how much more an expert is used
on its own domain than on average. Needs torch and the router-trained MoE; no
answer key.

QUALITY is generation against held-out references. It needs an answer key, and
whoever wrote the corpus is the only one who has it — which is exactly why it
is the half meant to be overridden.

Both are overrideable from the recipe. We provide the floor:

  eval:
    script: my_custom_eval.py      # replace ours entirely
    held_out_fraction: 0.1         # fraction held back from training
    num_samples: 20                # samples per expert
    dead_threshold: 1.2            # min enrichment before an expert is dead

Anything this module cannot measure is reported UNMEASURABLE with a reason.
Unmeasurable is not pass. See evalrecord.py for the vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import threading
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config as cfg_module
from . import stages as st
from . import manifest as mf
from .evalrecord import ERROR, FAIL, PASS, UNMEASURABLE


@dataclass
class EvalResult:
    """Result of one eval run."""
    expert_name: str
    """Name of the expert (or 'moe' / 'base')."""
    domain: str
    """Domain / test category this expert was evaluated on."""
    exact_match: float = 0.0
    """Exact match score (0-1)."""
    rouge1: float = 0.0
    """ROUGE-1 score."""
    bleu: float = 0.0
    """BLEU score."""
    avg_length: float = 0.0
    """Average output token length."""
    status: str = "pending"
    """pending | done | failed | skipped."""
    note: str = ""
    """Free-text note."""


@dataclass
class EvalReport:
    """Full eval report across all experts."""
    stages: Dict[str, EvalResult] = field(default_factory=dict)
    dead_experts: List[str] = field(default_factory=list)
    """Experts flagged as dead."""
    ok: bool = False
    """True if no fatal errors."""
    message: str = ""
    routing: Dict[str, Any] = field(default_factory=dict)
    """Router-discrimination result — the dead-expert measurement."""
    undiscriminating: List[str] = field(default_factory=list)
    """Experts the router uses but shows no domain preference for.

    NOT the same failure as dead, and calling it dead was a false alarm on the
    first real run: both 0.5B experts sat at ~0.50 selection share - used on
    half of all tokens, exactly what uniform routing gives with top-1 of 2 -
    and got reported as DEAD EXPERTS because their enrichment was 1.02x. An
    expert that is never selected and an expert that is selected constantly
    without specialising are different problems with different fixes, and the
    stitch is only broken in the first case.
    """
    caveats: List[str] = field(default_factory=list)
    """Measured, but with a limit the reader has to know about."""
    experts: Dict[str, Any] = field(default_factory=dict)
    """The pre-stitch expert checks, re-run against an existing run dir.

    Same measurements the build gate makes - weight divergence, cross-domain
    loss, config audit - available afterward without a rebuild, because the
    question "was there ever anything to route on" is usually asked while
    staring at a disappointing routing table.
    """
    unmeasured: List[str] = field(default_factory=list)
    """Things we could not measure, and therefore did not score."""


def _trace(tag: str) -> None:
    """Print a memory line to stderr. ALWAYS ON, and that is deliberate.

    Two OOMs on a 128 GB Spark could not be attributed to anything, because
    nothing in this module ever said how much memory it was using. The kernel
    said 119Gi/121Gi and the CUDA allocator said "free: 7906271232" and
    neither number could be pinned to a line of our code. Silence is signal,
    and the signal was that we were not looking.

    Six short lines per run is a cheap price for never having to guess again.
    """
    _mark(tag)
    m = _mem_mb()
    if not m:
        return
    print("[mem] " + tag + ": "
          + " ".join(f"{k}={v:,.0f}MiB" for k, v in m.items()),
          file=sys.stderr, flush=True)


def _mem_mb() -> Dict[str, float]:
    """Host and device memory in MiB. Empty dict where it cannot be read.

    proc_rss is RssAnon, NOT VmRSS: on unified memory VmRSS counts shared CUDA
    pages and over-reports by tens of GB. host_avail is MemAvailable, which is
    the number that actually predicts the OOM killer - and on a box where host
    and device share one pool it is also, approximately, how much CUDA has
    left. That is why the allocator reported 7.9 GB free of 130 GB total: not
    a device that had filled up, a MACHINE that had.
    """
    out: Dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    out["proc_rss"] = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    out["host_avail"] = int(line.split()[1]) / 1024.0
                elif line.startswith("MemTotal:"):
                    out["host_total"] = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda_alloc"] = torch.cuda.memory_allocated() / 2 ** 20
            out["cuda_reserved"] = torch.cuda.memory_reserved() / 2 ** 20
    except Exception:
        pass
    return out


class _MemSampler:
    """Peak-tracker on a background thread. BOUNDARY SAMPLING CANNOT SEE THIS.

    The [mem] lines fire between phases, and every one of them looked healthy
    while the run died: 114 GB available at "after MoE load", then an OOM
    inside the very next call. Nothing was wrong with the boundaries. The cost
    lives BETWEEN them - the allocator inflates during generation and the
    tensors it holds are released before anything gets a chance to print.

    So this samples on a timer, tracks the peak per named phase, and the phase
    label is what turns "it OOMs" into "it OOMs in generate, at N GB, with
    these settings" - a bug you can fix instead of a mood.

    FOOTPRINT IS RESERVED + RssAnon, NOT ALLOCATED + RssAnon. On a discrete GPU
    the reservation lives in VRAM and the host never feels it; on GB10 it comes
    out of the same 121 GB the kernel is using, so `reserved` IS host pressure.
    Printing one column and reasoning over the other is how two rounds of this
    diagnosis went wrong.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.phase = "startup"
        self.peaks: Dict[str, Dict[str, float]] = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_MemSampler":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)

    def mark(self, phase: str) -> None:
        self.phase = phase

    def _run(self) -> None:
        while not self._stop.is_set():
            m = _mem_mb()
            p = self.peaks.setdefault(self.phase, {
                "cuda_alloc": 0.0, "cuda_reserved": 0.0,
                "proc_rss": 0.0, "min_avail": float("inf")})
            for k in ("cuda_alloc", "cuda_reserved", "proc_rss"):
                if k in m and m[k] > p[k]:
                    p[k] = m[k]
            if "host_avail" in m:
                p["min_avail"] = min(p["min_avail"], m["host_avail"])
            self._stop.wait(self.interval)

    def table(self) -> str:
        rows = ["", "PEAKS BY PHASE (MiB)  [footprint = cuda_reserved + RssAnon:",
                "                       on unified memory the allocator's",
                "                       RESERVATION is host RAM]",
                f"  {'phase':22} {'alloc':>10} {'reserved':>10} "
                f"{'RssAnon':>10} {'footprint':>11} {'min avail':>11}"]
        for phase, p in self.peaks.items():
            foot = p["cuda_reserved"] + p["proc_rss"]
            avail = p["min_avail"]
            rows.append(
                f"  {phase:22} {p['cuda_alloc']:10,.0f} "
                f"{p['cuda_reserved']:10,.0f} {p['proc_rss']:10,.0f} "
                f"{foot:11,.0f} "
                + (f"{avail:11,.0f}" if avail != float("inf") else f"{'-':>11}"))
        return "\n".join(rows)


_SAMPLER: Optional["_MemSampler"] = None


def _mark(phase: str) -> None:
    if _SAMPLER is not None:
        _SAMPLER.mark(phase)


def _balloon() -> Tuple[float, float]:
    """(reserved_MiB, allocated_MiB) — the two numbers that must be compared."""
    try:
        import torch
        if torch.cuda.is_available():
            return (torch.cuda.memory_reserved() / 2 ** 20,
                    torch.cuda.memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return (0.0, 0.0)


def _deflate(ratio: float = 3.0, floor_mib: float = 4096.0) -> bool:
    """Hand back reserved-but-unused device memory, if there is a lot of it.

    Called between samples rather than after every one: empty_cache() is not
    free, and a healthy allocator reusing its own blocks is exactly what you
    want. This only fires when reservation has run away from live tensors,
    which is the fragmentation signature, not normal operation.
    """
    reserved, alloc = _balloon()
    if reserved < floor_mib or reserved < max(alloc, 1.0) * ratio:
        return False
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        return False
    _trace(f"deflated allocator: {reserved:,.0f}MiB reserved for "
           f"{alloc:,.0f}MiB live -> {_balloon()[0]:,.0f}MiB")
    return True


def _is_oom(exc: BaseException) -> bool:
    """Is this exception the machine running out of memory?

    Matched by NAME and by message rather than by class, because
    torch.cuda.OutOfMemoryError moved to torch.OutOfMemoryError, and importing
    either to catch it would drag torch into a module that promises to run
    without it.
    """
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _iter_jsonl(path: str):
    """Stream a JSONL file, one non-empty line at a time.

    NEVER read_text() A CORPUS. Every reader in this module used
    Path(p).read_text().splitlines(), which materialises the file as one
    string AND AGAIN as a list of strings before a single sample is drawn -
    2-4x the file size resident, per call, per expert. On a box where host and
    device memory are the same pool that is not a tidiness problem: it is CUDA
    memory, spent on text, and it is why a 0.5B model that trains fine can OOM
    during EVAL. The floor this project promises is a $250 Nano; eval must not
    need more memory than the build that produced the model.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line
    except OSError:
        return


def _reservoir(path: str, n: int, seed: int = 42) -> List[str]:
    """Uniform sample of n lines in ONE pass and O(n) memory.

    Same distribution as random.sample over the whole file, without the whole
    file. For n=20 this is twenty strings instead of a corpus.
    """
    rnd = random.Random(seed)
    out: List[str] = []
    for i, line in enumerate(_iter_jsonl(path)):
        if i < n:
            out.append(line)
        else:
            j = rnd.randint(0, i)
            if j < n:
                out[j] = line
    return out


def _digest(text: str) -> str:
    """Content hash, for membership tests that must not retain the content."""
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _load_or_split(data_path: str, held_out: float, seed: int = 42) -> Tuple[str, str]:
    """Split a JSONL dataset into train / held-out.

    Returns (train_path, held_out_path).

    TWO STREAMING PASSES, NOT FOUR COPIES. This used to read the corpus, hold
    a second shuffled copy of it, slice that into two more lists and join each
    into another string before writing - peak somewhere north of 4x the file,
    once per expert, at the very top of every eval run. Now it counts lines,
    draws the held-out INDICES, and streams each line to whichever file it
    belongs in. Resident cost is one line plus the index set.

    Behaviour note: rows keep their original order within each output file
    instead of coming out shuffled. Membership is still a uniform random draw
    at the same seed, which is the property that matters; nothing downstream
    reads these in order.
    """
    train_path = data_path + ".train"
    held_path = data_path + ".heldout"

    total = sum(1 for _ in _iter_jsonl(data_path))
    if not total:
        return data_path, held_path

    n_held = max(1, int(total * held_out))
    held_idx = set(random.Random(seed).sample(range(total), n_held))

    with open(train_path, "w", encoding="utf-8") as tf, \
            open(held_path, "w", encoding="utf-8") as hf:
        for i, line in enumerate(_iter_jsonl(data_path)):
            (hf if i in held_idx else tf).write(line + "\n")

    return train_path, held_path


def _tokenize_simple(text: str) -> List[str]:
    """Very simple whitespace-based tokenizer for eval metrics."""
    return text.lower().split()


def _exact_match(generated: str, reference: str) -> float:
    """0 or 1 — exact string match (after lowercase + strip)."""
    return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0


def _rouge1(generated: str, reference: str) -> float:
    """ROUGE-1: fraction of reference tokens found in generated."""
    gen_tokens = set(_tokenize_simple(generated))
    ref_tokens = set(_tokenize_simple(reference))
    if not ref_tokens:
        return 0.0
    overlap = gen_tokens & ref_tokens
    return len(overlap) / len(ref_tokens)


def _bleu_simple(generated: str, reference: str) -> float:
    """BLEU-1: clipped unigram precision WITH a brevity penalty.

    THE BREVITY PENALTY IS NOT OPTIONAL AND ITS ABSENCE WAS VISIBLE IN THE
    OUTPUT. Without it this returns matches/len(generated), which is pure
    precision: emit three words that all appear in the reference and score
    1.00. On the first real 0.5B run csharp reported BLEU 0.877 against
    ROUGE-1 0.369 - the highest number on the board belonged to the metric
    that rewards saying almost nothing, sitting next to a recall number
    saying most of the reference never got written.

    BP = exp(1 - r/c) for c < r, 1 otherwise, per Papineni et al. It is the
    term that makes precision answerable for what it left out, and this
    project has a standing rule against numbers that report more than they
    know.
    """
    import math
    from collections import Counter

    gen_tokens = _tokenize_simple(generated)
    ref_tokens = _tokenize_simple(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0

    gen_counts = Counter(gen_tokens)
    ref_counts = Counter(ref_tokens)
    matches = sum(min(gen_counts[t], ref_counts[t]) for t in gen_counts)
    precision = matches / len(gen_tokens)

    c, r = len(gen_tokens), len(ref_tokens)
    brevity = 1.0 if c >= r else math.exp(1.0 - r / c)
    return precision * brevity


# ── the two questions ────────────────────────────────────────────────────────
#
# WHAT THIS FILE USED TO DO, because it is worth writing down. eval_expert and
# eval_moe accepted expert_dir / moe_dir / base_model and NEVER OPENED THEM.
# There was no torch, no transformers, no from_pretrained and no .generate()
# anywhere in this module. What they actually scored was
#
#     len(prompt_tokens & ref_tokens) / len(ref_tokens)
#
# - the lexical overlap between a prompt and its own reference answer, which is
# a property of the DATASET. Point it at an empty directory and the number does
# not move. eval_moe then added `0.1 * len(specialist_dirs) / 4.0` on top,
# commented "MoE gets a small boost from routing (real routing helps)", which
# meant the MoE was DEFINED to beat every expert by a constant.
#
# It printed per-expert scores, ROUGE, BLEU, a tick column and a "dead experts"
# banner, and not one of those numbers referred to a model. That is worse than
# a crash. A crash is honest; this produced a plausible report. The whole point
# of an evidence layer is that you see the result instead of reasoning about
# it, and a fabricated result is the exact inverse - it makes you think you
# saw something.
#
# It is replaced by two measurements that answer two different questions, and
# by an honest UNMEASURABLE whenever the machinery to answer them is absent.
# `unmeasurable` is not `pass`. The vocabulary is already in evalrecord.py and
# validators.py has been using it correctly all along.


def _torch_available() -> Tuple[bool, str]:
    """Can we load a model at all? Returns (ok, reason-if-not).

    Kept as a probe rather than a hard import so `ms-moe-maker eval` still
    RUNS on a laptop - it just says what it could not measure and why, which
    is the useful answer. Refusing to start would be less kind and less
    informative than refusing to score.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch not importable ({exc.__class__.__name__}: {exc})"
    try:
        import transformers  # noqa: F401
    except Exception as exc:
        return False, f"transformers not importable ({exc.__class__.__name__}: {exc})"
    return True, ""


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits. 0 = identical, 1 = disjoint."""
    import math
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0 and b > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def probe_router_discrimination(moe_dir: str,
                                held_paths: Dict[str, str],
                                expert_order: Sequence[str],
                                num_samples: int = 16,
                                dead_threshold: float = 1.2,
                                max_len: int = 1024,
                                device: str = "cpu",
                                trained_on: Optional[set] = None,
                                callback=None) -> Dict[str, Any]:
    """THE dead-expert measurement — ported from probe_router_discrimination.py.

    PROVENANCE, and why this is a port rather than an implementation. This is
    the probe that produced the proven 0.5B numbers: all five experts winning
    their own column, mean enrichment 2.12x, p=0.00032. A from-spec
    reimplementation of it was written first and was wrong in five separate
    ways, every one of which would still have printed a confident table:

      * it averaged gate PROBABILITIES instead of counting which experts the
        router actually SELECTS (top-k). Those are different questions.
      * enrichment divided by the mean share across ALL sources including the
        expert's own, which dilutes the very signal being measured. The correct
        denominator is the mean over the OTHER sources.
      * no JS divergence, so it could not answer the weaker but more
        fundamental question: does routing depend on the input AT ALL? Balance
        entropy near 1.0 cannot tell "well-balanced" from "completely random" -
        an input-blind router scores as perfectly healthy with zero dead
        experts. That number can only ever be a grade.
      * no held-out-by-construction. It used arbitrary splits that could
        overlap what the router trained on, which measures memorisation.
      * it ignored mlp_only_layers. router_logits has one entry per MoE LAYER,
        not per model layer, so with dense layers at the bottom the indices are
        low by however many, and phantom zeros pad the top and drag the mean JS
        down.

    Why COLUMN-WISE and not against top_k/num_experts: marginal expert usage is
    nowhere near uniform - one expert can absorb 24% of all traffic while
    another takes 15% - so a flat "chance" baseline compares each expert
    against a number that has nothing to do with it and reports every single
    one as below chance.
    """
    ok, reason = _torch_available()
    if not ok:
        return {"status": UNMEASURABLE, "reason": reason, "experts": {}}
    if not Path(moe_dir).is_dir():
        return {"status": UNMEASURABLE,
                "reason": f"no router-trained MoE at {moe_dir} - run build first",
                "experts": {}}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # One-element lists so the inner loop can accumulate without `nonlocal`.
    conf_sum = [0.0]
    conf_n = [0]

    trained_on = trained_on or set()

    # Held out BY CONSTRUCTION: drop any row the router actually trained on.
    #
    # `trained_on` is a set of CONTENT HASHES, not of texts. Holding the
    # training corpus in a Python set to answer "have I seen this string"
    # keeps every byte of it alive for the duration of the probe, at roughly
    # 1.5-3x the file size once str objects are counted - and that set was
    # built from a read_text() of mixed_all.jsonl, so the corpus was resident
    # twice before the model loaded. Forty hex characters answers the same
    # question.
    #
    # The read is streamed and stops at 400 rows per source, which is what the
    # loop always intended; it just used to slurp the file first and then
    # break.
    sources: Dict[str, List[str]] = {}
    for name, path in held_paths.items():
        rows: List[str] = []
        for line in _iter_jsonl(path):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text") or obj.get("content")
            if t and _digest(t) not in trained_on:
                rows.append(t)
            if len(rows) >= 400:
                break
        if rows:
            sources[name] = rows

    if len(sources) < 2:
        return {"status": UNMEASURABLE,
                "reason": (f"need >=2 sources with held-out rows, found "
                           f"{sorted(sources)}"),
                "experts": {}}

    try:
        model, tok, device = _load_model(moe_dir, device=device)
    except Exception as exc:
        return {"status": UNMEASURABLE,
                "reason": f"could not load MoE from {moe_dir}: {exc}",
                "experts": {}}

    cfg = model.config
    E = getattr(cfg, "num_experts", 0)
    K = getattr(cfg, "num_experts_per_tok", 0)
    if not E or not K:
        model = None
        release_memory()
        return {"status": UNMEASURABLE,
                "reason": "model has no num_experts - it is not a MoE",
                "experts": {}}

    # TOP-K == EXPERT COUNT IS UNMEASURABLE, NOT INPUT-BLIND.
    #
    # torch.topk(logits, K) with K == E returns EVERY expert, on every token,
    # whatever the logits say. The selection counts are then 1/E each by
    # arithmetic, enrichment is exactly 1.00x, and the pairwise JS divergence
    # is exactly 0.000 - the same numbers a completely input-blind router
    # would produce, and utterly indistinguishable from them.
    #
    # Reporting that as "the router ignores its input entirely" is a false
    # diagnosis of a possibly-healthy router, and it is the worst kind because
    # the numbers look decisive. The router was never asked to choose.
    #
    # A 2-expert MoE therefore needs experts_per_tok=1 to be measurable at all,
    # and the usual shape is 3+ experts with top-2.
    if K >= E:
        model = None
        release_memory()
        return {"status": UNMEASURABLE,
                "reason": (f"experts_per_tok={K} with num_experts={E}: top-{K} "
                           f"of {E} selects every expert on every token, so "
                           f"routing cannot discriminate BY CONSTRUCTION. The "
                           f"0.5/0.5 shares and JS=0 this would report are "
                           f"arithmetic, not measurement. Set "
                           f"moe.experts_per_tok below {E}, or add experts."),
                "experts": {}, "n_experts": E, "top_k": K}

    dense = set(getattr(cfg, "mlp_only_layers", None) or [])
    moe_layers = [i for i in range(cfg.num_hidden_layers) if i not in dense]
    L = len(moe_layers)

    # WHICH EXPERT IS WHICH. Not cosmetic - guessing this wrong inverts the
    # headline result. Expert index follows stitch order, which config.json
    # stamps as expert_names. Never re-derive it by sorting.
    names_from_cfg = list(getattr(cfg, "expert_names", None) or [])
    expert_names = names_from_cfg or list(expert_order)
    if len(expert_names) != E:
        model = None
        release_memory()
        return {"status": UNMEASURABLE,
                "reason": (f"expert order has {len(expert_names)} names but the "
                           f"model has {E} experts: {expert_names}"),
                "experts": {}}

    src_names = sorted(sources)
    counts = {s: [[0] * E for _ in range(L)] for s in src_names}
    totals = {s: [0] * L for s in src_names}
    rnd = random.Random(0)

    for s in src_names:
        for text in rnd.sample(sources[s], min(num_samples, len(sources[s]))):
            wrapped = text
            try:
                msgs = [{"role": "user", "content": f"Write {s}:"},
                        {"role": "assistant", "content": text}]
                wrapped = tok.apply_chat_template(msgs, tokenize=False) + (
                    tok.eos_token or "")
            except Exception:
                pass
            enc = tok(wrapped, return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            with torch.no_grad():
                out = model(**enc, output_router_logits=True)
            router_logits = getattr(out, "router_logits", None)
            if not router_logits:
                model = None
                release_memory()
                return {"status": UNMEASURABLE,
                        "reason": ("model returned no router_logits - not a MoE, "
                                   "or this transformers version does not expose "
                                   "them for this architecture"),
                        "experts": {}}
            for li, logits in enumerate(router_logits):
                if logits is None or li >= L:
                    continue
                probs = torch.softmax(logits.float(), dim=-1)
                top = torch.topk(probs, K, dim=-1)
                sel = top.indices.flatten()
                for e in sel.tolist():
                    counts[s][li][e] += 1
                totals[s][li] += sel.numel() // K
                # GATE CONFIDENCE - how sure the router is, separate from
                # whether it is RIGHT. With norm_topk_prob=false the selected
                # weight multiplies the expert's output, so p is a free scalar
                # gain on a frozen FFN: at init p=1/E halves (or worse) a
                # contribution the base model expects at full strength, and the
                # cheapest way to fix that is p -> 1 on every token regardless
                # of input. That is a collapsed, input-blind router arrived at
                # for a reason that has nothing to do with routing, and no
                # share-or-enrichment number can distinguish it from any other
                # collapse. This can: saturated confidence next to zero JS is
                # the signature.
                conf_sum[0] += float(top.values.sum())
                conf_n[0] += int(top.values.numel())
        if callback:
            callback("eval.routing", "running", f"routed {s}")

    model = None
    release_memory()

    # source x expert, pooled over layers
    avg: Dict[str, List[float]] = {}
    for s in src_names:
        tot = sum(totals[s]) * K
        avg[s] = [sum(counts[s][li][e] for li in range(L)) / max(tot, 1)
                  for e in range(E)]

    experts: Dict[str, Any] = {}
    enrich: List[float] = []
    hits = 0
    for e, en in enumerate(expert_names):
        col = {s: avg[s][e] for s in src_names}
        if en not in col:
            continue
        own = col[en]
        others = [v for s, v in col.items() if s != en]
        oavg = sum(others) / max(len(others), 1)
        top = max(col, key=col.get)
        r = own / max(oavg, 1e-9)

        # A STARVED EXPERT HAS NO ENRICHMENT, ONLY A RATIO OF TWO NOISES.
        #
        # A collapsed run printed `python own 0.001 others 0.000 enrich 2.15x`
        # - the largest enrichment on the board, from 0.001 / 0.0005, on an
        # expert the router had abandoned. It also pulled the MEAN from ~1.0
        # to 1.57, so the one summary number a reader is most likely to quote
        # was set by the expert with the least data behind it.
        #
        # Below the same floor `dead` uses, the division is unstable by
        # construction: numerator and denominator are both a handful of token
        # selections out of tens of thousands. Compute it, flag it, and keep
        # it out of the mean.
        marginal = sum(col.values()) / max(len(col), 1)
        starved = marginal < (K / max(E, 1)) * 0.2
        if not starved:
            enrich.append(r)
        hits += (top == en)
        rivals = sorted(((s, v) for s, v in col.items() if s != en),
                        key=lambda kv: kv[1], reverse=True)
        experts[en] = {
            "enrichment": r,
            # False when the expert is selected too rarely for the ratio above
            # to mean anything. Read the share instead.
            "enrichment_reliable": not starved,
            "own_share": own,
            # Selection share averaged over EVERY source - the number that
            # answers "is this expert used at all", which is what dead means.
            # own_share alone cannot answer it: an expert can be selected
            # constantly and still show no preference for its own ground.
            "marginal_share": sum(col.values()) / max(len(col), 1),
            "others_share": oavg,
            "own_is_column_max": top == en,
            "top_competitor": rivals[0][0] if rivals else "",
            "top_competitor_share": rivals[0][1] if rivals else 0.0,
            # An expert can clear the enrichment bar and still be outranked on
            # its own ground by a neighbour taking more of the traffic. That is
            # a different failure and a column-only read misses it.
            # A REAL MARGIN, NOT A STRICT >. On a collapsed router both
            # experts printed OUTRANKED ON ITS OWN GROUND with own and rival
            # identical to four decimals - a floating-point tie reported as a
            # finding. 2% of the own-share is below anything worth acting on.
            "outranked": bool(rivals) and rivals[0][1] > own * 1.02,
        }

    # n counts only the experts whose enrichment is readable, so the mean is
    # over the same set the table asks you to believe.
    n = len(enrich)
    mean_enrichment = sum(enrich) / max(n, 1)

    # The weaker, more fundamental claim: does routing depend on input at all?
    per_layer: List[float] = []
    for li in range(L):
        dists = []
        for s in src_names:
            tot = totals[s][li] * K
            dists.append([counts[s][li][e] / max(tot, 1) for e in range(E)])
        pairs = [_js_divergence(dists[i], dists[j])
                 for i in range(len(dists)) for j in range(i + 1, len(dists))]
        per_layer.append(sum(pairs) / max(len(pairs), 1))
    mean_js = sum(per_layer) / max(len(per_layer), 1)

    weak = [n_ for n_, e in experts.items()
            if e["enrichment"] < dead_threshold or e["outranked"]]

    return {
        "status": FAIL if (weak or mean_js < 1e-3) else PASS,
        "reason": ((f"{len(weak)} expert(s) below {dead_threshold}x enrichment "
                    f"or outranked: {', '.join(weak)}") if weak else
                   ("routing is input-blind (mean JS ~ 0): the router ignores "
                    "its input entirely" if mean_js < 1e-3 else "")),
        "experts": experts,
        "routing_matrix": avg,
        "expert_order": list(expert_names),
        "n_experts": E,
        "top_k": K,
        "moe_layers": L,
        "own_is_max_count": hits,
        "named_experts": n,
        "mean_enrichment": mean_enrichment,
        # p for all-n-of-n winning their own column by chance.
        "p_value": (1 / (n ** n)) if n else None,
        "mean_js_bits": mean_js,
        "js_per_layer": per_layer,
        # Mean softmax probability of the experts actually selected. Uniform
        # is 1/E (or K/E summed over the top-K); 1.0 means the gate is fully
        # saturated and the softmax has stopped being a distribution.
        "mean_gate_confidence": (conf_sum[0] / conf_n[0]) if conf_n[0] else None,
        "uniform_confidence": 1.0 / E if E else None,
    }


def release_memory() -> None:
    """Collect and hand memory back. TAKES NO ARGUMENTS, AND THAT IS THE FIX.

    This was `model_cleanup(model)`, which did `del model` on its own
    PARAMETER. That drops one name inside the function while the CALLER still
    holds the object, so the gc.collect() immediately after ran with the model
    fully reachable and freed nothing. Every cleanup in the eval path was a
    no-op, and on a 121 GB unified-memory box that ends at the OOM killer.

    It is the same wrong-scope `del` as `del torch` in builder.py - which this
    function's own docstring cited as a cautionary tale while committing the
    identical mistake one level up. Deleting a NAME is not freeing an OBJECT,
    and a function cannot drop a reference it does not own.

    So the contract is explicit now: the caller sets its own reference to None
    and then calls this. Uglier at the call site, and it actually frees.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_model(model_dir: str, dtype_name: str = "float16", device: str = ""):
    """Load a model for evaluation. One loader, one set of decisions.

    NO device_map="auto", AND NOT float32 - both were in here, and on unified
    memory both are traps this project has already paid for. fp32 doubles a
    checkpoint for no benefit when there are no gradients, and `auto` sharding
    on a box where host and device memory are the same pool produces an
    allocation profile that shows up in no process's RSS and takes the machine
    down instead of raising.

    Evaluation is forward passes only, so half precision is the right default
    and an explicit device beats a planner that assumes two memory spaces.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name, torch.float16)
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model = model.to(device).eval()

    # use_cache=False IS A TRAINING SETTING AND IT SURVIVED THE STITCH.
    #
    # The router-trained config.json carries `"use_cache": false`, which is
    # correct while training (the cache is dead weight when every position is
    # computed anyway) and actively hostile at generation time: with no KV
    # cache, generate() re-runs a FULL forward over the entire prefix for
    # every single new token. 256 new tokens on a 1024-token prompt stops
    # being 256 cheap steps and becomes 256 full-sequence forwards, each one
    # re-materialising the attention scores for the whole thing.
    #
    # Overridden here rather than fixed in the config on purpose: the saved
    # config is the artifact of a build we did not run, and eval should be
    # able to score a model somebody else stitched.
    model.config.use_cache = True
    # AND THE GENERATION CONFIG, WHICH IS THE ONE generate() ACTUALLY READS.
    # GenerationConfig.from_model_config() runs inside from_pretrained, so it
    # has already copied use_cache=False by the time we get here; setting only
    # model.config would have looked like a fix and changed nothing.
    gc_ = getattr(model, "generation_config", None)
    if gc_ is not None:
        gc_.use_cache = True
    return model, tok, device


def _sample_texts(path: str, num_samples: int) -> List[str]:
    """Pull up to num_samples raw text bodies out of a JSONL held-out file."""
    picked = _reservoir(path, num_samples)
    out: List[str] = []
    for line in picked:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = (item.get("text") or item.get("content")
                or item.get("prompt") or item.get("input") or "")
        if text:
            out.append(text)
    return out


def _prompt_and_reference(item: Dict[str, Any],
                          completion_split: float = 0.5) -> Tuple[str, str]:
    """Get (prompt, reference) out of a corpus row, whatever shape it is.

    TWO KINDS OF CORPUS, AND ONLY ONE HAS AN ANSWER KEY.

    A QA-shaped dataset has prompt/answer (or input/output, question/
    reference) and the reference is exactly what the model should say. That is
    the easy case and it was the only case handled - so a run on a `stack` or
    `gh` corpus, whose rows are just {"text": ...}, reported

        no sample had both a prompt and a reference

    for every expert. Which is true, and useless: the corpus most people will
    actually build an expert from cannot be scored at all.

    A RAW TEXT corpus has an answer key hiding in it. Hold back the second
    half of a held-out document and the first half becomes the prompt: score
    the continuation against what really followed. That is a completion task,
    it needs no annotation, and for code it is close to what you want to know -
    given the top of this file, does the model write the rest of it like this
    project does?

    It is a WEAKER claim than exact-match QA and should be read as one. Exact
    match on a continuation is near zero for anything but boilerplate; ROUGE
    and BLEU carry the signal.
    """
    for pk, rk in (("prompt", "answer"), ("input", "output"),
                   ("question", "reference"), ("prompt", "reference")):
        prompt, reference = item.get(pk), item.get(rk)
        if prompt and reference:
            return str(prompt), str(reference)

    text = item.get("text") or item.get("content") or ""
    if not text:
        return "", ""

    # Split on a line boundary so the prompt ends somewhere a model would
    # plausibly be asked to continue from, rather than mid-token.
    lines = str(text).splitlines(keepends=True)
    if len(lines) < 4:
        return "", ""
    cut = max(1, int(len(lines) * completion_split))
    return "".join(lines[:cut]), "".join(lines[cut:])


def eval_generation(model_dir: str, test_data_path: str,
                    label: str, domain: str,
                    num_samples: int = 10,
                    max_new_tokens: int = 256,
                    max_prompt_tokens: int = 1024,
                    callback=None,
                    loaded=None) -> EvalResult:
    """Generate real tokens from a real model and score them against references.

    Replaces the prompt/reference overlap proxy. Every number here comes from
    something the model actually emitted; if the model cannot be loaded the
    result is UNMEASURABLE with a reason, never a substituted score.
    """
    result = EvalResult(expert_name=label, domain=domain, status="pending")

    ok, reason = _torch_available()
    if not ok:
        result.status = UNMEASURABLE
        result.note = reason
        return result
    if not Path(model_dir).is_dir():
        result.status = UNMEASURABLE
        result.note = f"no model at {model_dir} - run build first"
        return result

    if not Path(test_data_path).is_file():
        result.status = UNMEASURABLE
        result.note = f"cannot read held-out data: {test_data_path}"
        return result
    samples = _reservoir(test_data_path, num_samples)
    if not samples:
        result.status = "skipped"
        result.note = "no test data"
        return result

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # `loaded` lets a caller hand in a model it wants to reuse across several
    # calls. Whoever loads it frees it: this function releases only what it
    # opened itself, so a shared model is not pulled out from under the next
    # call.
    owns_model = loaded is None
    if loaded is not None:
        model, tok, device = loaded
    else:
        try:
            model, tok, device = _load_model(model_dir)
        except Exception as exc:
            result.status = UNMEASURABLE
            result.note = f"could not load {model_dir}: {exc}"
            return result

    seen_first = False
    peak_mib = 0.0
    em: List[float] = []
    r1: List[float] = []
    bl: List[float] = []
    lengths: List[int] = []
    scored = 0
    completion_mode = False

    for line in samples:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt, reference = _prompt_and_reference(item)
        if not prompt or not reference:
            continue
        if "prompt" not in item and "input" not in item:
            completion_mode = True

        batch = tok(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)

        # MEASURE THE SEQUENCE, DO NOT TRUST IT.
        #
        # truncation=True with max_length says this cannot exceed the cap. The
        # memory says otherwise: an 18 GB allocation on this model is an eager
        # attention matrix for roughly 25,000 tokens, which is 25x a cap that
        # was supposedly enforced. One of those two is wrong and guessing which
        # has already cost two runs, so the tensor now reports its own length
        # and the cap is applied a second time where it cannot be argued with.
        #
        # If the trace below ever fires, tokenizer truncation was not doing
        # what its arguments claim. If it never fires and we still OOM,
        # sequence length is exonerated and the cause is elsewhere.
        n_tok = int(batch["input_ids"].shape[-1])
        if n_tok > max_prompt_tokens:
            _trace(f"{label}/{domain}: TOKENIZER DID NOT TRUNCATE - got "
                   f"{n_tok} tokens for max_length={max_prompt_tokens}; "
                   f"slicing")
            batch = {k: v[:, :max_prompt_tokens] for k, v in batch.items()}
            n_tok = max_prompt_tokens
        if scored == 0 and not seen_first:
            seen_first = True
            _trace(f"{label}/{domain}: first prompt {n_tok} tokens, "
                   f"generating {max_new_tokens}")

        batch = {k: v.to(model.device) for k, v in batch.items()}
        # EVAL MUST NOT TAKE THE MACHINE DOWN.
        #
        # An OOM here used to propagate out of run_eval, out of the CLI, and
        # on a unified-memory box frequently past the point where anything
        # could still be printed - the last two runs ended with the process
        # killed and an SSH session closing, which is the least informative
        # possible failure. Whatever the cause, the CORRECT behaviour is the
        # same and it is the one this project already has a vocabulary for:
        # say what could not be measured, say why, hand the memory back, and
        # keep going. Unmeasurable is not pass; it is also not a crash.
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **batch, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tok.eos_token_id)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            # ONE RETRY, AFTER HANDING THE RESERVATION BACK.
            # An OOM whose cause is fragmentation is not an OOM the second
            # time: the bytes were never in use, they were stranded in
            # segments the allocator could not reuse. If it fails again the
            # model genuinely does not fit and we say so.
            reserved, live = _balloon()
            retried = False
            if reserved > max(live, 1.0) * 2:
                try:
                    import torch as _t
                    _t.cuda.empty_cache()
                    with torch.no_grad():
                        out_ids = model.generate(
                            **batch, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tok.eos_token_id)
                    retried = True
                    _trace(f"{label}/{domain}: recovered after deflating "
                           f"{reserved:,.0f}MiB reserved / {live:,.0f}MiB live")
                except Exception:
                    retried = False
            if retried:
                pass
            else:
                batch = None
                if owns_model:
                    model = None
                release_memory()
                mem = _mem_mb()
                result.status = UNMEASURABLE
                result.note = (
                    f"out of memory after {scored} of {len(samples)} samples "
                    f"on a {n_tok}-token prompt (+{max_new_tokens} new): "
                    f"{exc}. Allocator held {reserved:,.0f}MiB reserved for "
                    f"{live:,.0f}MiB of live tensors "
                    f"({reserved / max(live, 1.0):.1f}x)"
                    + ("; that ratio is fragmentation, not a model that needs "
                       "the memory - try "
                       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                       if reserved > max(live, 1.0) * 3 else "")
                    + ". At the failure: "
                    + ", ".join(f"{k}={v:,.0f}MiB" for k, v in mem.items())
                    + ". Nothing is scored from a partial run.")
                _trace(f"OOM in {label}/{domain} after {scored} samples, "
                       f"prompt was {n_tok} tokens, "
                       f"reserved/live = {reserved:,.0f}/{live:,.0f}MiB")
                return result
        generated = tok.decode(
            out_ids[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)

        try:
            peak_mib = max(peak_mib,
                           torch.cuda.max_memory_allocated() / 2 ** 20)
        except Exception:
            pass

        em.append(_exact_match(generated, reference))
        r1.append(_rouge1(generated, reference))
        bl.append(_bleu_simple(generated, reference))
        lengths.append(len(_tokenize_simple(generated)))
        scored += 1
        if scored % 4 == 0:
            _deflate()
        if callback and scored % 5 == 0:
            callback("eval.quality", "running", f"{label}: {scored}/{len(samples)}")

    # Only free what this call opened. Releasing a BORROWED model here would
    # pull it out from under the caller's next call, which is the bug that
    # sharing was introduced to avoid.
    if owns_model:
        model = None
        release_memory()

    if not scored:
        result.status = UNMEASURABLE
        result.note = ("no sample yielded a prompt and a reference - rows need "
                       "prompt/answer keys, or a `text` field long enough "
                       "(4+ lines) to split into a completion task")
        return result

    result.exact_match = sum(em) / scored
    result.rouge1 = sum(r1) / scored
    result.bleu = sum(bl) / scored
    result.avg_length = sum(lengths) / scored
    result.status = "done"
    if peak_mib:
        _trace(f"{label}/{domain}: peak {peak_mib:,.0f}MiB over {scored} samples")
    result.note = (f"{scored} samples generated"
                   + (" (completion: second half of each held-out doc; "
                      "read ROUGE/BLEU, exact-match is near zero by nature)"
                      if completion_mode else ""))
    return result


def detect_dead_experts(report: EvalReport, threshold: float = 1.2,
                        dead_share_frac: float = 0.2) -> List[str]:
    """Flag experts the router does not prefer on their own ground.

    THE DEFINITION CHANGED, and the old one is why this function could not work.
    It used to compare generation SCORES: dead if the expert scored under a
    threshold and the MoE beat it. Two things were wrong with that. The lookup
    was `report.stages.get(f"moe/{domain}", result)` - defaulting to the very
    thing being compared, so the test was `expert_score > expert_score`, always
    false, and nothing could ever be flagged on any input. And the underlying
    idea was wrong anyway: a dead expert is not one that writes badly, it is
    one the ROUTER NEVER ROUTES TO. That is a routing fact, and it now comes
    from probe_router_discrimination.

    Quality is still reported, it just is not what "dead" means.
    """
    routing = report.routing or {}
    experts = routing.get("experts") or {}

    if routing.get("status") == UNMEASURABLE or not experts:
        reason = routing.get("reason") or "routing was not measured"
        report.unmeasured.append(f"dead-expert check: {reason}")
        report.dead_experts = []
        return []

    # INPUT-BLINDNESS KILLS ENRICHMENT, NOT SHARE, AND THIS USED TO KILL BOTH.
    #
    # The early return here threw the whole table away and reported zero dead
    # experts. On a real run that printed
    #
    #     csharp  own 0.045  share 0.043      <- 4.3% against 0.50 for uniform
    #     python  own 0.957  share 0.955
    #     INPUT-BLIND - the router ignores its input entirely
    #     Eval complete. 0 dead expert(s).
    #
    # which is a router collapsed onto one expert, reported as a clean bill.
    # The two numbers answer different questions and only one of them depends
    # on the input:
    #
    #   ENRICHMENT is own-source share divided by other-source share. If
    #   routing does not vary with the source, both terms are the same number
    #   and the ratio is noise. Correctly discarded.
    #
    #   SHARE is how often the expert is selected AT ALL. It is a marginal, it
    #   does not reference the source, and an expert selected on 4% of tokens
    #   when uniform is 50% is a passenger whether or not the router reads its
    #   input. That measurement was fine and we deleted it.
    #
    # So blindness now suppresses the SPECIALISATION verdict and nothing else.
    blind = routing.get("mean_js_bits", 1.0) < 1e-3
    if blind:
        report.unmeasured.append(
            "specialisation: routing is input-blind (mean JS ~ 0) - the router "
            "ignores its input, so per-expert enrichment is noise. Selection "
            "share is still measurable and is read below.")

    # TWO FAILURES, TWO WORDS. This used to call both of them dead.
    #
    # DEAD is a usage fact: the router does not route to it. That is the
    # failure Ms.MoE uniquely claims to prevent, it means the stitch produced
    # a passenger, and it is measured against what uniform routing would give
    # (top_k / num_experts), not against a fixed number - "below 20% of
    # uniform" means the same thing whether there are 2 experts or 12.
    #
    # UNDISCRIMINATING is a specialisation fact: it gets plenty of traffic and
    # shows no preference for its own domain. Worth knowing, worth fixing in
    # router training, and NOT a broken stitch.
    E = int(routing.get("n_experts") or len(experts) or 1)
    K = int(routing.get("top_k") or 1)
    uniform = K / max(E, 1)
    floor = uniform * dead_share_frac

    dead: List[str] = []
    weak: List[str] = []
    for name, info in experts.items():
        enrichment = info.get("enrichment", 0.0)
        share = info.get("marginal_share")
        if share is None:
            share = info.get("own_share", uniform)

        if share < floor:
            dead.append(name)
            why = (f"dead: selected on {share:.3f} of tokens against "
                   f"{uniform:.3f} for uniform routing - the router does not "
                   f"route to it")
        elif blind:
            # Used enough to be alive; whether it SPECIALISES is unanswerable
            # here, and saying nothing beats guessing from a noise ratio.
            continue
        elif enrichment < threshold or info.get("outranked"):
            weak.append(name)
            why = (f"not specialised: enrichment {enrichment:.2f}x < "
                   f"{threshold}x, but selected on {share:.3f} of tokens "
                   f"({uniform:.3f} is uniform) - used, just not preferentially")
            if info.get("outranked"):
                why += (f"; outranked on its own domain by "
                        f"{info.get('top_competitor')} "
                        f"({info.get('top_competitor_share', 0):.3f} vs "
                        f"{info.get('own_share', 0):.3f})")
        else:
            continue
        result = report.stages.get(name)
        if result is not None:
            result.note = why

    # THE TEST HAS A POWER FLOOR AND IT IS A FUNCTION OF E ALONE.
    # "own-expert is the column maximum for n/n" has probability 1/E^E by
    # chance. At E=2 that is 0.25 - the headline can never be significant, no
    # matter how clean the table looks. Say it next to the result rather than
    # leaving the reader to notice p=0.25 on their own.
    # COLLAPSE IS ITS OWN DIAGNOSIS AND DESERVES ITS OWN SENTENCE. "n-1 dead
    # experts" is technically what happened; "the router put 96% of every
    # source's tokens through one expert" is what a person can act on, and it
    # points at the load-balancing coefficient rather than at the stitch.
    hog = max(experts.items(),
              key=lambda kv: kv[1].get("marginal_share",
                                       kv[1].get("own_share", 0.0)),
              default=(None, {}))
    hog_share = hog[1].get("marginal_share", hog[1].get("own_share", 0.0)) if hog[0] else 0.0
    if hog[0] and E >= 2 and hog_share > 1.0 - floor:
        report.caveats.append(
            f"router collapsed onto {hog[0]}: it takes {hog_share:.1%} of all "
            f"tokens regardless of source ({uniform:.1%} is uniform). This is "
            f"rich-get-richer in a top-{K} router, not a stitch problem - the "
            f"load-balancing loss is what holds it off, and "
            f"router.aux_loss_coef controls how hard. Switch used 0.01, "
            f"Mixtral 0.02.")

    p_floor = 1.0 / (E ** E) if E else 1.0
    if p_floor > 0.05:
        report.caveats.append(
            f"own-column test cannot reach significance with {E} experts: "
            f"all-{E}-of-{E} happens by chance with p={p_floor:.3f}. The "
            f"enrichment numbers are real; the 'n/n won their column' "
            f"headline is not evidence at this width.")

    report.dead_experts = dead
    report.undiscriminating = weak
    return dead


def run_eval(config, spec: Optional[Dict[str, Any]] = None) -> EvalReport:
    """Run the eval pipeline.

    Args:
        config: PipelineConfig
        spec: eval spec, from the recipe's `eval:` block. Keys:
            - script: custom eval script path (replaces ours entirely)
            - mode: "routing" | "quality" | "all" (default "all")
            - held_out_fraction: fraction held back (default 0.1)
            - num_samples: samples per expert (default 20)
            - dead_threshold: min enrichment before dead (default 1.2)

    Returns:
        EvalReport. Anything unmeasurable is reported as such, never scored.
    """
    spec = spec or {}
    report = EvalReport(ok=True, message="")

    held_out = spec.get("held_out_fraction", 0.1)
    num_samples = spec.get("num_samples", 20)
    dead_threshold = spec.get("dead_threshold", 1.2)
    mode = spec.get("mode", "all")
    if mode not in ("routing", "quality", "experts", "all"):
        report.ok = False
        report.message = (f"unknown eval mode {mode!r} "
                          f"(expect routing|quality|experts|all)")
        return report

    # Custom script replaces us entirely. We provide the floor; this is the
    # door out of it.
    custom_script = spec.get("script")
    if custom_script:
        return _run_custom_eval(custom_script, config, held_out, num_samples,
                                report)

    # ── locate corpora and artifacts ───────────────────────────────────────
    code_paths: Dict[str, str] = {}
    data_root = Path(config.data_root)
    for f in sorted(data_root.glob("*.jsonl")):
        name = f.stem.replace("_code", "")
        if name:
            code_paths[name] = str(f)

    if not code_paths:
        report.ok = False
        report.message = f"no data files found in {data_root} — cannot eval"
        return report

    output_root = Path(config.output_root)
    moe_dir = str(output_root / st.ARTIFACTS[st.ROUTER])

    # THE GATE AXIS BELONGS TO THE MODEL, NOT THE RECIPE.
    #
    # This used to read: "Expert order is the recipe's order, which is STITCH
    # order, which is the expert axis of the router's gate matrix." Two of
    # those three are the same thing and the first is not. The recipe's order
    # is the stitch order only if THIS recipe produced THAT stitch, and a
    # skeleton on disk survives a recipe edit.
    #
    # It broke exactly that way: expert list reordered, moe_trained deleted,
    # moe_untrained left in place, so the router retrained on a skeleton whose
    # gate axis was still the old order. eval labelled column 0 with the new
    # recipe's first name. Every routing number printed under the wrong
    # expert - collapse attributed to the wrong side, a dead expert named as
    # the live one - and nothing anywhere said a word.
    #
    # The skeleton stamps `expert_names`. That IS the axis. Read it, believe
    # it over the recipe, and say loudly when they disagree.
    expert_order = [n for n in (config.expert_names or list(code_paths))
                    if n in code_paths]
    routing_refused = ""
    try:
        with open(Path(moe_dir) / "config.json", encoding="utf-8") as _fh:
            moe_names = [n for n in (json.load(_fh).get("expert_names") or [])]
    except (OSError, ValueError):
        moe_names = []
    if moe_names and moe_names != expert_order:
        if sorted(moe_names) == sorted(expert_order):
            report.caveats.append(
                f"expert ORDER on disk {moe_names} differs from the recipe "
                f"{expert_order}; the model's order is the gate axis and has "
                f"been used. The skeleton predates this recipe - rebuild with "
                f"--force, or delete moe_untrained, if that was not intended.")
            expert_order = [n for n in moe_names if n in code_paths]
        else:
            routing_refused = (
                f"the MoE was stitched from {moe_names} but the recipe names "
                f"{expert_order}. These are different models; nothing can be "
                f"attributed to an expert. Rebuild the stitch.")
            report.caveats.append(f"REFUSED to label routing: {routing_refused}")

    global _SAMPLER
    t_start = time.time()
    if os.environ.get("MSMOE_MEM_SAMPLE", "1") != "0":
        _SAMPLER = _MemSampler(
            float(os.environ.get("MSMOE_MEM_INTERVAL", "0.25"))).start()
    _trace("eval start")
    held_paths: Dict[str, str] = {}
    for expert_name in expert_order:
        _, held_path = _load_or_split(code_paths[expert_name], held_out)
        held_paths[expert_name] = held_path
    _trace("held-out splits written")

    if routing_refused:
        report.routing = {"status": UNMEASURABLE, "reason": routing_refused,
                          "experts": {}}
    do_routing = mode in ("routing", "all") and not routing_refused
    do_quality = mode in ("quality", "all")
    do_experts = mode in ("experts", "all")

    # ── experts: was there ever anything to route on? ──────────────────────
    #
    # FIRST, because it is the question the other two make you ask. A routing
    # table at 1.00x enrichment sends you to router hyperparameters; whether
    # that is the right place to go depends entirely on whether the experts
    # differ and whether routing correctly lowers the loss - and those are
    # measured here.
    if do_experts:
        from . import experts as experts_mod
        expert_dirs = {
            n: str(output_root / st.FINETUNE_ARTIFACT.format(expert=n))
            for n in expert_order}
        _trace("expert checks")
        gate = experts_mod.run_experts(
            config, expert_dirs, held_paths=held_paths, moe_dir=moe_dir,
            spec={"num_samples": num_samples})
        report.experts = gate.to_dict()
        report.caveats.extend(gate.findings)
        report.unmeasured.extend(f"experts/{u}" for u in gate.unmeasured)

    # ── routing: the dead-expert measurement ───────────────────────────────
    if do_routing:
        # HELD OUT BY CONSTRUCTION. mixed_all.jsonl is exactly what the router
        # trained on, so excluding those rows is the difference between
        # measuring discrimination and measuring memorisation.
        trained_on = set()
        mixed = Path(config.output_root) / "mixed_all.jsonl"
        if not mixed.exists():
            mixed = Path(config.data_root) / "mixed_all.jsonl"
        if mixed.exists():
            for line in _iter_jsonl(str(mixed)):
                try:
                    trained_on.add(_digest(json.loads(line)["text"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        _trace(f"trained-on set built ({len(trained_on)} rows)")

        report.routing = probe_router_discrimination(
            moe_dir=moe_dir,
            held_paths=held_paths,
            expert_order=expert_order,
            num_samples=num_samples,
            dead_threshold=dead_threshold,
            trained_on=trained_on,
        )
        if report.routing.get("status") == UNMEASURABLE:
            report.unmeasured.append(
                f"routing: {report.routing.get('reason', 'unknown')}")

    # ── quality: real generation against held-out references ───────────────
    if do_quality:
        # ONE MODEL IN MEMORY AT A TIME, AND THE MoE LOADED ONCE.
        #
        # This loop used to call eval_generation(moe_dir, ...) INSIDE the
        # per-expert loop, so the MoE was loaded once per expert - plus once
        # more for the routing probe. With two experts that is five loads for
        # three distinct models, and because the old cleanup freed nothing,
        # every one of them stayed resident. On a 121 GB unified-memory Spark
        # the OOM killer arrived on the third.
        #
        # Now: each specialist is loaded, scored on its own held-out set, and
        # released before the next one; then the MoE is loaded ONCE and scored
        # against every domain. Peak is one model, and the cost of eval stops
        # scaling with the number of experts.
        for expert_name in expert_order:
            expert_dir = str(output_root /
                             st.FINETUNE_ARTIFACT.format(expert=expert_name))
            _trace(f"before {expert_name}")
            res = eval_generation(
                model_dir=expert_dir, test_data_path=held_paths[expert_name],
                label=expert_name, domain=expert_name,
                num_samples=num_samples)
            report.stages[expert_name] = res
            if res.status == UNMEASURABLE:
                report.unmeasured.append(f"quality/{expert_name}: {res.note}")
            release_memory()
            _trace(f"after {expert_name}")

        moe_model = moe_tok = None
        _trace("before MoE load")
        try:
            moe_model, moe_tok, moe_device = _load_model(moe_dir)
            _trace("after MoE load")
        except Exception as exc:
            for expert_name in expert_order:
                report.stages[f"moe/{expert_name}"] = EvalResult(
                    expert_name="moe", domain=expert_name,
                    status=UNMEASURABLE,
                    note=f"could not load {moe_dir}: {exc}")
                report.unmeasured.append(
                    f"quality/moe/{expert_name}: could not load the MoE")

        if moe_model is not None:
            for expert_name in expert_order:
                moe_res = eval_generation(
                    model_dir=moe_dir, test_data_path=held_paths[expert_name],
                    label="moe", domain=expert_name,
                    num_samples=num_samples,
                    loaded=(moe_model, moe_tok, moe_device))
                report.stages[f"moe/{expert_name}"] = moe_res
                if moe_res.status == UNMEASURABLE:
                    report.unmeasured.append(
                        f"quality/moe/{expert_name}: {moe_res.note}")
            moe_model = moe_tok = None
            release_memory()
            _trace("after MoE released")

    # ── verdict ────────────────────────────────────────────────────────────
    if do_routing:
        detect_dead_experts(report, threshold=dead_threshold)
    else:
        report.unmeasured.append(
            f"dead-expert check: needs --mode routing or all (ran {mode})")

    if _SAMPLER is not None:
        _SAMPLER.stop()
        print(_SAMPLER.table(), file=sys.stderr, flush=True)
        _SAMPLER = None

    elapsed = time.time() - t_start
    report.message = (f"Eval complete in {elapsed:.0f}s. "
                      f"mode={mode}. "
                      f"{len(report.dead_experts)} dead expert(s), "
                      f"{len(report.undiscriminating)} undiscriminating. "
                      f"{len(report.unmeasured)} thing(s) unmeasurable.")
    # ok means "we ran and did not error", NOT "everything passed" and NOT
    # "everything was measurable". The caller decides the exit code from
    # dead_experts and unmeasured, so a silent no-op can never read as success.
    report.ok = True
    return report


def _run_custom_eval(custom_script: str, config, held_out: float,
                     num_samples: int, report: EvalReport) -> EvalReport:
    """Hand the whole question to the user's own script."""
    script_path = Path(custom_script)
    if not script_path.is_file():
        report.ok = False
        report.message = f"eval.script {custom_script!r} does not exist"
        return report
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--data-root", config.data_root,
             "--output-root", config.output_root,
             "--held-out", str(held_out),
             "--num-samples", str(num_samples)],
            capture_output=True, text=True, timeout=3600,
        )
    except Exception as exc:
        report.ok = False
        report.message = f"custom eval script error: {exc}"
        return report
    report.ok = result.returncode == 0
    report.message = (f"custom eval script {'succeeded' if report.ok else 'failed'}"
                      f" (exit {result.returncode})")
    if not report.ok and result.stderr:
        report.message += f": {result.stderr[:300]}"
    return report


def eval_from_manifest(run_dir: Path) -> EvalReport:
    """Read eval results from a run's manifest/eval record.

    Used after a build to load previous eval results.
    """
    report = EvalReport(ok=True, message="")

    eval_path = run_dir / "eval_report.json"
    if not eval_path.is_file():
        report.ok = False
        report.message = "no eval report found"
        return report

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    report.ok = data.get("ok", False)
    report.message = data.get("message", "")
    report.dead_experts = data.get("dead_experts", [])
    report.undiscriminating = data.get("undiscriminating", [])
    report.caveats = data.get("caveats", [])
    report.experts = data.get("experts", {})

    for name, info in data.get("stages", {}).items():
        r = EvalResult(
            expert_name=name,
            domain=info.get("domain", ""),
            exact_match=info.get("exact_match", 0.0),
            rouge1=info.get("rouge1", 0.0),
            bleu=info.get("bleu", 0.0),
            avg_length=info.get("avg_length", 0.0),
            status=info.get("status", "pending"),
            note=info.get("note", ""),
        )
        report.stages[name] = r

    return report


def save_eval_report(report: EvalReport, path: Path) -> None:
    """Save eval report to disk as JSON."""
    data = {
        "ok": report.ok,
        "message": report.message,
        "dead_experts": report.dead_experts,
        "undiscriminating": report.undiscriminating,
        "caveats": report.caveats,
        "experts": report.experts,
        "stages": {
            name: {
                "expert_name": r.expert_name,
                "domain": r.domain,
                "exact_match": r.exact_match,
                "rouge1": r.rouge1,
                "bleu": r.bleu,
                "avg_length": r.avg_length,
                "status": r.status,
                "note": r.note,
            }
            for name, r in report.stages.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
