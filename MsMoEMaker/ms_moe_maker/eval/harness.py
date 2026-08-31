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
    max_new_tokens: -1             # -1 = 256, or 1024 when the run reasons

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

from ..config import pipeline as cfg_module
from ..run import stages as st
from ..run import manifest as mf
from .record import ERROR, FAIL, PASS, UNMEASURABLE


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
    # HOW MANY ROWS THE AVERAGES ABOVE ARE ACTUALLY OVER. Its own field,
    # because this used to live only in `note` - free text that
    # detect_dead_experts overwrites, so the JSON lost it too - and a 3-row
    # mean printed next to a 20-row mean with nothing to tell them apart is
    # not a comparison. A `stack` corpus of mostly short files yields 3 usable
    # rows out of 20: _prompt_and_reference needs 4+ lines to make a
    # completion task, and everything shorter is skipped in silence.
    scored_samples: int = 0
    """Rows that yielded a prompt AND a reference, and were scored."""
    attempted_samples: int = 0
    """Rows drawn from the held-out file, scored or not."""
    # Fraction of outputs that emitted a NON-EMPTY reasoning block. -1 = not
    # applicable (the model under eval is not a reasoning model); 0..1 when the
    # eval split the trace from the answer. Reasons-but-wrong and
    # doesn't-reason-at-all are different outcomes and must not collapse.
    reasoned: float = -1.0
    # HOW MANY GENERATIONS RAN OUT THE CLOCK instead of stopping on their own.
    # Here so two identical-looking numbers can be told apart: a low `reasoned`
    # because the model never closes its think block, and a low `reasoned`
    # because the budget ended mid-think and there was no close tag left to
    # find. The first is a finding about the model, the second is a finding
    # about eval.max_new_tokens, and reporting one as the other is the lie
    # this field exists to prevent.
    capped_generations: int = 0
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


def _own_column_p(hits: int, n: int):
    """P(at least `hits` of `n` experts top their own column by chance).

    THE OLD NUMBER ANSWERED A QUESTION NOBODY HAD ASKED. It was hardcoded to
    1/n^n - the probability that ALL n experts win their own column - and it
    got printed whatever the actual count was. On a table where one expert of
    five topped its column the report still announced p=0.00032, a
    significance figure for an event that did not occur, sitting directly
    under a failing table. That is the worst kind of wrong number: decisive
    looking, and about something else.

    The upper binomial tail asks what the count actually poses - if every
    column's maximum were a coin toss between the n experts on the table, how
    often would at least this many land on their own? - under exactly the same
    independence assumption 1/n^n already made. And it COLLAPSES to 1/n^n at
    hits == n, so the proven 0.5B headline (5 of 5, p=0.00032) is unchanged.

    Returns None when there is nothing to test.
    """
    import math

    if n <= 0 or hits < 0:
        return None
    hits = min(hits, n)
    p = 1.0 / n
    return sum(math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
               for k in range(hits, n + 1))


def _pooled_shares(counts_by_layer, totals_by_layer, n_experts, top_k):
    """Selection share per expert, pooled over layers. SLOTS, not tokens.

    The denominator is total selection SLOTS - tokens x top_k - which is why
    the shares sum to 1.0 across experts and uniform routing gives each of
    them 1/E rather than K/E. Pulled out of the pooled loop so the per-segment
    numbers below are computed by this arithmetic and not by a second copy of
    it that can drift; getting that denominator wrong once already turned a
    healthy expert into a STARVED one.
    """
    tot = sum(totals_by_layer) * top_k
    return [sum(layer[e] for layer in counts_by_layer) / max(tot, 1)
            for e in range(n_experts)]


def think_token_segments(text: str, offsets, style):
    """Token ranges INSIDE the think block(s) and AFTER the last one.

    Returns (inside, after) as lists of [start, end) TOKEN index pairs, or
    None when there is no complete block in this text to segment on.

    HOW THE CHARACTERS WERE MAPPED TO TOKENS, because that is the part that
    can quietly be wrong. reasoning.split() answers "did this reason" in
    STRINGS - it cannot say where - and re-tokenizing the two halves
    separately would not line up with the sequence the model actually routed,
    because a tokenizer merges across any boundary you cut. So the delimiters
    are found in the SAME string that was tokenized, and char -> token comes
    from the tokenizer's own `offset_mapping`: the fast tokenizer telling us
    exactly which characters each token covers. A token belongs to a segment
    when its span OVERLAPS that segment. Special tokens carry (0, 0), cover no
    characters, and are skipped rather than being silently filed under the
    first segment that starts at 0.

    The delimiters come off the run's resolved ReasoningStyle - the same
    `open`/`close` reasoning.split() reads - so this cannot drift from the
    table the traces were written with.

    `after` comes back empty when max_len cut the sequence mid-think; that is
    a truncated probe, not a routing finding, and the caller drops it.
    """
    if style is None or not text or not offsets:
        return None
    opening = getattr(style, "open", "")
    closing = getattr(style, "close", "")
    if not opening or not closing:
        return None

    # Every complete block, so an `interwoven` style (many blocks around tool
    # calls) is segmented the same way the splitter reads it: all interiors are
    # "inside", and "after" is whatever follows the LAST close tag.
    spans, i = [], 0
    while True:
        a = text.find(opening, i)
        if a == -1:
            break
        b = text.find(closing, a + len(opening))
        if b == -1:
            break
        spans.append((a + len(opening), b))
        i = b + len(closing)
    if not spans:
        return None
    after_start = i

    inside_idx, after_idx = [], []
    for t, span in enumerate(offsets):
        try:
            s0, s1 = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s1 <= s0:
            continue
        if any(s0 < end and s1 > start for start, end in spans):
            inside_idx.append(t)
        elif s0 >= after_start:
            after_idx.append(t)

    def _ranges(idx):
        out = []
        for t in idx:
            if out and out[-1][1] == t:
                out[-1][1] = t + 1
            else:
                out.append([t, t + 1])
        return [tuple(r) for r in out]

    return _ranges(inside_idx), _ranges(after_idx)


def summarize_think_swing(think, after, min_swing: float = 0.05):
    """Relay or duet? The DELTA is the finding, so the delta is what is reported.

    Two share tables printed side by side make the reader do the subtraction,
    and the subtraction IS the question the reasoning expert exists to raise:
    if reasoning behaves as a RELAY, the reasoning expert's share should drop
    the moment `</think>` closes and the domain expert's should rise. If it
    behaves as a DUET - both contributing throughout - nothing moves and the
    delta is noise.

    Nobody has measured this; the configuration is new. So this reports the
    number and names the two readings rather than asserting one of them.

    `min_swing` is a difference in SHARE, not a ratio: below 0.05 of the
    selection slots there is nothing to attribute to anything.
    """
    names = [n for n in think if n in after]
    delta = {n: think[n] - after[n] for n in names}
    out = {"think": dict(think), "after": dict(after), "delta": delta,
           "swing_to": "", "swing": 0.0, "yields_to": "",
           "verdict": "unmeasured"}
    if not delta:
        return out
    swing_to = max(delta, key=lambda n: delta[n])
    yields_to = min(delta, key=lambda n: delta[n])
    out["swing"] = delta[swing_to]
    if delta[swing_to] >= min_swing:
        out["verdict"] = "relay"
        out["swing_to"] = swing_to
        out["yields_to"] = yields_to
    else:
        out["verdict"] = "duet"
    return out


def probe_router_discrimination(moe_dir: str,
                                held_paths: Dict[str, str],
                                expert_order: Sequence[str],
                                num_samples: int = 16,
                                dead_threshold: float = 1.2,
                                max_len: int = 1024,
                                device: str = "cpu",
                                trained_on: Optional[set] = None,
                                raw_text_sources: Optional[Sequence[str]] = None,
                                callback=None,
                                reasoning_style=None) -> Dict[str, Any]:
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

    # AN EXPERT THAT LOST ITS HELD-OUT SET IS NOT AN EXPERT THAT PASSED.
    #
    # Rows are excluded when they appear in the router's training mix - held
    # out by construction, which is correct - but a large enough mix can
    # consume a source ENTIRELY. When that happened the probe carried on with
    # whatever was left: three experts, two rows in the table, "column maximum
    # for 2/2", and p silently regressed from 0.037 to 0.250 because the test
    # had narrowed. The missing expert was then reported as neither dead nor
    # undiscriminating, i.e. it passed a check that never ran.
    #
    # Narrowing the test is allowed. Narrowing it QUIETLY is not.
    excluded = [n for n in expert_order if n not in sources]
    if len(sources) < 2:
        return {"status": UNMEASURABLE,
                "reason": (f"need >=2 sources with held-out rows, found "
                           f"{sorted(sources)}"
                           + (f"; {excluded} had every held-out row consumed "
                              f"by the router mix - lower "
                              f"corpus.router_mix_total or rebuild so the "
                              f"mix draws from the .train split"
                              if excluded else "")),
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

    # DOES ROUTING SWING AT THE TAG BOUNDARY? The pooled share above cannot
    # answer it: it averages over the whole sequence, so a reasoning expert
    # that owns `<think>...</think>` and hands off at the close tag, and one
    # that contributes evenly throughout, produce the SAME number. Relay and
    # duet are different architectures with the same pooled row.
    #
    # Strictly additive. Every accumulator here stays at zero unless a
    # reasoning style was passed AND a sample has a complete block AND the
    # tokenizer can give offsets; the pooled path above does not read any of
    # it. No block, no offsets, no style - the table is exactly what it was.
    seg_counts = {k: {s: [[0] * E for _ in range(L)] for s in src_names}
                  for k in ("think", "after")}
    seg_totals = {k: {s: [0] * L for s in src_names}
                  for k in ("think", "after")}
    seg_samples = {s: 0 for s in src_names}
    seg_errors: Dict[str, str] = {}

    # PROBE IN THE FORMAT THE ROUTER WAS TRAINED IN, OR MEASURE FORMATTING.
    #
    # This asked every source with `Write {safe_name}:` inside a chat
    # template. Router training does neither of those things:
    #
    #   * it uses _make_code_prompt with the DISPLAY name ("C#", not
    #     "csharp") and one of six templates, so the probe's single made-up
    #     line is a string the router has literally never seen; and
    #   * for the tools expert and every reasoning expert its format_fn
    #     returns ex["text"] RAW - no chat template at all - while the probe
    #     wrapped them anyway.
    #
    # The mismatch is therefore WORSE for some experts than others: the code
    # experts get an approximate match and the tools/synth expert - the
    # largest domain contrast a Ms.MoE can have - gets probed under a format
    # it never saw once, and its enrichment comes back depressed by a pure
    # formatting artefact wearing a routing result's clothes.
    #
    # _make_code_prompt is imported from the trainer rather than copied, so
    # the two cannot drift; it is a function-local import because train.router
    # reaches back into this module (_load_or_split) and pulls torch in on the
    # way past.
    from ..config.pipeline import DISPLAY_LANG
    from ..train.router import _make_code_prompt

    raw_sources = {s for s in (raw_text_sources or ()) if s}
    # unnamed_fraction=0.0 ON PURPOSE, and it is the one place this
    # deliberately differs from training. Training mixes in 25% un-named
    # prompts ("Write code:") so the router cannot lean entirely on the
    # language name; a PROBE built from those is asking the router to
    # discriminate on a prompt that names no source, which measures nothing
    # and only adds variance to the column it lands in.
    prompt_rnd = random.Random(0)
    format_errors: Dict[str, str] = {}

    for s in src_names:
        for text in rnd.sample(sources[s], min(num_samples, len(sources[s]))):
            if s in raw_sources:
                # The raw path, exactly as router.format_fn takes it for the
                # tools expert and the reasoning experts: no template, no eos.
                wrapped = text
            else:
                prompt = _make_code_prompt(s, DISPLAY_LANG.get(s, s),
                                           unnamed_fraction=0.0, rnd=prompt_rnd)
                msgs = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": text}]
                try:
                    wrapped = tok.apply_chat_template(msgs, tokenize=False) + (
                        tok.eos_token or "")
                except Exception as exc:
                    # SILENCE IS SIGNAL, AND THIS USED TO BE `pass`. Falling
                    # back to raw text is survivable; doing it invisibly means
                    # one column was measured under a different format from
                    # the rest of the table and the reader compares them
                    # anyway. Record it once per source and report it.
                    wrapped = text
                    format_errors.setdefault(
                        s, f"{exc.__class__.__name__}: {exc}")
                    _trace(f"routing probe: no chat template for {s} "
                           f"({exc.__class__.__name__}: {exc}) - "
                           f"falling back to raw text")
            # OFFSETS ARE ASKED FOR ONLY WHEN THEY WILL BE USED, and never
            # at the cost of the pooled number: a slow tokenizer has no
            # offset_mapping and raises, so record it once per source and fall
            # through to exactly the call this always made.
            offsets, enc = None, None
            if reasoning_style is not None:
                try:
                    enc = tok(wrapped, return_tensors="pt", truncation=True,
                              max_length=max_len, return_offsets_mapping=True)
                    offsets = [tuple(x)
                               for x in enc.pop("offset_mapping")[0].tolist()]
                    enc = enc.to(device)
                except Exception as exc:
                    seg_errors.setdefault(s, f"{exc.__class__.__name__}: {exc}")
                    offsets, enc = None, None
            if enc is None:
                enc = tok(wrapped, return_tensors="pt", truncation=True,
                          max_length=max_len).to(device)
            n_seq = int(enc["input_ids"].shape[-1])
            seg = None
            if offsets and len(offsets) == n_seq:
                seg = think_token_segments(wrapped, offsets, reasoning_style)
                # A block with nothing after it is a sequence max_len cut
                # mid-think, not a hand-off measured at zero. Drop it rather
                # than average a swing against an empty half.
                if seg and not (seg[0] and seg[1]):
                    seg = None
            if seg:
                seg_samples[s] += 1
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
                # Same selections, split by where in the sequence they
                # happened. `top.indices` is [tokens, K] with one row per
                # sequence position, which is what makes a token-index slice
                # meaningful here; the length check is because a router that
                # ever reports something other than one row per token would
                # otherwise mis-attribute every segment silently.
                if seg is not None and int(top.indices.shape[0]) == n_seq:
                    for _key, _ranges in (("think", seg[0]), ("after", seg[1])):
                        for _a, _b in _ranges:
                            part = top.indices[_a:_b]
                            if not part.numel():
                                continue
                            for e in part.flatten().tolist():
                                seg_counts[_key][s][li][e] += 1
                            seg_totals[_key][s][li] += int(part.shape[0])
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
    avg: Dict[str, List[float]] = {
        s: _pooled_shares(counts[s], totals[s], E, K) for s in src_names}

    # The same arithmetic, restricted to tokens inside the think block and to
    # tokens after it. Only sources that actually had complete blocks appear,
    # so an empty dict means "nothing here reasoned", not "no swing".
    segments: Dict[str, Any] = {}
    for s in src_names:
        if not seg_samples[s]:
            continue
        think = _pooled_shares(seg_counts["think"][s], seg_totals["think"][s],
                               E, K)
        after = _pooled_shares(seg_counts["after"][s], seg_totals["after"][s],
                               E, K)
        summary = summarize_think_swing(dict(zip(expert_names, think)),
                                        dict(zip(expert_names, after)))
        summary["samples"] = seg_samples[s]
        segments[s] = summary

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
        # UNIFORM IS 1/E, NOT K/E. `avg` above divides by
        # `sum(totals[s]) * K` - i.e. by total selection SLOTS - so the shares
        # already sum to 1.0 across experts and uniform routing gives each of
        # them 1/E. Using K/E made this cutoff K times too high, so at top-2
        # a healthy expert on 0.30 of slots (90% of uniform) was blanked to
        # STARVED and dropped from mean_enrichment. Invisible in tests because
        # every routing test uses top_k=1, where K/E == 1/E.
        marginal = sum(col.values()) / max(len(col), 1)
        starved = marginal < (1.0 / max(E, 1)) * 0.2
        # COUNT THE HITS OVER THE SAME SET THE p-VALUE IS COMPUTED OVER.
        #
        # `hits` counted every expert that had a column; `n` counted only the
        # ones whose enrichment is readable. Three experts with one starved
        # expert topping its own row printed "column maximum for 3/2" - a
        # fraction above 1, next to a p computed at a width its own numerator
        # had never used.
        #
        # A starved expert's column maximum is the same handful of selections
        # its enrichment is, and we already refuse to read that as enrichment.
        # It does not get a vote in a test it cannot be evidence for.
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
        # p FOR THE EVENT THAT ACTUALLY HAPPENED. See _own_column_p: this used
        # to be 1/n^n regardless of `hits`, so a table where one of five
        # experts won its column still printed the all-five-of-five figure.
        "p_value": _own_column_p(hits, n),
        # What that number is the probability OF, in words, so the printed
        # sentence and the number cannot drift apart again. The reader should
        # never have to know which event the harness had in mind.
        "p_value_event": (f"at least {hits} of {n} by chance" if n else ""),
        "mean_js_bits": mean_js,
        "js_per_layer": per_layer,
        # Experts the probe could not score at all. Reported separately from
        # `experts` so a reader cannot mistake a shorter table for a cleaner
        # one, and so the p-value can be read against the width it was
        # actually computed at.
        "excluded": excluded,
        "excluded_reason": (
            "every held-out row was consumed by the router's training mix "
            "(held out by construction). Lower corpus.router_mix_total, or "
            "rebuild on a version whose router mix draws from the .train "
            "split." if excluded else ""),
        # Mean softmax probability of the experts actually selected. Uniform
        # is 1/E (or K/E summed over the top-K); 1.0 means the gate is fully
        # saturated and the softmax has stopped being a distribution.
        "mean_gate_confidence": (conf_sum[0] / conf_n[0]) if conf_n[0] else None,
        "uniform_confidence": 1.0 / E if E else None,
        # Sources probed as RAW TEXT because that is how the router trained on
        # them. Reported so a reader can see which columns were built which
        # way rather than assuming one format across the table.
        "raw_text_sources": sorted(raw_sources & set(src_names)),
        # Sources whose chat template refused, with the reason. Empty is the
        # normal case; non-empty means those columns fell back to raw text.
        "prompt_format_errors": format_errors,
        # Selection share INSIDE `<think>...</think>` vs AFTER it, per source
        # that had complete blocks, with the delta and a relay/duet reading.
        # Empty whenever the run does not reason, the tokenizer cannot give
        # offsets, or nothing sampled had a closed block - this is additive
        # and never narrows the pooled table above.
        "think_segments": segments,
        # Sources where offsets were asked for and refused (slow tokenizer),
        # so a missing segment row cannot be read as "no think blocks here".
        "think_segment_errors": seg_errors,
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


def _truncate_reference(reference: str, max_new_tokens: int,
                        tok=None) -> Tuple[str, bool]:
    """Cut the reference down to the budget the model was actually given.

    BLEU WAS MEASURING HELD-OUT FILE LENGTH. `_prompt_and_reference` hands
    back the whole second half of a document - unbounded - while generation is
    capped at max_new_tokens. _bleu_simple's brevity penalty is exp(1 - r/c)
    with c stuck under that cap, so two experts writing equally well scored
    0.055 and 1.00 purely because one corpus has ~1000-token files and the
    other ~250. A 20x gap in the same column, indistinguishable from a real
    quality gap. _rouge1 has the mirror of it: set-recall over EVERY reference
    token, so a long reference caps the achievable score no matter what the
    model writes.

    TRUNCATION CHANGES WHAT THE METRIC MEANS, so the note says so. The score
    is now "how well does it continue the next max_new_tokens of this file",
    not "does it reproduce the rest of the file". That is the question the
    generation budget was already asking; the reference just stopped
    disagreeing with it.

    Returns (reference, was_truncated). The tokenizer is the honest unit
    because it is the one the budget is denominated in. Whitespace words are
    the fallback for anything that will not tokenize, and they are a GENEROUS
    bound - a subword token is never more than a word - so the fallback can
    only ever leave the reference longer than the model's real budget, never
    shorter than it.
    """
    if max_new_tokens <= 0 or not reference:
        return reference, False
    if tok is not None:
        try:
            ids = tok(reference, add_special_tokens=False)["input_ids"]
            if ids and not isinstance(ids[0], int):
                ids = ids[0]
            if len(ids) <= max_new_tokens:
                return reference, False
            cut = tok.decode(ids[:max_new_tokens], skip_special_tokens=True)
            if cut:
                return cut, True
        except Exception:
            pass
    words = reference.split()
    if len(words) <= max_new_tokens:
        return reference, False
    return " ".join(words[:max_new_tokens]), True


def _split_reasoning_answer(text: str, style) -> Tuple[str, bool]:
    """(answer, reasoned). The shared splitter, in eval's shape.

    THE SPLIT LIVES IN ONE PLACE NOW. There were two implementations - this
    one, which scored the answer, and data's, which validated a generated
    trace - and a scorer that splits differently from the writer is measuring
    a different artifact than the one on disk. reasoning.split() is both.
    """
    from ..config.reasoning import split
    _think, answer, reasoned = split(text, style)
    return answer, reasoned


def eval_generation(model_dir: str, test_data_path: str,
                    label: str, domain: str,
                    num_samples: int = 10,
                    # The historical cap, kept so a direct caller behaves the
                    # way it always did. run_eval never leaves it at this: it
                    # passes the value resolved from the recipe and the run's
                    # shape (pipeline.eval_max_new_tokens), because 256 cuts a
                    # thinking trace off before it reaches the answer.
                    max_new_tokens: int = 256,
                    max_prompt_tokens: int = 1024,
                    callback=None,
                    loaded=None,
                    reasoning_style=None) -> EvalResult:
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
    reasoned_count: List[int] = []
    scored = 0
    truncated_refs = 0
    capped = 0
    completion_mode = False

    for line in samples:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt, reference = _prompt_and_reference(item)
        if not prompt or not reference:
            continue
        # LIKE-FOR-LIKE OR IT IS NOT A COMPARISON. An unbounded reference next
        # to a hardcoded max_new_tokens turns the brevity penalty into a
        # measurement of the held-out file's length - see _truncate_reference.
        reference, was_cut = _truncate_reference(reference, max_new_tokens, tok)
        truncated_refs += 1 if was_cut else 0
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
        # COUNT THE ONES THAT RAN OUT THE CLOCK. A generation stopped by the
        # budget has no ending - no close tag, no answer - and from the score
        # alone that is indistinguishable from a model that never writes one.
        # Somebody has to count, or the caveat below cannot tell the reader
        # which of the two they are looking at.
        new_ids = out_ids[0][batch["input_ids"].shape[-1]:]
        if int(new_ids.shape[-1]) >= max_new_tokens:
            capped += 1
        generated = tok.decode(new_ids, skip_special_tokens=True)

        try:
            peak_mib = max(peak_mib,
                           torch.cuda.max_memory_allocated() / 2 ** 20)
        except Exception:
            pass

        # SCORE THE ANSWER, NOT THE THINKING. A reasoning model's trace is not
        # the deliverable; the answer after the close tag is. Track separately
        # whether it reasoned at all, so "reasons but wrong" and "never reasons"
        # stay distinct.
        answer, reasoned = _split_reasoning_answer(generated, reasoning_style)
        em.append(_exact_match(answer, reference))
        r1.append(_rouge1(answer, reference))
        bl.append(_bleu_simple(answer, reference))
        lengths.append(len(_tokenize_simple(answer)))
        reasoned_count.append(1 if reasoned else 0)
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
    if reasoning_style is not None:
        result.reasoned = sum(reasoned_count) / scored
    result.capped_generations = capped
    # THE DENOMINATOR TRAVELS WITH THE AVERAGE. Every number above is a mean
    # over `scored`, not over the rows we drew, and the difference is the
    # whole difference between a result and a rumour.
    result.scored_samples = scored
    result.attempted_samples = len(samples)
    result.status = "done"
    if peak_mib:
        _trace(f"{label}/{domain}: peak {peak_mib:,.0f}MiB over {scored} samples")
    result.note = (f"{scored} of {len(samples)} sampled rows scored"
                   + (" (completion: second half of each held-out doc; "
                      "read ROUGE/BLEU, exact-match is near zero by nature)"
                      if completion_mode else "")
                   # Truncation changes what BLEU/ROUGE mean, so it is stated
                   # wherever the scores are read, not just in the source.
                   + (f"; the reference was cut to the {max_new_tokens}-token "
                      f"generation budget on {truncated_refs} of them - these "
                      f"score the continuation the model had room to write, "
                      f"not the whole rest of the file"
                      if truncated_refs else ""))
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

    # SAY WHO WAS NOT ON THE TABLE, BEFORE ANYTHING IS CONCLUDED FROM IT.
    for name in routing.get("excluded") or []:
        report.unmeasured.append(
            f"routing/{name}: not scored - "
            f"{routing.get('excluded_reason', 'no held-out rows')}")
    if routing.get("excluded"):
        report.caveats.append(
            f"the routing table covers "
            f"{len(routing.get('experts') or {})} of "
            f"{len(routing.get('excluded') or []) + len(routing.get('experts') or {})} "
            f"experts: {routing['excluded']} had no held-out rows left. The "
            f"p-value below is for the NARROWER test, and every summary on "
            f"this table is an average over the experts that survived.")

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
    # UNIFORM IS 1/E. Shares are normalised over selection SLOTS upstream
    # (`tot = sum(totals[s]) * K`), so they sum to 1.0 across experts no matter
    # what K is. `K / E` therefore overstated uniform by exactly K, and at
    # top-2-of-3 - the shipped flow recipe - it reported live experts as DEAD
    # with exit code 2, printing "0.120 against 0.667 for uniform" directly
    # under three shares that visibly summed to 1.0.
    uniform = 1.0 / max(E, 1)
    floor = uniform * dead_share_frac
    # The most any single expert can take when every token spends K slots.
    # At K=1 this is 1.0 and the collapse test below is unchanged; at K>=2 the
    # old `> 1.0 - floor` trigger was arithmetically unreachable, so a router
    # shoving every token through one expert plus a random second could never
    # be diagnosed as collapsed and the reader was pointed at the stitch
    # instead of at router.aux_loss_coef.
    hog_ceiling = 1.0 / max(K, 1)

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
            # APPEND. This used to assign, and the note it clobbered was the
            # only place the scored-sample count and the completion-mode
            # caveat existed - written by eval_generation, destroyed here, and
            # gone from the JSON too. A routing verdict is not entitled to
            # delete a quality fact it did not write.
            result.note = f"{result.note}; {why}" if result.note else why

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
    # Proportional to the ceiling, not offset from it: at K=1 this is
    # exactly the old `1.0 - floor` trigger, and at K>=2 it asks the same
    # question ("is this expert taking essentially all the mass it CAN take?")
    # of a ceiling that is 1/K instead of 1. Subtracting a 1/E-derived floor
    # from a 1/K ceiling instead would tighten as K grows and flag a healthy
    # top-2 leader at 46% of slots.
    if hog[0] and E >= 2 and hog_share > hog_ceiling * (1.0 - floor):
        report.caveats.append(
            f"router collapsed onto {hog[0]}: it takes {hog_share:.1%} of all "
            f"selection slots regardless of source ({uniform:.1%} is uniform, "
            f"{hog_ceiling:.1%} is the top-{K} maximum). This is "
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


def reasoning_discipline_caveat(enrichment: float, reasoned: float,
                                capped_fraction: float, expert: str = "",
                                enrichment_floor: float = 1.2,
                                reasoned_floor: float = 0.5,
                                capped_ceiling: float = 0.25) -> str:
    """The register-without-the-discipline reading, or '' when it does not fit.

    HIGH ENRICHMENT NEXT TO LOW `reasoned` IS A NAMED FAILURE, NOT A PUZZLE,
    and it is the one a reader is least equipped to interpret on their own.
    The router is doing its job - it prefers this expert on its own ground -
    while the expert produces think-block-FLAVOURED text without reliably
    opening, sustaining and closing the block. It learned the register of
    deliberation, not the discipline of it.

    That is the PREDICTED failure mode of putting reasoning in a routed FFN
    expert: chain-of-thought is a sequence-level policy that lives in
    attention, and router training only trains the gates. So it is exactly
    what a user should be told when the numbers take this shape, instead of
    being handed two good-looking columns and a bad one.

    `capped_fraction` is the guard against saying it when it is not true. A
    generation that ran out its token budget has no close tag because it was
    CUT, not because the model failed to write one, and those two produce an
    identical `reasoned`. When enough of the run hit the budget the honest
    answer is "raise eval.max_new_tokens and ask again", which is a different
    caveat - so this one stands down rather than conflating them.

    Pure and scalar on purpose: the diagnosis is the part worth testing, and
    everything it needs is a number.
    """
    if enrichment < enrichment_floor:
        return ""
    if reasoned < 0 or reasoned >= reasoned_floor:
        return ""
    if capped_fraction > capped_ceiling:
        return ""
    who = expert or "the reasoning expert"
    return (
        f"{who}: routing works ({enrichment:.2f}x enrichment on its own "
        f"ground) but only {reasoned:.0%} of its outputs opened AND closed a "
        f"think block, with just {capped_fraction:.0%} of generations cut off "
        f"by the token budget - so truncation does not explain it. That shape "
        f"has a name: the expert learned the REGISTER of deliberation without "
        f"the DISCIPLINE - think-block-flavoured text that does not reliably "
        f"open, sustain and close the block. It is the predicted failure mode "
        f"of putting reasoning in a routed FFN expert: chain-of-thought is a "
        f"sequence-level policy and lives in attention, while router training "
        f"only trains the gates. Read the quality scores in that row as "
        f"scores on an unstructured answer, not on a reasoned one.")


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
            - max_new_tokens: generation budget per sample (default -1 =
              you decide, resolved from whether this run writes thinking
              traces - see pipeline.eval_max_new_tokens)

    Returns:
        EvalReport. Anything unmeasurable is reported as such, never scored.
    """
    spec = spec or {}
    report = EvalReport(ok=True, message="")

    held_out = spec.get("held_out_fraction", 0.1)
    num_samples = spec.get("num_samples", 20)
    dead_threshold = spec.get("dead_threshold", 1.2)
    # RESOLVED ONCE, HERE, because this is the first place that has both the
    # recipe's answer and the run's shape. A caller that never sets it (the
    # builder's post-build eval) gets the same automatic number the recipe's
    # -1 gets, instead of the old hardcoded 256 that no key could reach.
    max_new_tokens = cfg_module.eval_max_new_tokens(
        config, spec.get("max_new_tokens", -1))
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
    # THE FILENAME IS A RE-DERIVATION, AND IT DROPPED AN EXPERT.
    #
    # builder hands the router `expert_corpus_paths` precisely so nothing
    # downstream has to guess a corpus path from a name - and then eval guessed
    # anyway, stripping only `_code`. A `reasoning: true` expert writes
    # `<name>_reasoning.jsonl`, so it resolved to the expert "math_reasoning",
    # matched nothing in expert_names, and vanished from BOTH the quality table
    # and the routing columns without a word. Silence is signal: strip every
    # suffix the generators actually write, and say so below when an expert
    # still has no corpus.
    for f in sorted(data_root.glob("*.jsonl")):
        name = f.stem
        for _suffix in ("_reasoning", "_code"):
            if name.endswith(_suffix) and len(name) > len(_suffix):
                name = name[:-len(_suffix)]
                break
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
    # AN EXPERT WITH NO CORPUS IS A NARROWED TEST, NOT A SMALLER ONE. Anything
    # the recipe names and eval cannot find is stated, not skipped - the same
    # rule that made the router mix stop silently eating held-out rows.
    for _missing in (n for n in (config.expert_names or []) if n not in code_paths):
        report.unmeasured.append(
            f"corpus/{_missing}: no .jsonl in {data_root} - this expert is "
            f"absent from every table below")
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
        from ..train import experts as experts_mod
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

        # WHICH SOURCES THE ROUTER SAW AS RAW TEXT. router.format_fn skips
        # the chat template entirely for the tools expert and the reasoning
        # experts, so probing them through one measures the format rather than
        # the routing. The probe needs the same list the trainer branched on.
        raw_text = {n for n in
                    ([getattr(config, "tools_expert_name", "")]
                     + list(getattr(config, "reasoning_experts", None) or []))
                    if n}

        report.routing = probe_router_discrimination(
            moe_dir=moe_dir,
            held_paths=held_paths,
            expert_order=expert_order,
            num_samples=num_samples,
            dead_threshold=dead_threshold,
            trained_on=trained_on,
            raw_text_sources=raw_text,
            # Segment by think-block position when this run writes traces at
            # all. Strictly additive: no style, no offsets or no closed block
            # and the pooled table is byte-for-byte what it was.
            reasoning_style=cfg_module.reasoning_style_of_config(config),
        )
        if report.routing.get("status") == UNMEASURABLE:
            report.unmeasured.append(
                f"routing: {report.routing.get('reason', 'unknown')}")
        fmt_errors = report.routing.get("prompt_format_errors") or {}
        if fmt_errors:
            report.caveats.append(
                "the routing probe could not apply the chat template for "
                + ", ".join(f"{k} ({v})" for k, v in sorted(fmt_errors.items()))
                + " and probed those sources as raw text instead. Those "
                  "columns were measured in a different format from the rest "
                  "of the table; their enrichment is not comparable with it.")

    # ── quality: real generation against held-out references ───────────────
    if do_quality:
        from ..config.pipeline import reasoning_style_of_config
        # None → score the whole output. Otherwise the tags this RUN wrote its
        # traces with, carried on the config rather than looked up again: the
        # table is a file now, and a scorer splitting on different delimiters
        # than the generator used is measuring a different artifact.
        reasoning_style = reasoning_style_of_config(config)

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
                num_samples=num_samples, max_new_tokens=max_new_tokens,
                reasoning_style=reasoning_style)
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
                    num_samples=num_samples, max_new_tokens=max_new_tokens,
                    loaded=(moe_model, moe_tok, moe_device),
                    reasoning_style=reasoning_style)
                report.stages[f"moe/{expert_name}"] = moe_res
                if moe_res.status == UNMEASURABLE:
                    report.unmeasured.append(
                        f"quality/moe/{expert_name}: {moe_res.note}")
            moe_model = moe_tok = None
            release_memory()
            _trace("after MoE released")

        # A MEAN OVER THREE ROWS IS NOT A MEAN OVER TWENTY, AND IT USED TO
        # PRINT THE SAME WAY. num_samples rows get drawn; any that cannot
        # yield both a prompt and a reference are skipped in silence, so a
        # corpus of short files scores 3 of 20 and lands in the table beside a
        # full row with nothing to distinguish them. Say it where the exit
        # code can see it, not only in a note.
        for _name, _res in report.stages.items():
            if _res.status != "done" or not _res.attempted_samples:
                continue
            if (_res.scored_samples < 5
                    or _res.scored_samples * 2 < _res.attempted_samples):
                report.caveats.append(
                    f"quality/{_name}: averaged over {_res.scored_samples} of "
                    f"{_res.attempted_samples} sampled rows - the rest had no "
                    f"prompt/reference pair (a `text` row needs 4+ lines to "
                    f"split into one). Read that row as a thin sample, not as "
                    f"a score comparable with a full one.")

        # DID YOU PICK THE RIGHT TAG STYLE?
        #
        # This is the one place a misconfiguration is INDISTINGUISHABLE from a
        # real finding. `_split_reasoning_answer` returns "no clean split" when
        # the delimiters do not match the output - which is exactly what a
        # model that simply did not reason looks like. So a table that is one
        # release behind a new model family reports "never reasons", every
        # score silently includes the think block, and the numbers look like a
        # result instead of a mistake.
        #
        # We cannot tell the two apart. We CAN say so, which is the whole job.
        if reasoning_style is not None:
            scored = [r for r in report.stages.values() if r.reasoned >= 0]
            if scored and (sum(r.reasoned for r in scored) / len(scored)) < 0.1:
                report.caveats.append(
                    f"almost nothing emitted a think block, and this run "
                    f"expected {reasoning_style.open!r}…{reasoning_style.close!r}. "
                    f"That is either a model that does not reason or the WRONG "
                    f"TAG STYLE - the two look identical from here, and every "
                    f"quality score above includes the trace if it is the "
                    f"second. Check the family for this base in reasoning.yaml "
                    f"(drop a corrected one at ~/.msmoe/reasoning.yaml; no "
                    f"release needed).")

        # RAN OUT THE CLOCK, OR NEVER FINISHED THE SENTENCE?
        #
        # These two produce the same low `reasoned` and the same short answer,
        # and only one of them is about the model. A row where half the
        # generations stopped because the budget ended is a row scored on
        # unfinished text - say so first, because it also disqualifies the
        # discipline reading below.
        for _name, _res in report.stages.items():
            if _res.status != "done" or not _res.scored_samples:
                continue
            if _res.capped_generations * 2 >= _res.scored_samples:
                report.caveats.append(
                    f"quality/{_name}: {_res.capped_generations} of "
                    f"{_res.scored_samples} generations ran the full "
                    f"{max_new_tokens}-token budget, so they were cut off "
                    f"rather than finished. Every score in that row is on a "
                    f"truncated output, and `reasoned` is a LOWER BOUND - a "
                    f"trace cut before its close tag reads as 'did not "
                    f"reason'. Raise eval.max_new_tokens and ask again.")

        # THE DIAGNOSIS THE READER CANNOT BE EXPECTED TO INVENT. High routing
        # enrichment next to a low `reasoned` is the predicted failure of
        # putting reasoning in a routed FFN expert, and it is only readable by
        # holding the routing table and the quality table together - which is
        # exactly what a person reading one column at a time does not do. Only
        # for experts the recipe declared reasoning: a reasoning BASE with no
        # reasoning expert cannot be failing this way.
        _routing_experts = (report.routing or {}).get("experts") or {}
        for _name in (getattr(config, "reasoning_experts", None) or []):
            _res = report.stages.get(_name)
            _ent = _routing_experts.get(_name)
            if _res is None or _ent is None or _res.status != "done":
                continue
            if not _res.scored_samples or not _ent.get("enrichment_reliable",
                                                       True):
                continue
            _caveat = reasoning_discipline_caveat(
                enrichment=float(_ent.get("enrichment", 0.0)),
                reasoned=_res.reasoned,
                capped_fraction=_res.capped_generations / _res.scored_samples,
                expert=_name, enrichment_floor=dead_threshold)
            if _caveat:
                report.caveats.append(_caveat)

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
            scored_samples=info.get("scored_samples", 0),
            attempted_samples=info.get("attempted_samples", 0),
            reasoned=info.get("reasoned", -1.0),
            capped_generations=info.get("capped_generations", 0),
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
                # The denominator, in the artifact too - a mean over 3 rows
                # and a mean over 20 are not the same claim.
                "scored_samples": r.scored_samples,
                "attempted_samples": r.attempted_samples,
                # The discipline check, in the artifact too. It is a column in
                # the printed table now, and a table read back from a run dir
                # that shows "-" for a run that DID measure it is the same
                # kind of quiet loss scored_samples used to suffer.
                "reasoned": r.reasoned,
                "capped_generations": r.capped_generations,
                "status": r.status,
                "note": r.note,
            }
            for name, r in report.stages.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
