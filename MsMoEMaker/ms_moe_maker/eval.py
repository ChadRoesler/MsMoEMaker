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

import json
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
    unmeasured: List[str] = field(default_factory=list)
    """Things we could not measure, and therefore did not score."""


def _load_or_split(data_path: str, held_out: float, seed: int = 42) -> Tuple[str, str]:
    """Split a JSONL dataset into train / held-out.

    Returns (train_path, held_out_path).
    """
    lines = Path(data_path).read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return data_path, data_path + ".heldout"

    random.seed(seed)
    n_held = max(1, int(len(lines) * held_out))
    shuffled = list(lines)
    random.shuffle(shuffled)
    held = shuffled[:n_held]
    train = shuffled[n_held:]

    train_path = data_path + ".train"
    held_path = data_path + ".heldout"

    Path(train_path).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(held_path).write_text("\n".join(held) + "\n", encoding="utf-8")

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
    """Very simple 1-gram BLEU. Not accurate but fast for local eval."""
    gen_tokens = _tokenize_simple(generated)
    ref_tokens = _tokenize_simple(reference)
    if not gen_tokens or not ref_tokens:
        return 0.0
    # Count 1-gram matches
    from collections import Counter
    gen_counts = Counter(gen_tokens)
    ref_counts = Counter(ref_tokens)
    matches = sum(min(gen_counts[t], ref_counts[t]) for t in gen_counts)
    return matches / len(gen_tokens)


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

    trained_on = trained_on or set()

    # Held out BY CONSTRUCTION: drop any row the router actually trained on.
    sources: Dict[str, List[str]] = {}
    for name, path in held_paths.items():
        rows: List[str] = []
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get("text") or json.loads(line).get("content")
            except json.JSONDecodeError:
                continue
            if t and t not in trained_on:
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
                sel = torch.topk(logits.float(), K, dim=-1).indices.flatten()
                for e in sel.tolist():
                    counts[s][li][e] += 1
                totals[s][li] += sel.numel() // K
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
        enrich.append(r)
        hits += (top == en)
        rivals = sorted(((s, v) for s, v in col.items() if s != en),
                        key=lambda kv: kv[1], reverse=True)
        experts[en] = {
            "enrichment": r,
            "own_share": own,
            "others_share": oavg,
            "own_is_column_max": top == en,
            "top_competitor": rivals[0][0] if rivals else "",
            "top_competitor_share": rivals[0][1] if rivals else 0.0,
            # An expert can clear the enrichment bar and still be outranked on
            # its own ground by a neighbour taking more of the traffic. That is
            # a different failure and a column-only read misses it.
            "outranked": bool(rivals) and rivals[0][1] > own,
        }

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
    return model.to(device).eval(), tok, device


def _sample_texts(path: str, num_samples: int) -> List[str]:
    """Pull up to num_samples raw text bodies out of a JSONL held-out file."""
    try:
        lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    random.seed(42)
    picked = random.sample(lines, min(num_samples, len(lines))) if lines else []
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

    try:
        lines = Path(test_data_path).read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        result.status = UNMEASURABLE
        result.note = f"cannot read held-out data: {exc}"
        return result
    if not lines:
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

    random.seed(42)
    samples = random.sample(lines, min(num_samples, len(lines)))

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

        batch = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **batch, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
        generated = tok.decode(
            out_ids[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)

        em.append(_exact_match(generated, reference))
        r1.append(_rouge1(generated, reference))
        bl.append(_bleu_simple(generated, reference))
        lengths.append(len(_tokenize_simple(generated)))
        scored += 1
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
    result.note = (f"{scored} samples generated"
                   + (" (completion: second half of each held-out doc; "
                      "read ROUGE/BLEU, exact-match is near zero by nature)"
                      if completion_mode else ""))
    return result


def detect_dead_experts(report: EvalReport, threshold: float = 1.2) -> List[str]:
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

    # Input-blind routing is not "no dead experts", it is every expert dead at
    # once and the enrichment table meaning nothing. Say so before reading it.
    if routing.get("mean_js_bits", 1.0) < 1e-3:
        report.unmeasured.append(
            "dead-expert check: routing is input-blind (mean JS ~ 0) - the "
            "router ignores its input, so per-expert enrichment is noise")
        report.dead_experts = []
        return []

    dead: List[str] = []
    for name, info in experts.items():
        enrichment = info.get("enrichment", 0.0)
        if enrichment < threshold or info.get("outranked"):
            dead.append(name)
            result = report.stages.get(name)
            if result is not None:
                why = f"enrichment {enrichment:.2f}x < {threshold}x"
                if info.get("outranked"):
                    why += (f"; outranked on its own domain by "
                            f"{info.get('top_competitor')} "
                            f"({info.get('top_competitor_share', 0):.3f} vs "
                            f"{info.get('own_share', 0):.3f})")
                result.note = f"dead: {why}"

    report.dead_experts = dead
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
    if mode not in ("routing", "quality", "all"):
        report.ok = False
        report.message = f"unknown eval mode {mode!r} (expect routing|quality|all)"
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

    # Expert order is the recipe's order, which is STITCH order, which is the
    # expert axis of the router's gate matrix. Never re-derive it by sorting.
    expert_order = [n for n in (config.expert_names or list(code_paths))
                    if n in code_paths]

    t_start = time.time()
    held_paths: Dict[str, str] = {}
    for expert_name in expert_order:
        _, held_path = _load_or_split(code_paths[expert_name], held_out)
        held_paths[expert_name] = held_path

    do_routing = mode in ("routing", "all")
    do_quality = mode in ("quality", "all")

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
            for line in mixed.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        trained_on.add(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        pass

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
            res = eval_generation(
                model_dir=expert_dir, test_data_path=held_paths[expert_name],
                label=expert_name, domain=expert_name,
                num_samples=num_samples)
            report.stages[expert_name] = res
            if res.status == UNMEASURABLE:
                report.unmeasured.append(f"quality/{expert_name}: {res.note}")
            release_memory()

        moe_model = moe_tok = None
        try:
            moe_model, moe_tok, moe_device = _load_model(moe_dir)
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

    # ── verdict ────────────────────────────────────────────────────────────
    if do_routing:
        detect_dead_experts(report, threshold=dead_threshold)
    else:
        report.unmeasured.append(
            f"dead-expert check: needs --mode routing or all (ran {mode})")

    elapsed = time.time() - t_start
    report.message = (f"Eval complete in {elapsed:.0f}s. "
                      f"mode={mode}. "
                      f"{len(report.dead_experts)} dead expert(s). "
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
