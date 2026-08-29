"""Do the experts differ, and is there anything for the router to learn?

THREE QUESTIONS, ASKED BEFORE YOU PAY FOR THE ANSWER. Every one of these was
a standalone probe first, and every one of them was written only after the
thing it detects had already cost a build:

  1. WEIGHT DIVERGENCE - did the specialists move apart, and in independent
     directions? (from probe_expert_divergence.py, plus a base-free pairwise
     check that probe did not have)
  2. CROSS-DOMAIN LOSS - is each expert measurably better on its own ground?
     This is the router's ENTIRE training signal; if it is absent, no router
     hyperparameter can conjure it.
  3. MoE CONFIG AUDIT - does the stitched config permit the router to train
     at all? A config can be structurally perfect and mathematically inert.

WHY IN THE TOOL RATHER THAN BESIDE IT. stages.plan() argues that probes do
not belong in the pipeline because "folding them in would make the build the
thing that grades itself". That argument is correct and it does not apply
here, because these grade the build's INPUTS, not its outputs:

    preflight   grades the machine    before we use it
    THIS        grades the experts    before we stitch them
    eval        grades the MoE        after it exists          <- self-grading

The first two are gates. Only the third is a report card, and it stays where
it is.

WHY THE BASE IS A LOOKUP AND NOT AN ARGUMENT. The divergence probe takes
--base, and on its first real use it was handed the tier default instead of
the base the recipe actually named (Qwen2.5-Coder-0.5B where the recipe said
huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3). Both are 0.5B Qwens; nothing
errored. The result was 92% "movement" and cos=1.000 - a confident, decisive,
completely wrong verdict of "these experts are identical", which pointed the
next day of work at the wrong thing.

A measurement whose reference can be supplied wrong will eventually be
supplied wrong. Here `base` comes from config.base and there is no parameter
to get it wrong with.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

OK = "ok"
WARN = "warn"
UNMEASURABLE = "unmeasurable"

PROJ = ("gate_proj", "up_proj", "down_proj")

# Below this, training barely moved the weights and every cosine computed from
# those deltas is dominated by noise. Not a quality bar - a TRUST bar. The
# divergence probe says it in prose ("small movement makes every cosine
# untrustworthy"); this is the same statement with a number attached so the
# code can act on it instead of the reader having to.
MIN_MOVEMENT = 0.002       # 0.2% of the base weight norm

# Two independently-moved experts SHOULD look unrelated. The question is how
# unrelated is suspicious, and the honest baseline is not 0 - it is what two
# RANDOM directions score in this many dimensions, which is ~1/sqrt(d). For a
# 896x4864 projection that is 0.0005, so a "low" cosine of 0.03 is actually
# 60x the chance floor and worth reporting as such rather than as "0".
HIGH_COS = 0.6             # above this, they learned the same thing


# ── weights ──────────────────────────────────────────────────────────────

def _shard_map(path: str) -> Dict[str, str]:
    """key -> file holding it. Sharded and single-file layouts both."""
    from safetensors import safe_open

    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            return {k: os.path.join(path, v)
                    for k, v in json.load(fh)["weight_map"].items()}
    single = os.path.join(path, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(f"no safetensors in {path}")
    with safe_open(single, framework="pt") as fh:
        return {k: single for k in fh.keys()}


def _tensor(smap: Dict[str, str], key: str):
    from safetensors import safe_open
    with safe_open(smap[key], framework="pt") as fh:
        return fh.get_tensor(key).float()


def _layer_of(key: str) -> int:
    for part in key.split("."):
        if part.isdigit():
            return int(part)
    return -1


def _ffn_keys(smap: Dict[str, str], stride: int) -> List[str]:
    keys = [k for k in smap
            if any(p in k for p in PROJ) and k.endswith(".weight")]
    keys.sort(key=lambda k: (_layer_of(k), k))
    return [k for k in keys if _layer_of(k) % max(stride, 1) == 0]


def pairwise_delta(expert_dirs: Dict[str, str],
                   stride: int = 4) -> Dict[str, Any]:
    """|A - B| / |A| for every pair. NO BASE INVOLVED, and that is the point.

    This is the check that untangled the wrong-base disaster: it is a property
    of the two checkpoints alone, so no reference can be supplied wrong. When
    the base-relative probe said cos=1.000 ("identical"), this said 1.26%
    ("genuinely different"), and the disagreement is what proved the base was
    the problem rather than the experts.

    Cheap enough to always run, and it answers the crudest question - are
    these actually two different files - which is the one nobody thinks to
    ask until it turns out to be the answer.
    """
    try:
        import torch  # noqa: F401
        from safetensors import safe_open  # noqa: F401
    except ImportError as exc:
        return {"status": UNMEASURABLE, "reason": f"needs torch + safetensors: {exc}"}

    names = sorted(expert_dirs)
    if len(names) < 2:
        return {"status": UNMEASURABLE,
                "reason": f"needs 2+ experts to compare, got {len(names)}"}

    maps: Dict[str, Dict[str, str]] = {}
    for n in names:
        try:
            maps[n] = _shard_map(expert_dirs[n])
        except (FileNotFoundError, OSError, KeyError) as exc:
            return {"status": UNMEASURABLE, "reason": f"{n}: {exc}"}

    pairs: Dict[str, float] = {}
    worst = ("", 1.0)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys = [k for k in _ffn_keys(maps[a], stride) if k in maps[b]]
            if not keys:
                return {"status": UNMEASURABLE,
                        "reason": f"{a} and {b} share no FFN tensors - "
                                  f"different architectures?"}
            num = den = 0.0
            for k in keys:
                A, B = _tensor(maps[a], k), _tensor(maps[b], k)
                num += float((A - B).norm()) ** 2
                den += float(A.norm()) ** 2
            rel = (num ** 0.5) / max(den ** 0.5, 1e-12)
            pairs[f"{a}|{b}"] = rel
            if rel < worst[1]:
                worst = (f"{a}|{b}", rel)

    identical = [p for p, v in pairs.items() if v < 1e-6]
    return {"status": OK, "pairs": pairs,
            "closest_pair": worst[0], "closest": worst[1],
            "identical_pairs": identical, "n_experts": len(names)}


def movement_and_direction(base: str, expert_dirs: Dict[str, str],
                           stride: int = 4) -> Dict[str, Any]:
    """How far each expert moved from THE RECIPE'S base, and whether they
    moved the same way. All pairs, not just the first two.

    movement_e = |W_e - W_base| / |W_base|
    cos(a, b)  = <d_a, d_b> / (|d_a| |d_b|)

    Read them TOGETHER: large movement with low cosine is specialisation.
    Small movement makes every cosine noise. Low cosine on its own proves
    nothing, because in d dimensions two random vectors already score
    ~1/sqrt(d) - which this reports, so "0.03" can be seen for what it is
    (60x the chance floor) instead of being read as zero.
    """
    try:
        import torch
        from safetensors import safe_open  # noqa: F401
    except ImportError as exc:
        return {"status": UNMEASURABLE, "reason": f"needs torch + safetensors: {exc}"}

    if not base:
        return {"status": UNMEASURABLE,
                "reason": "no base in the config - cannot measure movement"}

    base_dir = base
    if not os.path.isdir(base_dir):
        try:
            from huggingface_hub import snapshot_download
            base_dir = snapshot_download(
                base, allow_patterns=["*.safetensors", "*.json"])
        except Exception as exc:
            return {"status": UNMEASURABLE,
                    "reason": f"could not resolve base {base!r}: {exc}"}

    names = sorted(expert_dirs)
    try:
        bmap = _shard_map(base_dir)
        emaps = {n: _shard_map(expert_dirs[n]) for n in names}
    except (FileNotFoundError, OSError, KeyError) as exc:
        return {"status": UNMEASURABLE, "reason": str(exc)}

    keys = [k for k in _ffn_keys(bmap, stride)
            if all(k in m for m in emaps.values())]
    if not keys:
        return {"status": UNMEASURABLE,
                "reason": "no FFN tensors shared by the base and every expert"}

    deltas: Dict[str, List] = {n: [] for n in names}
    movement: Dict[str, float] = {}
    dims = 0
    num = {n: 0.0 for n in names}
    den = 0.0
    for k in keys:
        W = _tensor(bmap, k)
        den += float(W.norm()) ** 2
        dims += W.numel()
        for n in names:
            d = _tensor(emaps[n], k) - W
            deltas[n].append(d.flatten())
            num[n] += float(d.norm()) ** 2
    for n in names:
        movement[n] = (num[n] ** 0.5) / max(den ** 0.5, 1e-12)

    flat = {n: torch.cat(deltas[n]) for n in names}
    deltas.clear()
    cos: Dict[str, float] = {}
    worst = ("", -1.0)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = flat[a], flat[b]
            c = float((va @ vb) / max(float(va.norm()) * float(vb.norm()), 1e-12))
            cos[f"{a}|{b}"] = c
            if c > worst[1]:
                worst = (f"{a}|{b}", c)
    flat.clear()

    chance = 1.0 / math.sqrt(max(dims, 1))
    low_movement = [n for n, m in movement.items() if m < MIN_MOVEMENT]
    return {"status": OK, "base": base, "base_dir": base_dir,
            "movement": movement, "cos": cos,
            "most_aligned_pair": worst[0], "most_aligned": worst[1],
            "chance_cos": chance, "low_movement": low_movement,
            "dims": dims}


# ── behaviour ────────────────────────────────────────────────────────────

def cross_domain_loss(expert_dirs: Dict[str, str],
                      held_paths: Dict[str, str],
                      num_samples: int = 16,
                      max_len: int = 1024,
                      device: str = "",
                      callback=None) -> Dict[str, Any]:
    """The matrix that decides whether the router CAN learn.

    THE ROUTER'S ONLY TRAINING SIGNAL IS LM LOSS. The stitch zeroes the gate,
    so it starts at uniform, and the only thing that can move it is "routing
    this token to expert A lowered the loss more than expert B would have".
    If each expert is not measurably better on its own ground, that difference
    is zero, the gradient is zero, and the gate stays uniform at any learning
    rate for any number of epochs.

    So: mean per-token loss of every expert on every domain's held-out text.
    Diagonal lower than its column = signal exists = near-uniform routing is a
    ROUTER problem. Diagonal not lower = the experts are interchangeable in
    the only way that matters, and the fix is upstream (rank, steps, corpus,
    or full-FFN finetune instead of LoRA).

    Weight-space divergence cannot answer this. Deltas can point in different
    directions and produce the same outputs.

    One model resident at a time, fp16, reservoir-sampled corpora - the same
    discipline eval.py learned the hard way on a unified-memory box.
    """
    from ..eval import harness as ev

    ok, reason = ev._torch_available()
    if not ok:
        return {"status": UNMEASURABLE, "reason": reason}

    domains = sorted(set(expert_dirs) & set(held_paths))
    if len(domains) < 2:
        return {"status": UNMEASURABLE,
                "reason": (f"needs 2+ domains with both an expert and held-out "
                           f"data; got {sorted(set(expert_dirs))} experts and "
                           f"{sorted(set(held_paths))} data")}

    import torch

    corpora: Dict[str, List[str]] = {}
    for d in domains:
        rows = []
        for line in ev._reservoir(held_paths[d], num_samples):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (obj.get("text") or obj.get("content") or obj.get("output")
                 or obj.get("answer") or "")
            if t:
                rows.append(str(t))
        if not rows:
            return {"status": UNMEASURABLE,
                    "reason": f"no usable text rows in {held_paths[d]}"}
        corpora[d] = rows

    matrix: Dict[str, Dict[str, float]] = {}
    for ename in domains:
        try:
            model, tok, dev = ev._load_model(expert_dirs[ename], device=device)
        except Exception as exc:
            return {"status": UNMEASURABLE,
                    "reason": f"could not load {ename}: {exc}"}
        matrix[ename] = {}
        try:
            for dname in domains:
                tot_loss = 0.0
                tot_tok = 0
                for text in corpora[dname]:
                    enc = tok(text, return_tensors="pt", truncation=True,
                              max_length=max_len)
                    ids = enc["input_ids"].to(model.device)
                    if ids.shape[-1] < 8:
                        continue
                    try:
                        with torch.no_grad():
                            out = model(input_ids=ids, labels=ids)
                    except Exception as exc:
                        if not ev._is_oom(exc):
                            raise
                        model = None
                        ev.release_memory()
                        return {"status": UNMEASURABLE,
                                "reason": f"out of memory scoring {ename} on "
                                          f"{dname}: {exc}"}
                    n = int(ids.shape[-1]) - 1
                    tot_loss += float(out.loss) * n
                    tot_tok += n
                if not tot_tok:
                    return {"status": UNMEASURABLE,
                            "reason": f"every {dname} sample was under 8 tokens"}
                matrix[ename][dname] = tot_loss / tot_tok
                if callback:
                    callback("gate.experts", "running", f"{ename} on {dname}")
        finally:
            model = None
            ev.release_memory()

    gaps: Dict[str, Dict[str, Any]] = {}
    signal = True
    for d in domains:
        col = {e: matrix[e][d] for e in domains}
        own = col[d]
        rival_name = min((e for e in col if e != d), key=lambda e: col[e])
        gap = col[rival_name] - own
        wins = own <= min(col.values())
        signal &= wins
        gaps[d] = {"own": own, "best_rival": rival_name,
                   "best_rival_loss": col[rival_name], "gap": gap,
                   "own_wins": wins}

    return {"status": OK, "matrix": matrix, "gaps": gaps,
            "signal": signal, "domains": domains,
            "mean_gap": sum(g["gap"] for g in gaps.values()) / len(gaps)}


# ── the stitched artifact ────────────────────────────────────────────────

def audit_moe_config(moe_dir: str) -> Dict[str, Any]:
    """Read the stitched config and refuse the combinations that cannot train.

    Structurally perfect and mathematically inert is a real state and nothing
    else in this tool could see it. verify_stitch checks every tensor and
    passes; smoke generates tokens and passes; eval reports balanced routing
    and zero dead experts. All correct, all blind to a config that has severed
    the router from the loss.

    This reads the ARTIFACT, not the recipe, which matters because a stitch
    can arrive from anywhere - an older build, someone else's run, a hand-edit.
    recipe.validate() catches it at authoring time; this catches it on disk.
    """
    path = os.path.join(moe_dir, "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": UNMEASURABLE, "reason": f"cannot read {path}: {exc}"}

    E = cfg.get("num_experts")
    K = cfg.get("num_experts_per_tok")
    norm = cfg.get("norm_topk_prob")
    findings: List[str] = []

    # THE SEVERED GRADIENT. Qwen2MoeSparseMoeBlock does
    #     routing_weights, sel = topk(routing_weights, top_k, dim=-1)
    #     if norm_topk_prob: routing_weights /= routing_weights.sum(-1, keepdim=True)
    # and at top_k=1 that sum IS routing_weights - it divides by itself. Every
    # weight becomes the constant 1.0, d(w/w)/dw = 0, and the gate receives no
    # gradient from the LM loss at all. Only the aux loss remains, and it pulls
    # toward uniform: training the router harder makes it WORSE. Measured on
    # the first real 0.5B build - 1.02x enrichment down to 1.00x after tripling
    # the steps, with a 0.49-nat cross-domain gap sitting unused.
    if K == 1 and norm:
        findings.append(
            "experts_per_tok=1 with norm_topk_prob=true severs the router from "
            "the LM loss (normalising one top-k weight divides it by itself, so "
            "every routing weight is the constant 1.0 and the gate gets zero "
            "gradient). Only the aux loss can move it, and that pulls toward "
            "uniform. Restitch with norm_topk_prob=false - Switch Transformer's "
            "top-1 formulation - or use top-k>=2 with 3+ experts.")

    if isinstance(E, int) and isinstance(K, int) and E and K >= E:
        findings.append(
            f"experts_per_tok={K} with num_experts={E}: every expert is "
            f"selected on every token, so routing cannot discriminate by "
            f"construction and `eval --mode routing` will be UNMEASURABLE.")

    if isinstance(E, int) and E >= 2:
        p_floor = 1.0 / (E ** E)
        if p_floor > 0.05:
            findings.append(
                f"with {E} experts the own-column routing test cannot reach "
                f"significance: all-{E}-of-{E} happens by chance at p="
                f"{p_floor:.3f}. Enrichment is still real; the headline is not "
                f"evidence at this width. 3 experts puts p at 0.037.")

    if cfg.get("use_cache") is False:
        findings.append(
            "use_cache=false is stamped in the config. Correct for training, "
            "expensive at generation - every new token re-runs a full forward "
            "over the whole prefix. eval overrides it at load; anything else "
            "reading this directory will not.")

    return {"status": WARN if findings else OK,
            "num_experts": E, "experts_per_tok": K, "norm_topk_prob": norm,
            "use_cache": cfg.get("use_cache"),
            "expert_names": cfg.get("expert_names"),
            "findings": findings}


# ── the gate ─────────────────────────────────────────────────────────────

@dataclass
class ExpertsReport:
    status: str = OK
    divergence: Dict[str, Any] = field(default_factory=dict)
    pairwise: Dict[str, Any] = field(default_factory=dict)
    cross_loss: Dict[str, Any] = field(default_factory=dict)
    config_audit: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    unmeasured: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "divergence": self.divergence,
                "pairwise": self.pairwise, "cross_loss": self.cross_loss,
                "config_audit": self.config_audit, "findings": self.findings,
                "unmeasured": self.unmeasured}


def _interpret(rep: ExpertsReport) -> None:
    """Turn measurements into findings. WARN, never refuse.

    Mandate is not ethos: someone may genuinely want a dense ensemble of
    near-identical experts, and this tool's job is to make sure they know
    that is what they are getting - not to decide for them.
    """
    pw = rep.pairwise
    if pw.get("status") == OK and pw.get("identical_pairs"):
        rep.findings.append(
            f"identical experts: {', '.join(pw['identical_pairs'])} differ by "
            f"less than 1e-6 - these are the same weights. Nothing can route "
            f"between them. Check that two specialist dirs were not written "
            f"from one checkpoint.")

    dv = rep.divergence
    if dv.get("status") == OK:
        if dv.get("low_movement"):
            rep.findings.append(
                f"barely trained: {', '.join(dv['low_movement'])} moved less "
                f"than {MIN_MOVEMENT:.1%} from the base, so every cosine below "
                f"is noise-dominated and cannot be trusted either way.")
        if dv.get("most_aligned", 0.0) > HIGH_COS:
            rep.findings.append(
                f"aligned deltas: {dv['most_aligned_pair']} at cos="
                f"{dv['most_aligned']:.3f} - they moved in nearly the same "
                f"direction, i.e. learned the same thing twice. Buy divergence "
                f"before stitching: higher LoRA rank, more steps, or full-FFN "
                f"finetune.")

    cl = rep.cross_loss
    if cl.get("status") == OK and not cl.get("signal"):
        losers = [d for d, g in cl["gaps"].items() if not g["own_wins"]]
        rep.findings.append(
            f"NO ROUTER SIGNAL: {', '.join(losers)} scored better under a "
            f"foreign expert than its own. Routing correctly does not lower "
            f"the loss, so the gate has nothing to descend and no router "
            f"hyperparameter will help. The fix is upstream.")

    rep.findings.extend(rep.config_audit.get("findings", []))

    for name, block in (("divergence", dv), ("pairwise", pw),
                        ("cross-loss", cl), ("config", rep.config_audit)):
        if block.get("status") == UNMEASURABLE:
            rep.unmeasured.append(f"{name}: {block.get('reason', 'unknown')}")

    rep.status = WARN if rep.findings else OK


def run_experts(config,
                expert_dirs: Dict[str, str],
                held_paths: Optional[Dict[str, str]] = None,
                moe_dir: str = "",
                spec: Optional[Dict[str, Any]] = None,
                callback=None) -> ExpertsReport:
    """Run whichever checks the inputs allow, and say what could not run.

    Partial input is normal - before the stitch there is no moe_dir, and a
    caller may not have held-out splits. Each block reports UNMEASURABLE with
    a reason rather than being silently skipped, because a check that vanishes
    reads as a check that passed.
    """
    spec = spec or {}
    stride = int(spec.get("stride", 4))
    num_samples = int(spec.get("num_samples", 16))
    rep = ExpertsReport()

    present = {n: d for n, d in expert_dirs.items() if Path(d).is_dir()}
    for n, d in expert_dirs.items():
        if n not in present:
            rep.unmeasured.append(f"expert {n}: nothing at {d}")

    if len(present) >= 2:
        rep.pairwise = pairwise_delta(present, stride=stride)
        rep.divergence = movement_and_direction(
            getattr(config, "base", ""), present, stride=stride)
    else:
        why = f"needs 2+ specialist dirs on disk, found {len(present)}"
        rep.pairwise = {"status": UNMEASURABLE, "reason": why}
        rep.divergence = {"status": UNMEASURABLE, "reason": why}

    if held_paths and len(present) >= 2:
        rep.cross_loss = cross_domain_loss(
            present, held_paths, num_samples=num_samples, callback=callback)
    else:
        rep.cross_loss = {"status": UNMEASURABLE,
                          "reason": "no held-out data for the expert domains"}

    if moe_dir and Path(moe_dir).is_dir():
        rep.config_audit = audit_moe_config(moe_dir)
    else:
        rep.config_audit = {"status": UNMEASURABLE,
                            "reason": "no stitched MoE yet"}

    _interpret(rep)
    return rep


def format_report(rep: ExpertsReport) -> str:
    """Human-readable. Numbers first, verdict last, nothing inferred."""
    out: List[str] = ["", "  EXPERT DIVERGENCE - do the specialists differ?"]

    pw = rep.pairwise
    if pw.get("status") == OK:
        out.append(f"  {'pair':32} {'|A-B|/|A|':>12}   (no base involved)")
        for pair, v in sorted(pw["pairs"].items(), key=lambda kv: kv[1]):
            out.append(f"  {pair:32} {v:12.6f}")
    else:
        out.append(f"    unmeasurable: {pw.get('reason', '?')}")

    dv = rep.divergence
    if dv.get("status") == OK:
        out.append("")
        out.append(f"  movement from {dv['base']}")
        for n, m in sorted(dv["movement"].items()):
            out.append(f"    {n:24} {m:8.3%}")
        out.append(f"  direction (chance floor at {dv['dims']:,} dims = "
                   f"{dv['chance_cos']:.5f})")
        for pair, c in sorted(dv["cos"].items(), key=lambda kv: -kv[1]):
            ratio = abs(c) / max(dv["chance_cos"], 1e-12)
            out.append(f"    {pair:24} cos {c:+.4f}  ({ratio:,.0f}x chance)")
    elif dv.get("reason"):
        out.append(f"    movement/direction unmeasurable: {dv['reason']}")

    cl = rep.cross_loss
    out.append("")
    out.append("  CROSS-DOMAIN LOSS - is there anything to route on?")
    if cl.get("status") == OK:
        doms = cl["domains"]
        out.append("  " + "expert".ljust(20)
                   + "".join(f"{d + '-text':>16}" for d in doms))
        for e in doms:
            out.append("  " + (e + "-exp").ljust(20)
                       + "".join(f"{cl['matrix'][e][d]:16.4f}" for d in doms))
        out.append("")
        for d, g in cl["gaps"].items():
            mark = "own expert wins" if g["own_wins"] else "<-- OWN EXPERT LOSES"
            out.append(f"    {d + '-text':16} own {g['own']:.4f}  "
                       f"rival {g['best_rival_loss']:.4f} ({g['best_rival']})  "
                       f"gap {g['gap']:+.4f}  {mark}")
    else:
        out.append(f"    unmeasurable: {cl.get('reason', '?')}")

    ca = rep.config_audit
    if ca.get("status") in (OK, WARN):
        out.append("")
        out.append(f"  MoE CONFIG - experts={ca.get('num_experts')} "
                   f"top_k={ca.get('experts_per_tok')} "
                   f"norm_topk_prob={ca.get('norm_topk_prob')} "
                   f"use_cache={ca.get('use_cache')}")

    if rep.findings:
        out.append("")
        out.append("  [~] FINDINGS")
        for f in rep.findings:
            out.append(f"      - {f}")
    if rep.unmeasured:
        out.append("")
        out.append("  [?] NOT MEASURED")
        for u in rep.unmeasured:
            out.append(f"      - {u}")
    if not rep.findings and not rep.unmeasured:
        out.append("")
        out.append("  [ok] Experts diverged and each owns its ground.")
    return "\n".join(out)
