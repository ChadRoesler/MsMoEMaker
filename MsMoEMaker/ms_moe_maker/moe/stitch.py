"""MoE stitching — assemble the MoE skeleton from a base model + specialists.

Reads the anchor specialist's config.json for architecture params, then streams
every tensor from the specialist weight files straight into output shards via
the vendored `_moe_stitch`. Peak memory is one shard, not the full model.

THERE IS NO FALLBACK, AND THAT IS DELIBERATE. There used to be one, for when
`import moe_stitch` failed. It failed always — moe_stitch was never a PyPI
package, just a local module beside fraunkenstein_universal.py — and the
fallback had therefore never run. When read, it turned out to look up
`mlp.experts.N.*` keys inside a DENSE specialist checkpoint, where such keys
cannot exist, so its copy loop could never execute: it would have produced an
MoE whose every expert was an identical copy of the anchor, and the old
config-only verify_stitch would have passed it. The router would then have
trained on N identical experts and every enrichment number would have come out
at 1.0 with no way to tell a routing failure from a splice that never happened.

A fallback that silently produces a wrong-but-plausible artifact is worse than
no fallback. Deleted. If streaming cannot run, that is a loud error.
"""
from __future__ import annotations

from ..run import stages as st

import json
import os
from typing import Dict, List, Optional


def stitch_dir(config) -> str:
    return f"{config.output_root}/" + st.ARTIFACTS[st.STITCH]


def stitch_is_done(config) -> bool:
    """Is the MoE skeleton already on disk AND does it match this recipe?

    PRESENCE IS NOT ENOUGH, AND THAT COST A RUN. The skip used to test only
    that a config.json existed. Someone editing the expert list, deleting
    moe_trained and rebuilding got the OLD skeleton silently reused - so a
    recipe saying [csharp, python] trained a router whose gate axis was still
    [python, csharp], and every routing number downstream carried the wrong
    name. Nothing failed. The build said 8/8 stages done.

    The skeleton stamps `expert_names` precisely so this is answerable. If the
    names or their ORDER differ from the recipe, the artifact on disk is not
    the artifact this recipe describes and the skip has to decline - order
    matters as much as membership, because it IS the expert axis of the
    router's gate matrix.
    """
    if config.force:
        return False
    cfg_path = os.path.join(stitch_dir(config), "config.json")
    if not os.path.exists(cfg_path):
        return False

    want = list(getattr(config, "expert_names", []) or [])
    if not want:
        return True
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            have = list(json.load(fh).get("expert_names") or [])
    except (OSError, ValueError):
        return True          # unreadable: let the stitch decide, not the skip
    if have and have != want:
        print(f"[restitch] skeleton on disk was built for {have}, recipe says "
              f"{want} - the gate axis would not match the names. Restitching.")
        return False
    return True


def _specialist_dir(output_root: str, expert: str) -> str:
    """Where fine-tune put this expert. Quoted from stages, never spelled out,
    so the name lives in exactly one place."""
    return f"{output_root}/" + st.FINETUNE_ARTIFACT.format(expert=expert)


def stitch_moe(config, safe_names: List[str]) -> str:
    """Build the MoE skeleton from specialist checkpoints.

    Each specialist is at OUTPUT_ROOT/specialist_{safe}.  The output skeleton
    is at OUTPUT_ROOT/moe_untrained.

    Returns the output directory path.
    """
    import torch

    out_dir = stitch_dir(config)
    if stitch_is_done(config):
        print(f"[skip] MoE skeleton already present at {out_dir}")
        return out_dir

    print(f"\nStitching {len(safe_names)} experts into MoE...")

    # Use anchor's config (the first specialist)
    anchor_name = safe_names[0]
    anchor_dir = _specialist_dir(config.output_root, anchor_name)

    with open(os.path.join(anchor_dir, "config.json")) as f:
        config_dict = json.load(f)

    # Fail fast: a quantised specialist can't be stitched.
    #
    # `.pop()` had no default here, so on a NON-quantised specialist - the only
    # kind that can be stitched, as this very error says - it raised
    # KeyError: 'quantization_config' and killed the build at stage 4.
    # save_pretrained only writes that key for a quantised model, so the happy
    # path was the crashing path.
    if config_dict.pop("quantization_config", None) is not None:
        raise RuntimeError(
            f"specialist {anchor_name!r} is QUANTISED on disk — its weights are "
            f"bitsandbytes-packed, not real matrices, and cannot be stitched. "
            f"Fix: set load_in_4bit=False, delete the specialist directories, "
            f"and retrain.")

    # Remove fields that the MoE config overrides
    for key in ("architectures", "model_type", "num_experts",
                "num_experts_per_tok", "shared_expert_intermediate_size",
                "moe_intermediate_size", "tie_word_embeddings"):
        config_dict.pop(key, None)

    # Build MoE config
    from transformers import Qwen2MoeConfig
    moe_config = Qwen2MoeConfig(
        **{k: v for k, v in config_dict.items() if k not in (
            "architectures", "model_type", "num_experts",
            "num_experts_per_tok", "shared_expert_intermediate_size",
            "moe_intermediate_size", "tie_word_embeddings",
        )},
        tie_word_embeddings=False,
        num_experts=len(safe_names),
        num_experts_per_tok=config.experts_per_tok,
        moe_intermediate_size=config_dict.get("intermediate_size", 11008),
        mlp_only_layers=config.mlp_only_layers,
        shared_expert_intermediate_size=config.shared_expert_width,
        model_type="qwen2_moe",
        router_aux_loss_coef=config.router_aux_loss_coef,
        norm_topk_prob=config.norm_topk_prob,
        expert_names=list(safe_names),
        architectures=["Qwen2MoeForCausalLM"],
    )

    # Stream it. One path, no try/except around the import: the module is
    # vendored at ms_moe_maker/moe/_moe_stitch.py, so if this import fails the
    # install is broken and that should be loud, not quietly downgraded.
    #
    # The old code did `import moe_stitch` INSIDE this function and then called
    # `moe_stitch.stream_stitch` from a DIFFERENT function, where the name was
    # never bound - a NameError, which `except ImportError` does not catch. So
    # even with the module importable, the streaming path could not run.
    from . import _moe_stitch
    from transformers import Qwen2MoeForCausalLM

    print("   streaming tensors from disk (peak memory ~1 shard)")
    _moe_stitch.stream_stitch(
        out_dir=out_dir,
        anchor_dir=anchor_dir,
        spec_dirs=[_specialist_dir(config.output_root, n) for n in safe_names],
        expert_names=list(safe_names),
        model_cls=Qwen2MoeForCausalLM,
        moe_config=moe_config,
        shared_gate_fill=config.shared_expert_gate_fill,
        router_init=getattr(config, 'router_init', 'zero'),
        # THE FORGOTTEN ARGUMENT. stream_stitch has always taken
        # router_init_std; this call never forwarded it, so the recipe's value
        # sat in the fingerprint (busting resume on change) while the
        # streamer's own 0.02 default produced bit-identical weights. Now it
        # flows to both the stitch and the verifier.
        router_init_std=getattr(config, 'router_init_std', 0.02),
        router_seed=getattr(config, 'seed', 42),
        num_layers=config_dict.get("num_hidden_layers", 32),
        tokenizer_src=anchor_dir,
    )

    print(f"MoE skeleton saved to {out_dir}")
    return out_dir


def _smap(path: str) -> Dict[str, str]:
    """key -> the safetensors file holding it, so tensors read ONE AT A TIME."""
    from safetensors import safe_open
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return {k: os.path.join(path, v)
                    for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(path, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(f"no safetensors in {path}")
    with safe_open(single, framework="pt") as f:
        return {k: single for k in f.keys()}


def _get(m: Dict[str, str], k: str):
    from safetensors import safe_open
    with safe_open(m[k], framework="pt") as f:
        return f.get_tensor(k)


def verify_stitch(out_dir: str, output_root: Optional[str] = None,
                  gate_fill: float = 0.02,
                  router_init: str = "zero",
                  router_init_std: float = 0.02) -> bool:
    """Is every tensor in the stitched model bit-identical to its source?

    PORTED FROM verify_stitch_complete.py, the Lab script that actually
    verified the proven 0.5B rung. What used to be here read config.json,
    checked that three keys existed, printed "stitch OK" and returned True. It
    never opened a single tensor.

    That mattered more than it sounds. The stitch path it was gating had never
    run, and the fallback it fell into could not copy expert weights at all -
    so the exact artifact this was guarding against, an MoE whose every expert
    is an identical copy of the anchor, is precisely what a config-key check
    waves through. Structurally valid, completely wrong, and then the router
    trains on it and every enrichment number comes back at 1.0 with no way to
    tell a routing failure from a splice that never happened.

    Each tensor category has a different notion of correct, which is why this
    cannot be one loop with one comparison:

      expert weights  each expert equals ITS OWN specialist. For the fused
                      layout the row block is cat([gate, up]) - assembled in
                      the wrong order the model still runs and still scores
                      plausibly, it just computes a different function.
      router gate     must be exactly zero (untrained).
      shared gate     must contain NO zero: silu(0)/0 is NaN after GGUF export,
                      which is a bug that only appears outside Python.
      shared expert   inert by construction, so exactly zero.
      dense layers    in mlp_only_layers the FFN is the MEAN of all
                      specialists, accumulated in fp32 then cast back. Summed
                      in a different order the bf16 result differs, so the
                      check reproduces the order exactly.
      lm_head         equals embed_tokens when the anchor was tied.
      backbone        equals the anchor.
      anything else   came from nowhere, which is a failure by itself.

    Returns True only if nothing mismatched and nothing was dropped.
    """
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"   config.json missing at {out_dir}")
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)

    names = cfg.get("expert_names")
    if not names:
        print("   config.json has no expert_names - cannot map experts to sources")
        return False
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
        if key not in cfg:
            print(f"   missing MoE config field: {key}")
            return False

    root = output_root if output_root is not None else os.path.dirname(
        os.path.abspath(out_dir))
    spec_dirs = [_specialist_dir(root, n) for n in names]
    for d in spec_dirs:
        if not os.path.isdir(d):
            print(f"   missing specialist {d} - cannot verify against source")
            return False

    # ORDER MATTERS HERE. Everything above is structural and needs nothing
    # installed, so a missing config or an absent specialist gets a clear
    # answer on any machine. Only the tensor comparison below needs torch -
    # and if torch is missing we return FALSE, not True. "I could not check"
    # must never be reported as "I checked and it was fine"; that conflation
    # is the whole reason the old config-only verify was dangerous.
    try:
        import torch
    except ImportError:
        print("   cannot verify: torch is required to compare tensors, and an "
              "unverified stitch must not be reported as verified")
        return False

    E = cfg["num_experts"]
    try:
        M = _smap(out_dir)
        S = [_smap(d) for d in spec_dirs]
    except FileNotFoundError as exc:
        print(f"   {exc}")
        return False
    A = S[0]                                # anchor == expert 0's specialist

    fused = any(k.endswith("experts.gate_up_proj") for k in M)
    print(f"   verifying {E} experts, layout: "
          f"{'fused' if fused else 'ModuleList'}")

    bad: List[str] = []
    checked = dict(backbone=0, expert=0, router=0, shexp=0, head=0, dense_avg=0)

    def fail(key: str, why: str) -> None:
        bad.append(key)
        print(f"   MISMATCH {key}: {why}")

    for key in M:
        if ".mlp.experts." in key:
            li = int(key.split("model.layers.")[1].split(".")[0])
            t = _get(M, key)
            if fused and key.endswith("gate_up_proj"):
                for e in range(E):
                    g = _get(S[e], f"model.layers.{li}.mlp.gate_proj.weight")
                    u = _get(S[e], f"model.layers.{li}.mlp.up_proj.weight")
                    if not torch.equal(t[e], torch.cat([g, u], dim=0)):
                        fail(key, f"expert {e} ({names[e]}) gate/up mismatch")
                        break
            elif fused:
                for e in range(E):
                    d = _get(S[e], f"model.layers.{li}.mlp.down_proj.weight")
                    if not torch.equal(t[e], d):
                        fail(key, f"expert {e} ({names[e]}) down mismatch")
                        break
            else:
                bits = key.split(".")
                e = int(bits[bits.index("experts") + 1])
                proj = bits[-2]
                if not torch.equal(
                        t, _get(S[e], f"model.layers.{li}.mlp.{proj}.weight")):
                    fail(key, f"expert {e} ({names[e]}) {proj} mismatch")
            checked["expert"] += 1

        elif key.endswith(".mlp.gate.weight"):
            t = _get(M, key)
            # WHAT "CORRECT" MEANS HERE DEPENDS ON WHAT WAS ASKED FOR.
            #
            # zero  -> exactly zero, and the check stays as strict as it was.
            # random -> small noise. Still checkable, just not by equality: it
            #   must be non-zero (or the init silently did nothing) and it must
            #   be SMALL (or the untrained MoE is routing on garbage before a
            #   single step). A range check is a weaker claim than bit-equality
            #   and is labelled as one rather than quietly widened.
            if router_init == "random":
                amax = t.abs().max().item()
                if amax == 0.0:
                    fail(key, "router_init=random but the gate is all zeros - "
                              "the init did not run")
                elif amax > router_init_std * 12:
                    fail(key, f"router noise absmax {amax:.3e} is far beyond "
                              f"std {router_init_std} - the untrained MoE will "
                              f"route on this")
            elif t.abs().sum().item() != 0.0:
                fail(key, f"router not zero (absmax {t.abs().max().item():.3e})")
            checked["router"] += 1

        elif key.endswith(".mlp.shared_expert_gate.weight"):
            t = _get(M, key)
            if t.abs().min().item() == 0.0:
                fail(key, "shared_expert_gate contains ZERO -> NaN after GGUF export")
            elif abs(t.float().mean().item() - gate_fill) > 1e-3:
                fail(key, f"gate fill {t.float().mean().item():.4f} != {gate_fill}")
            checked["shexp"] += 1

        elif ".mlp.shared_expert." in key:
            if _get(M, key).abs().sum().item() != 0.0:
                fail(key, "shared expert should be zero (inert by construction)")
            checked["shexp"] += 1

        elif key == "lm_head.weight" and key not in A:
            if not torch.equal(_get(M, key), _get(A, "model.embed_tokens.weight")):
                fail(key, "lm_head != embed_tokens (anchor was tied)")
            checked["head"] += 1

        elif key.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight",
                           ".mlp.down_proj.weight")):
            acc = None
            for sp in S:
                v = _get(sp, key).float()
                acc = v if acc is None else acc + v
            want = (acc / len(S)).to(_get(M, key).dtype)
            if not torch.equal(_get(M, key), want):
                fail(key, "dense FFN != mean of specialists")
            checked["dense_avg"] += 1

        elif key in A:
            if not torch.equal(_get(M, key), _get(A, key)):
                fail(key, "backbone tensor differs from anchor")
            checked["backbone"] += 1

        else:
            fail(key, "NO SOURCE - this tensor came from nowhere")

    # The other direction: did the stitch DROP anything the anchor had?
    skip = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")
    missing = [k for k in A if k not in M and not any(x in k for x in skip)]

    print(f"   checked  backbone {checked['backbone']}  experts {checked['expert']}"
          f"  router {checked['router']}  shared {checked['shexp']}"
          f"  head {checked['head']}  dense-avg {checked['dense_avg']}")
    if missing:
        print(f"   anchor tensors NOT present in the MoE: {missing[:8]}")

    if bad or missing:
        print(f"   STITCH FAILED - {len(bad)} mismatched, {len(missing)} missing.")
        print("   Do NOT start router training on this. Every one of these is a")
        print("   tensor that would train fine and produce confident nonsense.")
        return False

    print("   stitch OK: every tensor bit-identical to its source. Backbone from")
    print("   the anchor, each expert from its own specialist, router zeroed,")
    print("   shared expert inert, gate safe for GGUF export.")
    return True
