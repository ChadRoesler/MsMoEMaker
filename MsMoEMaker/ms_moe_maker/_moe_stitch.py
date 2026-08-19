"""Streaming MoE stitcher — VENDORED, not reimplemented.

PROVENANCE. This is FraunkensteinsLab/moe_stitch.py, moved into the package
verbatim (shebang aside). It is the code that actually built the proven 0.5B
rung — the one that came back with every tensor bit-identical to its source and
2.12x mean router enrichment at p=0.00032.

WHY IT HAD TO MOVE. `moe_stitch` was never a PyPI package. It is a local module
that sits next to fraunkenstein_universal.py, so `import moe_stitch` resolved
only when Python was running with the Lab directory on sys.path — which is
exactly how every real build ran, because the legacy path forked that script
with cwd inside the Lab.

A pip-installed ms-moe-maker has no Lab directory. The import could never
succeed, so stitch.py always fell through to an in-process fallback that had
never once executed: it looked up `mlp.experts.N.*` keys in a DENSE specialist
checkpoint (where they cannot exist), so it copied no expert weights at all,
and it read pytorch_model.bin, which save_pretrained stopped writing by default
years ago. An MoE with N identical copies of the anchor would have come out the
other side and passed the config-only verify_stitch.

So the fallback is gone and this is the only path. Vendoring beats depending
here: the module is 279 lines with no dependencies beyond torch, and the
alternative is a published CLI that silently needs a private repo checked out
next to it.

Keep this file in sync by REPLACING it, not by editing it in place — if the Lab
copy changes, copy it over again and re-run the stitch tests.

Original module docstring follows.
"""

"""
Ms.Moe - streaming MoE stitcher (GPL-3.0)

WHY THIS EXISTS
---------------
The obvious way to build the stitched MoE is:

    moe = Qwen2MoeForCausalLM(moe_config)      # <- allocates the whole thing
    moe_state = moe.state_dict()
    ... copy tensors in ...
    moe.save_pretrained(out_dir)

At 0.5B that is fine. At 14B it is not, and the arithmetic is brutal:

    stitched MoE .............................. 55.5B parameters
    in torch's DEFAULT float32 ................ 222 GB
    in bfloat16 ............................... 111 GB
    plus the anchor held open ..................  30 GB
    plus one specialist at a time ..............  30 GB
    ------------------------------------------------------
    peak ...................................... 170 GB
    machine ................................... 121 GB (+15 GB swap)

It does not raise MemoryError. It goes into swap and sits there. On the real
run it printed "Loading weights: 100%" and then produced nothing for eighteen
hours, which reads exactly like a hang and is impossible to distinguish from
one without checking `free`.

So this module never materialises the model. It:

  1. instantiates the skeleton on the META device - names, shapes and layout,
     zero bytes of storage - purely to learn what tensors the MoE expects;
  2. streams each tensor from its source safetensors file straight into output
     shards, one at a time.

Peak memory is roughly one shard, ~4 GB, regardless of model size. The same
code path works at 0.5B and would work at 70B.

It also keeps every safety property the in-RAM version had: unknown source is
fatal, shape mismatch is fatal, and config.json is written LAST so a partial
directory is never mistaken for a finished one by _done().
"""

import json
import os
import re

import torch

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def safetensors_map(path):
    """key -> the file holding it, so tensors can be read ONE AT A TIME."""
    from safetensors import safe_open
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return {k: os.path.join(path, v)
                    for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(path, "model.safetensors")
    if not os.path.exists(single):
        raise RuntimeError(f"no safetensors found in {path}")
    with safe_open(single, framework="pt") as f:
        return {k: single for k in f.keys()}


def st_get(smap, key):
    """One tensor off disk. Nothing else stays resident."""
    from safetensors import safe_open
    with safe_open(smap[key], framework="pt") as f:
        return f.get_tensor(key)


class ShardWriter:
    """Write a sharded safetensors checkpoint incrementally.

    Tensors arrive one at a time, buffer up to a shard, then flush. The final
    rename to model-00001-of-0000N is done at the end, once the shard count is
    actually known.
    """

    def __init__(self, out_dir, max_bytes=4_000_000_000):
        self.out_dir, self.max = out_dir, max_bytes
        os.makedirs(out_dir, exist_ok=True)
        self.buf, self.buf_bytes = {}, 0
        self.shards, self.weight_map, self.total = [], {}, 0

    def add(self, key, tensor):
        n = tensor.numel() * tensor.element_size()
        if self.buf and self.buf_bytes + n > self.max:
            self._flush()
        self.buf[key] = tensor.contiguous()
        self.buf_bytes += n
        self.total += n

    def _flush(self):
        from safetensors.torch import save_file
        name = f"tmp-{len(self.shards) + 1:05d}.safetensors"
        save_file(self.buf, os.path.join(self.out_dir, name),
                  metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = name
        self.shards.append(name)
        print(f"      shard {len(self.shards)}: {self.buf_bytes / 1e9:5.2f} GB, "
              f"{len(self.buf)} tensors", flush=True)
        self.buf, self.buf_bytes = {}, 0

    def finish(self):
        if self.buf:
            self._flush()
        n = len(self.shards)
        rename = {}
        for i, old in enumerate(self.shards, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(os.path.join(self.out_dir, old),
                       os.path.join(self.out_dir, new))
            rename[old] = new
        wm = {k: rename[v] for k, v in self.weight_map.items()}
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": self.total}, "weight_map": wm},
                      f, indent=2)
        return n, self.total


def plan_from_meta(model_cls, moe_config):
    """Names + shapes of every tensor the MoE wants, at zero memory cost.

    torch.device("meta") makes every factory call allocate no storage, so this
    builds a full 55.5B-parameter module description for nothing. We only ever
    read .shape off it.
    """
    with torch.device("meta"):
        skeleton = model_cls(moe_config)
    plan = {k: tuple(v.shape) for k, v in skeleton.state_dict().items()}
    del skeleton
    return plan


def stream_stitch(out_dir, anchor_dir, spec_dirs, expert_names,
                  model_cls, moe_config, shared_gate_fill,
                  num_layers, tokenizer_src=None, max_shard_bytes=4_000_000_000):
    """Build the MoE checkpoint on disk. Never holds the model in memory.

    spec_dirs / expert_names are parallel and IN EXPERT INDEX ORDER.
    """
    plan = plan_from_meta(model_cls, moe_config)
    n_params = sum(int(torch.tensor(s).prod()) if s else 0 for s in plan.values())
    print(f"   planned on meta device: {len(plan)} tensors, "
          f"{n_params / 1e9:.1f}B params (0 bytes allocated)")

    anchor_map = safetensors_map(anchor_dir)
    spec_maps = [safetensors_map(d) for d in spec_dirs]

    # Write in the SOURCE dtype. The meta skeleton reports float32 because that
    # is torch's default, and honouring that would double the checkpoint on
    # disk for no reason - 222 GB instead of 111 GB.
    probe = st_get(anchor_map, "model.embed_tokens.weight")
    dtype = probe.dtype
    del probe
    print(f"   writing in {dtype} (source dtype, not torch's float32 default)")

    # Detect the layout from ANY expert key, not from layer 0. With
    # mlp_only_layers set, the bottom of the stack is dense and layer 0 has no
    # experts at all - probing it specifically reports "unrecognised layout"
    # for a model that is perfectly well formed.
    expert_keys = [k for k in plan if ".mlp.experts." in k]
    if not expert_keys:
        raise RuntimeError("no expert tensors in the plan at all - is "
                           "mlp_only_layers covering every layer?")
    fused = any(k.endswith(".mlp.experts.gate_up_proj") for k in expert_keys)
    listed = any(".mlp.experts.0." in k for k in expert_keys)
    if not (fused or listed):
        raise RuntimeError("unrecognised expert layout; expert keys seen: "
                           + str(expert_keys[:6]))
    moe_layers = sorted({int(_LAYER_RE.search(k).group(1)) for k in expert_keys})
    dense_layers = [i for i in range(num_layers) if i not in moe_layers]
    print(f"   expert layout: {'fused 3-D params' if fused else 'ModuleList'}")
    print(f"   MoE layers {len(moe_layers)} (from {moe_layers[0]}), "
          f"dense layers {len(dense_layers)} {dense_layers if dense_layers else ''}")

    writer = ShardWriter(out_dir, max_shard_bytes)
    stats = dict(backbone=0, expert=0, router=0, shexp=0, head=0, dense_avg=0)

    for key, shape in plan.items():
        m = _LAYER_RE.search(key)
        li = int(m.group(1)) if m else None

        if ".mlp.experts." in key:
            if fused and key.endswith("gate_up_proj"):
                # forward does linear(x, gate_up_proj[e]).chunk(2, -1), so rows
                # are [gate ; up] on dim 0. Backwards produces garbage silently.
                parts = []
                for sm in spec_maps:
                    g = st_get(sm, f"model.layers.{li}.mlp.gate_proj.weight")
                    u = st_get(sm, f"model.layers.{li}.mlp.up_proj.weight")
                    parts.append(torch.cat([g, u], dim=0))
                t = torch.stack(parts, dim=0)
                del parts
            elif fused and key.endswith("down_proj"):
                t = torch.stack(
                    [st_get(sm, f"model.layers.{li}.mlp.down_proj.weight")
                     for sm in spec_maps], dim=0)
            else:
                bits = key.split(".")
                ei = int(bits[bits.index("experts") + 1])
                proj = bits[-2]
                t = st_get(spec_maps[ei], f"model.layers.{li}.mlp.{proj}.weight")
            stats["expert"] += 1

        elif key.endswith(".mlp.gate.weight"):
            # The router. Qwen2MoeTopKRouter initialises this to zeros anyway,
            # and train_router is what fills it.
            t = torch.zeros(shape, dtype=dtype)
            stats["router"] += 1

        elif ".mlp.shared_expert." in key:
            # Inert by construction - see SHARED_EXPERT_WIDTH in the pipeline.
            t = torch.zeros(shape, dtype=dtype)
            stats["shexp"] += 1

        elif key.endswith(".mlp.shared_expert_gate.weight"):
            # NEVER zero: llama.cpp computes this sigmoid as silu(x)/x, which is
            # 0/0 at exactly zero and NaNs the whole model after export.
            t = torch.full(shape, shared_gate_fill, dtype=dtype)
            stats["shexp"] += 1

        elif key == "lm_head.weight" and key not in anchor_map:
            # Anchor had tied embeddings, so it has no lm_head to copy.
            # Materialise it from the embeddings - that is what tying meant.
            t = st_get(anchor_map, "model.embed_tokens.weight")
            stats["head"] += 1

        elif key.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight",
                           ".mlp.down_proj.weight")):
            # A DENSE layer (this index is in mlp_only_layers). It has one FFN
            # shared by everything, so taking it from the anchor would silently
            # make the bottom of the stack a powershell model. BTX averages the
            # non-expert parameters across branches; do that.
            acc = None
            for sm in spec_maps:
                v = st_get(sm, key).float()
                acc = v if acc is None else acc + v
            t = (acc / len(spec_maps))
            stats["dense_avg"] += 1

        elif key in anchor_map:
            t = st_get(anchor_map, key)
            stats["backbone"] += 1

        else:
            raise RuntimeError(
                f"no source for MoE tensor {key!r}. Refusing to write a "
                f"half-initialised model - a silently random tensor here costs "
                f"a GPU-week and produces confident nonsense with no error.")

        if tuple(t.shape) != tuple(shape):
            raise RuntimeError(
                f"shape mismatch on {key}: MoE wants {tuple(shape)}, source "
                f"gave {tuple(t.shape)}")
        writer.add(key, t.to(dtype))
        del t

    n_shards, total = writer.finish()
    print(f"   backbone {stats['backbone']}  experts {stats['expert']}  "
          f"router {stats['router']}  shared {stats['shexp']}  head {stats['head']}"
          f"  dense-avg {stats['dense_avg']}")
    print(f"   wrote {n_shards} shards, {total / 1e9:.1f} GB")

    # CONFIG LAST. _done() treats a directory as finished when config.json is
    # present, so writing it first would make an interrupted stitch look
    # complete and get silently skipped on the next run.
    moe_config.torch_dtype = str(dtype).replace("torch.", "")
    moe_config.save_pretrained(out_dir)
    if tokenizer_src:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tokenizer_src).save_pretrained(out_dir)
    return out_dir
