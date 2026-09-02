"""What `_prepare_gguf_src` actually writes.

THE HALF THAT WAS NOT TESTED IS THE HALF THAT BROKE. `_remap_plan` is pure and
has had tests since it was written; `_prepare_gguf_src` does the IO, had none,
and shipped a bug that only appears at `save_file` - after the whole build.

    for dst in dsts:
        out[dst] = t          # nine names, ONE tensor object

safetensors refuses a file where several names alias one storage, and it is
right to: there is no way to say which of the nine a reader should get. The
export died at stage 6 of 17, hours in.

No torch here, deliberately. The bug is about OBJECT IDENTITY in a dict, which
a stub models perfectly, and a test that needs a GPU is a test nobody runs.
"""
import json
import os
import sys
import types

import pytest

from ms_moe_maker.moe import export as ex


class FakeTensor:
    """Enough tensor to satisfy the remap: shape, dtype, clone, float, to.

    `clone` returns a NEW object, which is the entire property under test -
    a stub whose clone returned self would pass this test and ship the bug.
    """

    def __init__(self, name, shape=(4, 4), dtype="bf16", scale=1.0):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.scale = scale

    def clone(self):
        return FakeTensor(self.name, self.shape, self.dtype, self.scale)

    def float(self):
        return FakeTensor(self.name, self.shape, "f32", self.scale)

    def to(self, dtype):
        return FakeTensor(self.name, self.shape, dtype, self.scale)

    def __mul__(self, other):
        return FakeTensor(self.name, self.shape, self.dtype, self.scale * other)


def _checkpoint_keys(num_experts=3, moe_layer=2, dense=(0, 1)):
    keys = []
    for li in dense:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"model.layers.{li}.mlp.{proj}.weight")
    for e in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"model.layers.{moe_layer}.mlp.experts.{e}.{proj}.weight")
    keys += [
        f"model.layers.{moe_layer}.mlp.gate.weight",
        f"model.layers.{moe_layer}.mlp.shared_expert.gate_proj.weight",
        f"model.layers.{moe_layer}.mlp.shared_expert.up_proj.weight",
        f"model.layers.{moe_layer}.mlp.shared_expert.down_proj.weight",
        f"model.layers.{moe_layer}.mlp.shared_expert_gate.weight",
        "model.embed_tokens.weight",
    ]
    return keys


@pytest.fixture
def fake_stack(monkeypatch):
    """torch + safetensors.torch, stubbed. Records what save_file was handed."""
    seen = {}

    torch = types.ModuleType("torch")
    torch.zeros = lambda shape, dtype=None: FakeTensor("zeros", shape, dtype)
    torch.full = lambda shape, val, dtype=None: FakeTensor(
        f"full:{val}", shape, dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)

    class _Reader:
        def __init__(self, keys):
            self._keys = keys

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def keys(self):
            return list(self._keys)

        def get_tensor(self, k):
            return FakeTensor(k)

    st = types.ModuleType("safetensors.torch")
    st.safe_open = lambda path, framework=None: _Reader(seen["keys"])
    st.save_file = lambda tensors, path: seen.update(saved=dict(tensors),
                                                     path=path)
    safetensors = types.ModuleType("safetensors")
    safetensors.torch = st
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st)
    return seen


def _run(tmp_path, fake_stack, num_experts=3, down_scale=1.0):
    src = tmp_path / "moe_trained"
    src.mkdir()
    (src / "model.safetensors").write_text("", encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    fake_stack["keys"] = _checkpoint_keys(num_experts=num_experts)
    cfg = {"num_experts": num_experts, "num_experts_per_tok": 2,
           "mlp_only_layers": [0, 1], "architectures": ["Qwen2MoeForCausalLM"]}
    config = types.SimpleNamespace(shared_expert_gate_fill=0.05)
    work = ex._prepare_gguf_src(config, str(src), cfg, [0, 1], down_scale)
    return work, fake_stack["saved"]


def test_every_fanned_out_expert_is_its_own_tensor(tmp_path, fake_stack):
    """THE BUG. Nine expert names must not be nine references to one object.

    `save_file` is what caught this in production, after the whole build. The
    assertion is on object identity because that IS the defect - the values
    were correct, and there was exactly one of them.
    """
    _, saved = _run(tmp_path, fake_stack, num_experts=3)

    fanned = [k for k in saved
              if k.startswith("model.layers.0.mlp.experts.")
              and k.endswith("down_proj.weight")]
    assert len(fanned) == 3, fanned

    ids = [id(saved[k]) for k in fanned]
    assert len(set(ids)) == len(ids), (
        "the fanned-out experts share a tensor object - safetensors will "
        "refuse this file, and it will refuse it at the END of the build")


def test_no_two_keys_in_the_whole_file_share_an_object(tmp_path, fake_stack):
    """The general form, so the next fan-out cannot reintroduce it elsewhere."""
    _, saved = _run(tmp_path, fake_stack, num_experts=3)
    by_id = {}
    for k, v in saved.items():
        by_id.setdefault(id(v), []).append(k)
    shared = {i: ks for i, ks in by_id.items() if len(ks) > 1}
    assert not shared, f"aliased tensors: {list(shared.values())}"


def test_the_dense_layers_are_gone_and_the_experts_are_there(tmp_path,
                                                             fake_stack):
    """The remap's actual job, asserted on the written payload rather than the plan."""
    _, saved = _run(tmp_path, fake_stack, num_experts=3)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"model.layers.0.mlp.{proj}.weight" not in saved
        for e in range(3):
            assert f"model.layers.0.mlp.experts.{e}.{proj}.weight" in saved
    assert "model.layers.0.mlp.gate.weight" in saved
    assert "model.layers.0.mlp.shared_expert_gate.weight" in saved
    # untouched keys survive
    assert "model.embed_tokens.weight" in saved


def test_the_written_config_declares_no_dense_layers(tmp_path, fake_stack):
    """mlp_only_layers must not survive: the remapped checkpoint is all-MoE,
    and a converter that DOES understand the field would try to map dense
    names onto the MoE-shaped tensors we just wrote."""
    work, _ = _run(tmp_path, fake_stack)
    with open(os.path.join(work, "config.json"), encoding="utf-8") as fh:
        written = json.load(fh)
    assert "mlp_only_layers" not in written
    assert written["num_experts"] == 3


def test_sidecar_files_are_carried_over_but_weights_are_not(tmp_path,
                                                            fake_stack):
    """The tokenizer has to come along; the original weights must not, or the
    converter would find two checkpoints in one directory."""
    work, _ = _run(tmp_path, fake_stack)
    assert os.path.exists(os.path.join(work, "tokenizer.json"))
    assert not os.path.exists(
        os.path.join(work, "model.safetensors.orig"))


def test_the_down_scale_rides_only_on_down_proj(tmp_path, fake_stack):
    """A non-renormalising runtime needs num_experts/experts_per_tok on down.

    gate and up must stay untouched - scaling either fights the SiLU, and the
    error would be a quiet FFN magnitude shift rather than a crash.
    """
    _, saved = _run(tmp_path, fake_stack, num_experts=3, down_scale=1.5)
    assert saved["model.layers.0.mlp.experts.0.down_proj.weight"].scale == 1.5
    assert saved["model.layers.0.mlp.experts.0.gate_proj.weight"].scale == 1.0
    assert saved["model.layers.0.mlp.experts.0.up_proj.weight"].scale == 1.0
