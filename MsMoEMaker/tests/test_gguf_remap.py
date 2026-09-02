"""The GGUF dense-layer remap, guarded without torch.

transformers renders `moe.dense_layers` as plain Qwen2MLP blocks, and
llama.cpp's converter has no tensor name for a dense FFN inside a qwen2_moe
checkpoint - the first gauntlet export died on "Can not map tensor
'model.layers.0.mlp.down_proj.weight'".

The export rewrites the checkpoint before converting: each dense layer
becomes an all-MoE block whose experts are identical copies of the dense
FFN, with the down projection scaled to cancel the sum of the top-k softmax
weights. Which scale is exact depends on whether the llama.cpp CHECKOUT
renormalises Qwen2MoE top-k weights (b10499 does not: its qwen2moe builder
passes norm_w=false), so the export reads that regime out of the checkout's
own source and refuses when it cannot.

The decision logic is pure - no torch, no IO - so it is tested here exactly
as it runs, on the base install. A wrong key, shape, or scale in this plan
is a silent FFN scale factor in the exported model, which is precisely the
class of bug the export stage exists to catch.
"""
import pytest

from ms_moe_maker.moe import export as ex


def _keys(num_experts=8, dense=(0, 1), moe=(2, 3)):
    keys = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for li in range(4):
        keys += [f"model.layers.{li}.self_attn.q_proj.weight",
                 f"model.layers.{li}.input_layernorm.weight"]
    for li in dense:
        keys += [f"model.layers.{li}.mlp.gate_proj.weight",
                 f"model.layers.{li}.mlp.up_proj.weight",
                 f"model.layers.{li}.mlp.down_proj.weight"]
    for li in moe:
        keys.append(f"model.layers.{li}.mlp.gate.weight")
        keys += [f"model.layers.{li}.mlp.shared_expert.gate_proj.weight",
                 f"model.layers.{li}.mlp.shared_expert.up_proj.weight",
                 f"model.layers.{li}.mlp.shared_expert.down_proj.weight"]
        keys.append(f"model.layers.{li}.mlp.shared_expert_gate.weight")
        for e in range(num_experts):
            keys += [f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight",
                     f"model.layers.{li}.mlp.experts.{e}.up_proj.weight",
                     f"model.layers.{li}.mlp.experts.{e}.down_proj.weight"]
    return keys


def test_a_dense_mlp_fans_out_to_every_expert():
    """The block must compute MLP(x) once per expert so the top-k softmax
    weights can be cancelled by one scale on the down projection."""
    copies, _ = ex._remap_plan([0, 1], 8, _keys(), down_scale=4.0)
    assert set(copies) == {
        f"model.layers.{li}.mlp.{p}.weight"
        for li in (0, 1) for p in ("gate_proj", "up_proj", "down_proj")}
    dsts, _ = copies["model.layers.0.mlp.gate_proj.weight"]
    assert dsts == [
        f"model.layers.0.mlp.experts.{e}.gate_proj.weight" for e in range(8)]


def test_the_scale_rides_only_on_the_down_projection():
    """The expert computes down(SiLU(gate x) * up x): scaling gate or up
    would fight the SiLU nonlinearity; scaling down scales the output
    exactly."""
    copies, _ = ex._remap_plan([0, 1], 8, _keys(), down_scale=4.0)
    for li in (0, 1):
        for proj in ("gate_proj", "up_proj"):
            _, scale = copies[f"model.layers.{li}.mlp.{proj}.weight"]
            assert scale == 1.0
        _, scale = copies[f"model.layers.{li}.mlp.down_proj.weight"]
        assert scale == 4.0


def test_the_plan_adds_inert_moe_plumbing_per_dense_layer():
    """Zero router (uniform probs), zero shared expert, filled shared gate -
    the same shapes the real MoE layers already carry."""
    _, extras = ex._remap_plan([0, 1], 8, _keys(), down_scale=4.0)
    assert len(extras) == 10   # 5 per dense layer
    by_dst = {dst: (kind, ref) for dst, kind, ref in extras}
    assert by_dst["model.layers.0.mlp.gate.weight"] == (
        "zeros", "model.layers.2.mlp.gate.weight")
    assert by_dst["model.layers.0.mlp.shared_expert.gate_proj.weight"] == (
        "zeros", "model.layers.2.mlp.shared_expert.gate_proj.weight")
    assert by_dst["model.layers.0.mlp.shared_expert_gate.weight"] == (
        "fill", "model.layers.2.mlp.shared_expert_gate.weight")
    # every reference is the FIRST MoE layer, the one whose shapes are real
    assert {ref for _, _, ref in extras} <= {
        k for k in _keys() if "layers.2." in k}


def test_a_missing_remap_source_is_refused_not_guessed():
    keys = [k for k in _keys() if k != "model.layers.0.mlp.down_proj.weight"]
    with pytest.raises(RuntimeError, match="remap source"):
        ex._remap_plan([0, 1], 8, keys, down_scale=4.0)


def test_a_missing_reference_tensor_is_refused_not_guessed():
    keys = [k for k in _keys() if k != "model.layers.2.mlp.shared_expert_gate.weight"]
    with pytest.raises(RuntimeError, match="remap reference"):
        ex._remap_plan([0, 1], 8, keys, down_scale=4.0)


def test_dense_layers_without_any_experts_refuses():
    """mlp_only_layers covering the whole stack leaves nothing to borrow
    shapes from - and is a different model than the remap can express."""
    with pytest.raises(RuntimeError, match="no MoE expert tensors"):
        ex._remap_plan([0, 1], 8, _keys(moe=()), down_scale=4.0)


def test_destinations_that_clash_with_live_keys_refuse():
    """A checkpoint that already has expert tensors on a 'dense' layer is
    inconsistent; overwriting them silently would corrupt the export."""
    keys = _keys(dense=(0, 1), moe=(2, 3)) + [
        f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(8)]
    keys += [                      # layer 0 must also carry the full MoE
        "model.layers.0.mlp.gate.weight",                        # plumbing
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ]
    with pytest.raises(RuntimeError, match="clash"):
        ex._remap_plan([0, 1], 8, keys, down_scale=4.0)


def test_no_dense_layers_means_no_plan():
    copies, extras = ex._remap_plan([], 8, _keys(), down_scale=4.0)
    assert copies == {} and extras == []


def test_the_remap_requires_matching_expert_width():
    cfg = {"mlp_only_layers": [0, 1], "intermediate_size": 896,
           "moe_intermediate_size": 768}
    msg = ex._refuse_unremappable_dense(cfg)
    assert msg and "moe_intermediate_size" in msg
    cfg["moe_intermediate_size"] = 896
    assert ex._refuse_unremappable_dense(cfg) is None


def test_a_checkpoint_without_dense_layers_is_never_refused():
    assert ex._refuse_unremappable_dense({"intermediate_size": 896}) is None


def test_the_converter_config_drops_mlp_only_layers():
    """The remapped checkpoint is all-MoE; a converter that understands
    mlp_only_layers must not try dense mappings on MoE-shaped tensors."""
    cfg = {"model_type": "qwen2_moe", "mlp_only_layers": [0, 1],
           "num_experts": 8, "nested": {"kept": True}}
    out = ex._gguf_src_config(cfg)
    assert "mlp_only_layers" not in out
    assert out["num_experts"] == 8 and out["nested"]["kept"] is True
    assert "mlp_only_layers" in cfg, "the input must not be mutated"


# ── The runtime-regime detector ─────────────────────────────────────────────

_B10499_STYLE = """
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
"""

_OLD_STYLE_CASE = """
        case LLM_ARCH_QWEN2MOE:
            {
                cur = build_moe_ffn(cur,
                        layer.ffn_gate_inp,
                        layer.ffn_up_exps,
                        layer.ffn_gate_exps,
                        layer.ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        0.0f,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
            } break;
        case LLM_ARCH_QWEN3MOE:
"""


def test_the_detector_reads_b10499s_false():
    assert ex._norm_w_from_source(_B10499_STYLE) is False


def test_the_detector_reads_the_old_layouts_true():
    assert ex._norm_w_from_source(_OLD_STYLE_CASE) is True


def test_the_detector_returns_none_when_the_source_does_not_say():
    assert ex._norm_w_from_source("int main() { return 0; }") is None
    no_flag = "build_moe_ffn(cur, gate_inp, up, gate, down, nullptr, 1, 2, "
    no_flag += "LLM_FFN_SILU, hparams.x, 1.0f, SOFTMAX, il);"
    assert ex._norm_w_from_source(no_flag) is None


def test_the_detector_prefers_the_per_model_builder(monkeypatch):
    import builtins
    real_open = builtins.open
    files = {"qwen2moe.cpp": _B10499_STYLE, "llama-graph.cpp": _OLD_STYLE_CASE}

    def fake_open(path, *a, **kw):
        for name, body in files.items():
            if str(path).endswith(name):
                import io
                return io.StringIO(body)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(ex.os.path, "isfile",
                        lambda p: any(str(p).endswith(n) for n in files))
    assert ex._detect_qwen2moe_norm_w("/llama") is False


def test_the_detector_falls_back_to_the_graph_case(monkeypatch):
    import builtins
    real_open = builtins.open
    files = {"llama-graph.cpp": _OLD_STYLE_CASE}

    def fake_open(path, *a, **kw):
        for name, body in files.items():
            if str(path).endswith(name):
                import io
                return io.StringIO(body)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(ex.os.path, "isfile",
                        lambda p: any(str(p).endswith(n) for n in files))
    assert ex._detect_qwen2moe_norm_w("/llama") is True


def test_an_unreadable_layout_is_refused_not_guessed(monkeypatch):
    import builtins
    real_open = builtins.open
    files = {"llama-graph.cpp": "int main() { return 0; }"}

    def fake_open(path, *a, **kw):
        for name, body in files.items():
            if str(path).endswith(name):
                import io
                return io.StringIO(body)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(ex.os.path, "isfile",
                        lambda p: any(str(p).endswith(n) for n in files))
    with pytest.raises(RuntimeError, match="top-k regime"):
        ex._detect_qwen2moe_norm_w("/llama")
