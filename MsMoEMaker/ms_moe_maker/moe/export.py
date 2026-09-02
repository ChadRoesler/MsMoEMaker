"""GGUF export — convert the trained MoE to GGUF and smoke-test it.

Proves the model generates OUTSIDE Python.  Three of this project's nastiest
bugs were invisible inside transformers and only appeared past that boundary:
  * shared_expert_intermediate_size=0 → zero-element GGUF tensor overflow
  * tie_word_embeddings=true → no output.weight written
  * all-zero shared-expert gate → NaN in every layer → same token forever

The smoke test runs llama-cli (or llama.cpp CLI) and checks:
  1. The process exits cleanly
  2. It produces at least one alphanumeric token
  3. No degenerate run of identical characters (NaN signature)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional


# Characters that are legitimately repeated in code / llama-cli output.
# Excluded from the degenerate-run detector.
_GGUF_SEP = set("=-_ *#~/\\.+|<>")


def _longest_char_run(text: str):
    """Longest run of one repeated, non-structural character.

    Returns (length, character).  A long run of identical alnum chars is
    the signature of NaN in the residual stream.
    """
    run = best = 0
    prev, worst = "", ""
    for ch in text:
        if ch.isspace() or ch in _GGUF_SEP or 0x2500 <= ord(ch) <= 0x259F:
            run, prev = 0, ""
            continue
        run = run + 1 if ch == prev else 1
        prev = ch
        if run > best:
            best, worst = run, ch
    return best, worst


def resolve_llama_binary(llama_cpp_dir: str, name: str) -> str:
    """Find a llama.cpp executable. One resolver, every caller.

    There were two. export_gguf checked
    `<llama_cpp_dir>/build/bin/<name>` and then PATH; smoke_gguf checked PATH
    and nothing else. So a llama.cpp built in the ordinary way - binaries in
    `build/bin`, not installed system-wide - was findable during a build and
    invisible to `ms-moe-maker smoke`, which is the one command whose entire
    job is to run it. Two functions answering the same question differently is
    the shape of half the bugs in this project.

    Order: the build tree, then an installed prefix, then PATH. `build/bin` is
    first because that is where `cmake --build build` puts things and almost
    nobody installs llama.cpp system-wide.
    """
    for cand in (os.path.join(llama_cpp_dir, "build", "bin", name),
                 os.path.join(llama_cpp_dir, "bin", name),
                 os.path.join(llama_cpp_dir, name)):
        if llama_cpp_dir and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which(name) or ""


def gguf_path_for(config) -> str:
    return f"{config.output_root}/msmoe-{config.size}-f16.gguf"


def export_is_done(config) -> bool:
    """Has the GGUF been converted AND proven to generate?

    "DONE" means CONVERTED + SMOKE-PASSED. A file that converted but has no
    smoke log was never verified - and re-running only the smoke test must not
    cost a full re-conversion, which is why the two conditions are separate.

    ONE PREDICATE, TWO CALLERS. The stage function and the orchestrator both
    need this answer, and deriving it twice is how they drift. (Runner and
    build_config each worked out the run directory independently and
    disagreed - the manifest landed in one folder and every artifact in
    another. Same trap, so: one function.)
    """
    if config.force:
        return False
    gguf_path = gguf_path_for(config)
    # ".smokepass.txt", NOT ".smoketest.txt". The latter is the LOG, and it is
    # written unconditionally so a failed run still leaves its evidence - which
    # meant a build that failed the NaN check was nonetheless "done", and every
    # subsequent run printed "[skip] GGUF export already done" and reported
    # OK - 8/8 stages for a model that emits one token forever. The proof is
    # now a separate artifact written only after every check passes.
    return (os.path.exists(gguf_path)
            and os.path.getsize(gguf_path) > 0
            and os.path.exists(gguf_path + ".smokepass.txt"))


def _gguf_src_workdir(src_dir: str) -> str:
    return src_dir + ".gguf-src"


def _refuse_unremappable_dense(cfg_dict: dict):
    """Wire-or-refuse gates for the dense-layer remap. None = go.

    transformers renders `moe.dense_layers` as PLAIN Qwen2MLP blocks at the
    bottom of the Qwen2MoE stack, and llama.cpp's convert_hf_to_gguf.py has
    no tensor mapping for a dense FFN inside a qwen2_moe checkpoint - the
    first gauntlet export died on "Can not map tensor
    'model.layers.0.mlp.down_proj.weight'".

    The export therefore rewrites the checkpoint into a converter-shaped one
    (see _remap_plan): each dense layer becomes an all-MoE block whose
    experts are identical copies of the dense FFN, with the down projection
    scaled to cancel the sum of the top-k softmax weights (see
    _detect_qwen2moe_norm_w). The only thing that cannot be cancelled by a
    scale is a shape mismatch: the expert slots must fit the dense matrices.
    """
    dense = cfg_dict.get("mlp_only_layers") or []
    if not dense:
        return None
    inter = cfg_dict.get("intermediate_size")
    moe_inter = cfg_dict.get("moe_intermediate_size", inter)
    if moe_inter != inter:
        return (
            f"moe_intermediate_size ({moe_inter}) != intermediate_size "
            f"({inter}): the dense FFN matrices would not fit the expert "
            f"slots the GGUF remap needs. Make them equal, or clear "
            f"moe.dense_layers.")
    return None


def _gguf_src_config(cfg_dict: dict) -> dict:
    """The converter-shaped config: identical, minus mlp_only_layers.

    The remapped checkpoint is all-MoE. Declaring dense layers would make a
    converter that DOES understand mlp_only_layers try to map dense names and
    fail on the MoE-shaped tensors we wrote; old converters ignore the field,
    and both must see a plain all-MoE model.
    """
    out = json.loads(json.dumps(cfg_dict))
    out.pop("mlp_only_layers", None)
    return out


def _remap_plan(dense_layers, num_experts: int, source_keys, down_scale: float):
    """Fan the dense FFNs out into identical experts + inert MoE plumbing.

    PURE - no torch, no IO - so the decision is testable on a laptop.
    Returns (copies, extras):
      copies: {src_key: ([expert destination keys], scale)} - the dense
              matrix fans out to every expert. The scale rides on the DOWN
              projection only: the expert computes down(SiLU(gate x) * up x),
              so scaling down's weights scales the output exactly, while
              scaling gate or up would fight the SiLU nonlinearity.
              down_scale = num_experts / experts_per_tok cancels the sum of
              the top-k softmax weights on the runtimes that do NOT
              renormalise them (a zero router gives exactly 1/num_experts per
              expert for every input). On renormalising runtimes the scale
              is 1.0.
      extras: [(dst_key, "zeros"|"fill", ref_key)] - the router gate (zero:
              uniform probs), the shared expert (zero: the stitched model's
              shared experts are already inert 1-wide zeros), and the shared
              gate (fill: irrelevant, it multiplies zero), each borrowing the
              shape and dtype of the REAL MoE layer's tensor at ref_key.
    Refuses if a consumed or referenced key is missing - guessing a shape
    here is a silent FFN scale in the exported model.
    """
    if not dense_layers:
        return {}, []
    keys = set(source_keys)
    # THE REFERENCE LAYER IS THE FIRST MoE LAYER, DETERMINISTICALLY. The old
    # scan took whichever expert key a SET happened to yield first, so the
    # borrowed shapes could come from layer 2 or layer 3 depending on hash
    # order - harmless today (identical shapes) but a trap the moment the
    # stack becomes irregular, and it made the remap unrepeatable.
    moe_layers = {int(m.group(1)) for k in keys
                  if (m := re.match(r"model\.layers\.(\d+)\.mlp\.experts\.", k))}
    if not moe_layers:
        raise RuntimeError(
            "moe.dense_layers is set but no MoE expert tensors exist in the "
            "checkpoint - cannot build the remap reference shapes. Is "
            "mlp_only_layers covering every layer?")
    moe_layer = min(moe_layers)
    copies: dict = {}
    extras: list = []
    for li in sorted(dense_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            src = f"model.layers.{li}.mlp.{proj}.weight"
            if src not in keys:
                raise RuntimeError(f"remap source {src!r} missing")
            scale = down_scale if proj == "down_proj" else 1.0
            copies[src] = ([
                f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"
                for e in range(num_experts)], scale)
        refs = {
            "router": f"model.layers.{moe_layer}.mlp.gate.weight",
            "shexp_g": f"model.layers.{moe_layer}.mlp.shared_expert.gate_proj.weight",
            "shexp_u": f"model.layers.{moe_layer}.mlp.shared_expert.up_proj.weight",
            "shexp_d": f"model.layers.{moe_layer}.mlp.shared_expert.down_proj.weight",
            "shexp_t": f"model.layers.{moe_layer}.mlp.shared_expert_gate.weight",
        }
        for label, key in refs.items():
            if key not in keys:
                raise RuntimeError(
                    f"remap reference {label} {key!r} missing from the "
                    f"checkpoint - refusing to guess a shape")
        extras += [
            (f"model.layers.{li}.mlp.gate.weight", "zeros", refs["router"]),
            (f"model.layers.{li}.mlp.shared_expert.gate_proj.weight", "zeros", refs["shexp_g"]),
            (f"model.layers.{li}.mlp.shared_expert.up_proj.weight", "zeros", refs["shexp_u"]),
            (f"model.layers.{li}.mlp.shared_expert.down_proj.weight", "zeros", refs["shexp_d"]),
            (f"model.layers.{li}.mlp.shared_expert_gate.weight", "fill", refs["shexp_t"]),
        ]
    dsts = {d for cs, _ in copies.values() for d in cs} | {d for d, _, _ in extras}
    clash = dsts & (keys - set(copies))
    if clash:
        raise RuntimeError(
            f"remap destinations clash with live checkpoint keys: "
            f"{sorted(clash)}")
    return copies, extras


def _norm_w_from_source(src_text: str):
    """Extract the QWEN2MOE builder's norm_w literal from its C++ source.

    PURE - so the detection is testable. The remap's correctness depends on
    the runtime's top-k regime, and both llama.cpp layouts state it in the
    same place: the build_moe_ffn call, at the bool right after
    LLM_FFN_SILU. b10499 passes `false` there (Qwen2MoE gets NO top-k
    renormalisation); older checkouts pass `true`. Returns True/False, or
    None when the source does not say - a wrong guess is a silent 4x scale
    on the dense FFN in the exported model, so the caller refuses.
    """
    m = re.search(r"build_moe_ffn\(\s*(.{0,1200}?)\)\s*;", src_text, re.S)
    if not m:
        return None
    flag = re.search(r"LLM_FFN_SILU\s*,\s*(true|false)\b", m.group(1))
    if not flag:
        return None
    return flag.group(1) == "true"


def _detect_qwen2moe_norm_w(llama_cpp_dir: str) -> bool:
    """True if THIS llama.cpp checkout renormalises QWEN2MOE top-k weights.

    Looks in the per-model builder first (the layout since the models/
    split), then in the old single-graph layout's QWEN2MOE case. Refuses
    when the layout is unrecognised - the remap cannot be exact in both
    regimes, and guessing wrong exports a dense FFN scaled by 4x (or 1/4x).
    """
    model_file = os.path.join(llama_cpp_dir, "src", "models", "qwen2moe.cpp")
    if os.path.isfile(model_file):
        with open(model_file, encoding="utf-8", errors="replace") as fh:
            found = _norm_w_from_source(fh.read())
        if found is not None:
            return found
    graph_file = os.path.join(llama_cpp_dir, "src", "llama-graph.cpp")
    if os.path.isfile(graph_file):
        with open(graph_file, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        case = re.search(
            r"case LLM_ARCH_QWEN2MOE:\s*(.{0,4000}?)\n\s*case LLM_ARCH_",
            src, re.S)
        if case:
            found = _norm_w_from_source(case.group(1))
            if found is not None:
                return found
    raise RuntimeError(
        "cannot determine this llama.cpp checkout's QWEN2MoE top-k regime "
        f"from {llama_cpp_dir!r}: found neither src/models/qwen2moe.cpp nor "
        "a readable QWEN2MOE case in src/llama-graph.cpp. The dense-layer "
        "remap must know whether the runtime renormalises top-k weights. "
        "Update llama.cpp (or report the layout), or clear moe.dense_layers "
        "in the recipe.")


def _prepare_gguf_src(config, src_dir: str, cfg_dict: dict,
                      dense_layers, down_scale: float) -> str:
    """Write the remapped, converter-shaped checkpoint next to the original.

    Returns the work dir. Torch comes in via safetensors.torch, so this runs
    only on a build box - the same place the converter itself needs torch.
    """
    from safetensors.torch import safe_open, save_file
    import torch

    work = _gguf_src_workdir(src_dir)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            files = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        files = [os.path.join(src_dir, "model.safetensors")]

    keys = []
    for f in files:
        with safe_open(f, framework="pt") as sf:
            keys.extend(sf.keys())
    copies, extras = _remap_plan(dense_layers, cfg_dict["num_experts"], keys,
                                 down_scale)
    consumed = set(copies)

    out = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k in consumed:
                    dsts, scale = copies[k]
                    t = sf.get_tensor(k)
                    if scale != 1.0:
                        # Multiply in fp32: power-of-two scales stay exact,
                        # and a non-power-of-two cannot come from a
                        # num_experts/experts_per_tok ratio this code allows.
                        t = (t.float() * scale).to(t.dtype)
                    for dst in dsts:
                        out[dst] = t
                else:
                    out[k] = sf.get_tensor(k)
    for dst, kind, ref in extras:
        ref_t = out[ref]
        if kind == "zeros":
            out[dst] = torch.zeros(ref_t.shape, dtype=ref_t.dtype)
        else:
            out[dst] = torch.full(ref_t.shape, config.shared_expert_gate_fill,
                                  dtype=ref_t.dtype)

    save_file(out, os.path.join(work, "model.safetensors"))
    del out

    with open(os.path.join(work, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_gguf_src_config(cfg_dict), fh, indent=2)
    for name in os.listdir(src_dir):
        if name == "config.json" or name.endswith((".safetensors", ".bin")):
            continue
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(work, name))
    return work


def export_gguf(config, final_dir: str) -> Optional[str]:
    """Convert the trained MoE to GGUF and PROVE it generates.

    Returns the GGUF path if successful, None if skipped (no llama.cpp).
    """
    import subprocess

    gguf_path = gguf_path_for(config)
    smoke_log = gguf_path + ".smoketest.txt"
    if export_is_done(config):
        print(f"[skip] GGUF export already done at {gguf_path}")
        return gguf_path

    if os.path.exists(gguf_path):
        print(f"   {gguf_path} exists but has no smoke log — re-running smoke test")

    # Find llama.cpp converter
    conv = os.path.join(config.llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(conv):
        print(f"\n[gguf] SKIPPED — no converter at {conv}")
        print("[gguf] The model is fine for transformers but UNPROVEN outside")
        print("[gguf] Python, which is where this project's nastiest bugs live.")
        print("[gguf]   git clone --depth 1 https://github.com/ggml-org/llama.cpp")
        print("[gguf]   cd llama.cpp && cmake -B build -DGGML_CUDA=ON \\")
        print("[gguf]       && cmake --build build --config Release -j")
        return None

    # Convert if needed
    if not os.path.exists(gguf_path):
        print(f"\nExporting GGUF → {gguf_path}")

        # DENSE LAYERS HAVE NO NAME IN THE CONVERTER'S MAP. transformers
        # renders moe.dense_layers as plain Qwen2MLP blocks, and
        # convert_hf_to_gguf.py dies on them ("Can not map tensor
        # 'model.layers.0.mlp.down_proj.weight'") - the first gauntlet export
        # failed at exactly that line. The remap below rewrites each dense
        # layer as an all-MoE block whose experts are identical copies of the
        # dense FFN, with the down projection scaled to cancel the sum of the
        # top-k softmax weights - the scale is chosen from the CHECKOUT's own
        # norm_w regime (_detect_qwen2moe_norm_w), so the function is the
        # same on renormalising and non-renormalising runtimes alike.
        with open(os.path.join(final_dir, "config.json"),
                  encoding="utf-8") as fh:
            src_cfg = json.load(fh)
        dense_layers = src_cfg.get("mlp_only_layers") or []
        if dense_layers:
            refusal = _refuse_unremappable_dense(src_cfg)
            if refusal:
                raise RuntimeError(f"GGUF export refused: {refusal}")
        conv_src = final_dir
        if dense_layers:
            norm_w = _detect_qwen2moe_norm_w(config.llama_cpp_dir)
            n_experts = int(src_cfg["num_experts"])
            n_used = int(src_cfg.get("num_experts_per_tok", 1))
            down_scale = 1.0 if norm_w else (n_experts / n_used)
            conv_src = _prepare_gguf_src(config, final_dir, src_cfg,
                                         dense_layers, down_scale)
            print(f"   dense layers {dense_layers} remapped to "
                  f"identical-expert MoE blocks for the converter "
                  f"(checkout norm_w={norm_w} -> down_proj x{down_scale:g}; "
                  f"the same function either way) → {conv_src}")

        # BOTH OF THESE USED TO BE `config.base`, which is the HUGGINGFACE
        # MODEL ID. It was argv[0] - so the run tried to EXECUTE
        # "goblinModeMan/Qwen2.5-0.5B-Instruct-abliterated-v3" as a program -
        # and it was PYTHONPATH as well. The errno names the model id, which
        # reads like a missing model and is actually a missing interpreter.
        #
        # What the two slots actually want:
        #   argv[0]     an interpreter that can run convert_hf_to_gguf.py.
        #               Ours by default; the converter needs torch, and if we
        #               are running the build we have it.
        #   PYTHONPATH  llama.cpp's gguf-py, because convert_hf_to_gguf.py
        #               does `import gguf` and that package ships INSIDE the
        #               llama.cpp checkout rather than on PyPI. Without it the
        #               converter fails with ModuleNotFoundError: gguf, which
        #               is its own confusing errand.
        gguf_py = os.path.join(config.llama_cpp_dir, "gguf-py")
        env = {**os.environ}
        if os.path.isdir(gguf_py):
            env["PYTHONPATH"] = os.pathsep.join(
                [gguf_py] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        r = subprocess.run(
            [sys.executable, conv, conv_src,
             "--outfile", gguf_path, "--outtype", "f16"],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            tail = "\n".join((r.stderr or r.stdout).splitlines()[-25:])
            raise RuntimeError(f"GGUF conversion failed:\n{tail}")
        # The work dir served its purpose. It stays behind on failure so the
        # remapped tensors are inspectable; on success it is just a copy.
        if dense_layers and os.path.isdir(conv_src):
            shutil.rmtree(conv_src, ignore_errors=True)
        size_gb = os.path.getsize(gguf_path) / 1e9
        print(f"   converted OK ({size_gb:.2f} GB)")
    else:
        print(f"\n[skip] GGUF already converted "
              f"({os.path.getsize(gguf_path) / 1e9:.2f} GB)")

    # smoke.script REPLACES THE BUILT-IN SMOKE ENTIRELY. The script receives
    # the GGUF path as argv[1]; exit 0 = pass, anything else = fail with its
    # last output lines. llama-cli is neither required nor consulted on this
    # path - "replace" means replace.
    if getattr(config, "gguf_smoke_script", ""):
        return _run_custom_smoke(config, gguf_path, smoke_log)

    # Find llama-cli
    cli = resolve_llama_binary(config.llama_cpp_dir, "llama-cli")
    if not os.path.exists(cli):
        cli = shutil.which("llama-cli") or ""
    if not os.path.exists(cli):
        print("[gguf] converted, but llama-cli not found — cannot verify")
        print("[gguf]   cd llama.cpp && cmake -B build -DGGML_CUDA=ON \\")
        print("[gguf]       && cmake --build build --config Release -j")
        return gguf_path

    # Smoke test
    print("   smoke test: generating...")

    base_cmd = [cli, "-m", gguf_path, "-p", config.gguf_smoke_prompt,
                "-n", str(config.gguf_smoke_tokens), "-ngl", "99"]

    # stdbuf forces line buffering on the binary we don't control
    stdbuf_cmd = []
    if shutil.which("stdbuf"):
        stdbuf_cmd = ["stdbuf", "-oL", "-eL"]

    # Try single-turn flags first, fall back gracefully
    all_extra = [["-st"], ["--single-turn"], ["-no-cnv"], []]

    out = ""
    gen = ""
    proc_info = None
    for extra in all_extra:
        try:
            cmd = stdbuf_cmd + base_cmd + extra
            p = subprocess.run(
                cmd, capture_output=True, text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                timeout=config.gguf_smoke_timeout,
            )
            # STDOUT IS THE GENERATION. STDERR IS THE BANNER.
            # llama.cpp writes generated tokens to stdout and `build:`,
            # `llama_model_loader:`, `print_info:`,
            # `llama_perf_context_print:` to stderr. These were concatenated
            # BEFORE the "did it emit anything" check, so the load banner
            # alone satisfied it and the proof could not fail for any model
            # that loads - including one that emits zero tokens. `out` stays
            # combined for the log, because the banner is exactly the evidence
            # you want when something breaks. `gen` is what gets judged.
            gen = p.stdout or ""
            out = (p.stdout or "") + (p.stderr or "")
            low = out.lower()
            bad_flag = any(s in low for s in (
                "unrecognized", "invalid argument", "unknown argument",
                "error while handling argument",
            ))
            if p.returncode == 0 or not bad_flag:
                proc_info = {"returncode": p.returncode, "cmd": cmd}
                break
            print(f"   (llama-cli rejected {' '.join(extra) or '<no flag>'}; trying next)")
        except subprocess.TimeoutExpired as e:
            # KEEP BYTES — they contain the evidence of WHERE it stalled
            def _txt(v):
                if v is None:
                    return ""
                return v if isinstance(v, str) else v.decode("utf-8", "replace")

            partial = _txt(e.stdout) + _txt(e.stderr)
            hang_log = gguf_path + ".hang.txt"
            with open(hang_log, "w", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(stdbuf_cmd + base_cmd + extra)}\n"
                         f"TIMEOUT after {config.gguf_smoke_timeout}s\n\n{partial}")

            lines = [ln for ln in partial.splitlines() if ln.strip()]
            print(f"   *** llama-cli did not finish in {config.gguf_smoke_timeout}s. "
                  f"It emitted {len(lines)} lines before the clock ran out:")
            for ln in lines[-25:]:
                print("   | " + ln[:108])

            if not lines:
                print("   | (NOTHING AT ALL. Two causes: the load stalled, "
                      "OR the output is block-buffered and never flushed. "
                      "Run the same command by hand in a terminal to tell them apart.)")

            raise RuntimeError(
                f"llama-cli hung during GGUF smoke test. Partial output → {hang_log}")

    # Write smoke log (always, not just on failure)
    with open(smoke_log, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(stdbuf_cmd + base_cmd)}\n"
                 f"returncode={proc_info['returncode'] if proc_info else '?'}\n\n"
                 f"{out}")
    tail = [ln for ln in out.splitlines() if ln.strip()][-12:]
    print("   " + "-" * 24 + " smoke test output (tail) " + "-" * 24)
    for ln in tail:
        print("   | " + ln[:108])
    print("   " + "-" * 74)
    print(f"   full output → {smoke_log}")

    # THE EXIT CODE IS PART OF THE PROOF.
    # This function's own docstring has always listed "the process exits
    # cleanly" as check #1, and nothing checked it: returncode was captured
    # and only ever interpolated into a log line. GGML_ASSERT, CUDA OOM and an
    # unsupported quant all exit non-zero with a banner-rich stderr, which the
    # old combined-stream check accepted without complaint.
    if proc_info is None:
        raise RuntimeError(
            f"GGUF smoke test: llama-cli rejected every argument form tried "
            f"({len(all_extra)} of them). See {smoke_log}.")
    if proc_info["returncode"] != 0:
        raise RuntimeError(
            f"GGUF smoke test FAILED: llama-cli exited "
            f"{proc_info['returncode']} without generating. See {smoke_log}.")

    # Must produce at least one alphanumeric character ON STDOUT
    if not any(c.isalnum() for c in gen):
        raise RuntimeError(
            f"GGUF smoke test produced no generated output on stdout "
            f"(rc={proc_info['returncode']}). The model loaded and emitted "
            f"nothing - stderr had {len(out) - len(gen)} bytes of banner. "
            f"See {smoke_log}.")

    # Check for degenerate run (NaN signature)
    best, ch = _longest_char_run(gen)
    print(f"   longest repeated-character run: {best}" +
          (f" ({ch!r})" if best else ""))

    if best >= config.gguf_degenerate_run:
        raise RuntimeError(
            f"GGUF smoke test FAILED: {best} identical {ch!r} characters "
            f"in a row. That is the signature of NaN in the residual stream — "
            f"the model loads, runs at full speed, and emits one token forever. "
            f"Prime suspect: a tensor that is exactly zero where llama.cpp "
            f"divides by it (see SHARED_EXPERT_GATE_FILL).")

    print("   smoke test PASSED — the GGUF generates real tokens")

    # WRITTEN LAST, AND ONLY HERE. Every check above can raise; none of them
    # can leave this file behind. export_is_done() reads THIS, not the log.
    with open(gguf_path + ".smokepass.txt", "w", encoding="utf-8") as fh:
        fh.write(f"returncode={proc_info['returncode']}\n"
                 f"generated_stdout_chars={len(gen)}\n"
                 f"longest_repeated_run={best}\n"
                 f"log={smoke_log}\n")

    print(f"\n   try it:  {cli} -m {gguf_path} -ngl 99")
    return gguf_path


def _run_custom_smoke(config, gguf_path: str, smoke_log: str) -> Optional[str]:
    """smoke.script — the user's own proof that the GGUF generates.

    Replaces the llama-cli smoke entirely: the script gets the GGUF path as
    argv[1], and its exit code IS the verdict. Same smokepass marker as the
    built-in, so export_is_done() and resume behave identically either way.
    """
    import subprocess

    script = config.gguf_smoke_script
    print(f"   smoke.script: {script} {gguf_path}")
    try:
        r = subprocess.run([sys.executable, script, gguf_path],
                           capture_output=True, text=True, timeout=3600)
    except Exception as exc:
        raise RuntimeError(f"smoke.script {script!r} failed to run: {exc}")
    with open(smoke_log, "w", encoding="utf-8") as fh:
        fh.write((r.stdout or "") + "\n" + (r.stderr or ""))
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).splitlines()[-25:])
        raise RuntimeError(
            f"smoke.script {script!r} failed (exit {r.returncode}):\n{tail}")
    print("   smoke test PASSED — the user's own script proved it")
    with open(gguf_path + ".smokepass.txt", "w", encoding="utf-8") as fh:
        fh.write(f"custom script {script}\n")
    return gguf_path


def smoke_gguf(gguf_path: str,
               tokens: int = 48, timeout: int = 300,
               prompt: str = "Write a function that works.",
               degenerate_run: int = 32,
               llama_cpp_dir: str = "") -> bool:
    """Smoke-test a standalone GGUF file.  Returns True on pass.

    Proves the model generates OUTSIDE Python — the boundary where
    our nastiest bugs live (shared_expert=0, tie_word_embeddings,
    all-zero gates → NaN forever).

    `llama_cpp_dir` was not a parameter, so this looked on PATH alone while
    export_gguf looked in the build tree first. A perfectly normal llama.cpp
    checkout was therefore usable by the build and not by `ms-moe-maker
    smoke` - the command that exists to run it.
    """
    import subprocess

    cli = resolve_llama_binary(llama_cpp_dir, "llama-cli")
    if not cli:
        looked = (f" (looked in {llama_cpp_dir}/build/bin, "
                  f"{llama_cpp_dir}/bin and PATH)") if llama_cpp_dir else ""
        print(f"[smoke] llama-cli not found{looked}")
        print("[smoke] Point at it with runtime.llama_cpp in the recipe, or")
        print("[smoke] MSMOE_LLAMA_CPP, or put it on PATH.")
        print("[smoke]   git clone --depth 1 https://github.com/ggml-org/llama.cpp")
        print("[smoke]   cd llama.cpp && cmake -B build -DGGML_CUDA=ON")
        print("[smoke]       && cmake --build build --config Release -j")
        return False

    print("   smoke test: generating...")

    base_cmd = [cli, "-m", gguf_path, "-p", prompt,
                "-n", str(tokens), "-ngl", "99"]

    stdbuf_cmd = []
    if shutil.which("stdbuf"):
        stdbuf_cmd = ["stdbuf", "-oL", "-eL"]

    all_extra = [["-st"], ["--single-turn"], ["-no-cnv"], []]

    out = ""
    gen = ""
    rc = None
    for extra in all_extra:
        try:
            cmd = stdbuf_cmd + base_cmd + extra
            p = subprocess.run(
                cmd, capture_output=True, text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                timeout=timeout,
            )
            # See export_gguf: stdout is the generation, stderr is the
            # banner, and judging the concatenation makes this unfailable.
            gen = p.stdout or ""
            out = (p.stdout or "") + (p.stderr or "")
            low = out.lower()
            bad_flag = any(s in low for s in (
                "unrecognized", "invalid argument", "unknown argument",
                "error while handling argument",
            ))
            if p.returncode == 0 or not bad_flag:
                rc = p.returncode
                break
            print(f"   (llama-cli rejected {extra or '<no flag>'}; trying next)")
        except subprocess.TimeoutExpired as e:
            def _txt(v):
                return v if isinstance(v, str) else v.decode("utf-8", "replace") if v else ""
            partial = _txt(e.stdout) + _txt(e.stderr)
            hang_log = gguf_path + ".hang.txt"
            with open(hang_log, "w", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(stdbuf_cmd + base_cmd + extra)}\n"
                         f"TIMEOUT after {timeout}s\n\n{partial}")
            print(f"   *** llama-cli did not finish in {timeout}s. "
                  f"Partial output → {hang_log}")
            raise

    tail = [ln for ln in out.splitlines() if ln.strip()][-12:]
    print("   " + "-" * 24 + " smoke test output (tail) " + "-" * 24)
    for ln in tail:
        print("   | " + ln[:108])
    print("   " + "-" * 74)

    if rc is None:
        print(f"   ✗ llama-cli rejected every argument form tried "
              f"({len(all_extra)} of them) — smoke test FAILED")
        return False
    if rc != 0:
        print(f"   ✗ llama-cli exited {rc} — smoke test FAILED")
        return False

    # Must produce at least one alphanumeric character ON STDOUT
    if not any(c.isalnum() for c in gen):
        print("   ✗ No generated output on stdout (the model loaded and "
              "emitted nothing) — smoke test FAILED")
        return False

    # Check for degenerate run (NaN signature)
    best, ch = _longest_char_run(gen)
    print(f"   longest repeated-character run: {best}"
          + (f" ({ch!r})" if best else ""))

    if best >= degenerate_run:
        print(f"   ✗ Degenerate run: {best} identical {ch!r} characters "
              f"in a row — NaN signature. Smoke test FAILED.")
        return False

    print("   smoke test PASSED — the GGUF generates real tokens")
    return True
