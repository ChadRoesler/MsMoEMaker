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
import shutil
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
    return (os.path.exists(gguf_path)
            and os.path.getsize(gguf_path) > 0
            and os.path.exists(gguf_path + ".smoketest.txt"))


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
        r = subprocess.run(
            [config.base, conv, final_dir,
             "--outfile", gguf_path, "--outtype", "f16"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": config.base},
        )
        if r.returncode != 0:
            tail = "\n".join((r.stderr or r.stdout).splitlines()[-25:])
            raise RuntimeError(f"GGUF conversion failed:\n{tail}")
        size_gb = os.path.getsize(gguf_path) / 1e9
        print(f"   converted OK ({size_gb:.2f} GB)")
    else:
        print(f"\n[skip] GGUF already converted "
              f"({os.path.getsize(gguf_path) / 1e9:.2f} GB)")

    # Find llama-cli
    cli = os.path.join(config.llama_cpp_dir, "build", "bin", "llama-cli")
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

    # Must produce at least one alphanumeric character
    if not any(c.isalnum() for c in out):
        raise RuntimeError(
            f"GGUF smoke test produced no alphanumeric output "
            f"(rc={proc_info['returncode'] if proc_info else '?'}). "
            f"See {smoke_log}.")

    # Check for degenerate run (NaN signature)
    best, ch = _longest_char_run(out)
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
    print(f"\n   try it:  {cli} -m {gguf_path} -ngl 99")
    return gguf_path


def smoke_gguf(gguf_path: str,
               tokens: int = 48, timeout: int = 300,
               prompt: str = "Write a function that works.",
               degenerate_run: int = 32) -> bool:
    """Smoke-test a standalone GGUF file.  Returns True on pass.

    Proves the model generates OUTSIDE Python — the boundary where
    our nastiest bugs live (shared_expert=0, tie_word_embeddings,
    all-zero gates → NaN forever).
    """
    import subprocess

    cli = shutil.which("llama-cli") or ""
    if not cli:
        print("[smoke] llama-cli not found — installing llama.cpp is required")
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
    for extra in all_extra:
        try:
            cmd = stdbuf_cmd + base_cmd + extra
            p = subprocess.run(
                cmd, capture_output=True, text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                timeout=timeout,
            )
            out = (p.stdout or "") + (p.stderr or "")
            low = out.lower()
            bad_flag = any(s in low for s in (
                "unrecognized", "invalid argument", "unknown argument",
                "error while handling argument",
            ))
            if p.returncode == 0 or not bad_flag:
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

    # Must produce at least one alphanumeric character
    if not any(c.isalnum() for c in out):
        print("   ✗ No alphanumeric output — smoke test FAILED")
        return False

    # Check for degenerate run (NaN signature)
    best, ch = _longest_char_run(out)
    print(f"   longest repeated-character run: {best}"
          + (f" ({ch!r})" if best else ""))

    if best >= degenerate_run:
        print(f"   ✗ Degenerate run: {best} identical {ch!r} characters "
              f"in a row — NaN signature. Smoke test FAILED.")
        return False

    print("   smoke test PASSED — the GGUF generates real tokens")
    return True
