# Troubleshooting Signatures

Validated against commit/tag: main (unreleased)

Canonical error signatures and their remediation. The
[Troubleshooting FAQ](Troubleshooting-FAQ) is the operator decision index; this
is the exact-match reference.

## `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

The trainer's loss was computed under `no_grad` — something left
`torch.set_grad_enabled(False)` globally on before the finetune stage. The
`abliterate.base` stage runs in a subprocess specifically so this cannot happen;
if you see it, another in-process step is mutating grad mode.

## `ERROR: torch is not installed` (preflight)

The `[train]` extras are missing. `pip install "ms-moe-maker[train]"`.

## `convert_hf_to_gguf.py not found`

llama.cpp isn't on the search path. Set `runtime.llama_cpp` (or
`MSMOE_LLAMA_CPP`). Export is skipped with a warning; you still get the HF
checkpoint.

## `ERROR: {expert}: only N samples from {repo} (min M)`

The corpus floor beat the source. `min_samples` rises to meet `router_mix_total`;
lower `router_mix_total`, use a richer dataset, or raise `max_samples`.
`--plan` prints the raised floor before the run starts.

## `RepositoryNotFoundError` / 401 / 404 on a model or dataset

A deleted or gated model. A deleted repo and an invalid token raise the same
exception — run `hf auth whoami` first, then check the id. Mirror what you
depend on (`mirror_bases.py`) so a deleted upstream can't end a build.

## Gated model or private dataset 401/404

No Hugging Face token. Put `HF_TOKEN=...` in a `.env` next to the recipe
(auto-loaded), or `huggingface-cli login`. The shell always wins over `.env`.

## Resume refused: build mismatch

The run directory was built by a different resolved config. Continue the exact
lineage with the original `--defaults`, or fork it with `--force`.
