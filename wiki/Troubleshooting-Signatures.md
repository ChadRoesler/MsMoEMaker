# Troubleshooting Signatures

Validated against commit/tag: main (unreleased)

Canonical error signatures and their remediation. The
[Troubleshooting FAQ](Troubleshooting-FAQ) is the operator decision index; this
is the exact-match reference.

## `almost nothing emitted a think block, and this run expected ...`

**The only signature on this page that is a wrong ANSWER rather than a
failure.** Eval prints it when a reasoning run finished but almost nothing it
generated carried a thinking delimiter.

There are two causes and from the outside they look identical:

- the base does not actually reason, or
- the base reasons fine and the **tag style is wrong**.

If it is the second, the splitter found no delimiters, reported "did not
reason", and scored the whole think block as if it were the answer - so every
quality number in that run silently includes the trace. The build succeeds, the
GGUF is fine, the measurements are garbage. This is exactly the thing the
[Architecture](Architecture) invariants call worse than a loud failure, which is
why eval says "the two look identical from here" instead of picking one.

Remediation, in order:

1. Ask the box which table it actually merged:

   ```bash
   ms-moe-maker describe      # .reasoning holds the merged styles + families
   ```

2. If your family is missing or points at the wrong style, add it - you do not
   need a release for this. Drop a file at `~/.msmoe/reasoning.yaml`, or point
   `$MSMOE_REASONING` at one:

   ```yaml
   Families:
     - Key: acme
       FamilyName: Acme Thinkers
       Models: [Acme-R2, acme-thinker]    # write what is on the model card
       PreferredStyle: xml
   ```

   Layers merge **by name**, so adding one family never costs you the shipped
   ones. Model names match loosely - case, spaces, dots and hyphens are ignored
   on both sides - and the **longest** match wins, so the answer never depends
   on the order of the file.

3. If the base genuinely does not reason and you want it to, that is not a tag
   problem: set `reasoning: true` on the sources you want traces baked into.
   If it does not reason and should not, set `base_kind: nonreasoning`.

Correcting the table changes the resolved config, and the resolved delimiters
are stamped into it - so this changes `build_id`. The affected run needs a
rebuild, not a re-score.

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
