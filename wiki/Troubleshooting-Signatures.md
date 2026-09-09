# Troubleshooting Signatures

Validated against commit/tag: main (unreleased)

Canonical error signatures and their remediation. The
[Troubleshooting FAQ](Troubleshooting-FAQ) is the operator decision index; this
is the exact-match reference.

## `almost nothing emitted a think block across the N row(s) asked to, and this run expected ...`

**The only signature on this page that is a wrong ANSWER rather than a
failure.** Eval prints it when a reasoning run finished but almost nothing it
generated carried a thinking delimiter.

**If you saw this before the row count was in the message, check whether it was
ever about you.** The average behind it used to run over EVERY expert with a
`reasoned` number, and every non-reasoning expert had one — `0.00` — because the
run's tag style was handed to all of them. On a nine-expert run with one
reasoning expert that is `0.38` averaged with seventeen zeros, so the warning
fired by construction on every realistic run, three lines under a row whose
`0.38` proved the tags parse fine. An alarm that always fires is an alarm that
never fires. Only rows that were ASKED to reason count now, `reasoned` prints
`-` rather than `0.00` for the rest, and they are no longer stamped "does not
reliably reason" for declining to do something nobody asked of them. If your run
predates that, the tags were quite possibly never the problem.

When it does fire, there are two causes and from the outside they look
identical:

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

## `NOTE: 66% of generations hit the token cap and were discarded.`

Not an error, and not cosmetic either — this is the loudest thing the synth
stage says, and it is easy to watch scroll past. Printed by any generator loop
when more than a quarter of a batch ran out of budget.

**A generation cut at the cap is discarded, not trimmed.** That is the part
worth sitting with: a low ceiling does not shorten the corpus, it keeps whichever
material happened to fit and deletes the rest. A real gauntlet at `512` tokens
dropped 490 of 740 domain generations, so the lore corpus became the third of the
narrative short enough to fit, the specialist learned atypically short lore, and
the eval reported `NO ROUTER SIGNAL: lore scored better under a foreign expert
than its own ... the fix is upstream`. It was upstream. It was here.

Remediation, in order:

1. **Raise the budget belonging to the loop that is actually being cut.** The
   NOTE names it; there are four and they are not interchangeable:
   `agent_teacher_max_new` (MCP tool calls), `domain_teacher_max_new` (plain
   prose), `reasoning_teacher_max_new` (think block + answer), and
   `teacher_max_new` as the fallback for any loop without its own. Raising the
   fallback when one loop is starving moves only the loops that have not been
   given a ceiling.

2. **`0` means unbounded** — as far as the engine window allows. On the vLLM path
   that window is `runtime.vllm_max_len` *minus the prompt*, and it defaults to
   `4096`, so asking for unbounded without raising it can hand you LESS room than
   naming a number. Preflight warns about exactly that case.

3. **If the budget IS the constraint** — which is the whole point of the Nano
   floor — stop discarding instead of raising. `budget.teacher_overrun:
   answer-only` finishes a generation whose answer was cut;
   `close-and-answer` also closes a thought still in progress and buys its
   answer with `budget.teacher_answer_reserve`. Yield goes to ~100% at any
   budget, the happy path costs nothing, and only the overruns pay for a second
   batched call. See the [Recipe Options
   Reference](Recipe-Options-Reference) for what each policy promises and where
   it degrades.

4. `budget.teacher_tell_budget: true` puts the available room into the teacher's
   prompt so it aims to fit. Cheap, and it bounds nothing — no model obeys a
   length instruction exactly.

The progress line is the feedback loop: `cut short` should fall and `salvaged`
should rise. If `cut short` stays high under a salvage policy, those generations
overran, were given a reserve, and ran out of that too — raise the reserve.

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
