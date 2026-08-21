# Open questions

Things this project knows it has not answered, written down at the moment they
were noticed rather than reconstructed later. Each one names what would settle
it, so picking one up does not start with re-deriving why it is open.

---

## The tool-calling expert, and the teacher-model path

**Status: designed, never run end to end inside MsMoEMaker.**

`corpus.Kind("synth")` exists, `generate_agent_traces` exists, `stages.DATA_SYNTH`
is in the public vocabulary. None of it has executed in this package — the
machinery was carved out of `fraunkenstein_universal.py` and has only ever run
there.

Why it matters: a tool-calling expert is the largest domain contrast available
(chat-formatted JSON against raw source), and it is the first rung of the
`propose_tool` path. It is also the only expert whose corpus is *generated*
rather than scraped, which exercises a whole branch of the pipeline.

What is unresolved:

- **Which teacher, and on what.** `_VLLMTeacher` wants vLLM; `_HFTeacher` wants
  transformers + bitsandbytes. The Lab's notes say vLLM and unsloth pin
  incompatible transformers versions and should not share a venv, so the
  generation stage may have to be a separate environment joined by files on
  disk. That is already how `_done()` is designed to work, but MsMoEMaker's
  in-process builder has never been asked to straddle two environments.
- **vLLM on aarch64 Blackwell.** Unverified. The Lab comment says to check the
  wheel installs before betting an evening on it (sbsa build).
- **Teacher size.** The Lab found a 0.5B teacher too weak to emit schema-valid
  tool calls and settled on 7B as the smallest that clears the rejection
  sampler. On a 0.5B rung that means the TEACHER is bigger than the model being
  built, which is fine but worth stating.
- **How the corpus is scored.** Tool-call traces are templated by construction,
  so `corpushealth` reframes line reuse rather than warning on it
  (`Kind.generated`). But cross-domain loss on chat-formatted rows against raw
  code rows is comparing two different distributions, and it is not obvious the
  gap means the same thing.
- **`agent_mix_fraction`.** Now a recipe knob (was hardcoded 0.15). With three
  experts and one synth expert, 0.15 shows the router the tool-calling expert
  a third as often as the others. Probably wants to be ~1/n for a build whose
  point is to prove that expert can be routed to.

Settles it: one 0.5B build with `[python, csharp, agentcore]`, teacher on the
Spark, `--mode all`. If the cross-loss gap for agentcore is materially larger
than the ~0.05 nats every scraped-text pair produces, domain contrast is not
the ceiling after all and the whole picture changes.

---

## Why the router will not discriminate

**Status: located, not solved.**

Reproduced across three builds: 3/3 own-column, p=0.037, JS ~0.005, enrichment
1.06–1.08x, balanced and unsaturated. Real, significant, and small.

Ruled out by measurement:

- corpus monoculture (fixed; the gap barely moved)
- domain contrast (markdown vs code is the same ~0.05 nats as python vs csharp)
- expert strength (4x tokens: experts markedly better and MORE independent,
  routing unchanged)
- collapse and symmetry (top-2 normalised + random init + aux 0.02 hold it)

Not yet tested: **the router's own budget.** It got 150 optimiser steps in
every run while the specialists went from 150 to 600. Mix size, aux
coefficient and router LR have never been swept.

Settles it: router-only rebuilds, one knob at a time — `router_mix_total`
first, then `aux_loss_coef`, then `lr`. ~15 minutes each.

---

## MCP

The plug-and-play tool registry (`propose_tool`, `tools/proposed/`, the HITL
approve/reject gate) is designed and unbuilt. It is downstream of the
tool-calling expert in the sense that matters: an MoE that cannot emit a tool
call has nothing to propose with.
