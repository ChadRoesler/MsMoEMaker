# Glossary

Validated against commit/tag: main (unreleased)

## Core config terms

- **Recipe**: User-authored build configuration document (`.yaml`/`.json`).
- **Defaults layers**: Non-recipe config sources merged under recipe.
- **Explicit defaults**: A defaults file passed with `--defaults`.
- **Runtime roots**: Data/output locations where corpora and artifacts are
  written.

## Identity and reproducibility terms

- **`recipe_id`**: Identifier for recipe content identity.
- **`build_id`**: Identifier for resolved build configuration identity.
- **Build lineage**: The continuity of runs that share compatible resolved
  settings and artifacts.
- **Resume refusal**: Safety refusal when existing run artifacts do not match
  current resolved build context.

## Pipeline and execution terms

- **Stage**: Contracted unit of pipeline work (for example `data.corpus`).
- **Stage vocabulary**: Stable stage IDs consumed by downstream tooling.
- **`abliterate.base`**: Optional stage that decensors the base model (vendored
  Heretic core) before any specialist trains from it. Runs in its own process.
- **Event stream**: JSON Lines events emitted under `--json`.
- **Manifest**: Run-state metadata used for tracking and compatibility checks.

## Data and model terms

- **Expert**: Specialist component trained on domain-focused corpus inputs.
- **Generated expert**: Expert corpus produced synthetically rather than
  fetched/scraped.
- **Router mix**: Curated dataset used to train router discrimination.
- **Per-repo cap**: Corpus diversity control limiting contribution from a
  single repository.

## Reasoning terms

- **`base_kind`**: Whether the *base model* already reasons (`auto` |
  `reasoning` | `nonreasoning`). `auto` sniffs the model id against the known
  families. It changes how prompts are formatted and how eval reads output; it
  does **not** make a non-reasoning base reason.
- **`reasoning: true`** (on a source): Bake reasoning into a specialist that
  lacks it - a reasoning teacher writes trace-plus-answer pairs on that
  expert's domain and the specialist is fine-tuned on them. Works on any base.
- **Reasoning teacher**: The model that writes those traces. Defaults to
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (the `-1.5B` on a dryrun);
  override with `teacher:` on the source.
- **`templates:`** (on a source): A YAML `Prompts:` list of questions the teacher
  answers for generated experts (reasoning + plain synth). `{domain}` is swapped
  for the expert's display name. A bare `code`/`dnd`/`math`/`culinary`/`generic`
  resolves to the packaged file; empty means `generic_templates.yaml`.
- **Tag style**: The convention separating a thinking trace from an answer -
  opening tag, closing tag, and whether blocks interleave with tool calls.
  Shipped styles: standard XML, DeepSeek R1, interleaved agentic XML, markdown
  fence, Llama system-header.
- **Reasoning table**: The layered `reasoning.yaml` mapping model families to
  tag styles. A file rather than a dict on purpose: a wrong tag style is a
  silent wrong answer, so a new family must be addable without a release.
- **Interwoven**: A tag style whose reasoning blocks appear many times per turn
  (reasoning interleaved with tool calls). The splitter strips them all.

## Teacher budget terms

- **Teacher budget**: How many tokens a teacher may write for one generated
  example. There are three loops with opposite appetites and one fallback:
  `agent_teacher_max_new` (an MCP tool call, small), `domain_teacher_max_new`
  (a page of prose), `reasoning_teacher_max_new` (a think block *plus* an
  answer), and `teacher_max_new` for any loop not given its own.
- **Unbounded** (`0`): "As far as the engine window allows" - the spelling every
  generation budget accepts, including `eval.max_new_tokens`. On the vLLM path
  the window is `runtime.vllm_max_len` minus the prompt, so unbounded is only as
  generous as that ceiling.
- **Overrun**: A generation that hit its budget instead of finishing. It is
  **discarded, not trimmed**, which is why a cap set too low does not shorten a
  corpus - it keeps whichever material happened to fit.
- **Overrun policy** (`budget.teacher_overrun`): What to do about one.
  `discard` throws it away (the default, and what every earlier release did);
  `answer-only` finishes a generation that already ended its reasoning and had
  its answer cut; `close-and-answer` also closes a thought still in progress and
  buys its answer. Published in `describe` as `overrun_policies`.
- **Answer reserve** (`budget.teacher_answer_reserve`): Tokens held back to buy
  that ending. `-1` = a quarter of *that loop's* budget, floored at 64, never
  more than half - a reserve that eats the budget leaves nothing to think with.
- **Salvaged**: A row that overran and was finished with the reserve rather than
  thrown away. Counted separately from `kept` and `cut short` in the progress
  line, because the three are different facts about the same batch.
- **Budget hint** (`budget.teacher_tell_budget`): Puts the room available into
  the teacher's own prompt, in words. Lowers the overrun rate; bounds nothing.
  Teacher-side only - it never reaches a corpus row, or the specialist would
  learn "be brief" as part of its identity.

## Validation and quality terms

- **Dryrun**: Real smallest-rung build path for quick structural validation.
- **Smoke test**: Post-build generation sanity check.
- **Routing eval**: Measurement of whether routing prefers appropriate experts.
- **Quality eval**: Measurement of output quality on evaluation prompts.
- **`reasoned`**: Share of an expert's outputs that opened AND closed a think
  block - the DISCIPLINE check, read against routing enrichment. High
  enrichment beside low `reasoned` is a named failure: the register of
  deliberation without the structure. Prints `-`, not `0.00`, for an expert that
  was never asked to reason.
- **Capped generation**: An eval sample that ran its whole `max_new_tokens`
  budget, so it was cut off rather than finished. Every score in such a row is
  computed on an amputated answer and `reasoned` becomes a lower bound rather
  than a measurement, which is why the report says so in its caveats.
