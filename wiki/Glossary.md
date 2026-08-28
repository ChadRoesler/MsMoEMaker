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

## Validation and quality terms

- **Dryrun**: Real smallest-rung build path for quick structural validation.
- **Smoke test**: Post-build generation sanity check.
- **Routing eval**: Measurement of whether routing prefers appropriate experts.
- **Quality eval**: Measurement of output quality on evaluation prompts.
