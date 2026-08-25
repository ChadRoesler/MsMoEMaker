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

## Validation and quality terms

- **Dryrun**: Real smallest-rung build path for quick structural validation.
- **Smoke test**: Post-build generation sanity check.
- **Routing eval**: Measurement of whether routing prefers appropriate experts.
- **Quality eval**: Measurement of output quality on evaluation prompts.
