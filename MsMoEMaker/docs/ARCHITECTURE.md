# Architecture

## High-level shape

`ms-moe-maker` is a contract-first CLI pipeline. The user-facing contract is:

1. command vocabulary (`describe`, `build`, `validate`, etc.)
2. stage vocabulary (`preflight`, `data.corpus`, ...)
3. JSON event vocabulary under `--json`
4. run manifest schema compatibility

## Stage model

Stage IDs are defined in `ms_moe_maker/stages.py` and are public contract values.

Fixed stages:

- `preflight`
- `abliterate.base` (optional)
- `data.corpus`
- `data.synth` (when generated experts exist)
- `gate.experts` (when enabled)
- `stitch`
- `router`
- `export.gguf`

Per-expert stages:

- `finetune.<expert_name>`

Order is meaningful and used by downstream readers.

## Sequence diagram: build stage flow

```mermaid
flowchart TD
    A[build recipe.yaml] --> B[resolve recipe + defaults layers]
    B --> C[plan stages]
    C --> D[preflight]

    D --> E{abliteration enabled?}
    E -- yes --> F[abliterate.base]
    E -- no --> G[data.corpus]
    F --> G

    G --> H{generated experts present?}
    H -- yes --> I[data.synth]
    H -- no --> J[finetune.<expert_1..n>]
    I --> J

    J --> K{experts gate enabled?}
    K -- yes --> L[gate.experts]
    K -- no --> M[stitch]
    L --> M

    M --> N[router]
    N --> O[export.gguf]
    O --> P[done]
```

## Sequence diagram: JSON event flow (`--json`)

```mermaid
sequenceDiagram
    participant U as User/Caller
    participant CLI as ms-moe-maker CLI
    participant R as Runner
    participant P as Pipeline Stages
    participant M as Run Manifest

    U->>CLI: ms-moe-maker build recipe.yaml --json
    CLI->>R: resolve config + build translation
    R-->>CLI: started

    loop for each planned stage
        R->>P: run stage
        P-->>R: stage updates/progress
        R-->>CLI: stage/progress/warning/error events
    end

    R->>M: write/update manifest state
    R-->>CLI: defaults (provenance)
    R-->>CLI: done
    CLI-->>U: JSON Lines on stdout
```

## CLI and compatibility

- Canonical command/mode/event lists live in `ms_moe_maker/_describe.py`.
- `--describe` is designed to answer with minimal side effects.
- Additive change policy: adding commands/modes/events can be compatible; renaming/removing is breaking for downstream consumers.

## Defaults model

Defaults are layered data (floor + packaged + user + explicit + recipe wins).
This supports reproducibility while allowing machine-specific tuning.

## Integration boundaries

- `README.md`: fast path for users
- `docs/CLI.md`: command semantics
- `docs/TROUBLESHOOTING.md`: failure signatures and fixes
- `docs/SOURCE_OF_TRUTH.md`: anti-drift ownership map
