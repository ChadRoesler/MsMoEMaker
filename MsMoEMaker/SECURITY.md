# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub Security Advisories or direct maintainer contact in the repository profile.

Do not file exploitable details in public issues.

## What to include

- Affected version/commit
- Reproduction steps
- Impact assessment
- Suggested remediation (if known)

## Response expectations

- Initial acknowledgment target: within 7 days
- Remediation target depends on severity and reproducibility

## Scope notes

This project is a CLI pipeline and does not expose a default network service surface. Most security risk comes from dependency, file-path, and execution environment handling; report anything that can cause unsafe code execution, credential exposure, or artifact tampering.

### Models you name run code

The base model, the abliteration target and any Hub-hosted synth teacher are loaded with `trust_remote_code=True`. That is what lets a recipe point at model families whose architecture ships as Python inside the repository, and it means the repository's code runs in the build process with the build's permissions. A recipe is a statement of trust in every model id it names: point it only at repositories you would run a script from. Reports about a model repository behaving maliciously belong upstream with the Hub; reports about this tool loading something it was not told to belong here.
