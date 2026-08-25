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
