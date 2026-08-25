# MsMoEMaker Wiki

Validated against commit/tag: `<fill-me>`

This wiki is the operational handbook layer for MsMoEMaker.
Use it for deep how-to guidance, tuning playbooks, and runbook practices.

For contract-sensitive behavior, defer to canonical repo docs:

- CLI contract: `MsMoEMaker/docs/CLI.md`
- Architecture/contracts: `MsMoEMaker/docs/ARCHITECTURE.md`
- Troubleshooting canon: `MsMoEMaker/docs/TROUBLESHOOTING.md`
- Source-of-truth policy: `MsMoEMaker/docs/SOURCE_OF_TRUTH.md`

## How to use this wiki

If this is your first run:

1. Start with [How-To: First Dryrun](How-To-First-Dryrun)
2. Continue to [How-To: First Full Build](How-To-First-Full-Build)
3. Use [Troubleshooting FAQ](Troubleshooting-FAQ) when blocked

If you are tuning or operating repeated builds:

1. Read [Recipe Options Reference](Recipe-Options-Reference)
2. Read [Recipe Deep Dive: Defaults + Reproducibility](Recipe-Deep-Dive-Defaults-and-Reproducibility)
3. Read [Recipe Deep Dive: Corpus Strategy](Recipe-Deep-Dive-Corpus-Strategy)
4. Work through [Tuning Playbook: Router](Tuning-Playbook-Router)

If you need shared terminology:

- See [Glossary](Glossary)

## Handbook map

### Onboarding runbooks

- [How-To: First Dryrun](How-To-First-Dryrun)
- [How-To: First Full Build](How-To-First-Full-Build)

### Deep dives

- [Recipe Options Reference](Recipe-Options-Reference)
- [Recipe Deep Dive: Defaults + Reproducibility](Recipe-Deep-Dive-Defaults-and-Reproducibility)
- [Recipe Deep Dive: Corpus Strategy](Recipe-Deep-Dive-Corpus-Strategy)

### Tuning playbooks

- [Tuning Playbook: Router](Tuning-Playbook-Router)

### Operations support

- [Troubleshooting FAQ](Troubleshooting-FAQ)
- [Glossary](Glossary)

## Operating model

- Repo docs define behavior; wiki explains implementation practice.
- Prefer runbooks/checklists over prose when documenting operations.
- Update this wiki after behavior changes are documented in repo docs.
- Keep `Validated against commit/tag` current during release cycles.

## Suggested maintenance cadence

- Every release: validate all runbooks against tagged behavior.
- Monthly: refresh troubleshooting with recent failure signatures.
- Quarterly: prune stale guidance and simplify duplicated sections.
