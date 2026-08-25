# MsMoEMaker Wiki

Validated against commit/tag: main (unreleased)

This wiki is the operational handbook layer for MsMoEMaker.
Use it for deep how-to guidance, tuning playbooks, and runbook practices.

For contract-sensitive behavior, defer to the canonical reference pages:

- CLI contract: [CLI Reference](CLI-Reference)
- Architecture/contracts + ownership: [Architecture](Architecture)
- Failure signatures: [Troubleshooting Signatures](Troubleshooting-Signatures)

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

### Reference

- [CLI Reference](CLI-Reference)
- [Architecture](Architecture)
- [Troubleshooting Signatures](Troubleshooting-Signatures)

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

- The reference pages define behavior; the rest of the wiki explains practice.
- Prefer runbooks/checklists over prose when documenting operations.
- Update this wiki in the same PR that changes behavior.
- Keep `Validated against commit/tag` current during release cycles.

## Suggested maintenance cadence

- Every release: validate all runbooks against tagged behavior.
- Monthly: refresh troubleshooting with recent failure signatures.
- Quarterly: prune stale guidance and simplify duplicated sections.
