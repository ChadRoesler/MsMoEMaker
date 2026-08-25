# Wiki Bootstrap Plan

Use this to initialize the GitHub Wiki without duplicating canonical repo docs.

## Ready-to-publish starter pack

A standard starter set now lives in:

- `MsMoEMaker/wiki-starter/`

Included pages:

- `Home.md`
- `How-To-First-Dryrun.md`
- `How-To-First-Full-Build.md`
- `Recipe-Deep-Dive-Defaults-and-Reproducibility.md`
- `Recipe-Deep-Dive-Corpus-Strategy.md`
- `Tuning-Playbook-Router.md`
- `Troubleshooting-FAQ.md`
- `Glossary.md`
- `_Sidebar.md`

## Publish flow

Automatic publish is wired through `.github/workflows/wiki-publish.yml`.

1. Enable Wiki in repository settings (one-time).
2. Ensure the wiki exists (open Wiki tab once if needed).
3. Push changes under `MsMoEMaker/wiki-starter/**`.
4. Workflow syncs starter files into `${repo}.wiki.git`.
5. Fill `Validated against commit/tag` on each page after major updates.

## Required page banner

Each wiki page should include:

- `Validated against commit/tag: <sha-or-tag>`
- Link to canonical repo page for any contract claim

## Anti-drift rules

- Repo docs are canonical for behavior and command semantics.
- Wiki can explain and exemplify; it should not redefine command contracts.
- When command semantics change, update repo docs first, then wiki.
