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

1. Open repo Wiki and create pages matching starter filenames.
2. Paste page contents from `wiki-starter` files.
3. Set `Home` as landing page and `_Sidebar` for navigation.
4. Fill `Validated against commit/tag` on each page.

## Required page banner

Each wiki page should include:

- `Validated against commit/tag: <sha-or-tag>`
- Link to canonical repo page for any contract claim

## Anti-drift rules

- Repo docs are canonical for behavior and command semantics.
- Wiki can explain and exemplify; it should not redefine command contracts.
- When command semantics change, update repo docs first, then wiki.
