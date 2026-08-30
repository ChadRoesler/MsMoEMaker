"""The corpus registry, and the domain-neutrality it exists to protect.

The point of these tests is not that a dict can hold entries. It is that
nothing in the build assumes the training text is CODE - because the day that
assumption ships in a public contract is the day every person building a
Ms.MoE for lore, or exam material, or case law is told they are using someone
else's tool.
"""
from __future__ import annotations

import pytest

from ms_moe_maker.data import corpus
from ms_moe_maker.config import recipe as rp
from ms_moe_maker.run import stages as st


# ── the registry ────────────────────────────────────────────────────────────

def test_the_builtin_kinds_are_registered():
    assert set(corpus.names()) >= {"hf", "stack", "synth", "local"}


def test_only_one_builtin_kind_is_code_specific():
    """The generality argument, asserted rather than claimed.

    `hf` is a repo plus a text field, `local` is a path, `synth` is a teacher.
    None of those three know what the text means. Only `stack` - which scans a
    code corpus by language - is about code at all, and it is one entry among
    four rather than a third of the world.
    """
    code_specific = {k.name for k in corpus.all_kinds()
                     if "code" in k.summary.lower()}
    assert code_specific == {"stack"}, (
        f"{code_specific} look code-specific. If a new kind genuinely is, that "
        f"is fine - but it must not be the ONLY way to get text, or a "
        f"non-code Ms.MoE becomes unbuildable.")


def test_registering_a_kind_does_not_silently_shadow_an_existing_one():
    """A plugin quietly redefining `hf` would change the meaning of every
    recipe on the box, including ones its author never saw."""
    with pytest.raises(ValueError):
        corpus.register(corpus.Kind(name="hf", summary="hijacked"))


def test_a_third_party_kind_validates_without_this_package_knowing_it():
    """The actual openness claim: someone else's kind works here.

    Registered by hand rather than through an entry point because installing a
    second distribution inside a unit test proves less than it costs; the entry
    point loader is a thin wrapper around exactly this call.
    """
    corpus.register(corpus.Kind(
        name="_test_obsidian", summary="an Obsidian vault",
        requires=("path",)), replace=True)
    try:
        errs, _ = corpus.check("_test_obsidian",
                               rp.Source(kind="_test_obsidian", path="/vault"))
        assert errs == []
        errs, _ = corpus.check("_test_obsidian",
                               rp.Source(kind="_test_obsidian"))
        assert errs and "needs a path" in errs[0]
    finally:
        corpus._REGISTRY.pop("_test_obsidian", None)


def test_an_unknown_kind_says_what_is_available_and_how_to_add_one():
    """A refusal that does not tell you the next move is just a wall."""
    errs, _ = corpus.check("obsidian", rp.Source(kind="obsidian"))
    assert errs
    assert "hf" in errs[0] and "entry point" in errs[0]


# ── the recipe no longer hardcodes the list ─────────────────────────────────

def test_describe_reports_the_live_registry():
    """It was a frozen literal in three places, which is how a kind gets
    supported but never offered."""
    names = {k["name"] for k in rp.DESCRIBE["kinds"]}
    assert names == set(corpus.names())


def test_a_non_code_recipe_validates_clean():
    """The whole point, as an executable claim.

    No language, no stack, no code corpus anywhere - a lore expert from a
    HuggingFace dataset, a notes expert off local disk, and a generated one.
    If this ever fails, a domain assumption has crept back in.
    """
    rec, warnings = rp.parse({
        "schema_version": 1,
        "name": "msmoe-dungeonmaster",
        "size": "0.5B",
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "experts": [
            {"name": "bestiary",
             "source": {"kind": "hf", "repo": "x/monster-manual",
                        "text_field": "text"}},
            {"name": "lore",
             "source": {"kind": "local", "path": "~/notes", "glob": "**/*.md"}},
            {"name": "encounters",
             "source": {"kind": "synth", "teacher": "Qwen/Qwen2.5-7B-Instruct",
                        "generator": "encounter_design", "examples": 2000}},
        ],
        "budget": {"target_steps": 150},
        "moe": {"dense_layers": []},
    })
    errs, _ = rp.validate(rec)
    assert errs == [], f"a non-code recipe was rejected: {errs}"


# ── the stage plan follows the registry, not a name ─────────────────────────

def test_a_generated_expert_gets_a_data_stage_whatever_it_is_called():
    """This was `if "agentcore" in experts`.

    It worked for exactly as long as the only generated expert in the world
    was the MCP-trace one. Any other synth expert got no data.synth stage, so
    the longest phase of that build sat at `pending` in the viewer for its
    entire duration - the failure mode being SILENCE, which is the worst shape
    for it.
    """
    ids = [i for i, _ in st.plan(["bestiary", "encounters"], ["encounters"])]
    assert st.DATA_SYNTH in ids


def test_no_generated_expert_means_no_synth_stage():
    ids = [i for i, _ in st.plan(["bestiary", "lore"])]
    assert st.DATA_SYNTH not in ids
    assert st.DATA_CORPUS in ids


def test_the_stage_vocabulary_carries_no_domain_assumption():
    """The ids are a public contract, and `data.code` told every non-code
    builder they were a guest in someone else's tool. Renamed while Theatre
    was the only consumer; this keeps it renamed."""
    all_ids = {i for i, _ in st.plan(["a", "b"], ["b"])}
    for stage_id in all_ids:
        assert "code" not in stage_id, (
            f"stage id {stage_id!r} names a domain. Everything between the "
            f"corpus and the GGUF is domain-blind; the ids must be too.")


# ── the source KIND decides the collector, not the expert NAME ──────────────

def test_source_kind_decides_the_collector_not_the_expert_name(monkeypatch):
    """THE FIX, as an executable claim.

    A `kind: hf` expert whose NAME is also a CODE_LANGUAGE must go to the HF
    collector, not the stack scan. `remaining_languages` used to be every
    expert name, so "powershell" (kind=hf) had its HF corpus overwritten by a
    shard scan keyed on the code-language "PowerShell", and "shell"
    (kind=synth) was scanned as if it were the Shell language.

    The source kind is the contract; the name is a label.

    And the label is what the corpus is KEYED by: `dotnet` scans the language
    C# and must come back as `dotnet`. Keyed by language it came back as
    `csharp`, while run/builder.py and cli/_common.py look corpora up by expert
    name - so the build died on "No data path for expert dotnet" after the full
    shard scan. The old version of this test only covered the case where the
    expert name and the language name happen to be the same word, which is the
    one case the bug could not reach.
    """
    import types

    from ms_moe_maker.data import synth as d
    from ms_moe_maker.config.recipe import Source

    calls = {"hf": [], "shards": [], "names": []}

    def fake_hf(repo, text_field, split, out_path, config,
                callback=None, lang=""):
        calls["hf"].append(repo)
        return out_path

    def fake_shards(languages, config, callback=None, names=None,
                    caps=None, targets=None):
        calls["shards"].append(list(languages))
        calls["names"].append(dict(names or {}))
        from ms_moe_maker.config import pipeline as cfg_mod
        # Keyed the way the real scan keys: by the EXPERT the language was
        # scanned for. `names.get(l, l)` is exactly the old language-keyed
        # behaviour, so a caller that stops passing `names` fails this test.
        return {cfg_mod.safe_name((names or {}).get(l, l)): f"stack:{l}"
                for l in languages}

    monkeypatch.setattr(d, "_collect_hf", fake_hf)
    monkeypatch.setattr(d, "_collect_from_shards", fake_shards)

    config = types.SimpleNamespace(data_root="test_data", force=False)
    sources = {
        "python": Source(kind="stack", language="Python"),
        "dotnet": Source(kind="stack", language="C#"),
        "powershell": Source(
            kind="hf", repo="SaeedRahmani/codeparrot_github_code_powershell",
            text_field="code"),
        "shell": Source(kind="synth", reasoning=True, teacher="x/y"),
    }

    results = d.collect_corpus(
        config, languages=["python", "dotnet", "powershell", "shell"],
        sources=sources)

    # The hf expert went to the HF collector, once, with its repo.
    assert calls["hf"] == ["SaeedRahmani/codeparrot_github_code_powershell"]
    # The stack scan was asked for ONLY the two stack languages — never
    # "PowerShell" (an hf expert) or "Shell" (a synth expert).
    assert calls["shards"] == [["Python", "C#"]]
    # And it was told WHICH EXPERT each language is being scanned for.
    assert calls["names"] == [{"Python": "python", "C#": "dotnet"}]
    # And the hf result survived — it was not overwritten by a stack path.
    assert results["powershell"] == "test_data/powershell_code.jsonl"
    # Keyed by the expert that asked, not by the language that was scanned.
    assert results["dotnet"] == "stack:C#"
    assert "csharp" not in results, (
        "the corpus is keyed by expert name; `csharp` here is the language "
        "leaking into the key, which is what builder.py cannot look up")


def test_the_shard_scan_files_and_keys_by_expert_name(monkeypatch, tmp_path):
    """The same claim against the REAL function, not a fake.

    The already-on-disk skip is the only branch of _collect_from_shards that
    can be reached without a 45 GB download, and it derives its answer from the
    same safe_map the write does - so it pins both the returned key and the
    filename the scan would have written.
    """
    import sys
    import types

    from ms_moe_maker.data import synth as d

    # The heavy import is at the top of the function; stub it so the branch
    # below it is reachable. Anything that actually touched the network here
    # would be a bug in the skip path.
    stub = types.ModuleType("huggingface_hub")
    stub.hf_hub_download = stub.list_repo_files = None
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    (tmp_path / "dotnet_code.jsonl").write_text('{"text": "x"}\n',
                                                encoding="utf-8")
    config = types.SimpleNamespace(data_root=str(tmp_path), force=False,
                                   max_shards=1)
    got = d._collect_from_shards(["C#"], config, names={"C#": "dotnet"})
    assert got == {"dotnet": str(tmp_path / "dotnet_code.jsonl")}


def test_every_unsourced_expert_is_scanned_not_just_the_first(monkeypatch):
    """`elif not normalized:` guarded on the LIST being empty.

    So the second and every later expert with no source and a name that is not
    a code language was dropped without a word: three experts went in, one
    corpus came out, and the stage printed "collected 1 corpora".
    """
    import types

    from ms_moe_maker.data import synth as d

    seen = {}

    def fake_shards(languages, config, callback=None, names=None):
        seen["languages"] = list(languages)
        seen["names"] = dict(names or {})
        return {n: f"stack:{l}" for l, n in (names or {}).items()}

    monkeypatch.setattr(d, "_collect_from_shards", fake_shards)
    config = types.SimpleNamespace(data_root="test_data", force=False)

    results = d.collect_corpus(
        config, languages=["monster_manual", "lore", "dm_guide"], sources={})

    assert seen["languages"] == ["monster_manual", "lore", "dm_guide"]
    assert sorted(results) == ["dm_guide", "lore", "monster_manual"]


def test_per_source_max_shards_and_token_budget_reach_the_scan(monkeypatch):
    """Source.max_shards and Expert.tokens were declared, validated, and then
    overwritten by the run-wide values. Both now ride through collect_corpus
    into the shard scan, keyed by the scanned language."""
    import types

    from ms_moe_maker.config.recipe import Source
    from ms_moe_maker.data import synth as d

    seen = {}

    def fake_shards(languages, config, callback=None, names=None,
                    caps=None, targets=None):
        seen["languages"] = list(languages)
        seen["caps"] = dict(caps or {})
        seen["targets"] = dict(targets or {})
        return {n: f"stack:{l}" for l, n in (names or {}).items()}

    monkeypatch.setattr(d, "_collect_from_shards", fake_shards)
    config = types.SimpleNamespace(data_root="test_data", force=False)

    d.collect_corpus(
        config,
        languages=["python", "powershell"],
        sources={
            "python": Source(kind="stack", language="Python", max_shards=12),
            "powershell": Source(kind="stack", language="PowerShell"),
        },
        token_budgets={"python": 5_000_000},
        shard_caps={"python": 12})

    assert seen["caps"] == {"Python": 12}, (
        "source.max_shards must narrow the scan window for that expert")
    assert seen["targets"] == {"Python": 5_000_000}, (
        "expert.tokens must override the token budget for that expert")
    assert "PowerShell" not in seen["caps"]
    assert "PowerShell" not in seen["targets"]


def test_a_gh_source_without_a_glob_gets_the_md_default(monkeypatch):
    """Source.glob used to default to a truthy '**/*.txt', which made the
    '**/*.md' fallback unreachable - a doc-repo source collected zero files
    while describe and the README promised markdown."""
    import types

    from ms_moe_maker.config.recipe import Source
    from ms_moe_maker.data import synth as d

    seen = {}

    def fake_gh(repo, glob_pat, ref, subdir, out_path, config,
                callback=None, lang=""):
        seen["glob"] = glob_pat
        return out_path

    monkeypatch.setattr(d, "_collect_gh", fake_gh)
    d.collect_corpus(types.SimpleNamespace(data_root="d", force=False),
                     languages=["docs"],
                     sources={"docs": Source(kind="gh", repo="o/r")})
    assert seen["glob"] == "**/*.md"
