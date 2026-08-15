"""The corpus registry, and the domain-neutrality it exists to protect.

The point of these tests is not that a dict can hold entries. It is that
nothing in the build assumes the training text is CODE - because the day that
assumption ships in a public contract is the day every person building a
Ms.MoE for lore, or exam material, or case law is told they are using someone
else's tool.
"""
from __future__ import annotations

import pytest

from ms_moe_maker import corpus
from ms_moe_maker import recipe as rp
from ms_moe_maker import stages as st


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
