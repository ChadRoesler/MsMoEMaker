"""Freezing a recipe so it builds the same thing on somebody else\'s box.

TWO DIFFERENT FAILURES ARE GUARDED HERE AND THEY NEED DIFFERENT TESTS.

  omission   a field that should be stamped and is not. The round-trip check
             CANNOT see this: on one box the missing field resolves the same
             way both times, so the fingerprints agree and everything looks
             fine. It is caught by insisting the triage is TOTAL - every direct
             fingerprint field is either pinnable or explicitly named as
             unpinnable, with a reason.
  corruption a field stamped with the wrong value. That the round trip does
             see, because re-resolving produces a different fingerprint.

Testing only one of them would leave the other silent, and the silent one is
the omission - which is also the one that grows every time somebody adds a
field to PipelineConfig.
"""
from __future__ import annotations

import dataclasses

import pytest
import yaml

from ms_moe_maker.bundle import stamp as S
from ms_moe_maker.config import recipe as R
from ms_moe_maker.config.knobs import KNOBS, UNPINNABLE, pinnable, recipe_path
from ms_moe_maker.config.pipeline import build_config, build_fingerprint, build_id

RECIPE = """
name: handoff
size: 0.5B
experts:
  - name: python
    source: { kind: hf, repo: owner/py, text_field: text }
  - name: markdown
    source: { kind: gh, repo: owner/docs, glob: 'docs/**/*.md' }
budget:
  target_steps: 800
router:
  epochs: 8
"""

# The same recipe with the router block REMOVED, so `router.epochs` is still a
# hole for the defaults layer to fill. The two-boxes tests need a hole - a
# recipe that already answers every question cannot demonstrate a question
# being answered differently somewhere else, and a version of those tests built
# on RECIPE passed for that reason rather than for the right one.
BARE = """
name: handoff
size: 0.5B
experts:
  - name: python
    source: { kind: hf, repo: owner/py, text_field: text }
  - name: markdown
    source: { kind: gh, repo: owner/docs, glob: 'docs/**/*.md' }
budget:
  target_steps: 800
"""


@pytest.fixture
def recipe_file(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(RECIPE, encoding="utf-8")
    return p


def _resolved(path, defaults=None):
    rec, _ = R.load(str(path), defaults_path=str(defaults) if defaults else None,
                    include_user_defaults=False)
    return build_config(rec)


# ── the triage is total ──────────────────────────────────────────────────────

def test_every_direct_field_is_either_pinnable_or_named_unpinnable():
    """THE OMISSION GUARD, and the only one that can catch an omission.

    A field added to PipelineConfig lands in build_fingerprint, so it changes
    what the build produces. If nobody decides whether a recipe can express it,
    it silently becomes a thing that follows the far box - and the round-trip
    check on the author\'s machine will happily pass, because there both
    resolutions agree.
    """
    direct = {name for name, k in KNOBS.items() if k.derived_from is None}
    pins = set(pinnable())
    unpinned = set(UNPINNABLE)
    unaccounted = direct - pins - unpinned
    assert not unaccounted, (
        f"these fingerprint fields have no recipe path and no entry in "
        f"UNPINNABLE: {sorted(unaccounted)}. Give each one a `recipe=` path, "
        f"or name it in UNPINNABLE with the reason it cannot have one.")
    both = pins & unpinned
    assert not both, f"claimed pinnable AND unpinnable: {sorted(both)}"
    stray = unpinned - direct
    assert not stray, (
        f"UNPINNABLE names fields that are not direct: {sorted(stray)}. A "
        f"DERIVED field is not unpinnable - it is recomputed, which is right.")


def test_no_derived_field_is_ever_stamped():
    """The load-bearing exclusion.

    A value computed from other values has to be recomputed over there. Freezing
    `collect_token_target` would leave it fighting the `collect_headroom` it is
    supposed to follow, and there is no recipe key to write it into anyway.
    """
    derived = {name for name, k in KNOBS.items() if k.derived_from is not None}
    assert not (derived & set(pinnable())), (
        "a derived field is in the stamp set; stamped, it would override the "
        "formula it is supposed to be the output of")


def test_every_recipe_path_resolves_against_the_real_recipe():
    """A path that does not exist stamps into a key nothing reads.

    The failure is completely silent: the exported recipe carries a line, the
    loader ignores it, and the far box uses its own default anyway. It would
    look exactly like a bundle that worked.
    """
    bad = []
    for field, path in sorted(pinnable().items()):
        cur, parts = R.Recipe, path.split(".")
        for i, part in enumerate(parts):
            if not dataclasses.is_dataclass(cur):
                bad.append((field, path, f"{parts[i-1]} is not a block"))
                break
            names = {f.name: f for f in dataclasses.fields(cur)}
            if part not in names:
                bad.append((field, path, f"no {part!r} on {cur.__name__}"))
                break
            if i < len(parts) - 1:
                t = names[part].type
                cur = getattr(R, t if isinstance(t, str) else t.__name__, None)
    assert not bad, f"recipe paths that do not exist: {bad}"


def test_every_pinnable_path_is_exactly_two_levels():
    """`render` attaches its comments by matching keys at indent two.

    A three-level path would stamp correctly and then be marked on the wrong
    line, or on none - so this is a real constraint on the data and not a
    restatement of it.
    """
    deep = {f: p for f, p in pinnable().items() if len(p.split(".")) != 2}
    assert not deep, f"paths that are not <block>.<key>: {deep}"


# ── the round trip ───────────────────────────────────────────────────────────

def test_a_stamped_recipe_rebuilds_to_the_same_build_id(recipe_file, tmp_path):
    """THE ACCEPTANCE TEST for the whole feature, in one line of assertion."""
    config = _resolved(recipe_file)
    before = build_id(config)
    raw = yaml.safe_load(RECIPE)
    stamped, marks = S.stamp(raw, config)
    text = S.render(stamped, marks)
    out = tmp_path / "s.yaml"
    out.write_text(text, encoding="utf-8")
    assert build_id(_resolved(out)) == before


def test_verify_reports_nothing_for_a_good_stamp(recipe_file):
    config = _resolved(recipe_file)
    raw = yaml.safe_load(RECIPE)
    stamped, marks = S.stamp(raw, config)
    assert S.verify(S.render(stamped, marks), build_fingerprint(config)) == {}


def test_verify_catches_a_corrupted_stamp(recipe_file):
    """THE CORRUPTION GUARD. A wrong value must not reach a bundle.

    The exporter runs this before writing, so the person who would otherwise
    discover it is the one the bundle was given to.
    """
    config = _resolved(recipe_file)
    raw = yaml.safe_load(RECIPE)
    stamped, marks = S.stamp(raw, config)
    text = S.render(stamped, marks).replace("experts_per_tok: 2",
                                            "experts_per_tok: 4")
    drift = S.verify(text, build_fingerprint(config))
    assert "experts_per_tok" in drift
    assert drift["experts_per_tok"]["theirs"] == 2
    assert drift["experts_per_tok"]["ours"] == 4


# ── the feature it actually exists for ───────────────────────────────────────

def test_a_bare_recipe_builds_a_different_model_under_someone_elses_defaults(
        tmp_path):
    """The bug, demonstrated. This is what the verb is FOR.

    Same file, two boxes, two models, no error anywhere. If this test ever
    starts passing-by-being-equal, the defaults layering stopped working and
    the whole feature is answering a question nobody has.
    """
    r = tmp_path / "r.yaml"
    r.write_text(BARE, encoding="utf-8")
    mine = tmp_path / "mine.yaml"
    mine.write_text("router:\n  epochs: 3\n", encoding="utf-8")
    theirs = tmp_path / "theirs.yaml"
    theirs.write_text("router:\n  epochs: 12\n", encoding="utf-8")

    assert _resolved(r, mine).router_epochs == 3.0
    assert _resolved(r, theirs).router_epochs == 12.0
    assert build_id(_resolved(r, mine)) != build_id(_resolved(r, theirs))


def test_a_stamped_recipe_survives_someone_elses_defaults(tmp_path):
    """And the fix, demonstrated against the same two boxes."""
    r = tmp_path / "r.yaml"
    r.write_text(BARE, encoding="utf-8")
    mine = tmp_path / "mine.yaml"
    mine.write_text("router:\n  epochs: 3\n", encoding="utf-8")
    theirs = tmp_path / "theirs.yaml"
    theirs.write_text("router:\n  epochs: 12\n", encoding="utf-8")

    config = _resolved(r, mine)
    stamped, marks = S.stamp(yaml.safe_load(BARE), config)
    out = tmp_path / "s.yaml"
    out.write_text(S.render(stamped, marks), encoding="utf-8")

    assert build_id(_resolved(out, theirs)) == build_id(config), (
        "the stamped recipe picked up the other box\'s defaults, which is the "
        "entire failure this verb exists to prevent")


# ── what it must not do ──────────────────────────────────────────────────────

def test_an_authored_key_is_never_overwritten():
    """A stamp fills blanks. Anything else is an editorial pass on somebody
    else\'s file, and the one thing worse than a recipe that builds
    differently over there is one that builds differently HERE after export."""
    raw = yaml.safe_load(RECIPE)
    config = build_config(R.parse(raw)[0])
    stamped, marks = S.stamp(raw, config)
    assert stamped["router"]["epochs"] == 8
    assert stamped["budget"]["target_steps"] == 800
    assert ("router", "epochs") not in marks
    assert ("budget", "target_steps") not in marks


def test_the_stamp_does_not_mutate_the_input():
    raw = yaml.safe_load(RECIPE)
    before = yaml.safe_dump(raw, sort_keys=True)
    S.stamp(raw, build_config(R.parse(raw)[0]))
    assert yaml.safe_dump(raw, sort_keys=True) == before, (
        "stamp edited the caller\'s dict; a caller that then writes it back "
        "has silently rewritten the author\'s own recipe")


def test_only_the_stamped_lines_are_marked():
    """The comments ARE the feature. Ninety explicit keys where six were
    chosen: if everything is explicit then nothing is emphasised, and "these
    are the knobs I picked" is exactly what the reader needs."""
    raw = yaml.safe_load(RECIPE)
    config = build_config(R.parse(raw)[0])
    stamped, marks = S.stamp(raw, config)
    text = S.render(stamped, marks)
    for line in text.split("\n"):
        if line.strip().startswith("epochs:"):
            assert "# default" not in line, "an authored line was marked"
        if line.strip().startswith("aux_loss_coef:"):
            assert "# default" in line, "a stamped line was not marked"
    assert text.count("# default") == len(marks)


def test_an_unknown_block_survives_the_round_trip():
    """A key this version has never heard of is somebody\'s extension or a
    newer schema. Dropping it would be the worst possible way to help two
    people share a recipe."""
    raw = yaml.safe_load(RECIPE)
    raw["some_future_block"] = {"a": 1}
    config = build_config(R.parse(yaml.safe_load(RECIPE))[0])
    stamped, marks = S.stamp(raw, config)
    assert "some_future_block" in S.render(stamped, marks)


# ── the sixteen ──────────────────────────────────────────────────────────────

def test_the_unpinnable_snapshot_records_all_of_them(recipe_file):
    snap = S.unpinnable_snapshot(_resolved(recipe_file))
    assert set(snap) == set(UNPINNABLE)


def test_the_diff_reports_a_field_only_one_side_has():
    """A field one box has and the other does not is a VERSION difference,
    which is precisely the thing worth being told about."""
    out = S.diff_fingerprints({"a": 1, "gone": 2}, {"a": 1, "new": 3})
    assert out["gone"]["ours"] is None
    assert out["new"]["theirs"] is None
    assert "a" not in out


def test_the_diff_explains_an_unpinnable_field_with_its_reason():
    """DRAWN FROM THE REGISTRY, not named here.

    This used to hardcode use_vllm as its example, and broke the day that
    field earned a recipe key - correctly, but the failure read as a puzzle
    about a missing env var rather than as a finding. A test about "every
    unpinnable field explains itself" should ask the list: a field leaving
    it is then a non-event, and a field arriving in it is covered for free.
    """
    from ms_moe_maker.config.knobs import UNPINNABLE

    assert UNPINNABLE, "nothing is unpinnable, which cannot be right"
    for field, reason in UNPINNABLE.items():
        out = S.diff_fingerprints({field: True}, {field: False})
        assert out[field]["why"] == reason, (
            f"{field} differs between two boxes and the diff does not say "
            f"WHY it could not travel - that is a mystery, not an action")

