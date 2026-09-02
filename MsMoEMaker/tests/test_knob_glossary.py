"""A number in a playbill that nothing explains, and two copies of the sentence
that would have explained it.

`build_fingerprint` emits 75 resolved values and seren-theatre renders them off
the manifest. `collect_token_target 29491200` is a true row and an unreadable
one, so `config/knobs.py` carries a summary and a derivation for every field.

Two failures this file exists to make loud:

  A KNOB WITH NO WORDS. Add a field to PipelineConfig, forget the glossary
  entry, and the playbill grows a row nobody can read - silently, because
  missing documentation does not raise. The fingerprint is fail-closed on
  purpose (test_reproducibility::test_the_fingerprint_is_fail_closed); the
  glossary has to be closed against the same set or it is not a glossary, it
  is a sample.

  TWO COPIES OF ONE SENTENCE. About 36 of these knobs are also described in the
  README's knob tables. Two hand-maintained copies of one explanation drift,
  and the drift is invisible: both files still read fine. So the README's
  "What it does" column IS the glossary summary, and the check below says so.
  The README may add emphasis, backticks and em dashes; it may not add, drop
  or reword.

EMPTY SETS ARE NOT AGREEMENT. Every loop here asserts it compared something -
a check that passes because it found nothing to check is the failure mode this
repo keeps closing, and it is the one a documentation test is most prone to.
"""
import dataclasses
import re
from pathlib import Path

import pytest

from ms_moe_maker.config import knobs as K
from ms_moe_maker.config import pipeline as C
from ms_moe_maker.config import recipe as R
from ms_moe_maker.run import manifest as mf

# The shipped README (next to pyproject.toml) and, in a git checkout, the copy
# at the repository root. Both are checked; neither may drift.
PKG_README = Path(__file__).resolve().parent.parent / "README.md"
ROOT_README = PKG_README.parent.parent / "README.md"
READMES = [p for p in (PKG_README, ROOT_README) if p.is_file()]


def _rec(**extra):
    body = {"schema_version": 1, "name": "t", "size": "0.5B",
            "experts": [{"name": "a", "source": {"kind": "hf", "repo": "o/d"}},
                        {"name": "b", "source": {"kind": "hf", "repo": "o/e"}}]}
    body.update(extra)
    rec, _ = R.parse(body)
    return rec


def _fingerprint():
    return C.build_fingerprint(C.build_config(_rec(), dryrun=False))


class TestEveryKnobIsExplained:
    """THE STRUCTURAL POINT. A new knob without a description fails the suite."""

    def test_the_fingerprint_is_worth_checking_against(self):
        """The guard on every other test here: if this set were empty, all of
        them would pass by comparing nothing."""
        cfg = C.build_config(_rec(), dryrun=False)
        fields = set(_fingerprint())
        declared = {f.name for f in dataclasses.fields(cfg)}
        assert fields == declared - C._FINGERPRINT_EXCLUDE
        assert len(fields) > 50, (
            "the fingerprint went nearly empty - the glossary tests below "
            "would then be asserting nothing")

    def test_every_emitted_field_has_a_non_empty_summary(self):
        fields = sorted(_fingerprint())
        assert fields
        missing = [f for f in fields
                   if K.entry(f) is None or not K.entry(f).summary.strip()]
        assert not missing, (
            "these fields land in the manifest and the playbill with nothing "
            "to explain them - add an entry to config/knobs.py:\n  "
            + "\n  ".join(missing))

    def test_the_glossary_describes_nothing_the_fingerprint_does_not_emit(self):
        """Rot in the other direction. An entry for a field that was renamed or
        excluded is a sentence nobody will ever see and nobody will ever fix."""
        stale = sorted(set(K.KNOBS) - set(_fingerprint()))
        assert not stale, (
            "config/knobs.py explains fields build_fingerprint no longer "
            "emits: " + ", ".join(stale))

    def test_the_detector_catches_a_new_undescribed_field(self):
        """A lint nobody has seen fail is a lint nobody should trust."""
        fields = {"lora_r", "a_knob_nobody_wrote_up"}
        missing = [f for f in sorted(fields)
                   if K.entry(f) is None or not K.entry(f).summary.strip()]
        assert missing == ["a_knob_nobody_wrote_up"]


class TestDerivations:
    """`derived_from` is the half a non-technical reader needs most: a field
    nobody typed is the one they stare at."""

    def test_a_derivation_is_absent_or_a_real_expression(self):
        checked = 0
        for name, k in sorted(K.KNOBS.items()):
            if k.derived_from is None:
                continue
            checked += 1
            assert k.derived_from.strip(), f"{name}: empty derived_from"
        assert checked > 10, (
            "almost nothing claims to be derived - either the glossary lost "
            "its formulas or this test stopped finding them")

    def test_every_field_named_in_a_formula_still_exists(self):
        """A wrong formula is worse than none, and the way one goes wrong
        quietly is a rename: the arithmetic stays readable while naming a field
        that no longer exists. Any underscored, lower-case name in a formula
        has to be a real fingerprint field or a declared recipe key.
        (ALL-CAPS names are env vars and are left alone.)"""
        fields = set(_fingerprint())
        assert fields
        unknown, checked = [], 0
        for name, k in sorted(K.KNOBS.items()):
            if not k.derived_from:
                continue
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", k.derived_from):
                if "_" not in tok or tok.isupper():
                    continue
                checked += 1
                if tok not in fields and tok not in K.RECIPE_TERMS:
                    unknown.append(f"{name}: {tok}")
        assert checked > 10, "no formulas were inspected"
        assert not unknown, (
            "these formulas name something that is neither a fingerprint "
            "field nor a declared recipe key:\n  " + "\n  ".join(unknown))

    def test_the_expensive_derivations_are_present(self):
        """The three numbers a reader is most likely to stare at, because
        nobody typed any of them."""
        for f in ("collect_token_target", "expert_token_budget",
                  "warmup_steps", "min_samples_per_expert"):
            assert K.entry(f) is not None and K.entry(f).derived_from, (
                f"{f} is computed, not typed - it needs a derived_from")


class TestTheWireContract:
    """The shape seren-theatre's playbill is being built against. Fixed."""

    def test_for_fields_emits_exactly_summary_and_derived_from(self):
        out = K.for_fields(_fingerprint())
        assert out
        for name, row in out.items():
            assert set(row) == {"summary", "derived_from"}, name
            assert isinstance(row["summary"], str) and row["summary"].strip()
            assert row["derived_from"] is None or isinstance(
                row["derived_from"], str)

    def test_a_field_with_no_entry_is_absent_rather_than_blank(self):
        """Partial coverage has to degrade to "no affordance", never to an
        empty tooltip - which reads as a broken page instead of a gap."""
        out = K.for_fields(["lora_r", "a_field_with_no_entry"])
        assert "a_field_with_no_entry" not in out
        assert out["lora_r"]["summary"]

    def test_describe_matches_kinds_and_validators_beside_it(self):
        rows = K.describe()
        assert rows
        for row in rows:
            assert set(row) == {"name", "summary", "derived_from"}
        assert [r["name"] for r in rows] == sorted(r["name"] for r in rows)

    def test_describe_carries_the_glossary(self):
        from ms_moe_maker.__main__ import DESCRIBE
        assert DESCRIBE["knobs"], "--describe lost the glossary"
        names = {r["name"] for r in DESCRIBE["knobs"]}
        assert "collect_token_target" in names

    def test_the_manifest_round_trips_it(self, tmp_path):
        m = mf.Manifest(recipe_id="abc", build_id="deadbeef",
                        resolved={"lora_r": 64},
                        knobs=K.for_fields(["lora_r"]))
        mf.write(tmp_path, m)
        back = mf.read(tmp_path)
        assert back.knobs["lora_r"]["summary"]
        assert "derived_from" in back.knobs["lora_r"]

    def test_an_old_manifest_without_knobs_still_reads(self, tmp_path):
        """Additive only. seren-theatre has an independent reader for this
        format and a schema bump would blind it."""
        import json
        (tmp_path / mf.MANIFEST_NAME).write_text(json.dumps({
            "schema_version": 1, "recipe_id": "abc", "name": "t",
            "size": "0.5B", "base": "", "experts": ["a"], "stages": [],
        }), encoding="utf-8")
        back = mf.read(tmp_path)
        assert back.knobs == {}
        assert mf.SCHEMA_VERSION == 1, "these fields must not need a bump"

    def test_a_run_stamps_the_glossary_for_what_it_resolved(self, tmp_path):
        """The whole point of stamping: an archived run explains itself to a
        viewer that has no ms-moe-maker installed."""
        from ms_moe_maker.run.events import Events
        from ms_moe_maker.run.runner import Runner, Translation
        run = Runner(_rec(), None, Translation(), Events(enabled=False),
                     cwd=tmp_path, dryrun=True)
        assert run.manifest.resolved, "no fingerprint to explain"
        assert set(run.manifest.knobs) == set(run.manifest.resolved)
        assert run.manifest.knobs["lora_r"]["summary"]


class TestOneSourceNotTwo:
    """The README's knob tables and this glossary would be the same sentences,
    written twice. They are not: the README's column IS the summary."""

    SPLIT = re.compile(r"(?<!\\)\|")
    HEAD = re.compile(r"^####\s+`([A-Za-z_]+):`")

    @staticmethod
    def _norm(text):
        """Words only. The README may add backticks, bold and em dashes."""
        t = text.replace("\\|", "|").replace("`", "").replace("*", "")
        t = t.replace("—", "-").replace("–", "-")
        return re.sub(r"\s+", " ", t).strip()

    @classmethod
    def _table_cells(cls, path):
        """{"<block>.<row>": what-it-does} for every knob row in a README."""
        out, blocks, section = {}, set(), ""
        for line in path.read_text(encoding="utf-8").split("\n"):
            head = cls.HEAD.match(line)
            if head:
                section = head.group(1)
                blocks.add(section)
                continue
            if not line.startswith("| `"):
                continue
            parts = cls.SPLIT.split(line)
            if len(parts) != 5:      # '', name, default, what it does, ''
                continue
            cells = [c.strip() for c in parts[1:-1]]
            out[f"{section}.{cells[0].strip('`')}"] = cells[2]
        return out, blocks

    def test_there_are_readmes_to_check(self):
        assert PKG_README.is_file(), (
            "the shipped README is gone - the agreement check below would "
            "silently compare nothing")
        assert READMES

    def test_the_readme_column_is_the_glossary_summary(self):
        rows = K.readme_rows()
        assert rows, "no knob claims a README row - nothing would be compared"
        mismatched, found, checked = [], set(), 0
        for path in READMES:
            cells, _ = self._table_cells(path)
            assert cells, f"{path.name}: no knob tables found"
            for key, field in sorted(rows.items()):
                if key not in cells:
                    continue
                found.add(key)
                checked += 1
                want = self._norm(K.KNOBS[field].summary)
                got = self._norm(cells[key])
                if want != got:
                    mismatched.append(
                        f"{path.name} [{key}]\n     README: {got}\n"
                        f"  knobs.py: {want}")
        assert checked >= len(rows), (
            f"only {checked} README cells were compared against {len(rows)} "
            f"mapped knobs")
        assert not mismatched, (
            "a README knob table and config/knobs.py disagree. The glossary "
            "is the source; paste its sentence into the cell:\n  "
            + "\n  ".join(mismatched))

    def test_every_mapped_row_is_actually_in_a_readme(self):
        """The other direction: a `readme=` pointing at a row that was renamed
        or deleted means the mapping is comparing nothing and saying nothing."""
        rows = K.readme_rows()
        found, blocks = set(), set()
        for path in READMES:
            cells, seen = self._table_cells(path)
            found |= set(cells)
            blocks |= seen
        # Only rows whose table is present in some README can be checked; an
        # sdist without the repo-root copy legitimately has fewer tables.
        expected = {k for k in rows if k.split(".", 1)[0] in blocks}
        assert expected, "no knob-table sections found in any README"
        assert expected <= found, (
            "config/knobs.py claims these README rows exist and they do not: "
            + ", ".join(sorted(expected - found)))
