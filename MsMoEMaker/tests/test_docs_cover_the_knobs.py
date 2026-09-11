"""Every knob a recipe can set is mentioned in the docs somewhere.

WHY A TEST AND NOT A REVIEW. A recipe field is added in three places - the
dataclass, the resolution, the registry - and documented in a fourth that no
code path touches. Nothing fails when the fourth is skipped, so the drift is
invisible until somebody reads a knob's name in a stack trace and goes looking
for what it does. Nine fields had gone missing by the time anyone counted, four
of them from releases back.

This is a RATCHET, not a wall. `UNDOCUMENTED` below is a list of known gaps,
not a policy: it exists so this test can pass today while still failing the
moment a NEW field is added without a line of prose. Shrinking it is the work;
adding to it should feel like a decision.

The docs are not shipped in the wheel, so a test run from an installed package
finds nothing and skips. That is deliberate - this checks a repo, not an
install.
"""

import dataclasses
import re
from pathlib import Path

import pytest

from ms_moe_maker.config import recipe as recipe_mod


#: Recipe fields with no prose yet. DEBT, not policy - see the module docstring.
#: Each entry is a promise to write a line, not permission to skip one.
#:
#: EMPTY, AND THAT IS THE POINT. It held nine entries when this test was
#: written - five from the per-loop budgets and the overrun policy, four older
#: than anyone realised. All nine now have prose in the wiki's
#: Recipe-Options-Reference, so the ratchet did its job twice: it named the
#: debt, and then it failed the build when the debt was paid and the list was
#: not updated. Keep it empty if you can. Adding to it should feel like a
#: decision, and the next person to add a field will find out here.
UNDOCUMENTED: set = set()


def _doc_text():
    """Every doc in this checkout, concatenated. Empty if this is an install."""
    pkg = Path(__file__).resolve().parent.parent      # .../MsMoEMaker
    repo = pkg.parent                                  # .../MsMoEMaker (outer)
    candidates = [pkg / "README.md", repo / "README.md"]
    wiki = repo / "wiki"
    if wiki.is_dir():
        candidates += sorted(wiki.glob("*.md"))
    found = {}
    for path in candidates:
        if path.is_file():
            found[str(path)] = path.read_text(encoding="utf-8", errors="replace")
    return found


def _recipe_fields():
    """{"block.field"} for every field a recipe can set."""
    out = set()
    for field in dataclasses.fields(recipe_mod.Recipe):
        cls = field.type
        if not dataclasses.is_dataclass(cls):
            name = str(cls).replace("Optional[", "").rstrip("]").split(".")[-1]
            cls = getattr(recipe_mod, name, None)
        if not dataclasses.is_dataclass(cls):
            continue
        for sub in dataclasses.fields(cls):
            out.add("%s.%s" % (field.name, sub.name))
    return out


@pytest.fixture(scope="module")
def docs():
    found = _doc_text()
    if not found:
        pytest.skip("no README or wiki in this checkout - nothing to check")
    return found


def _mentioned(dotted, docs):
    """Dotted path, or the bare field name under any heading."""
    bare = dotted.split(".", 1)[1]
    for text in docs.values():
        if dotted in text:
            return True
        if re.search(r"\b%s\b" % re.escape(bare), text):
            return True
    return False


class TestEveryRecipeFieldIsWrittenDownSomewhere:
    def test_the_ratchet_holds(self, docs):
        """A field that is neither documented nor a known gap fails here.

        The message names the field and the file to put it in, because the
        useful version of this failure is "write a line in the budget table",
        not "coverage decreased".
        """
        missing = sorted(f for f in _recipe_fields()
                         if f not in UNDOCUMENTED and not _mentioned(f, docs))
        assert not missing, (
            "recipe fields with no documentation:\n  "
            + "\n  ".join(missing)
            + "\n\nAdd a row to that block's table in README.md (and the wiki's "
              "Recipe-Options-Reference.md), or add it to UNDOCUMENTED in this "
              "file with a reason if it genuinely does not need prose.")

    def test_the_debt_list_does_not_rot(self, docs):
        """An entry that HAS been documented must leave the list.

        Otherwise the list quietly becomes permission rather than debt, and the
        ratchet stops meaning anything.
        """
        now_documented = sorted(f for f in UNDOCUMENTED if _mentioned(f, docs))
        assert not now_documented, (
            "these are documented now - delete them from UNDOCUMENTED:\n  "
            + "\n  ".join(now_documented))

    def test_the_debt_list_only_names_real_fields(self):
        """A typo or a renamed field would sit here forever, exempting nothing
        and hiding the real gap behind a name that no longer exists."""
        stale = sorted(UNDOCUMENTED - _recipe_fields())
        assert not stale, (
            "UNDOCUMENTED names fields the recipe does not have:\n  "
            + "\n  ".join(stale))


class TestThePublishedVocabularyIsAlsoWrittenDown:
    """`--describe` is a contract with front-ends; the docs are the contract
    with people. A key in one and not the other is half a contract."""

    KEYS_NEEDING_PROSE = ("overrun_policies", "eval_modes", "started_fields")

    #: Same ratchet, same rules. `overrun_policies` and `eval_modes` are
    #: documented in CLI-Reference under `describe`, which is where a front-end
    #: author would look for the legal values of `eval.mode` and
    #: `budget.teacher_overrun`.
    #:
    #: `started_fields` is genuine debt: seren-theatre reads that event to find
    #: a run directory, so the keys are a contract with a real consumer, and
    #: nothing in any doc mentions them. Listed here so the ratchet stays green
    #: now and `test_the_key_debt_list_does_not_rot` fires the moment somebody
    #: writes the prose - at which point delete this line.
    UNDOCUMENTED_KEYS: set = {"started_fields"}

    #: WHY `stages` IS NOT IN THE LIST ABOVE, THOUGH IT IS EQUALLY UNDOCUMENTED.
    #:
    #: This ratchet matches a key NAME as a substring of the docs, and "stages"
    #: is ordinary English that already appears in seven of them ("the build runs
    #: nine stages"). An entry for it would be permanently, falsely green - a
    #: test that reports coverage it never checked, which is worse than no test
    #: because somebody would trust it.
    #:
    #: It cannot go in UNDOCUMENTED_KEYS either: the debt-rot check looks for the
    #: same substring and would fire immediately against that same prose.
    #:
    #: So the mechanism does not fit this key. Fixing it means matching something
    #: unambiguous - a backticked form, or a heading - and that is a change to
    #: how every entry here is checked, not a new entry.
    UNRATCHETABLE_KEYS: set = {"stages"}

    def test_the_ratchet_holds(self, docs):
        # THE MERGED PAYLOAD, not the identity card. The card carries eleven
        # keys; `__main__` publishes twelve more that only an installed CLI can
        # answer - kinds, validators, knobs, tiers, stages and the rest - and
        # reading only the card meant every one of those was invisible to this
        # check no matter what was added to the tuple above.
        from ms_moe_maker.__main__ import DESCRIBE
        missing = sorted(
            key for key in self.KEYS_NEEDING_PROSE
            if key in DESCRIBE and key not in self.UNDOCUMENTED_KEYS
            and not any(key in text for text in docs.values()))
        assert not missing, missing

    @staticmethod
    def published_vocabularies() -> set:
        """Every list-shaped key `--describe` actually publishes.

        A METHOD SO IT CAN BE ASSERTED ON DIRECTLY. Reading the identity card
        instead of the merged payload hides twelve keys, and no coverage check
        could tell the difference - every key published today is already
        accounted for either way, so the mistake would only surface on the next
        one added. Pulling it out gives the source of truth its own test.
        """
        from ms_moe_maker.__main__ import DESCRIBE
        return {k for k in DESCRIBE
                if isinstance(DESCRIBE[k], (list, tuple))}

    def test_the_check_reads_the_merged_payload_not_the_card(self):
        """The card is a SUBSET, and the difference is where the debt hides.

        `stages` is published by the installed CLI and not by the stdlib-only
        identity card, so it is the witness: if this check ever goes back to
        reading the card, `stages` vanishes from the set and so does every
        future key that only an installed CLI can answer.
        """
        from ms_moe_maker.box.describe import DESCRIBE as CARD
        got = self.published_vocabularies()
        assert "stages" in got, (
            "the coverage check is reading the identity card rather than the "
            "merged payload, so every key only an installed CLI can answer - "
            "kinds, validators, knobs, tiers, stages - is invisible to it.")
        assert "stages" not in CARD, (
            "stages moved into the stdlib-only card; this test's witness is "
            "stale, pick another __main__-only key.")
        assert got - set(CARD), "the merged payload published nothing extra"

    def test_every_published_key_is_accounted_for(self, docs):
        """A key nobody listed is a key nobody decided about.

        Not "every key needs prose" - most are self-evident rows a front-end
        renders. This asserts only that a key is either being checked, or
        recorded as debt, or recorded as not fitting the mechanism. The failure
        mode this closes is a new published vocabulary going in and nobody
        noticing it was never documented.
        """
        vocabularies = self.published_vocabularies()
        decided = (set(self.KEYS_NEEDING_PROSE) | self.UNDOCUMENTED_KEYS
                   | self.UNRATCHETABLE_KEYS)
        # Rows a front-end renders rather than words a person looks up.
        self_evident = {"commands", "requires", "kinds", "validators", "knobs",
                        "tiers", "gates", "templates", "modes", "events",
                        "registry_errors", "reasoning", "defaults"}
        missing = sorted(vocabularies - decided - self_evident)
        assert not missing, (
            f"{missing} are published vocabularies that nobody has decided "
            f"about. Add each to KEYS_NEEDING_PROSE (it should be documented), "
            f"UNDOCUMENTED_KEYS (it should be, and is not yet) or the "
            f"self_evident set above (it is rows, not prose).")

    def test_the_key_debt_list_does_not_rot(self, docs):
        documented = sorted(k for k in self.UNDOCUMENTED_KEYS
                            if any(k in t for t in docs.values()))
        assert not documented, (
            "documented now - delete from UNDOCUMENTED_KEYS: %s" % documented)


class TestEveryCliVerbIsWrittenDown:
    """A verb nobody can find is a verb nobody uses."""

    #: `bundle` shipped without a mention in ANY doc - not the wiki, not
    #: either README - and so did `export`, which only escaped this list
    #: because the word appears in prose about GGUF export. Both are written
    #: up in CLI-Reference now.
    UNDOCUMENTED_VERBS: set = set()

    def test_the_ratchet_holds(self, docs):
        from ms_moe_maker.box.describe import DESCRIBE
        missing = sorted(
            verb for verb in DESCRIBE["commands"]
            if verb not in self.UNDOCUMENTED_VERBS
            and not any(re.search(r"\b%s\b" % verb, t) for t in docs.values()))
        assert not missing, (
            "CLI verbs with no documentation: %s" % missing)

    def test_the_verb_debt_list_does_not_rot(self, docs):
        documented = sorted(
            v for v in self.UNDOCUMENTED_VERBS
            if any(re.search(r"\b%s\b" % v, t) for t in docs.values()))
        assert not documented, (
            "documented now - delete from UNDOCUMENTED_VERBS: %s" % documented)
