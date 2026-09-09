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
UNDOCUMENTED = {
    # Added this session with the per-loop budgets and the overrun policy.
    "budget.agent_teacher_max_new",
    "budget.domain_teacher_max_new",
    "budget.teacher_tell_budget",
    "budget.teacher_overrun",
    "budget.teacher_answer_reserve",
    # Older, and undocumented for longer than anyone realised.
    "abliterate.enabled",
    "corpus.synth_samples",
    "runtime.use_vllm",
    "runtime.vllm_max_len",
}


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

    KEYS_NEEDING_PROSE = ("overrun_policies", "eval_modes")

    #: Same ratchet, same rules.
    UNDOCUMENTED_KEYS = {"overrun_policies", "eval_modes"}

    def test_the_ratchet_holds(self, docs):
        from ms_moe_maker.box.describe import DESCRIBE
        missing = sorted(
            key for key in self.KEYS_NEEDING_PROSE
            if key in DESCRIBE and key not in self.UNDOCUMENTED_KEYS
            and not any(key in text for text in docs.values()))
        assert not missing, missing

    def test_the_key_debt_list_does_not_rot(self, docs):
        documented = sorted(k for k in self.UNDOCUMENTED_KEYS
                            if any(k in t for t in docs.values()))
        assert not documented, (
            "documented now - delete from UNDOCUMENTED_KEYS: %s" % documented)


class TestEveryCliVerbIsWrittenDown:
    """A verb nobody can find is a verb nobody uses."""

    #: `bundle` shipped without a mention in any doc.
    UNDOCUMENTED_VERBS = {"bundle"}

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
