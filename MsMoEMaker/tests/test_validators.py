"""The validator registry, and the rule it exists to make unbreakable.

Most of this file is one assertion said several ways: a tool this box does not
have must never produce a `fail`. C# scored 0/10 on a model that writes C#
fine, because a missing compiler was indistinguishable from a syntax error,
and nothing looked broken. These tests are the argument that cannot be lost.
"""
from __future__ import annotations

import shutil

import pytest

from ms_moe_maker import validators as v
from ms_moe_maker.evalrecord import ERROR, FAIL, MEASURED, PASS, UNMEASURABLE


# ── the funnel ──────────────────────────────────────────────────────────────

def test_an_unavailable_validator_is_unmeasurable_never_a_failure():
    """THE rule. Asserted against a validator whose tool cannot exist."""
    v.register(v.Validator(
        name="_test_missing", summary="needs a tool nobody has",
        tools=("definitely-not-a-real-binary-9f3a",),
        run=lambda gen, spec: v.Outcome(FAIL, "should never be reached")),
        replace=True)
    try:
        out = v.judge("_test_missing", "anything")
        assert out.verdict == UNMEASURABLE, (
            "a missing toolchain produced a verdict other than unmeasurable - "
            "this is the C# 0/10 bug reopening")
        assert out.verdict not in MEASURED
        assert "PATH" in out.reason
    finally:
        v._REGISTRY.pop("_test_missing", None)


def test_the_availability_gate_is_not_advisory():
    """The run() above returns FAIL, and judge() must never reach it."""
    v.register(v.Validator(
        name="_test_gate", tools=("definitely-not-a-real-binary-9f3a",),
        run=lambda gen, spec: v.Outcome(FAIL, "reached")), replace=True)
    try:
        assert v.judge("_test_gate", "x").reason != "reached"
    finally:
        v._REGISTRY.pop("_test_gate", None)


def test_an_unknown_validator_is_an_error_not_a_failure():
    """The HARNESS is wrong, which is still not the model being wrong - and it
    is distinct from unmeasurable so a person can tell 'install a compiler'
    from 'fix the eval' without reading the source."""
    out = v.judge("no-such-validator", "anything")
    assert out.verdict == ERROR
    assert out.verdict not in MEASURED


def test_a_validator_that_raises_is_an_error_not_a_failure():
    def boom(gen, spec):
        raise RuntimeError("harness bug")

    v.register(v.Validator(name="_test_boom", run=boom), replace=True)
    try:
        out = v.judge("_test_boom", "x")
        assert out.verdict == ERROR and "harness bug" in out.reason
    finally:
        v._REGISTRY.pop("_test_boom", None)


# ── the domain-neutral validators: a lore Ms.MoE is gradeable ───────────────

def test_contains_grades_an_answer_without_demanding_a_phrasing():
    spec = {"expected": ["dragon", "lair actions"]}
    assert v.judge("contains", "An ancient dragon has Lair Actions.",
                   spec).verdict == PASS
    assert v.judge("contains", "It is a big lizard.", spec).verdict == FAIL


def test_exact_is_whitespace_and_case_insensitive():
    spec = {"expected": "Armor Class 18"}
    assert v.judge("exact", "  armor   class 18 ", spec).verdict == PASS


def test_regex_matches():
    assert v.judge("regex", "AC is 18", {"pattern": r"\bAC\b.*\d+"}).verdict == PASS


def test_json_unwraps_a_fence_and_checks_required_keys():
    gen = '```json\n{"name": "Owlbear", "cr": 3}\n```'
    assert v.judge("json", gen, {"required_keys": ["name", "cr"]}).verdict == PASS
    assert v.judge("json", gen, {"required_keys": ["speed"]}).verdict == FAIL
    assert v.judge("json", "not json at all", {}).verdict == FAIL


def test_every_domain_neutral_validator_needs_no_tools():
    """If a lore validator ever grows a toolchain, a lore Ms.MoE becomes
    ungradeable on a box that lacks it - reintroducing the whole problem for
    the audience this registry was built for."""
    for name in ("exact", "contains", "regex", "json"):
        assert v.get(name).tools == (), f"{name} grew a tool dependency"
        assert v.availability(name).ok is True


# ── syntax: the code-specific one ───────────────────────────────────────────

def test_python_syntax_parses_and_does_not_execute():
    """compile() builds an AST. If this EXECUTED, the assignment below would
    raise and the test would report a failure instead of a pass."""
    spec = {"language": "python"}
    assert v.judge("syntax", "raise SystemExit('boom')", spec).verdict == PASS
    assert v.judge("syntax", "def f(:", spec).verdict == FAIL


def test_a_language_with_no_toolchain_here_reports_unmeasurable():
    """The C# case, reproduced. Skips if the box HAS a C# compiler, because
    then there is nothing to prove."""
    if shutil.which("csc") or shutil.which("mcs"):
        pytest.skip("this box can judge C#; the unmeasurable path needs a box "
                    "that cannot")
    out = v.judge("syntax", "public class A {}", {"language": "C#"})
    assert out.verdict == UNMEASURABLE, (
        "a missing C# compiler scored as a model failure - this is exactly the "
        "0/10 result that started all of this")
    assert "toolchain" in out.reason or "installed" in out.reason


def test_dotnet_is_not_in_the_csharp_toolchain():
    """It was listed once and its branch was `return False`, so a box with the
    .NET SDK but no mono read as AVAILABLE and then scored zero without a
    compiler ever seeing the code. Listing a tool nobody implemented is how a
    toolchain gap became a model result."""
    assert "dotnet" not in v.LANG_TOOLCHAIN["csharp"]


def test_an_unknown_language_is_unmeasurable_and_says_how_to_add_one():
    out = v.judge("syntax", "SELECT 1", {"language": "sql"})
    assert out.verdict == UNMEASURABLE
    assert "entry point" in out.reason


def test_language_aliases_resolve():
    assert v._lang({"language": "C#"}) == "csharp"
    assert v._lang({"language": "ps1"}) == "powershell"


# ── registry mechanics ──────────────────────────────────────────────────────

def test_registering_does_not_silently_shadow():
    with pytest.raises(ValueError):
        v.register(v.Validator(name="contains", summary="hijacked"))


def test_check_is_declarative_and_touches_no_tools():
    """Recipe-time validation has to run on a laptop with nothing installed -
    the same laptop promise the corpus registry keeps."""
    errs, _ = v.check("syntax", {})
    assert errs and "needs a language" in errs[0]
    assert v.check("syntax", {"language": "python"})[0] == []


def test_an_unknown_validator_names_what_is_available():
    errs, _ = v.check("vibes", {})
    assert errs and "contains" in errs[0] and "entry point" in errs[0]


def test_a_third_party_validator_works_without_this_package_knowing_it():
    v.register(v.Validator(
        name="_test_rhyme", summary="does it rhyme",
        run=lambda gen, spec: v.Outcome(PASS if gen.endswith("ay") else FAIL)),
        replace=True)
    try:
        assert v.judge("_test_rhyme", "hooray").verdict == PASS
    finally:
        v._REGISTRY.pop("_test_rhyme", None)
