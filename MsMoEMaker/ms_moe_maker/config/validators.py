"""Validator kinds - how an eval decides whether an answer was right.

The second registry, and the last code-locked thing in the package. The corpus
registry made it possible to BUILD a Ms.MoE about anything; this makes it
possible to GRADE one. Without it a lore model can be trained and then only
judged by a compiler, which is not a useful sentence.

────────────────────────────────────────────────────────────────────────────
THE ONE RULE, AND WHY IT IS STRUCTURAL RATHER THAN A CONVENTION

Every judgement goes through `judge()`, and `judge()` checks AVAILABILITY
before it runs anything. A validator this box cannot run returns
`unmeasurable`, with a reason. It cannot return `fail`. There is no path
through this module that turns a missing tool into a wrong answer.

That is not defensive tidiness. It is a scar, and the comment it comes from is
worth quoting because it was written in the wreckage:

    "The old check_syntax shelled out to `pwsh` and `mcs` inside a bare
     `except Exception: return False`. On a box without them that raises
     FileNotFoundError, gets swallowed, and scores a MISSING TOOLCHAIN
     exactly like a syntax error - so PowerShell reported 0/10 on a Linux
     DGX Spark and looked like a model failure. It was a package list."

    "Python is the only language that needs nothing (compile() is
     in-process), which is precisely why Python was the only one scoring."

    "A number that cannot tell 'wrong' from 'unknown' is worse than no
     number, because you will act on it."

C# scored 0/10 on a model that writes C# fine. Nothing looked broken. That is
the failure this whole viewer exists against: not being broken, being
confidently wrong.

────────────────────────────────────────────────────────────────────────────
PARSE, NEVER EXECUTE

Non-negotiable, and the reason is concrete: one of the PowerShell eval prompts
asks for a function that stops a process by name. Running an abliterated
model's output to "check" it means running process-killing code on the lab box,
as a test, on purpose.

So every syntax validator PARSES. Python uses compile(). Shell uses `bash -n`.
PowerShell uses the .NET parser against a file. C# compiles to a temp assembly
and never invokes it. And model text is always written to a TEMP FILE whose
path is passed as an argv element - never interpolated into a shell string,
because a generation is untrusted input and a shell is an interpreter.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json as _json
import os
import re as _re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..eval.record import ERROR, FAIL, PASS, UNMEASURABLE

# How long any external parser gets. A hung compiler must not hang the eval.
TOOL_TIMEOUT = 20


@dataclass(frozen=True)
class Availability:
    """Can this box run this validator, right now?"""
    ok: bool
    reason: str = ""
    tool: Optional[str] = None


@dataclass(frozen=True)
class Outcome:
    verdict: str
    reason: str = ""


@dataclass(frozen=True)
class Validator:
    """One way of deciding whether a generation was right.

    `requires` is declarative so `ms-moe-maker validate` can check a recipe on
    a laptop with no tools installed - the same laptop promise the corpus
    registry keeps. `probe` and `run` are callables because unlike corpus
    kinds, judging IS execution; this module is the execution layer.
    """
    name: str
    summary: str = ""
    requires: Tuple[str, ...] = ()
    # Any ONE of these on PATH makes it available. Empty means pure-python and
    # therefore always available - which is why the domain-neutral validators
    # below can never produce a toolchain-shaped zero.
    tools: Tuple[str, ...] = ()
    notes: str = ""
    probe: Optional[Callable[[Any], Availability]] = None
    run: Optional[Callable[[str, Any], Outcome]] = None


_REGISTRY: Dict[str, Validator] = {}
_LOAD_ERRORS: List[str] = []
_ENTRY_POINT_GROUP = "ms_moe_maker.validators"
_loaded = False


def register(validator: Validator, *, replace: bool = False) -> Validator:
    if validator.name in _REGISTRY and not replace:
        raise ValueError(
            f"validator {validator.name!r} is already registered. Pass "
            f"replace=True if you genuinely mean to redefine it - that changes "
            f"how every recipe on this machine is graded, including ones you "
            f"did not write.")
    _REGISTRY[validator.name] = validator
    return validator


def _load_entry_points() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        found = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - <3.10 signature
        found = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return
    for entry in found:
        try:
            obj = entry.load()
            for v in (obj if isinstance(obj, (list, tuple)) else [obj]):
                if isinstance(v, Validator) and v.name not in _REGISTRY:
                    _REGISTRY[v.name] = v
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not
            _LOAD_ERRORS.append(f"{entry.name}: {exc}")   # stop an eval


def get(name: str) -> Optional[Validator]:
    _load_entry_points()
    return _REGISTRY.get(name)


def names() -> List[str]:
    _load_entry_points()
    return sorted(_REGISTRY)


def all_validators() -> List[Validator]:
    _load_entry_points()
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def describe() -> List[Dict[str, object]]:
    """The registry as data, for --describe and Backstage's craft form."""
    return [{"name": v.name, "summary": v.summary,
             "requires": list(v.requires), "tools": list(v.tools),
             "notes": v.notes}
            for v in all_validators()]


def load_errors() -> List[str]:
    _load_entry_points()
    return list(_LOAD_ERRORS)


def check(name: str, spec: Any) -> Tuple[List[str], List[str]]:
    """Recipe-time schema check. No tools touched - runs on a laptop."""
    errs: List[str] = []
    warns: List[str] = []
    validator = get(name)
    if validator is None:
        errs.append(f"validator {name!r} is not registered. Available: "
                    f"{', '.join(names())}. A distribution can add more by "
                    f"publishing a {_ENTRY_POINT_GROUP} entry point.")
        return errs, warns
    for required in validator.requires:
        if _field(spec, required) in (None, "", [], {}):
            errs.append(f"validator={validator.name} needs a {required}")
    return errs, warns


def _field(spec: Any, key: str, default: Any = None) -> Any:
    """Read a field off a dataclass OR a plain dict. Eval specs arrive as both
    - from a parsed recipe and from a suite file - and making callers care
    which is how you get two code paths that drift."""
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def availability(name: str, spec: Any = None) -> Availability:
    """Can this box judge this item? The question that produces `unmeasurable`."""
    validator = get(name)
    if validator is None:
        return Availability(False, f"validator {name!r} is not registered")
    if validator.probe is not None:
        return validator.probe(spec)
    if not validator.tools:
        return Availability(True)          # pure python, always judgeable
    found = _first_tool(validator.tools)
    if found:
        return Availability(True, tool=found)
    return Availability(
        False,
        f"none of {list(validator.tools)} is on PATH, so this box cannot "
        f"judge {validator.name} items")


def judge(name: str, generation: str, spec: Any = None) -> Outcome:
    """THE FUNNEL. Every verdict comes through here, so the rule holds once.

    Unknown validator -> error (the harness is wrong).
    Unavailable       -> unmeasurable (the BOX is short a tool).
    Otherwise         -> whatever the validator decided.

    A missing tool has no path to `fail` from here, and that is the whole
    design. Callers must not call `validator.run` directly; the availability
    gate is not advisory.
    """
    validator = get(name)
    if validator is None:
        return Outcome(ERROR, f"validator {name!r} is not registered")

    avail = availability(name, spec)
    if not avail.ok:
        # UNMEASURABLE, never FAIL. See the module header.
        return Outcome(UNMEASURABLE, avail.reason)

    if validator.run is None:
        return Outcome(ERROR, f"validator {validator.name!r} declares no run()")
    try:
        return validator.run(generation, spec)
    except Exception as exc:  # noqa: BLE001
        # The HARNESS broke, which is also not the model's fault. Distinct from
        # unmeasurable so a person can tell "install a compiler" from "fix the
        # eval code" without reading the source.
        return Outcome(ERROR, f"{type(exc).__name__}: {exc}")


# ── domain-neutral validators ───────────────────────────────────────────────
# Pure python, no tools, therefore never able to produce a toolchain zero.
# These are what make a lore / study / rules Ms.MoE gradeable at all.

def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _run_exact(generation: str, spec: Any) -> Outcome:
    expected = _field(spec, "expected")
    if _norm(generation) == _norm(expected):
        return Outcome(PASS)
    return Outcome(FAIL, f"expected {str(expected)[:120]!r}")


def _run_contains(generation: str, spec: Any) -> Outcome:
    expected = _field(spec, "expected")
    wanted = expected if isinstance(expected, (list, tuple)) else [expected]
    haystack = _norm(generation)
    missing = [w for w in wanted if _norm(w) not in haystack]
    if not missing:
        return Outcome(PASS)
    return Outcome(FAIL, f"missing: {missing}")


def _run_regex(generation: str, spec: Any) -> Outcome:
    pattern = _field(spec, "pattern")
    flags = _re.IGNORECASE if _field(spec, "ignore_case", True) else 0
    if _re.search(str(pattern), generation or "", flags):
        return Outcome(PASS)
    return Outcome(FAIL, f"no match for {pattern!r}")


def _run_json(generation: str, spec: Any) -> Outcome:
    """Parses as JSON, and optionally carries the keys it was asked for.

    This is the MCP-trace validator generalised: 'did the model emit a
    well-formed structured answer' is the same question whether the structure
    is a tool call or a stat block.
    """
    text = (generation or "").strip()
    # Models fence their JSON more often than not; unwrapping it is not
    # leniency about correctness, it is refusing to fail a right answer for
    # a presentation habit the prompt did not forbid.
    fence = _re.search(r"```(?:json)?\s*(.+?)```", text, _re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = _json.loads(text)
    except ValueError as exc:
        return Outcome(FAIL, f"not valid JSON: {exc}")
    required = _field(spec, "required_keys") or []
    if isinstance(parsed, dict) and required:
        missing = [k for k in required if k not in parsed]
        if missing:
            return Outcome(FAIL, f"missing keys: {missing}")
    return Outcome(PASS)


register(Validator(
    name="exact", summary="generation equals the expected answer",
    requires=("expected",), run=_run_exact,
    notes="Whitespace-normalised and case-insensitive. Domain-neutral."))

register(Validator(
    name="contains", summary="every expected substring is present",
    requires=("expected",), run=_run_contains,
    notes="`expected` may be a string or a list. The workhorse for lore and "
          "study material, where an answer is right if it names the right "
          "things and does not have to be phrased one way."))

register(Validator(
    name="regex", summary="a pattern matches the generation",
    requires=("pattern",), run=_run_regex))

register(Validator(
    name="json", summary="parses as JSON, optionally with required keys",
    run=_run_json,
    notes="Unwraps a ``` fence first. Generalises the MCP-trace check: "
          "'did the model emit well-formed structure' is one question whether "
          "the structure is a tool call or a stat block."))


# ── syntax: the code-specific one, with the toolchain gate that started this ──
#
# The table is ported verbatim from eval_fraunkenstein.py, including the
# reasoning, because every line of it was bought.

LANG_TOOLCHAIN: Dict[str, Tuple[str, ...]] = {
    "python": (),                       # builtin - compile() is in-process
    "shell": ("bash",),
    "powershell": ("pwsh", "powershell"),
    # NO "dotnet". It was listed once and the branch for it was `return False`
    # - there is no single-file compile path. So on a box with the .NET SDK but
    # no mono, C# read as AVAILABLE, was counted in the average, and then
    # scored 0/10 without a compiler ever seeing the code. Listing a tool
    # nobody implemented is how a toolchain gap became a model result.
    "csharp": ("csc", "mcs"),
}
LANG_ALIASES = {"c#": "csharp", "cs": "csharp", "ps1": "powershell",
                "pwsh": "powershell", "bash": "shell", "sh": "shell",
                "py": "python"}


def _first_tool(candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if shutil.which(name):
            return name
    return None


def _lang(spec: Any) -> str:
    raw = str(_field(spec, "language", "") or "").strip().lower()
    return LANG_ALIASES.get(raw, raw)


def _probe_syntax(spec: Any) -> Availability:
    lang = _lang(spec)
    if not lang:
        return Availability(False, "no language given, so nothing can parse it")
    if lang not in LANG_TOOLCHAIN:
        return Availability(
            False,
            f"no syntax checker for {lang!r} on this box. Known: "
            f"{', '.join(sorted(LANG_TOOLCHAIN))}. Register your own with a "
            f"{_ENTRY_POINT_GROUP} entry point rather than scoring it zero.")
    tools = LANG_TOOLCHAIN[lang]
    if not tools:
        return Availability(True, tool="builtin")
    found = _first_tool(tools)
    if found:
        return Availability(True, tool=found)
    # The C# case, said out loud. This is a PACKAGE LIST, not a model result.
    return Availability(
        False,
        f"{lang}: none of {list(tools)} is installed, so this box cannot judge "
        f"{lang} syntax. This is a missing toolchain, not a wrong answer.")


def _strip_fence(text: str) -> str:
    fence = _re.search(r"```[a-zA-Z#+]*\s*(.+?)```", text or "", _re.S)
    return fence.group(1) if fence else (text or "")


def _parse_only(argv: List[str], suffix: str, code: str) -> Outcome:
    """Write the generation to a temp FILE and hand the parser a path.

    The generation never appears in a shell string. It is untrusted text and a
    shell is an interpreter; the only safe place for it is a file whose name we
    chose. See the PARSE, NEVER EXECUTE note in the header.
    """
    handle, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(code)
        proc = subprocess.run(argv + [path], capture_output=True, text=True,
                              timeout=TOOL_TIMEOUT)
        if proc.returncode == 0:
            return Outcome(PASS)
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return Outcome(FAIL, detail[0][:200] if detail else
                       f"exit {proc.returncode}")
    except subprocess.TimeoutExpired:
        # A hung parser is the harness failing, not the model being wrong.
        return Outcome(ERROR, f"{argv[0]} did not finish in {TOOL_TIMEOUT}s")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_syntax(generation: str, spec: Any) -> Outcome:
    lang = _lang(spec)
    code = _strip_fence(generation)

    if lang == "python":
        try:
            compile(code, "<generation>", "exec")   # PARSES. Does not run.
            return Outcome(PASS)
        except SyntaxError as exc:
            return Outcome(FAIL, f"line {exc.lineno}: {exc.msg}")

    if lang == "shell":
        # -n is 'read commands but do not execute'. That is the entire reason
        # this is safe to point at model output.
        return _parse_only(["bash", "-n"], ".sh", code)

    if lang == "powershell":
        tool = _first_tool(LANG_TOOLCHAIN["powershell"]) or "pwsh"
        # ParseFile is the .NET parser. It reads the file and builds an AST;
        # it does not invoke anything in it. The path is passed as its own
        # argv element so nothing from the generation reaches a command string.
        script = ("param($p); $e=$null; $t=$null; "
                  "[void][System.Management.Automation.Language.Parser]::"
                  "ParseFile($p,[ref]$t,[ref]$e); "
                  "if($e -and $e.Count -gt 0){ $e[0].Message; exit 1 }; exit 0")
        return _parse_only([tool, "-NoProfile", "-NonInteractive",
                            "-Command", script, "-p"], ".ps1", code)

    if lang == "csharp":
        tool = _first_tool(LANG_TOOLCHAIN["csharp"])
        out = tempfile.mktemp(suffix=".dll")
        try:
            # COMPILES to an assembly and never invokes it. Compilation is not
            # execution, which is what makes this acceptable where running the
            # output would not be.
            result = _parse_only([tool, "-target:library", f"-out:{out}"],
                                 ".cs", code)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass
        return result

    return Outcome(UNMEASURABLE, f"no parser for {lang!r}")


register(Validator(
    name="syntax",
    summary="the generation parses as valid CODE in a named language",
    requires=("language",),
    probe=_probe_syntax,
    run=_run_syntax,
    notes="The one code-specific validator. PARSES, never executes - one of "
          "the PowerShell prompts asks for a function that stops a process by "
          "name, and 'checking' it by running it would kill processes on the "
          "lab box as a test. A language whose toolchain is absent reports "
          "unmeasurable and is excluded from the score; it must never read as "
          "zero."))
