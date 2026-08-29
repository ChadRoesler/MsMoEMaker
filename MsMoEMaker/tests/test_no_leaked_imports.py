"""A module imported LOCALLY in one function and read as a GLOBAL in another.

Python does not complain until the line executes, so this hides in any branch
that a test does not take — and it hid in three places at once:

  stitch.py   `import moe_stitch` inside stitch_moe(), used in _stream_stitch()
              -> NameError, uncaught by the `except ImportError` beside it, so
                 the streaming path could not run even with the module present.

  data.py     `from . import config as cfg` inside collect_corpus() and three
              others, used in _collect_from_shards() which imported nothing
              -> killed the first real Spark build at stage 1, and the failure
                 arrived as a bare "name 'cfg' is not defined".

  data.py     AutoTokenizer imported in _HFTeacher.__init__, used in
              _VLLMTeacher.__init__ -> waiting for the first synth expert on
              the vLLM path.

Two were found by hand, one at a time, after each had cost something. This test
is the sweep, so the third kind never gets its turn.

WHY THE FUNCTION IS THE SCOPE UNIT. A nested def legitimately closes over names
bound in the function around it — `kl()` inside `_js_divergence()` reads the
`math` imported there and is perfectly correct. Treating a whole top-level
function subtree as one scope models that, and errs toward false NEGATIVES,
which is the right direction for a lint: a check that cries wolf gets disabled.
"""
import ast
import builtins
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "ms_moe_maker"
BUILTIN = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "annotations"}


def _module_level_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, ast.Try):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Import):
                    for a in sub.names:
                        out.add(a.asname or a.name.split(".")[0])
                elif isinstance(sub, ast.ImportFrom):
                    for a in sub.names:
                        out.add(a.asname or a.name)
    return out


def _bound_anywhere(node):
    """Every name bound anywhere in this subtree — the scope unit."""
    b = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                b.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                b.add(a.asname or a.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            b.add(n.id)
        elif isinstance(n, ast.arg):
            b.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            b.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            b.update(n.names)
    return b


def _top_level_scopes(tree):
    """Top-level functions, plus methods, as independent scope units."""
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n.name, n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{n.name}.{m.name}", m


def _leaks(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = _module_level_names(tree) | BUILTIN

    # Names that SOME scope imports locally: the candidates.
    candidates = set()
    for _, fn in _top_level_scopes(tree):
        for n in ast.walk(fn):
            if isinstance(n, ast.Import):
                for a in n.names:
                    candidates.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    candidates.add(a.asname or a.name)
    candidates -= top

    found = []
    for label, fn in _top_level_scopes(tree):
        bound = _bound_anywhere(fn)
        for n in ast.walk(fn):
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id in candidates and n.id not in bound):
                found.append((path.name, n.lineno, label, n.id))
    return sorted(set(found))


MODULES = sorted(p for p in PKG.rglob("*.py") if "heretic" not in p.parts)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_function_local_import_is_read_from_another_function(path):
    leaks = _leaks(path)
    assert not leaks, "\n".join(
        f"  {f}:{line}  {scope}() reads {name!r}, which is only imported "
        f"inside a different function"
        for f, line, scope, name in leaks)


def test_the_detector_catches_the_real_thing(tmp_path):
    """A lint nobody has seen fail is a lint nobody should trust."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def a():\n"
        "    import json\n"
        "    return json.dumps({})\n"
        "\n"
        "def b():\n"
        "    return json.dumps({})\n",
        encoding="utf-8")
    leaks = _leaks(bad)
    assert leaks and leaks[0][3] == "json", leaks


def test_the_detector_allows_a_closure(tmp_path):
    """kl() reading the math imported by its enclosing function is correct."""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "def outer():\n"
        "    import math\n"
        "    def inner(x):\n"
        "        return math.log2(x)\n"
        "    return inner(8)\n",
        encoding="utf-8")
    assert _leaks(ok) == []


def test_the_detector_allows_a_module_level_import(tmp_path):
    ok = tmp_path / "ok2.py"
    ok.write_text(
        "import json\n"
        "def a():\n"
        "    import json\n"
        "    return json\n"
        "def b():\n"
        "    return json.dumps({})\n",
        encoding="utf-8")
    assert _leaks(ok) == []
