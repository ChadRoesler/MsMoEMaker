"""Every knob a recipe can set must be read by something.

THE FAMILY THIS FILE EXISTS FOR. Four times now a field has been declared on
the recipe schema, validated by the parser, documented in the README — and
never actually consumed, with a hardcoded literal sitting where its value
belonged:

    corpus.max_shards      hardcoded at 80
    source.teacher         "declared and validated for the whole life of the
                            synth pipeline and then never read"
    --dryrun               applied after the values it was meant to influence
    budget.warmup_floor    `max(10, ...)` where `max(b.warmup_floor, ...)` goes

Every one of them is the same failure and it is the quiet kind: the recipe
parses, the run starts, and the setting does nothing. As the Corpus docstring
already puts it — silently ignoring a setting is worse than rejecting it,
because the run then fails for the reason the setting was there to prevent.

Nothing could catch that, because "accepted by the parser" and "used by the
pipeline" were never checked against each other. This checks them.

DELIBERATELY LENIENT. A bare word-boundary match anywhere in the package counts
as a read, so a field consumed through `getattr(src, "teacher")`, an f-string,
or a yaml key still passes. That means false NEGATIVES are possible and false
POSITIVES are not: anything this test flags is genuinely unreferenced. A guard
that cries wolf gets an allowlist entry and then gets ignored, so this one is
built to only ever be right when it complains.
"""
import ast
import pathlib
import re

import ms_moe_maker

PKG = pathlib.Path(ms_moe_maker.__file__).resolve().parent

# Fields that are legitimately unread by the package. Add ONLY with a reason,
# and prefer wiring the field up to adding it here — an entry is a promise that
# a knob doing nothing is intended.
ALLOWED_UNREAD: dict = {
    # "Block.field": "why it is legitimately not consumed",
    "Recipe.tools_expert": (
        "consumed by parse() itself when injecting the tools expert - it "
        "turns the flag into a real expert on the list, and nothing "
        "downstream reads the raw flag; tools_expert_name is the derived "
        "value the pipeline reads"),
}


def _schema_fields():
    tree = ast.parse((PKG / "config" / "recipe.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    yield node.name, stmt.target.id, stmt.lineno


def _consumers():
    """Everything in the package that can READ a recipe object.

    recipe.py itself is INCLUDED: it holds validate(), which reads rec.gates.*,
    rec.schema_version and friends. The old bare-identifier matcher had to
    exclude the file because field DECLARATIONS matched; the recipe-object
    pattern does not match declarations, so the file is back in - and a field
    read only by the parser's own validation finally counts as read.
    """
    out = {}
    for path in PKG.rglob("*.py"):
        out[str(path.relative_to(PKG))] = path.read_text(
            encoding="utf-8", errors="replace")
    for path in PKG.rglob("*.yaml"):
        out[str(path.relative_to(PKG))] = path.read_text(
            encoding="utf-8", errors="replace")
    return out


def _read_pattern(field: str) -> re.Pattern:
    """A READ of a recipe field, not a bare mention.

    The old check matched the bare identifier anywhere, so a field mentioned
    only in PipelineConfig's own definition, a KNOWN_FIELDS string tuple, or a
    docstring "passed" the guard while being unreachable from a recipe -
    Roots.data, MoE.shared_expert_gate_fill, Runtime.load_in_4bit,
    Source.max_shards, Source.examples, Source.generator, Expert.tokens and
    SmokeSpec.script all sailed through it. ALLOWED_UNREAD was empty, which
    read as proof the class was closed. It wasn't.

    A read now has to go THROUGH a recipe-shaped object:

        rec.moe.shared_expert_gate_fill        recipe.<block>.<field>
        getattr(recipe.budget, "warmup_floor", ...)
        src.max_shards                         source.<field>
        e.tokens                               expert.<field>

    Hardcoding a value in build_config as `PipelineConfig(...) field=0.02` or
    reading `config.field` downstream no longer counts: the recipe's field must
    be the thing being read.
    """
    name = re.escape(field)
    block = r"(?:rec|recipe|b|rt|bud|moe|_corpus|_abl|roots|gates)"
    leaf = r"(?:src|source|expert)"
    ident = r"[A-Za-z_]\w*"
    return re.compile(
        rf"\b{block}\s*\.\s*(?:{ident}\s*\.\s*)*{name}\b"
        rf"|\bgetattr\(\s*{block}\s*(?:\.\s*{ident})?\s*,\s*[\"']{name}[\"']"
        rf"|\b_corpus_knob\(\s*[\"']{name}[\"']"
        rf"|\b{leaf}\s*\.\s*(?:{ident}\s*\.\s*)*{name}\b"
        rf"|\bgetattr\(\s*{leaf}\s*,\s*[\"']{name}[\"']"
        rf"|\be\s*\.\s*{name}\b")


def test_every_recipe_field_is_read_by_something():
    consumers = _consumers()
    assert consumers, "found no package sources to scan — the finder is broken"

    unread = []
    for cls, field, line in _schema_fields():
        if f"{cls}.{field}" in ALLOWED_UNREAD:
            continue
        pat = _read_pattern(field)
        if not any(pat.search(text) for text in consumers.values()):
            unread.append(f"{cls}.{field}  (recipe.py:{line})")

    assert not unread, (
        "these recipe fields are accepted by the parser and read by nothing, "
        "so a recipe setting them is silently ignored:\n  "
        + "\n  ".join(unread)
        + "\n\nWire the field up, or add it to ALLOWED_UNREAD with a reason.")


def test_the_finder_actually_finds_fields():
    """Anti-silence. If the AST walk stops matching, the test above passes by
    finding nothing to check — which is exactly how a guard goes to sleep."""
    fields = list(_schema_fields())
    assert len(fields) > 50, (
        f"only {len(fields)} schema fields found; the parser or the schema "
        f"layout moved and this guard is no longer guarding anything")
    names = {f"{c}.{f}" for c, f, _ in fields}
    for expected in ("Budget.target_steps", "Corpus.max_samples",
                     "MoE.experts_per_tok"):
        assert expected in names, f"{expected} not found by the field finder"


def test_the_read_pattern_rejects_a_bare_mention():
    """A lint nobody has seen fail is a lint nobody should trust.

    The trio that let twelve dead knobs through: the field is DEFINED, its
    name sits in a KNOWN_FIELDS string tuple, and the only reads are off a
    non-recipe object (`config.X`). None of that may count as a read."""
    consumers = {
        "config/pipeline.py": "KNOWN_FIELDS = ('max_shards',)\n"
                              "shared_expert_gate_fill=0.02\n",
        "moe/stitch.py": "config.shared_expert_gate_fill\n"
                         "x = cfg.max_shards\n",
    }
    for field in ("max_shards", "shared_expert_gate_fill"):
        pat = _read_pattern(field)
        assert not any(pat.search(t) for t in consumers.values()), (
            f"{field}: a bare mention must not count as a read")


def test_the_read_pattern_accepts_a_recipe_object_read():
    pat = _read_pattern("max_shards")
    assert any(pat.search(t) for t in [
        "src = e.source\nif getattr(src, 'max_shards', 0): ...",
        "cap = e.source.max_shards",
        "x = rec.moe.shared_expert_gate_fill",
        "getattr(rt, 'direct_load', False)",
        "roots = resolve_roots(size, dryrun, recipe.roots)",
    ])


def test_the_allowlist_has_no_stale_entries():
    """An allowlist that outlives its field is how this guard goes to sleep.

    doc_ceiling was removed from the schema the same night it was allowlisted.
    Left behind, that entry would silently excuse any FUTURE field that
    happened to reuse the name — the exact shape of the staleness bug this
    file exists to catch, reproduced inside the catcher.
    """
    live = {f"{cls}.{field}" for cls, field, _ in _schema_fields()}
    stale = sorted(set(ALLOWED_UNREAD) - live)
    assert not stale, (
        "ALLOWED_UNREAD names fields that no longer exist on the schema; "
        "delete these entries so they cannot excuse a future field that "
        f"reuses the name: {stale}")
