"""Reading somebody else\'s archive, which is the dangerous direction.

A bundle is a file a person was handed. Two separate hazards live in that
sentence and both get tested here:

  the container   zip entries have been writing files outside their destination
                  since the format existed - `../..`, absolute paths, and
                  symlinks, which are a write primitive whose NAME looks
                  harmless.
  the contents    a recipe names an `eval.script`, and the harness runs it with
                  the interpreter. A recipe from someone else is executable
                  content BY DESIGN. That is fine between friends and not fine
                  silently, so it is reported rather than blocked.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from ms_moe_maker.bundle import pack


def _bundle(tmp_path, name="b.zip", recipe="name: x\n", meta=None, extra=()):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(pack.RECIPE_NAME, recipe)
        if meta is not None:
            zf.writestr(pack.MANIFEST_NAME, json.dumps(meta))
        for entry, body in extra:
            zf.writestr(entry, body)
    return p


# ── the container ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry, word", [
    ("../../etc/evil", "climbs out"),
    ("..\\..\\evil", "climbs out"),
    ("/etc/evil", "absolute"),
    ("C:/Windows/evil", "absolute"),
])
def test_an_entry_that_chooses_where_it_lands_is_refused(tmp_path, entry, word):
    """REFUSING THE WHOLE ARCHIVE, not skipping the entry. A bundle containing
    one of these is not a bundle with a flaw."""
    with pytest.raises(pack.UnreadableBundle) as exc:
        pack.read(_bundle(tmp_path, extra=[(entry, "x")]))
    assert word in str(exc.value)


def test_a_symlink_entry_is_refused(tmp_path):
    p = tmp_path / "link.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(pack.RECIPE_NAME, "name: x\n")
        info = zipfile.ZipInfo("data/link")
        info.external_attr = (0xA000 | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(pack.UnreadableBundle) as exc:
        pack.read(p)
    assert "symlink" in str(exc.value)


def test_extract_writes_only_inside_the_destination(tmp_path):
    src = _bundle(tmp_path, extra=[("data/python/a.jsonl", "{}\n")])
    dest = tmp_path / "out"
    written = pack.extract(src, dest)
    assert set(written) == {pack.RECIPE_NAME, "data/python/a.jsonl"}
    for path in dest.rglob("*"):
        assert dest.resolve() in path.resolve().parents


def test_a_recipe_the_size_of_a_dataset_is_refused(tmp_path):
    big = "x" * (pack.MAX_RECIPE_BYTES + 1)
    with pytest.raises(pack.UnreadableBundle) as exc:
        pack.read(_bundle(tmp_path, recipe=big))
    assert "document" in str(exc.value)


def test_a_zip_with_no_recipe_is_an_archive_not_a_bundle(tmp_path):
    p = tmp_path / "plain.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("something.txt", "hello")
    with pytest.raises(pack.UnreadableBundle):
        pack.read(p)


def test_a_newer_schema_refuses_rather_than_guessing(tmp_path):
    p = _bundle(tmp_path, meta={"schema_version": pack.SCHEMA_VERSION + 1})
    with pytest.raises(pack.UnreadableBundle) as exc:
        pack.read(p)
    assert "Upgrade" in str(exc.value)


# ── the contents ─────────────────────────────────────────────────────────────

def test_a_recipe_that_runs_a_script_says_so(tmp_path):
    """REPORTED, NEVER BLOCKED. `eval.script` is a documented feature; the
    failure being prevented is a person clicking Import without being told,
    not a person choosing to trust a friend."""
    got = pack.read(_bundle(
        tmp_path, recipe="name: x\neval:\n  script: ./their_grader.py\n"))
    assert got["executes"], "an executable field reached a reader silently"
    assert "their_grader.py" in got["executes"][0]


def test_an_ordinary_recipe_reports_nothing_to_execute(tmp_path):
    assert pack.read(_bundle(tmp_path))["executes"] == []


# ── degrading honestly ───────────────────────────────────────────────────────

def test_a_hand_rolled_zip_without_a_manifest_still_opens(tmp_path):
    """A zip with a recipe in it is a perfectly good thing to make by hand.
    Refusing it would turn the format into a gate instead of a convenience -
    but the reader is told there is no claim to compare against."""
    got = pack.read(_bundle(tmp_path))
    assert got["meta"] == {}
    assert "makes no claim" in got["meta_error"]
    assert got["recipe"].startswith("name: x")


def test_a_corrupt_manifest_does_not_lose_the_recipe(tmp_path):
    p = tmp_path / "b.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(pack.RECIPE_NAME, "name: x\n")
        zf.writestr(pack.MANIFEST_NAME, "{not json")
    got = pack.read(p)
    assert "not valid JSON" in got["meta_error"]
    assert got["recipe"].startswith("name: x"), (
        "the recipe is the point; a bad manifest must not take it down")


def test_write_then_read_round_trips(tmp_path):
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "a.jsonl").write_text('{"text":"hi"}\n', encoding="utf-8")
    out = pack.write(tmp_path / "out.zip", recipe_text="name: x\n",
                     meta={"name": "x", "build_id": "abc"},
                     data_dirs=[("python", data)], notes="# notes\n")
    got = pack.read(out)
    assert got["meta"]["build_id"] == "abc"
    assert got["meta"]["recipe_sha256"]
    assert got["data_experts"] == ["python"]
    assert got["notes"].startswith("# notes")
    assert got["meta"]["data_bytes"] > 0
