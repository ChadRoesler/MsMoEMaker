"""Tests for the `gh` corpus kind.

The HTTP half cannot be exercised in every environment (a sandbox or a
corporate network may block codeload entirely), which is exactly why the
download and the parsing are separate functions. Everything below builds a
tarball shaped the way codeload shapes one — a single `<name>-<sha>/` top
directory — and runs the real parser over it. No network, no skips.
"""
import io
import json
import tarfile
import types

import pytest

from ms_moe_maker import corpus
from ms_moe_maker.data import _corpus_from_tarball, _collect_gh
from ms_moe_maker.recipe import parse, validate


def _cfg(**over):
    base = dict(chars_per_token_est=3.2, collect_token_target=0,
                num_code_samples=10_000, min_samples_per_expert=1)
    base.update(over)
    return types.SimpleNamespace(**base)


def _tarball(files, prefix="repo-abc123"):
    """Build a codeload-shaped .tar.gz in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, body in files.items():
            data = body.encode("utf-8")
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


BODY = "line one\nline two\nline three\nline four\n"


class TestRegistration:
    def test_gh_is_a_registered_kind(self):
        assert "gh" in corpus.names()

    def test_it_requires_a_repo(self):
        assert corpus.get("gh").requires == ("repo",)

    def test_it_is_not_a_generated_kind(self):
        """Generated kinds get a data.synth stage; gh is fetched, not made."""
        assert corpus.get("gh").generated is False

    def test_a_gh_recipe_validates_on_a_laptop(self):
        """No network, no torch — the laptop promise."""
        rec, warns = parse({
            "schema_version": 1, "name": "docs",
            "experts": [
                {"name": "a", "source": {"kind": "gh", "repo": "o/r",
                                         "glob": "docs/**/*.md"}},
                {"name": "b", "source": {"kind": "gh", "repo": "o/r2"}},
            ],
        })
        errs, _ = validate(rec)
        assert errs == [], errs
        assert rec.experts[0].source.glob == "docs/**/*.md"

    def test_a_gh_source_without_a_repo_is_refused(self):
        rec, _ = parse({
            "schema_version": 1, "name": "docs",
            "experts": [
                {"name": "a", "source": {"kind": "gh"}},
                {"name": "b", "source": {"kind": "gh", "repo": "o/r"}},
            ],
        })
        errs, _ = validate(rec)
        assert any("repo" in e for e in errs), errs


class TestTarballParsing:
    def test_the_sha_prefix_is_stripped(self):
        """codeload's top directory changes with every commit. A recipe's glob
        is written against the repo, so the prefix must not reach it."""
        tar = _tarball({"docs/a.md": BODY}, prefix="llama.cpp-9f8e7d6")
        rows = _corpus_from_tarball(tar, "docs/**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_glob_is_relative_to_the_repo_root(self):
        tar = _tarball({"docs/a.md": BODY, "src/b.md": BODY})
        rows = _corpus_from_tarball(tar, "docs/**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_subdir_narrows_and_reroots_the_glob(self):
        # Distinct bodies on purpose: identical content is deduped by hash, so
        # reusing BODY here would hide the very thing being measured.
        tar = _tarball({"docs/guide/a.md": BODY + "a",
                        "docs/b.md": BODY + "b",
                        "other/c.md": BODY + "c"})
        rows = _corpus_from_tarball(tar, "**/*.md", "docs", _cfg())
        assert len(rows) == 2

    def test_double_star_slash_matches_zero_directories(self):
        """`docs/**/*.md` means markdown ANYWHERE under docs, including
        directly in it. fnmatch got this wrong and it looked like an empty
        repo."""
        tar = _tarball({"docs/README.md": BODY + "1",
                        "docs/a/b/deep.md": BODY + "2"})
        rows = _corpus_from_tarball(tar, "docs/**/*.md", None, _cfg())
        assert len(rows) == 2

    def test_single_star_does_not_cross_a_separator(self):
        tar = _tarball({"docs/a.md": BODY + "1", "docs/deep/b.md": BODY + "2"})
        rows = _corpus_from_tarball(tar, "docs/*.md", None, _cfg())
        assert len(rows) == 1

    def test_non_matching_extensions_are_skipped(self):
        tar = _tarball({"docs/a.md": BODY, "docs/b.png": BODY})
        rows = _corpus_from_tarball(tar, "**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_identical_files_are_deduped(self):
        tar = _tarball({"a.md": BODY, "copy/a.md": BODY})
        rows = _corpus_from_tarball(tar, "**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_tiny_files_are_dropped(self):
        tar = _tarball({"a.md": "hi\n", "b.md": BODY})
        rows = _corpus_from_tarball(tar, "**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_directories_are_not_rows(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            d = tarfile.TarInfo("repo-abc/docs")
            d.type = tarfile.DIRTYPE
            tf.addfile(d)
            data = BODY.encode()
            f = tarfile.TarInfo("repo-abc/docs/a.md")
            f.size = len(data)
            tf.addfile(f, io.BytesIO(data))
        rows = _corpus_from_tarball(buf.getvalue(), "docs/**/*.md", None, _cfg())
        assert len(rows) == 1

    def test_sample_cap_is_honoured(self):
        tar = _tarball({f"d/{i}.md": BODY + str(i) for i in range(20)})
        rows = _corpus_from_tarball(tar, "d/**/*.md", None,
                                    _cfg(num_code_samples=5))
        assert len(rows) == 5

    def test_garbage_is_not_a_crash(self):
        assert _corpus_from_tarball(b"not a tarball", "**/*", None, _cfg()) == []


class TestCollectGh:
    def test_a_repo_without_a_slash_is_refused_before_any_request(self, tmp_path):
        out = tmp_path / "o.jsonl"
        assert _collect_gh("notaslug", "**/*.md", None, None, str(out),
                           _cfg(), None, "x") is None
        assert not out.exists()

    def test_too_few_samples_is_a_failure_not_an_empty_corpus(
            self, tmp_path, monkeypatch):
        """An under-filled corpus must not be written and called done — the
        next stage would train an expert on almost nothing and report success.
        """
        import ms_moe_maker.data as data
        monkeypatch.setattr(data, "_fetch_gh_tarball",
                            lambda repo, ref: (_tarball({"a.md": BODY}), "HEAD"))
        out = tmp_path / "o.jsonl"
        got = data._collect_gh("o/r", "**/*.md", None, None, str(out),
                               _cfg(min_samples_per_expert=50), None, "x")
        assert got is None
        assert not out.exists()

    def test_a_good_fetch_writes_jsonl_rows(self, tmp_path, monkeypatch):
        import ms_moe_maker.data as data
        files = {f"docs/{i}.md": BODY + str(i) for i in range(4)}
        monkeypatch.setattr(data, "_fetch_gh_tarball",
                            lambda repo, ref: (_tarball(files), "main"))
        out = tmp_path / "o.jsonl"
        got = data._collect_gh("o/r", "docs/**/*.md", "main", None, str(out),
                               _cfg(), None, "x")
        assert got == str(out)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 4
        assert all("text" in r for r in rows)

    def test_a_failed_fetch_writes_nothing(self, tmp_path, monkeypatch):
        import ms_moe_maker.data as data
        monkeypatch.setattr(data, "_fetch_gh_tarball", lambda repo, ref: (None, ""))
        out = tmp_path / "o.jsonl"
        assert data._collect_gh("o/r", "**/*.md", None, None, str(out),
                                _cfg(), None, "x") is None
        assert not out.exists()
