"""The check that catches a corpus lying about its subject.

A C# corpus once reached its token quota from one Japanese enterprise
application: 78% of files importing the same proprietary namespace, 82% line
reuse, held-out perplexity 1.33. It trained. It diverged from its neighbour at
263x the chance floor. It won its own domain by 0.43 nats. Every instrument in
the package called it a healthy expert, and when the collector was fixed and
the same language came from 243 repos the signal fell to 0.058 - most of the
evidence had been boilerplate.
"""
import json

import pytest

from ms_moe_maker.data import health as ch


def _write(tmp_path, rows, name="c.jsonl"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _one_repo(n=400, repo="zynas/bizsystem"):
    head = "using Zynas.Framework.Utility;\nusing System.Reflection;\n"
    return [{"text": head + f"class C{i} {{ void M() {{ TraceLog.Start(); }} }}\n",
             "repo": repo, "path": f"src/C{i}.cs"} for i in range(n)]


def _many_repos(n_repos=60, per=5):
    return [{"text": "\n".join(f"def f{r}_{i}_{j}(): return {j}"
                               for j in range(8)),
             "repo": f"owner/proj{r}", "path": f"p{i}.py"}
            for r in range(n_repos) for i in range(per)]


class TestInspect:

    def test_a_single_repo_corpus_is_named_as_one(self, tmp_path):
        h = ch.inspect(_write(tmp_path, _one_repo() + _many_repos(10, 2)))
        assert h.top_repo == "zynas/bizsystem"
        assert h.top_repo_share > 0.9
        assert any("one repo" in f for f in h.findings)

    def test_a_diverse_corpus_is_clean(self, tmp_path):
        h = ch.inspect(_write(tmp_path, _many_repos()))
        assert h.repos == 60
        assert h.top_repo_share < 0.05
        assert h.findings == [], h.findings

    def test_exact_duplicates_are_counted(self, tmp_path):
        rows = _many_repos(20, 2)
        h = ch.inspect(_write(tmp_path, rows + rows[:10]))
        assert h.exact_dupes == 10

    def test_a_shared_header_is_flagged_even_across_repos(self, tmp_path):
        """The tell survives a corpus assembled from many repos that all
        vendored the same scaffold."""
        # A real license block is longer than HEADER_LINES, which is what
        # makes the signature identical across files that differ below it.
        # A SHORTER shared header does not produce a matching signature - it
        # shows up in line reuse instead, which is the other half of the pair.
        head = "".join(f"// Copyright ACME line {i}\n" for i in range(14))
        rows = [{"text": head + f"class X{i} {{}}\n", "repo": f"o/p{i % 40}"}
                for i in range(200)]
        h = ch.inspect(_write(tmp_path, rows))
        assert h.top_header_share > 0.9
        assert any("identical" in f for f in h.findings)

    def test_a_corpus_without_provenance_says_so(self, tmp_path):
        """Rows collected before provenance stamping are {"text": ...} only.
        Repo dominance - the defect that mattered - becomes unmeasurable, and
        that must be stated, not silently skipped."""
        rows = [{"text": f"def f{i}(): pass\n" * 5} for i in range(50)]
        h = ch.inspect(_write(tmp_path, rows))
        assert h.has_provenance is False
        assert any("no `repo` field" in u for u in h.unmeasured)

    def test_an_empty_file_is_unmeasured_not_healthy(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        h = ch.inspect(str(p))
        assert h.docs == 0
        assert h.unmeasured
        assert h.findings == []

    def test_a_missing_file_does_not_raise(self, tmp_path):
        h = ch.inspect(str(tmp_path / "nope.jsonl"))
        assert h.docs == 0
        assert h.unmeasured


class TestGeneratedCorpora:
    """A synth corpus is templated ON PURPOSE.

    corpus.Kind carries generated=True for exactly this. Warning that
    tool-call traces "look generated" would be the check failing to know what
    it was pointed at - and the tool-calling expert is the next one to build.
    """

    def _templated(self):
        return [{"text": '{"tool": "get_time", "args": {}}\n'
                         '{"result": "12:0%d"}\n' % (i % 10), "repo": "synth"}
                for i in range(200)]

    def test_line_reuse_is_not_a_finding_for_generated_text(self, tmp_path):
        path = _write(tmp_path, self._templated())
        assert any("lines are repeats" in f
                   for f in ch.inspect(path, generated=False).findings)
        gen = ch.inspect(path, generated=True)
        assert not any("lines are repeats" in f for f in gen.findings)

    def test_the_number_is_still_reported(self, tmp_path):
        h = ch.inspect(_write(tmp_path, self._templated()), generated=True)
        assert h.line_reuse > 0.3
        assert any("expected for synthesised text" in f for f in h.findings)


class TestProposeNeverCommits:

    def test_propose_writes_nothing(self, tmp_path):
        # Two labels, because the per-repo rule is withheld on a corpus whose
        # `repo` never varies - see TestAConstantLabelIsNotProvenance. The
        # claim here is that propose_prune touches nothing, and it needs a
        # corpus the rule can actually run on to make that claim about.
        path = _write(tmp_path, _one_repo() + _one_repo(60, repo="other/x"))
        before = open(path, encoding="utf-8").read()
        p = ch.propose_prune(path, per_repo_cap=20)
        assert p.drop > 300
        assert open(path, encoding="utf-8").read() == before, (
            "propose_prune must not touch the corpus")
        assert not list(tmp_path.glob("*.pruned.jsonl"))

    def test_write_pruned_leaves_the_original_alone(self, tmp_path):
        path = _write(tmp_path, _one_repo() + _many_repos(20, 3))
        before = open(path, encoding="utf-8").read()
        out = str(tmp_path / "c.pruned.jsonl")
        p = ch.write_pruned(path, out, per_repo_cap=20)
        assert open(path, encoding="utf-8").read() == before
        kept = sum(1 for _ in open(out, encoding="utf-8"))
        assert kept == p.keep
        assert p.keep < 400

    def test_pruning_actually_fixes_the_dominance(self, tmp_path):
        path = _write(tmp_path, _one_repo() + _many_repos(40, 5))
        out = str(tmp_path / "c.pruned.jsonl")
        ch.write_pruned(path, out, per_repo_cap=20)
        assert ch.inspect(path).top_repo_share > 0.6
        assert ch.inspect(out).top_repo_share < 0.25

    def test_prune_reports_what_it_cannot_fix(self, tmp_path):
        """Without provenance the per-repo rule cannot run. Saying nothing
        would let a caller believe dominance had been handled."""
        rows = [{"text": f"x{i}\n" * 6} for i in range(100)]
        p = ch.propose_prune(_write(tmp_path, rows))
        assert any("no `repo` field" in u for u in p.unmeasured)

    def test_reasons_are_itemised_not_a_total(self, tmp_path):
        rows = _one_repo(100) + _one_repo(5, repo="other/x")
        p = ch.propose_prune(_write(tmp_path, rows + rows[:3]),
                             per_repo_cap=10)
        assert p.reasons, "a drop with no stated reason is not a proposal"
        assert sum(p.reasons.values()) == p.drop


class TestAConstantLabelIsNotProvenance:
    """`hf`, `gh` and `local` stamp ONE label on every row they write.

    _collect_hf writes the dataset id, _collect_gh the tarball's top directory,
    _collect_local the source path. Counted naively that is "100% of this
    corpus is one repo" on every corpus any of them ever produced - a finding
    that can never be cleared, on a measurement that never happened. And
    `corpus --prune` believed it: a 5,000-document corpus came out at 20.
    """

    def _constant(self, tmp_path):
        rows = [{"text": f"def f{i}():\n    return {i}\n    # {i}\n",
                 "repo": "bigcode/the-stack-smol"} for i in range(300)]
        return _write(tmp_path, rows)

    def test_one_label_is_unmeasured_not_total_dominance(self, tmp_path):
        h = ch.inspect(self._constant(tmp_path))
        assert h.top_repo_share == 0.0
        assert not h.has_provenance
        assert not any("one repo" in f for f in h.findings)
        assert any("same `repo` label" in u for u in h.unmeasured)

    def test_the_cap_is_withheld_rather_than_applied(self, tmp_path):
        path = self._constant(tmp_path)
        p = ch.propose_prune(path, per_repo_cap=20)
        assert p.keep == 300, (
            "capping a label that does not vary just truncates the corpus - "
            "20 documents kept out of 300, under every min_samples floor")
        assert any("same `repo` label" in u for u in p.unmeasured)

    def test_write_pruned_writes_the_whole_corpus(self, tmp_path):
        path = self._constant(tmp_path)
        out = str(tmp_path / "c.pruned.jsonl")
        p = ch.write_pruned(path, out, per_repo_cap=20)
        assert sum(1 for _ in open(out, encoding="utf-8")) == 300 == p.keep
        assert any("same `repo` label" in u for u in p.unmeasured)

    def test_two_labels_are_still_measured(self, tmp_path):
        """The withholding is about ONE label, not about small corpora."""
        rows = _one_repo(100) + _one_repo(10, repo="other/x")
        p = ch.propose_prune(_write(tmp_path, rows), per_repo_cap=20)
        assert p.reasons[f"over the 20/repo cap"] == 80


def test_health_is_pure_stdlib():
    """`validate` promises to run on a laptop with no torch and no network.
    A corpus check that breaks that promise is a check that only runs after
    you have already paid for the build."""
    import ast
    import pathlib
    src = pathlib.Path(ch.__file__).read_text(encoding="utf-8")
    allowed = {"hashlib", "json", "re", "collections", "dataclasses",
               "typing", "__future__"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in allowed, a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert (node.module or "").split(".")[0] in allowed, node.module
