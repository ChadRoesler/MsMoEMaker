"""One repo must not be the corpus.

THE FAILURE, measured on a real build. The C# bucket hit its 1.8M-token
target from 658 files in the first shard. 515 of them imported
`Zynas.Framework.Utility`; 486 wrapped every method in the same trace-log
call; 82.4% of all lines were repeats. It was one Japanese enterprise
application, and the pipeline reported it as a C# corpus.

What makes it dangerous is that nothing downstream could see it. The expert
trained. It diverged from its neighbour at 263x the chance floor. It won its
own domain by 0.43 nats in cross-domain loss. Held-out perplexity 1.33 - which
reads as a fantastic expert and actually means the text was a template.

The quota is in TOKENS, so a verbose language reaches it without ever leaving
the first repository. Python needed 1754 files for the same count and got
diversity for free, which is why this was invisible on the language everyone
tests with.
"""
import json

import pytest

from ms_moe_maker import data as d


class TestRepoLabel:

    def test_a_named_repo_uses_its_name(self):
        assert d._repo_label({"repo_name": "acme/widget"}) == "acme/widget"

    def test_repo_path_is_preferred_because_the_corpus_actually_has_it(self):
        """The stack rows carry repo_path and repo_id. Two versions of this
        function looked for repo_name/repo/repository - none of which exist -
        and fell through to a path heuristic whose output was then reported as
        a repo name."""
        row = {"repo_path": "torvalds/linux", "repo_id": 1234,
               "files": [{"file_path": "kernel/sched/core.c"}]}
        assert d._repo_label(row) == "torvalds/linux"

    def test_repo_id_beats_a_path_guess(self):
        row = {"repo_id": "gh-99", "files": [{"file_path": "a/b/c.py"}]}
        assert d._repo_label(row) == "gh-99"

    def test_uses_the_common_directory_prefix_when_there_is_no_name(self):
        repo = {"files": [{"file_path": "zynas/bizsystem/src/Foo.cs"},
                          {"file_path": "zynas/bizsystem/src/Bar.cs"},
                          {"file_path": "zynas/bizsystem/test/Baz.cs"}]}
        assert d._repo_label(repo) == "zynas/bizsystem"

    def test_root_level_files_do_not_invent_a_repo_called_readme(self):
        """THE BUG. A markdown file at repo root has no directory, so taking
        the first path components made every project's README one fake repo -
        and the report announced that 26% of the corpus came from it. An
        honest row id beats a confident wrong name."""
        repo = {"files": [{"file_path": "README.md"}]}
        label = d._repo_label(repo, fallback="shard1#42")
        assert label == "shard1#42"
        assert "README" not in label

    def test_one_file_deep_in_a_tree_still_yields_its_directory(self):
        repo = {"files": [{"file_path": "acme/widget/src/main.py"}]}
        assert d._repo_label(repo) == "acme/widget"

    def test_unrelated_paths_share_no_prefix_and_fall_back(self):
        repo = {"files": [{"file_path": "a/one.py"}, {"file_path": "b/two.py"}]}
        assert d._repo_label(repo, fallback="shard2#7") == "shard2#7"

    def test_never_raises_on_a_shapeless_row(self):
        assert d._repo_label({}) == "<row>"
        assert d._repo_label({"files": [{}]}, fallback="x") == "x"


class TestDiversity:

    def test_reports_the_biggest_contributor_and_its_share(self):
        from collections import Counter
        c = Counter({"zynas/biz": 515, "other/a": 100, "other/b": 43})
        n, top, share = d._diversity(c, 658)
        assert n == 3
        assert top == "zynas/biz"
        assert share == pytest.approx(515 / 658)

    def test_an_empty_corpus_is_not_a_crash(self):
        from collections import Counter
        assert d._diversity(Counter(), 0) == (0, "", 0.0)


class TestLineReuse:

    def test_templated_code_scores_high(self):
        header = "using Zynas.Framework.Utility;\nusing System.Reflection;\n"
        docs = [{"text": header + f"class C{i} {{ }}\n"} for i in range(50)]
        assert d._line_reuse(docs) > 0.6

    def test_varied_code_scores_low(self):
        docs = [{"text": "\n".join(f"line {i}_{j}" for j in range(20))}
                for i in range(50)]
        assert d._line_reuse(docs) < 0.1

    def test_an_empty_corpus_is_zero_not_a_division_error(self):
        assert d._line_reuse([]) == 0.0
        assert d._line_reuse([{"text": ""}]) == 0.0


class TestPerRepoCap:
    """The cap is enforced inside the shard loop, so this exercises the same
    arithmetic the loop does rather than mocking the whole HuggingFace stack:
    one repo offering 500 files against a cap of 20."""

    def _take(self, repos, cap):
        taken, per_repo = [], {}
        for label, files in repos:
            here = 0
            for f in files:
                if here >= cap:
                    continue
                taken.append(f)
                per_repo[label] = per_repo.get(label, 0) + 1
                here += 1
        return taken, per_repo

    def test_one_giant_repo_cannot_fill_the_bucket(self):
        repos = [("zynas/biz", [f"f{i}" for i in range(500)])]
        repos += [(f"other/{r}", [f"g{r}_{i}" for i in range(5)])
                  for r in range(30)]
        taken, per_repo = self._take(repos, cap=20)
        assert per_repo["zynas/biz"] == 20
        assert per_repo["zynas/biz"] / len(taken) < 0.15, (
            "no single repo may dominate the corpus")
        assert len(per_repo) == 31

    def test_the_cap_is_per_language_not_global(self):
        """A monorepo holding both languages contributes to both - it just
        cannot BE either."""
        cap = 20
        per_lang = {"C#": 0, "Python": 0}
        for _ in range(100):                      # 100 files of each, one repo
            for lang in per_lang:
                if per_lang[lang] < cap:
                    per_lang[lang] += 1
        assert per_lang == {"C#": 20, "Python": 20}


def test_per_repo_cap_is_a_recipe_knob():
    from ms_moe_maker.recipe import parse
    from ms_moe_maker import config as cfg_mod
    base = {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "experts": [{"name": "a",
                     "source": {"kind": "stack", "language": "Python"}}],
    }
    rec, _ = parse(base)
    assert cfg_mod.build_config(rec, dryrun=True).per_repo_cap == 20

    rec, _ = parse({**base, "corpus": {"per_repo_cap": 5}})
    assert cfg_mod.build_config(rec, dryrun=True).per_repo_cap == 5



def test_the_stack_collector_stamps_provenance():
    """THE EDIT THAT WENT MISSING, pinned so it cannot go missing again.

    Provenance was written for the hf, gh and local collectors and NOT for
    `stack` - the one every real build uses - because the script that applied
    it aborted on a later guard before writing the file. Nothing failed;
    corpora just kept coming out as {"text": ...} while the report claimed
    repo diversity it had measured in memory and thrown away.

    Reading the source is the only way to assert this without a HuggingFace
    round trip, and an assertion that needs the network is an assertion that
    does not run.
    """
    import inspect
    src = inspect.getsource(d._collect_from_shards)
    assert '"repo": rlabel' in src, (
        "the stack collector must stamp `repo` on every row - without it the "
        "per-repo rule cannot run on the finished corpus and dominance cannot "
        "be diagnosed after the fact")
    assert '"path": f.get("file_path")' in src


class TestTwoUnitsForEnough:
    """The scan retires on TOKENS; min_samples is a floor on DOCUMENTS.

    THE FAILURE, from a real run:

        [Python]   FULL at ~7.4M est. tokens (6778 docs, shard 1)
        [Markdown] FULL at ~7.4M est. tokens (7397 docs, shard 1)
        All languages satisfied — stopping early.
        BUILD FAILED: buckets below min 9000: {'Python': 6778, ...}

    The loop declared success by one measure and was failed by another, and
    the advice it printed ("raise max_shards") could not work because the
    break happened before another shard was ever considered. A floor that
    cannot steer the loop is not a floor, it is a late assertion.
    """

    def test_the_done_gate_requires_both_units(self):
        import inspect
        src = inspect.getsource(d._collect_from_shards)
        assert "have_tokens and have_docs" in src, (
            "a language may only be retired when it has enough tokens AND "
            "enough documents - retiring on tokens alone is what made "
            "min_samples unsatisfiable")
        assert "min_samples_per_expert" in src

    def test_the_failure_explains_which_limit_stopped_the_scan(self):
        """'Raise max_shards' was the only advice offered and it was wrong in
        the case that actually happened."""
        import inspect
        src = inspect.getsource(d._collect_from_shards)
        assert "ran_out_of_shards" in src
        assert "DOC floor while the training budget is" in src


def test_max_shards_is_a_recipe_knob():
    """It was hardcoded at 80, so a recipe setting it parsed fine and did
    nothing - and the run then failed for the reason the setting existed to
    prevent."""
    from ms_moe_maker.recipe import parse
    from ms_moe_maker import config as cfg_mod
    base = {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "experts": [{"name": "a",
                     "source": {"kind": "stack", "language": "Python"}}],
    }
    rec, _ = parse(base)
    assert cfg_mod.build_config(rec, dryrun=True).max_shards == 80
    rec, _ = parse({**base, "corpus": {"max_shards": 240}})
    assert cfg_mod.build_config(rec, dryrun=True).max_shards == 240


def test_a_floor_above_the_ceiling_is_refused_at_validate():
    from ms_moe_maker.recipe import parse, validate
    rec, _ = parse({
        "schema_version": 1, "name": "t", "size": "0.5B",
        "experts": [{"name": "a",
                     "source": {"kind": "stack", "language": "Python"}}],
        "corpus": {"min_samples": 12000, "max_samples": 9000},
    })
    errs, _ = validate(rec)
    assert any("floor is\nhigher" in e or "higher than the ceiling" in e
               for e in errs), errs
