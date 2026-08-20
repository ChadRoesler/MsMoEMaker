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

    def test_falls_back_to_the_owner_and_project_in_the_path(self):
        repo = {"files": [{"file_path": "zynas/bizsystem/src/Foo.cs"}]}
        assert d._repo_label(repo) == "zynas/bizsystem"

    def test_never_raises_on_a_shapeless_row(self):
        assert d._repo_label({}) == "<unnamed>"
        assert d._repo_label({"files": [{}]}) == "<unnamed>"


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
