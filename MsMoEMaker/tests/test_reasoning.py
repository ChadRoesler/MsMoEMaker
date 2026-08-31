"""The reasoning table: a file, layered, merged by name.

A wrong tag style is a SILENT WRONG ANSWER — the splitter finds no delimiters,
reports "did not reason", and the whole think block gets scored as the answer.
That is why this table is data you can drop on a box rather than a dict you
wait for a release to change, and why the tests below care as much about the
loud paths as the working ones.
"""
import pytest

from ms_moe_maker.config import pipeline as C
from ms_moe_maker.config import reasoning as Rz


def _w(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestPackagedTable:
    def test_it_ships_and_parses_clean(self):
        styles, families, warns = Rz.load(include_user=False)
        assert warns == [], warns
        assert {"xml", "agentic_xml", "markdown", "llama"} <= set(styles)
        assert families, "the packaged table has no families"

    def test_the_floor_survives_a_missing_table(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Rz._defaults, "packaged_path",
                            lambda name: str(tmp_path / "gone.yaml"))
        monkeypatch.setenv(Rz.USER_ENV, str(tmp_path / "also-gone.yaml"))
        styles, families, warns = Rz.load()
        assert "xml" in styles, "a missing table must not take a build down"
        assert styles["xml"].open == "<think>"
        assert warns == []

    def test_the_floor_does_not_contradict_the_shipped_table(self):
        styles, _, _ = Rz.load(include_user=False)
        for key, floor in Rz.FLOOR_STYLES.items():
            assert styles[key] == floor, (
                f"{key}: floor and packaged reasoning.yaml disagree")


class TestSniffing:
    @pytest.mark.parametrize("model,expected", [
        ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
        # R1-Distill-Qwen is Qwen fine-tuned on R1 traces: it emits <think>,
        # NOT the DeepSeek-native <|reasoning|>. The table's longer
        # "R1-Distill-Qwen" hint must win over the DeepSeek family's "r1".
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "xml"),
        ("moonshotai/Kimi-K2-Instruct", "agentic_xml"),
        ("Qwen/QwQ-32B", "xml"),
        # a plain coder model reasons about nothing
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct", ""),
    ])
    def test_ids_land_on_the_right_family(self, model, expected):
        s, f, _ = Rz.load(include_user=False)
        assert Rz.style_for_base(model, "auto", s, f) == expected

    def test_punctuation_is_not_a_trivia_quiz(self):
        """The table is written the way a vendor writes a model card; ids come
        from a hub. Requiring those to agree on spaces and hyphens makes the
        table impossible to write by hand."""
        s, f, _ = Rz.load(include_user=False)
        assert Rz.style_for_base("meta-llama/Llama-3.1-8B", "auto", s, f) == "llama"
        assert Rz.style_for_base("Llama_3.1", "auto", s, f) == "llama"

    def test_nonreasoning_short_circuits(self):
        s, f, _ = Rz.load(include_user=False)
        assert Rz.style_for_base("Qwen/QwQ-32B", "nonreasoning", s, f) == ""

    def test_reasoning_falls_back_to_plain_xml(self):
        s, f, _ = Rz.load(include_user=False)
        assert Rz.style_for_base("some/unknown-thinker", "reasoning", s, f) == "xml"

    def test_the_longest_hint_wins_so_order_does_not_matter(self, tmp_path):
        """A table somebody else edits must not change meaning because of where
        they put their entry."""
        box = _w(tmp_path, "r.yaml",
                 "TagStyles:\n"
                 "  - Key: special\n    TagStyleName: Special\n"
                 "    OpeningTag: '<t>'\n    ClosingTag: '</t>'\n"
                 "Families:\n"
                 "  - Key: broad\n    FamilyName: Broad\n"
                 "    Models: [qwen]\n    PreferredStyle: xml\n"
                 "  - Key: narrow\n    FamilyName: Narrow\n"
                 "    Models: [Qwen3-Coder-Special]\n    PreferredStyle: special\n")
        s, f, _ = Rz.load(box, include_user=False)
        assert Rz.style_for_base("Qwen/Qwen3-Coder-Special-7B", "auto", s, f) \
            == "special"


class TestLayering:
    def test_a_box_adds_a_family_without_losing_the_others(self, tmp_path):
        box = _w(tmp_path, "r.yaml",
                 "Families:\n  - Key: acme\n    FamilyName: Acme\n"
                 "    Models: [acme-thinker]\n    PreferredStyle: xml\n")
        s, f, warns = Rz.load(box, include_user=False)
        assert warns == [], warns
        assert "acme" in f and "deepseek" in f, (
            "merge by name, never replace - adding one family must not cost "
            "you the other four")

    def test_a_box_can_correct_a_style_in_place(self, tmp_path):
        box = _w(tmp_path, "r.yaml",
                 "TagStyles:\n  - Key: xml\n    TagStyleName: Standard XML Style\n"
                 "    OpeningTag: '<reason>'\n    ClosingTag: '</reason>'\n")
        s, _, _ = Rz.load(box, include_user=False)
        assert s["xml"].open == "<reason>"
        assert "markdown" in s, "the rest of the table survived"

    def test_preferred_style_may_name_a_key_or_a_display_name(self, tmp_path):
        box = _w(tmp_path, "r.yaml",
                 "Families:\n  - Key: byname\n    FamilyName: ByName\n"
                 "    Models: [byname-model]\n"
                 "    PreferredStyle: Markdown Code Fence\n")
        _, f, warns = Rz.load(box, include_user=False)
        assert warns == [], warns
        assert f["byname"].style == "markdown"

    def test_a_broken_entry_is_ignored_out_loud(self, tmp_path):
        box = _w(tmp_path, "r.yaml",
                 "TagStyles:\n  - TagStyleName: No Tags Here\n"
                 "Families:\n  - FamilyName: Nowhere\n"
                 "    Models: [x]\n    PreferredStyle: does-not-exist\n")
        s, f, warns = Rz.load(box, include_user=False)
        assert len(warns) == 2, warns
        assert "no_tags_here" not in s and "nowhere" not in f

    def test_an_unreadable_explicit_table_is_loud(self, tmp_path):
        _, _, warns = Rz.load(str(tmp_path / "missing.yaml"), include_user=False)
        assert any("could not be read" in w for w in warns), warns


class TestSplit:
    def test_no_style_scores_the_whole_output(self):
        assert Rz.split("hello", None) == ("", "hello", False)

    def test_a_plain_split(self):
        s, _, _ = Rz.load(include_user=False)
        think, answer, reasoned = Rz.split(
            "<think>because</think>42", s["xml"])
        assert (think, answer, reasoned) == ("because", "42", True)

    def test_a_model_that_did_not_reason_is_a_finding_not_a_crash(self):
        s, _, _ = Rz.load(include_user=False)
        think, answer, reasoned = Rz.split("just 42", s["xml"])
        assert (think, answer, reasoned) == ("", "just 42", False)

    def test_interwoven_strips_every_block(self):
        """THE FLAG THAT WAS LOADED AND NEVER READ. An agentic model emits many
        think blocks around tool calls; taking the first close tag and calling
        the rest the answer leaves later think blocks inside the thing being
        scored."""
        s, _, _ = Rz.load(include_user=False)
        text = ("<think>plan</think>call_tool(x)"
                "<think>interpret</think>The answer is 42.")
        think, answer, reasoned = Rz.split(text, s["agentic_xml"])
        assert reasoned is True
        assert "<think>" not in answer and "interpret" not in answer
        assert "call_tool(x)" in answer and "The answer is 42." in answer
        assert "plan" in think and "interpret" in think

    def test_a_non_interwoven_style_keeps_the_old_behaviour(self):
        s, _, _ = Rz.load(include_user=False)
        text = "<think>a</think>tail<think>b</think>end"
        _, answer, _ = Rz.split(text, s["xml"])
        assert answer.startswith("tail")


class TestOneSplitterEverywhere:
    def test_eval_and_data_share_it(self):
        """A scorer that splits differently from the writer is measuring a
        different artifact than the one on disk. Both now call reasoning.split."""
        import inspect
        from ms_moe_maker.data import synth as data_mod
        from ms_moe_maker.eval.harness import _split_reasoning_answer
        s, _, _ = Rz.load(include_user=False)
        text = "<think>why</think>the answer"

        # eval's wrapper delegates to the shared splitter
        answer, reasoned = _split_reasoning_answer(text, s["xml"])
        think, d_answer, _ = Rz.split(text, s["xml"])
        assert answer == d_answer == "the answer"
        assert reasoned and think == "why"

        # the writer calls the SAME splitter, not a private copy
        src = inspect.getsource(data_mod._parse_teacher_output)
        assert "from ..config.reasoning import split" in src, (
            "the writer must split with reasoning.split, not a private copy")


class TestTheTagsTravelWithTheRun:
    def _rec(self, base="", reasoning_expert=False):
        from ms_moe_maker.config.recipe import parse
        src = {"kind": "stack", "language": "Python"}
        if reasoning_expert:
            src["reasoning"] = True
        rec, _ = parse({"schema_version": 1, "name": "t", "size": "0.5B",
                        "base": base,
                        "experts": [{"name": "python", "source": src},
                                    {"name": "csharp",
                                     "source": {"kind": "stack",
                                                "language": "C#"}}]})
        return rec

    def test_the_delimiters_are_stamped_not_looked_up_later(self):
        """The table is a file now. Looking the style up again at eval time
        could read a table edited while the build ran."""
        c = C.build_config(self._rec(reasoning_expert=True), dryrun=True)
        assert (c.reasoning_open, c.reasoning_close) == ("<think>", "</think>")
        assert C.reasoning_style_of_config(c).open == "<think>"

    def test_editing_the_table_changes_the_build_id(self, tmp_path, monkeypatch):
        """`I changed my reasoning table` is a change to the build, and the
        fingerprint is fail-closed, so it says so for free."""
        plain = C.build_id(C.build_config(self._rec("Qwen/QwQ-32B"), dryrun=True))
        box = _w(tmp_path, "r.yaml",
                 "TagStyles:\n  - Key: xml\n    TagStyleName: Standard XML Style\n"
                 "    OpeningTag: '<reason>'\n    ClosingTag: '</reason>'\n")
        monkeypatch.setenv(Rz.USER_ENV, box)
        moved = C.build_id(C.build_config(self._rec("Qwen/QwQ-32B"), dryrun=True))
        assert plain != moved


class TestTheRosterSpanningCorpus:
    """The reasoning expert's questions must span the ROSTER, not one subject.

    THE BUG THIS CLASS EXISTS FOR is the one that never crashes: pairing the
    template index with the domain index (`templates[i % T]` beside
    `domains[i % D]`) locks the two together, so the corpus holds lcm(T, D)
    distinct questions instead of T*D. At T=4 and D=4 that is FOUR prompts
    repeated for a whole run, and every row still looks perfectly well formed -
    you would only ever find it by counting.
    """

    T = ("Write a {domain} script.", "Explain {domain} step by step.",
         "Debug this {domain} problem.", "Design a {domain} solution.")
    D = ["Python", "C#", "PowerShell", "Bash"]

    def _rng(self, seed=42, expert="deliberation"):
        import types
        from ms_moe_maker.data import synth as d
        return d._expert_rng(types.SimpleNamespace(seed=seed), expert)

    def _prompts(self, n=200, seed=42, expert="deliberation"):
        from ms_moe_maker.data import synth as d
        rnd = self._rng(seed, expert)
        return [d._reasoning_msgs("sys", "Deliberation", i, list(self.T),
                                  list(self.D), rnd)[1]["content"]
                for i in range(n)]

    def test_the_draw_is_not_locked_by_two_modulos(self):
        """THE anti-lock assertion. Under double-modulo this is exactly
        lcm(4, 4) = 4 distinct prompts; independent draws reach toward 16."""
        distinct = set(self._prompts())
        assert len(distinct) > max(len(self.T), len(self.D)), (
            f"only {len(distinct)} distinct prompts from {len(self.T)} "
            f"templates x {len(self.D)} domains - the template and the domain "
            f"are advancing together, which is the double-modulo lock")
        assert len(distinct) >= 12, (
            f"{len(distinct)}/16 pairs seen in 200 draws; the corpus is not "
            f"really spanning the roster")

    def test_the_same_seed_reproduces_the_same_prompts(self):
        assert self._prompts(60, seed=42) == self._prompts(60, seed=42)

    def test_a_different_seed_does_not(self):
        assert self._prompts(60, seed=1) != self._prompts(60, seed=2)

    def test_two_experts_do_not_get_identical_prompts(self):
        assert (self._prompts(60, expert="deliberation")
                != self._prompts(60, expert="thinky"))

    def test_no_domains_is_byte_identical_to_the_old_behaviour(self):
        """A hand-written `reasoning: true` expert must generate exactly what
        it generated before this parameter existed - same template, same index,
        same single domain."""
        from ms_moe_maker.data import synth as d
        tpl = list(self.T)
        for i in range(12):
            msgs = d._reasoning_msgs("sys", "Bash", i, tpl)
            assert msgs[1]["content"] == tpl[i % len(tpl)].format(domain="Bash")
            assert msgs[0] == {"role": "system", "content": "sys"}

    def test_the_domains_actually_reach_the_prompts(self):
        seen = set()
        for text in self._prompts(200):
            seen.update(dom for dom in self.D if dom in text)
        assert seen == set(self.D), f"domains never drawn: {set(self.D) - seen}"
