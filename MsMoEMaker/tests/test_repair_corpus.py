"""The corpus repair, pinned against the rows it was written for.

The bug filed real answers under `think` and trained on a one-line summary.
Nothing was lost - the text is still in the row, under the wrong heading - so
these corpora are repairable and nobody has to buy the GPU hours twice.

THE ACCEPTANCE TEST IS THE ROUND TRIP. Not "the string changed" but: after the
repair, the SAME splitter eval uses reads the row back into a clean think and a
clean answer. A repair that produced something eval still mis-splits would be
a second wrong answer wearing a fix's clothes.
"""
import importlib.util
import json
import os
import sys

import pytest

from ms_moe_maker.config.reasoning import ReasoningStyle, split

XML = ReasoningStyle(name="xml", open="<think>", close="</think>")

_TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools", "repair_reasoning_corpus.py")
_spec = importlib.util.spec_from_file_location("repair_tool", _TOOL)
repair_tool = importlib.util.module_from_spec(_spec)
sys.modules["repair_tool"] = repair_tool
_spec.loader.exec_module(repair_tool)


# Chad's actual rows, trimmed but byte-faithful in shape.
SHAPE_B = (
    "<|im_start|>assistant\n"
    "<think>Okay, so the user is asking me to think through a task.\n"
    "</think>\n\n"
    "The Python function `calculate_average` computes the average but raises "
    "an error if the list is empty.\n\n"
    "```python\ndef calculate_average(numbers):\n    return sum(numbers)\n```"
    "</think>\n"
    "calculate_average<|im_end|>\n<|im_end|>")

SHAPE_A = (
    "<|im_start|>assistant\n"
    "<think>Okay, so I'm trying to figure out how I would notice.\n"
    "</think></think>\n"
    "To determine if a C# result is correct, check for compile-time errors."
    "<|im_end|>\n<|im_end|>")

CLEAN = ("<|im_start|>assistant\n<think>clean reasoning</think>\n"
         "a clean answer<|im_end|>\n")


class TestTheWorstRow:
    """The one whose entire training target was the word `calculate_average`,
    with the function it was supposed to teach sitting inside the think."""

    def test_the_buried_answer_is_promoted(self):
        new, note = repair_tool.repair(SHAPE_B)
        assert "promoted-buried-answer" in note
        assert "def calculate_average" in new
        think, answer, reasoned = split(new, XML)
        assert reasoned
        assert "def calculate_average" in answer, (
            "the function is still filed under think - the repair moved bytes "
            "around without fixing what the specialist would learn")
        assert "def calculate_average" not in think
        assert answer.strip() != "calculate_average", (
            "the one-word summary is still the training target")

    def test_the_summary_does_not_survive_as_the_answer(self):
        new, _ = repair_tool.repair(SHAPE_B)
        _, answer, _ = split(new, XML)
        assert not answer.strip().endswith("calculate_average\n"), answer[-80:]


class TestTheDoubledTag:
    def test_the_stray_closer_goes_and_the_answer_stays(self):
        new, note = repair_tool.repair(SHAPE_A)
        assert "dropped-stray-tag" in note
        assert new.count("</think>") == 1, new
        think, answer, reasoned = split(new, XML)
        assert reasoned
        assert answer.startswith("To determine if a C# result is correct")
        assert "</think>" not in answer, (
            "eval would score an answer that opens with a stray tag - a "
            "constant subtracted from every row, looking like a weak model")


class TestItDoesNotTouchWhatIsFine:
    def test_a_clean_row_is_left_alone(self):
        new, note = repair_tool.repair(CLEAN)
        assert note == "", note
        assert new == CLEAN

    def test_a_row_with_no_tags_at_all_is_left_alone(self):
        text = "<|im_start|>assistant\njust an answer<|im_end|>\n"
        assert repair_tool.repair(text) == (text, "")

    def test_an_unrepairable_row_is_refused_not_guessed_at(self):
        """Half a row is worse than a row you skipped and counted."""
        new, note = repair_tool.repair("<think></think></think>\n")
        assert new is None and note == "unrepairable"


class TestTheDoubledStop:
    def test_two_end_tokens_become_one(self):
        new, note = repair_tool.repair(CLEAN.rstrip() + "<|im_end|>")
        assert "double-eos" in note
        assert new.count("<|im_end|>") == CLEAN.count("<|im_end|>")

    def test_one_end_token_is_not_stripped(self):
        assert "double-eos" not in repair_tool.repair(CLEAN)[1]


class TestTheFileMode:
    """DRY RUN BY DEFAULT, because it rewrites something that cost GPU hours."""

    def _corpus(self, tmp_path):
        p = tmp_path / "reasoning.jsonl.partial"
        p.write_text("".join(
            json.dumps({"text": t}, ensure_ascii=False) + "\n"
            for t in (SHAPE_B, SHAPE_A, CLEAN)), encoding="utf-8")
        return p

    def test_a_dry_run_writes_nothing(self, tmp_path, capsys):
        p = self._corpus(tmp_path)
        before = p.read_bytes()
        assert repair_tool.main([str(p)]) == 0
        assert p.read_bytes() == before, "a dry run rewrote the corpus"
        assert "dry run" in capsys.readouterr().out

    def test_write_repairs_and_keeps_the_original(self, tmp_path, capsys):
        p = self._corpus(tmp_path)
        before = p.read_bytes()
        assert repair_tool.main([str(p), "--write"]) == 0
        assert p.read_bytes() != before
        assert (tmp_path / "reasoning.jsonl.partial.bak").read_bytes() == before

        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        assert len(rows) == 3, "the repair lost or duplicated a row"
        for row in rows:
            assert row["text"].count("</think>") == 1, row["text"][:120]
            think, answer, reasoned = split(row["text"], XML)
            assert reasoned and think and answer

    def test_a_repaired_corpus_is_idempotent(self, tmp_path):
        """Running it twice must not eat a second bite. A migration you cannot
        re-run safely is one nobody dares run at all."""
        p = self._corpus(tmp_path)
        repair_tool.main([str(p), "--write"])
        once = p.read_bytes()
        repair_tool.main([str(p), "--write"])
        assert p.read_bytes() == once

    def test_an_unrepairable_row_survives_the_rewrite(self, tmp_path):
        """COUNTED, NOT DELETED. A row this tool cannot make sense of is a
        row a person may still want to look at - and a repair pass that
        quietly shrinks a corpus is the worst possible kind, because the
        count is the only thing anyone checks afterwards."""
        p = tmp_path / "x.jsonl"
        broken = "<think></think></think>\n"
        p.write_text("".join(
            json.dumps({"text": t}, ensure_ascii=False) + "\n"
            for t in (SHAPE_A, broken, CLEAN)), encoding="utf-8")

        repair_tool.main([str(p), "--write"])
        rows = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 3, (
            f"the corpus lost a row it could not repair - {len(rows)} of 3")
        assert any(r["text"] == broken for r in rows), (
            "the unrepairable row was silently rewritten instead of kept")

    def test_an_unparseable_line_is_carried_through_not_dropped(
            self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"text": "' + CLEAN.replace("\n", "\\n") + '"}\n'
                     "not json at all\n", encoding="utf-8")
        repair_tool.main([str(p), "--write"])
        assert "not json at all" in p.read_text(encoding="utf-8"), (
            "a line it could not read was silently deleted")
