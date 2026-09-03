"""The reasoning split, pinned against the bytes that broke it.

WHAT HAPPENED, because every fixture here is a real row off a real run.

DeepSeek-R1 and its distills put `<think>\\n` at the END of the generation
prompt. The model wakes up already inside the block and emits only the CLOSER.
`split` required the pair, so the most reasoning-shaped output there is got
reported as "did not reason" - and the generator believed it, fell back to a
prompt asking for an `ANSWER:` line, and an R1 model obliged by producing FOUR
things: reasoning, its closer, a full answer, and then a one-line summary.

The marker cut at the last seam. So the corpus filed the real answer under
`think` and trained the specialist on the summary. One row's entire response
was the word `calculate_average`, with the whole function inside the think
block. Nine thousand rows, at a 92% accept rate, reported as a healthy run.

The accept rate was not lying. It was answering a different question: both
halves EXIST. Nothing asked whether either was the right half.
"""
import pytest

from ms_moe_maker.config.reasoning import ReasoningStyle, split
from ms_moe_maker.data.synth import (_finish, _has_delimiter,
                                     _parse_teacher_output, _speaks_tags)

# answer_marker left at its shipped default: the marker path has to stay
# reachable, it just must not be the FIRST thing tried.
XML = ReasoningStyle(name="Standard XML Style", open="<think>",
                     close="</think>", interwoven=False)


# -- the shape that started it -----------------------------------------------

LONE_CLOSER = (
    "Okay, so I need to figure out how to break down a large C# problem.\n"
    "</think>\n\n"
    "Breaking down a C# problem involves modularizing it into smaller, "
    "self-contained components.")


def test_a_lone_closer_is_reasoning_not_prose():
    """The opener was in the prompt. That is not the same as absent."""
    think, answer, reasoned = split(LONE_CLOSER, XML)
    assert reasoned, (
        "an R1-shaped completion read as 'did not reason', which is what sent "
        "the whole run down the marker path")
    assert think.startswith("Okay, so I need")
    assert answer.startswith("Breaking down a C# problem")
    assert "</think>" not in think, "the orphan closer stayed in the think half"
    assert "</think>" not in answer


def test_a_bare_closer_with_nothing_around_it_is_still_malformed():
    """Lenient, not credulous. A completion that is only a tag reasoned about
    nothing, and calling that a trace would replace one bad row with another."""
    assert split("</think>", XML)[2] is False
    assert split("   </think>   ", XML)[2] is False
    assert split("</think>\nonly an answer", XML)[2] is False, (
        "no thinking at all still has to read as no thinking")


def test_the_ordinary_pair_still_works():
    think, answer, reasoned = split("<think>reasoning</think>the answer", XML)
    assert (think, answer, reasoned) == ("reasoning", "the answer", True)


def test_no_delimiter_at_all_is_still_a_finding_not_a_crash():
    think, answer, reasoned = split("just prose, no tags", XML)
    assert reasoned is False and think == "" and answer == "just prose, no tags"


# -- the four-part output, which is the expensive one ------------------------

FOUR_PART = (
    "Okay, so I'm trying to figure out what the user is asking here.\n"
    "</think>\n\n"
    "The model someone is likely missing involves a structured approach:\n"
    "1. **Understanding Syntax and Semantics**\n"
    "2. **Functions and Methods**\n"
    "ANSWER: A comprehensive learning model for C# includes syntax and OOP.")


def test_the_full_answer_wins_over_the_summary():
    """THE ROW THAT MADE THIS A BUG AND NOT A WART.

    Told to write an ANSWER: line, an R1 teacher writes a proper answer AND
    the line. The old parser cut at the marker, so `think` swallowed the real
    answer and the specialist was trained on a one-sentence summary - in one
    case, on the single word `calculate_average`.
    """
    think, answer = _parse_teacher_output(FOUR_PART, XML)
    assert think.startswith("Okay, so I'm trying")
    assert "Understanding Syntax" in answer, (
        "the real answer is still filed under think - this is the bug that "
        "gutted ten thousand traces")
    assert "1. **Understanding" in answer
    assert "ANSWER:" not in answer, "the redundant summary was left on the end"
    assert "Understanding Syntax" not in think


MARKER_RIGHT_AFTER_CLOSER = (
    "Okay, so I'm trying to figure out how I would notice.\n"
    "</think>\n\n"
    "ANSWER: To determine if a C# result is correct, check for compile-time "
    "errors.")


def test_the_other_shape_loses_its_doubled_tag():
    """`canonical = f"{open}{think}{close}"` around a think that still held a
    closer produced `</think></think>` in the corpus. eval reads with this
    same splitter, cuts at the FIRST closer, and scores an answer beginning
    with a stray tag - a constant subtracted from every score, indistinguishable
    from a specialist that came out weak."""
    think, answer = _parse_teacher_output(MARKER_RIGHT_AFTER_CLOSER, XML)
    assert "</think>" not in think
    assert answer.startswith("To determine if a C# result is correct")
    canonical = f"{XML.open}{think}{XML.close}\n{answer}"
    assert canonical.count("</think>") == 1, canonical[:200]


def test_the_marker_path_scrubs_tags_it_finds_on_the_way_past():
    """THE CASE _clean EXISTS FOR, and it had no test until a mutation run
    showed the guard could be deleted with everything still green.

    A truncated completion - an opener, no closer, generation cut at
    max_new - fails the delimiter read and falls to the marker. Without
    scrubbing, that half-open tag rides into `think`, gets wrapped by the
    re-emit, and produces a corpus row with two openers and one closer.
    """
    think, answer = _parse_teacher_output(
        "<think>I reasoned but was cut off\nANSWER: the answer", XML)
    assert "<think>" not in think, (
        "a stray opener rode through the marker path into the corpus")
    assert think == "I reasoned but was cut off"
    assert answer == "the answer"


def test_a_genuinely_markerless_teacher_still_gets_the_marker_path():
    """The fallback is not deleted, it is demoted. A prose teacher that emits
    no delimiter at all is exactly what it was written for."""
    think, answer = _parse_teacher_output(
        "I thought about it carefully.\nANSWER: forty-two", XML)
    assert think == "I thought about it carefully."
    assert answer == "forty-two"


@pytest.mark.parametrize("text", [
    "</think>",
    "ANSWER:",
    "<think></think>",
    "",
])
def test_nothing_usable_is_a_reject_not_an_empty_trace(text):
    assert _parse_teacher_output(text, XML) == ("", "")


# -- the tripwire that would have caught all of it at trace 1 ----------------

class TestPresenceIsNotShape:
    """The old reject test was `if not think or not answer`. A `think` that is
    nothing but `"</think>"` passes it, which is how a 92% accept rate was a
    true statement about a corpus that was 94% malformed."""

    def test_a_half_carrying_a_delimiter_is_detected(self):
        assert _has_delimiter("reasoning\n</think>", XML) is True
        assert _has_delimiter("<think>reasoning", XML) is True
        assert _has_delimiter("clean prose", XML) is False
        assert _has_delimiter("", XML) is False

    def test_the_parser_never_hands_back_a_dirty_half(self):
        """Belt and braces: whatever the parser returns, the reject test in
        the loop must have nothing left to catch."""
        for raw in (LONE_CLOSER, FOUR_PART, MARKER_RIGHT_AFTER_CLOSER):
            think, answer = _parse_teacher_output(raw, XML)
            assert not _has_delimiter(think, XML), raw[:60]
            assert not _has_delimiter(answer, XML), raw[:60]


# -- one EOS, not two --------------------------------------------------------

class _Tok:
    def __init__(self, eos):
        self.eos_token = eos


def test_a_template_that_already_closed_the_turn_is_left_alone():
    """Every row of every corpus ended `<|im_end|>\\n<|im_end|>` because all
    three loops appended eos to a Qwen template that had already added it."""
    rendered = "<|im_start|>assistant\nhi<|im_end|>\n"
    assert _finish(rendered, _Tok("<|im_end|>")) == rendered


def test_a_template_that_did_not_close_still_gets_its_stop():
    """CHECKED, NOT ASSUMED. Dropping the append unconditionally would leave
    some corpora with no stop at all - a worse failure than a doubled one."""
    assert _finish("plain text", _Tok("</s>")) == "plain text</s>"


def test_a_tokenizer_with_no_eos_is_not_a_crash():
    assert _finish("plain text", _Tok("")) == "plain text"


# -- the speed lever that had no recipe key ---------------------------------

class TestVllmIsReachableFromARecipe:
    """`use_vllm` was env-only (MSMOE_VLLM) - the same mistake `llama_cpp` used
    to make, whose own docstring calls it out: the knob most likely to differ
    per box was the one a recipe could not carry. For a build started from a
    viewer that meant editing a systemd unit to make the longest stage faster.

    It is not only a speed knob either, which is why it belongs in the recipe:
    it moves the teacher batch from 96 to 512, so it changes what gets
    generated - and it was already fingerprinted for that reason.
    """

    def _translate(self, **runtime):
        from ms_moe_maker.config.levers import translate
        from ms_moe_maker.config.recipe import Runtime

        class _Rec:
            pass

        rec = _Rec()
        rec.runtime = Runtime(**runtime)
        rec.budget = None
        rec.moe = None
        rec.gates = None
        return translate(rec)

    def test_true_reaches_the_environment(self):
        tr = self._translate(use_vllm=True)
        assert tr.env.get("MSMOE_VLLM") == "1"
        assert any("runtime.use_vllm" in a for a in tr.agreed)

    def test_false_is_a_real_answer_and_beats_the_environment(self):
        """`use_vllm: false` is a deliberate 'plain transformers on this box'.
        A truthiness test would drop it and hand the author the opposite of
        what they wrote, silently, on a box with MSMOE_VLLM=1 exported."""
        from ms_moe_maker.config.pipeline import _env_bool

        tr = self._translate(use_vllm=False)
        assert tr.env.get("MSMOE_VLLM") == "0"
        assert _env_bool.__module__  # sanity: the reader exists
        import os
        os.environ["MSMOE_VLLM"] = tr.env["MSMOE_VLLM"]
        try:
            assert _env_bool("MSMOE_VLLM", True) is False, (
                "the recipe said no and the reader said yes")
        finally:
            os.environ.pop("MSMOE_VLLM", None)

    def test_unset_leaves_the_environment_alone(self):
        """Nothing changes for anyone who never writes the key."""
        assert "MSMOE_VLLM" not in self._translate().env

    def test_it_can_now_travel_in_a_bundle(self):
        from ms_moe_maker.config.knobs import UNPINNABLE, recipe_path

        assert recipe_path("use_vllm") == "runtime.use_vllm"
        assert "use_vllm" not in UNPINNABLE, (
            "a fingerprinted field that changes the corpus is still marked as "
            "unable to travel, so a bundle cannot say which generator made "
            "the data")


class TestTheProbeAsksTheRightQuestion:
    """THE ROOT CAUSE, and the one line that decided the shape of a corpus.

    The probe used to ask "does split() succeed", which needs BOTH tags. An
    R1-distill's opener lives in the generation prompt, so the completion has
    only the closer - and the most reasoning-shaped output there is came back
    as "does not reason". That sent the run to a marker prompt, and the rest
    is the four-part output above.
    """

    def test_a_lone_closer_means_the_teacher_speaks_tags(self):
        assert _speaks_tags("reasoning\n</think>\nthe answer", XML) is True

    def test_even_a_closer_with_an_empty_half_counts(self):
        """One unlucky draw says nothing about the teacher. It emitted the
        delimiter; that is the whole question being asked."""
        assert _speaks_tags("</think>\nanswer only", XML) is True

    def test_a_proper_pair_obviously_counts(self):
        assert _speaks_tags("<think>a</think>b", XML) is True

    def test_genuine_prose_still_falls_back_to_the_marker(self):
        """The fallback is demoted, not deleted - a teacher that emits no
        delimiter at all is exactly what it was written for."""
        assert _speaks_tags("I just rambled at you.", XML) is False

    def test_nothing_at_all_is_not_a_yes(self):
        assert _speaks_tags("", XML) is False
        assert _speaks_tags("text", None) is False


class TestTheLeverIsWiredAndNotJustAgreedTo:
    """THE TEST THE ONE ABOVE FAILED TO BE, and the distinction is the same one
    this codebase keeps re-learning: a contract test proves two ends AGREE, and
    proves nothing about whether they are CONNECTED.

    `test_true_reaches_the_environment` passed the whole time the lever was
    inert. translate() really did put MSMOE_VLLM in tr.env - and Runner applies
    that env AFTER build_config has already read it, so `config.use_vllm` was
    False, `teacher_batch` stayed 96, and the agreed list cheerfully reported
    the knob as honoured. `--plan` caught it by printing "teacher: transformers,
    batch 96" for a recipe that had just asked for vLLM.

    So these assert the RESOLVED CONFIG, which is the thing synth.py and
    preflight actually read.
    """

    def _config(self, use_vllm=None, env=None):
        import os
        import tempfile

        from ms_moe_maker.config.pipeline import build_config
        from ms_moe_maker.config.recipe import load

        src = ("schema_version: 1\nname: t\nsize: 0.5B\n"
               "base: Qwen/Qwen2.5-Coder-0.5B-Instruct\n"
               "experts:\n"
               "  - {name: a, source: {kind: hf, repo: x/y, text_field: text}}\n")
        if use_vllm is not None:
            src += f"runtime:\n  use_vllm: {str(use_vllm).lower()}\n"
        path = tempfile.mktemp(suffix=".yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        old = os.environ.get("MSMOE_VLLM")
        if env is None:
            os.environ.pop("MSMOE_VLLM", None)
        else:
            os.environ["MSMOE_VLLM"] = env
        try:
            rec, _ = load(path)
            return build_config(rec)
        finally:
            os.unlink(path)
            if old is None:
                os.environ.pop("MSMOE_VLLM", None)
            else:
                os.environ["MSMOE_VLLM"] = old

    def test_the_recipe_key_reaches_the_resolved_config(self):
        cfg = self._config(use_vllm=True)
        assert cfg.use_vllm is True, (
            "the recipe asked for vLLM and the resolved config says no - the "
            "lever is agreed-to and inert")

    def test_it_moves_the_teacher_batch_too(self):
        """Not a cosmetic flag: 96 -> 512 is why it is in the fingerprint."""
        assert self._config(use_vllm=True).teacher_batch == 512
        assert self._config(use_vllm=False).teacher_batch == 96

    def test_the_recipe_beats_the_environment_in_both_directions(self):
        """`use_vllm: false` on a box with MSMOE_VLLM=1 exported is somebody
        saying 'not on this one', and it has to win."""
        assert self._config(use_vllm=False, env="1").use_vllm is False
        assert self._config(use_vllm=True, env="0").use_vllm is True

    def test_a_silent_recipe_still_lets_the_environment_decide(self):
        """Nothing changes for anyone who never writes the key."""
        assert self._config(env="1").use_vllm is True
        assert self._config(env=None).use_vllm is False
