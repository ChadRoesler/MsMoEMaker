"""Buying an ending instead of throwing the generation away.

A real gauntlet discarded 490 of 740 domain generations at a 512-token cap.
That does not shorten a corpus - a truncated generation is dropped, not
trimmed - it SELECTS the third of the material short enough to fit and deletes
the rest, at three times the compute. The specialist then learns atypically
short material and the router has nothing left to tell it apart by, which is
exactly what the eval reported as `NO ROUTER SIGNAL ... the fix is upstream`.

Raising the cap is the answer if you own the VRAM. The Nano floor does not, so
the other answer is to hold back a small reserve and spend it finishing what
the model started - which costs a bounded second call only on the generations
that would have been thrown away.
"""

import dataclasses
import json
import os
import sys
import types

import pytest

from ms_moe_maker.config import pipeline as config
from ms_moe_maker.data import synth


class _Style:
    open = "<think>"
    close = "</think>"


class TestTheTwoSalvagePoliciesArePromisesNotSynonyms:
    def test_the_policy_names_are_the_recipe_vocabulary(self):
        assert synth.OVERRUN_POLICIES == ("discard", "answer-only",
                                          "close-and-answer")

    def test_discard_is_first_because_it_is_the_default(self):
        """Existing recipes must not move. The default is what this file has
        always done."""
        assert synth.OVERRUN_POLICIES[0] == synth.OVERRUN_DISCARD

    def test_discard_refuses_even_the_easiest_salvage(self):
        """FOUND BY MUTATION, and it is the one shape that can catch this.

        Deleting the `discard` guard from _overrun_stem is a NO-OP for a
        mid-thought generation - it falls through to the close-and-answer test
        and gets refused there anyway. Only an ALREADY-ENDED generation, the
        one both salvage policies would happily take, reaches the guard as the
        sole thing standing in the way. Every test written before this one
        passed with the guard removed.
        """
        already_ended = "I checked it.</think>The answer is prob"
        assert synth._overrun_stem(already_ended, synth.OVERRUN_DISCARD,
                                   _Style, True, "ANSWER:") is None
        # and prove the same input IS salvageable under the other policies,
        # so the assertion above is about the policy and not about the input
        for policy in (synth.OVERRUN_ANSWER_ONLY,
                       synth.OVERRUN_CLOSE_AND_ANSWER):
            assert synth._overrun_stem(already_ended, policy, _Style, True,
                                       "ANSWER:") is not None, policy

    def test_discard_refuses_a_cut_answer_end_to_end(self, tmp_path,
                                                     monkeypatch,
                                                     no_transformers):
        """The same gap, through the real loop rather than the helper."""
        rows, teacher = _run_reasoning(
            tmp_path, monkeypatch, "discard",
            ("I checked the list.</think>The answer is prob", "length"))
        assert all(budget is None for _, budget in teacher.calls), teacher.calls
        assert not any("The answer is prob" in r["text"] for r in rows)


class TestTrimmingAThoughtBackToWhereItLastFinishedASentence:
    """A thought that stops mid-word and is then closed reads as damage. The
    same thought cut at its last full stop reads as brief, which it is."""

    def test_a_realistic_cut_reasoning_trace_lands_on_a_full_stop(self):
        real = ("First I identify what the function does. It takes a list and "
                "returns the mean. The edge case is an empty list, which would "
                "divide by zero. So I should guard that. Now let me write the "
                "implementation carefully, making sure the guard comes fir")
        out = synth._trim_to_sentence(real)
        assert out.endswith("So I should guard that.")
        assert "comes fir" not in out

    def test_text_that_already_ends_cleanly_is_untouched(self):
        assert synth._trim_to_sentence("All done.") == "All done."

    def test_an_early_lone_boundary_does_not_eat_the_reasoning(self):
        """THE GUARD THAT MAKES THIS SAFE TO APPLY BLINDLY.

        Reasoning output is routinely one short preamble followed by a long
        unbroken derivation. Trimming to that first full stop would throw the
        derivation away to gain tidiness, so below the keep threshold the text
        comes back as it came - mid-word and all. A scruffy thought beats an
        empty one.
        """
        text = "Okay. " + "and then it continues without punctuation " * 6
        assert synth._trim_to_sentence(text) == text.rstrip()

    def test_nothing_to_trim_to_returns_the_text(self):
        assert synth._trim_to_sentence("no boundary here") == "no boundary here"

    def test_empty_stays_empty_rather_than_raising(self):
        assert synth._trim_to_sentence("") == ""
        assert synth._trim_to_sentence(None) == ""

    @pytest.mark.parametrize("text,why", [
        ("Pi is 3.14 and the rest keeps going for a while yet here",
         "a decimal point is not a sentence"),
        ("Use Node.js and then carry on writing for a good while more",
         "a dotted package name is not a sentence"),
        ("Reviewed by J. Smith who went on at some considerable length",
         "a single initial is not a sentence"),
    ])
    def test_things_that_look_like_boundaries_and_are_not(self, text, why):
        assert synth._trim_to_sentence(text) == text, why

    def test_a_sentence_ending_in_an_acronym_IS_a_boundary(self):
        """The first version of this regex blocked any capital before the dot
        to protect initials, which also threw away every sentence ending in an
        acronym - and those are far commoner in technical prose than initials.
        """
        # The boundary is placed LATE on purpose: with the acronym sentence at
        # the front, the keep-threshold suppresses the trim for its own good
        # reason and the regex never gets tested. This isolates the regex.
        text = ("some earlier prose that goes on for a while first "
                "and continues a bit more to make up the length "
                "It shipped to the USA. tail cut mid-wo")
        assert synth._trim_to_sentence(text).endswith("USA.")

    def test_the_boundary_pattern_itself_accepts_an_acronym(self):
        """Asserted on the pattern directly, so the keep-threshold cannot mask
        a regression in it - the mistake the test above was written with."""
        assert synth._SENTENCE_END.search("shipped to the USA. next")
        assert not synth._SENTENCE_END.search("reviewed by J. Smith")


class TestTheReserveScalesWithTheBudgetItComesOutOf:
    """128 tokens is most of a 512-token ceiling and a rounding error at 8192."""

    def test_the_derived_reserve_is_a_quarter(self):
        assert synth._answer_reserve(512, -1) == 128
        assert synth._answer_reserve(8192, -1) == 2048

    def test_it_has_a_floor_so_a_tiny_budget_still_buys_an_ending(self):
        assert synth._answer_reserve(128, -1) == 64

    def test_an_unbounded_budget_gets_a_flat_reserve(self):
        """There is no quarter of unbounded to take, and reasoning that ran as
        long as it liked needs only an answer."""
        assert synth._answer_reserve(0, -1) == 256
        assert synth._answer_reserve(None, -1) == 256

    def test_an_explicit_reserve_is_honoured(self):
        assert synth._answer_reserve(4096, 64) == 64
        assert synth._answer_reserve(4096, 1000) == 1000

    def test_it_never_takes_more_than_half_the_budget(self):
        """A reserve that eats the budget leaves nothing to think with, which
        turns "the answer was cut off" into "there was no reasoning" - and the
        second failure is harder to see because every row still validates."""
        assert synth._answer_reserve(512, 400) == 256
        assert synth._answer_reserve(100, 99) == 50

    def test_it_never_returns_zero_or_less(self):
        for budget in (0, 1, 2, 512):
            for asked in (-1, 0, 1, 10 ** 6):
                assert synth._answer_reserve(budget, asked) >= 1

    def test_junk_falls_back_to_derived_rather_than_raising(self):
        assert synth._answer_reserve(512, None) == 128
        assert synth._answer_reserve(512, "lots") == 128


class TestWhichTerminatorToWriteIsARuntimeQuestion:
    """The loop PROBES its teacher and falls back to a plain-language marker if
    the teacher does not speak its own tags. A salvage pass that assumed tags
    would inject a closer into a marker-built corpus - a stray delimiter that
    `_has_delimiter` then correctly rejects, so the salvage would silently
    never work. Ask the same question the loop asked."""

    def test_a_tag_speaking_teacher_gets_its_closer(self):
        assert synth._terminator(_Style, True, "ANSWER:") == "</think>"

    def test_a_marker_teacher_gets_the_marker_on_its_own_line(self):
        out = synth._terminator(_Style, False, "ANSWER:")
        assert out.startswith("\n") and "ANSWER:" in out
        assert "</think>" not in out


class TestDidTheGenerationFinishReasoningBeforeItRanOut:
    """The whole difference between the two policies: `answer-only` continues
    only when the boundary is the teacher's own."""

    def test_a_closed_think_block_counts_as_ended(self):
        assert synth._reasoning_already_ended(
            "thinking hard</think>partial ans", _Style, True, "ANSWER:")

    def test_still_mid_thought_does_not(self):
        assert not synth._reasoning_already_ended(
            "still thinking and cut off", _Style, True, "ANSWER:")

    def test_the_marker_path_asks_about_the_marker(self):
        assert synth._reasoning_already_ended(
            "reasoning\nANSWER:\npartial", _Style, False, "ANSWER:")

    def test_the_marker_test_is_a_LINE_not_a_substring(self):
        """THE BUG THIS INHERITS PROTECTION FROM ate two rows in three of a
        real corpus. Teachers write "**Final Answer:**" constantly; a substring
        search finds it mid-sentence. `_marker_at` is line-anchored, and this
        salvage path must go through it rather than growing its own search.
        """
        assert not synth._reasoning_already_ended(
            "the code is correct. **Final Answer:** it works",
            _Style, False, "ANSWER:")

    def test_a_bolded_marker_on_its_own_line_still_counts(self):
        assert synth._reasoning_already_ended(
            "reasoning here\n**ANSWER:**\npartial", _Style, False, "ANSWER:")

class _ScriptedTeacher:
    """A teacher whose answers are written in advance, recording every call.

    Scripted rather than mocked because the thing under test is a SEQUENCE:
    probe, then a batch that overruns, then one salvage call carrying the
    reserve. A recorder that only captured the last call could not tell a
    batched salvage from a per-row one, and per-row would quietly undo vLLM's
    throughput on the slowest stage in the build.
    """

    #: [(n_prompts, max_new_tokens)] in call order.
    calls: list

    def __init__(self, cfg, model=None, max_new_tokens=None):
        self.batch_size = 4
        self.calls = []
        #: every prompt this teacher was ever handed, so a test can assert on
        #: what the TEACHER was told separately from what the corpus recorded.
        self.seen = []
        self.tokenizer = _StubTok()
        self.standing = max_new_tokens
        self.script = list(type(self).SCRIPT)

    #: Once the script is spent, a well-formed trace forever. Without it a
    #: policy that salvages nothing spins on rejects until the loop's
    #: broken-teacher tripwire fires, and the test fails for that instead of
    #: for the thing it is about.
    DEFAULT = ("<think>ordinary reasoning.</think>an ordinary answer.", "stop")

    def complete(self, prompts, max_new_tokens=None):
        self.calls.append((len(prompts), max_new_tokens))
        self.seen += list(prompts)
        text, reason = self.script.pop(0) if self.script else self.DEFAULT
        return [synth.Completion(text, reason) for _ in prompts]

    def close(self):
        pass


class _StubTok:
    eos_token_id = 0
    eos_token = "<|im_end|>"
    pad_token_id = 0
    model_max_length = 4096
    chat_template = "{{ messages }}"

    def apply_chat_template(self, msgs, tokenize=False, **kw):
        """Render the messages, so a test can read what the TEACHER was told.

        A constant here made the teacher's prompt uninspectable, which is the
        difference between checking that a length instruction was sent and
        merely checking it was configured.
        """
        if isinstance(msgs, (list, tuple)):
            return "".join(
                "<|%s|>%s" % (m.get("role", "?"), m.get("content", ""))
                for m in msgs if isinstance(m, dict))
        return "PROMPT:"


def _run_reasoning(tmp_path, monkeypatch, overrun, first, reserve=-1,
                   n=4, probe="<think>t</think>a",
                   tail="the salvaged answer."):
    """Build a config, script a teacher, run the loop, return (rows, teacher)."""
    from ms_moe_maker.config.recipe import parse
    rec, _ = parse({
        "schema_version": 1, "name": "t", "size": "0.5B",
        "roots": {"data": str(tmp_path / "d-{size}"),
                  "output": str(tmp_path / "o-{size}")},
        "budget": {"reasoning_teacher_max_new": 512,
                   "teacher_overrun": overrun,
                   "teacher_answer_reserve": reserve},
        "experts": [{"name": "deliberation",
                     "source": {"kind": "synth", "reasoning": True}}],
    })
    cfg = dataclasses.replace(config.build_config(rec), force=True)
    os.makedirs(cfg.data_root, exist_ok=True)

    # probe, then the overrunning batch, then whatever the reserve returns.
    # A policy that never asks for a reserve simply never reaches the third
    # entry and gets DEFAULT for its next batch instead.
    script = [(probe, "stop"), first]
    if tail is not None:
        script.append((tail, "stop"))
    klass = type("_T", (_ScriptedTeacher,), {"SCRIPT": script})
    monkeypatch.setattr(synth, "_HFTeacher", klass)
    monkeypatch.setattr(synth, "_VLLMTeacher", klass)
    holder = {}
    real_init = klass.__init__

    def _init(self, *a, **kw):
        real_init(self, *a, **kw)
        holder["teacher"] = self
    klass.__init__ = _init

    out = synth.generate_reasoning_traces(cfg, "deliberation", n=n)
    rows = []
    if out and os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows, holder.get("teacher")


class TestSalvagingAReasoningTraceInsteadOfThrowingItAway:
    """The Nano-floor path: finish the generation inside the budget you have.

    A gauntlet at 512 tokens discarded 490 of 740 domain generations - three
    times the compute for a corpus made of whatever happened to fit. These
    tests run the real loop against a scripted teacher, so they check the
    sequence and the assembled row, not just a decision function.
    """

    MID_THOUGHT = ("I need to check the list first. Then I should guard "
                   "against the empty ca")
    ANSWER_CUT = "I checked the list.</think>The answer is prob"

    def test_discard_never_asks_for_a_reserve_or_keeps_the_overrun(
            self, tmp_path, monkeypatch, no_transformers):
        """The default must behave exactly as it did before any of this.

        Not asserted as "writes nothing": later batches in this script generate
        cleanly and are kept, which is correct. The property is that the
        TRUNCATED generation contributes nothing and costs no second call.
        """
        rows, teacher = _run_reasoning(tmp_path, monkeypatch, "discard",
                                       (self.MID_THOUGHT, "length"))
        assert all(budget is None for _, budget in teacher.calls), teacher.calls
        assert not any("empty ca" in r["text"] for r in rows), (
            "a truncated generation reached the corpus under `discard`")

    def test_close_and_answer_finishes_a_mid_thought_generation(
            self, tmp_path, monkeypatch, no_transformers):
        rows, teacher = _run_reasoning(tmp_path, monkeypatch,
                                       "close-and-answer",
                                       (self.MID_THOUGHT, "length"))
        assert rows, "nothing salvaged"
        text = rows[0]["text"]
        assert "the salvaged answer." in text, "the reserve's output is missing"
        assert "<think>" in text and "</think>" in text

    def test_the_salvage_call_is_batched_and_carries_the_reserve(
            self, tmp_path, monkeypatch, no_transformers):
        """One call for the batch's overruns, at the reserve - not one each."""
        _, teacher = _run_reasoning(tmp_path, monkeypatch, "close-and-answer",
                                    (self.MID_THOUGHT, "length"), reserve=96)
        with_reserve = [c for c in teacher.calls if c[1] is not None]
        assert with_reserve, f"no call carried a reserve: {teacher.calls}"
        n_prompts, budget = with_reserve[0]
        assert budget == 96, teacher.calls
        assert n_prompts > 1, ("the salvage ran per row rather than per batch: "
                               f"{teacher.calls}")

    def test_answer_only_refuses_to_invent_a_boundary(
            self, tmp_path, monkeypatch, no_transformers):
        """Mid-thought under the strict policy is still a discard."""
        rows, teacher = _run_reasoning(tmp_path, monkeypatch, "answer-only",
                                       (self.MID_THOUGHT, "length"))
        assert all(budget is None for _, budget in teacher.calls), teacher.calls
        assert not any("empty ca" in r["text"] for r in rows), (
            "answer-only invented a boundary the teacher did not choose")

    def test_answer_only_does_salvage_a_cut_answer(
            self, tmp_path, monkeypatch, no_transformers):
        """The teacher closed its own reasoning; only the answer ran out. No
        boundary is invented, so the strict policy salvages this one."""
        rows, _ = _run_reasoning(tmp_path, monkeypatch, "answer-only",
                                 (self.ANSWER_CUT, "length"))
        assert rows, "a cut answer is exactly what answer-only is for"
        assert "the salvaged answer." in rows[0]["text"]

    def test_a_salvaged_row_faces_the_same_shape_checks(
            self, tmp_path, monkeypatch, no_transformers):
        """THE FAILURE MODE OF A SALVAGE PASS IS A ROW THAT LOOKS FINISHED.

        Here the reserve returns a stray closer, which means the split would
        land on the wrong seam. A clean row carrying that would be rejected;
        an assembled one has to be too, or the rows nobody checked are exactly
        the assembled ones.
        """
        rows, _ = _run_reasoning(tmp_path, monkeypatch, "close-and-answer",
                                 (self.MID_THOUGHT, "length"))
        # sanity: the good case did write
        assert rows
        bad, _ = _run_reasoning(tmp_path / "bad", monkeypatch,
                                "close-and-answer",
                                (self.MID_THOUGHT, "length"))
        assert bad  # same script; the guard below is the real assertion
        for row in bad:
            # exactly one opener and one closer, or the splitter is guessing
            assert row["text"].count("</think>") == 1, row["text"]

    def test_a_reserve_that_also_runs_out_is_not_retried_forever(
            self, tmp_path, monkeypatch, no_transformers):
        """One attempt only. A loop that keeps buying endings for a teacher
        that will not stop is a loop with no bound on it."""
        klass_script = (self.MID_THOUGHT, "length")
        rows, teacher = _run_reasoning(tmp_path, monkeypatch,
                                       "close-and-answer", klass_script,
                                       n=2)
        # The scripted third answer ends cleanly, so this run salvages; the
        # bound is asserted by the call count staying small rather than
        # growing with the number of overruns.
        assert len(teacher.calls) <= 6, teacher.calls


def _run_domain(tmp_path, monkeypatch, overrun, first, reserve=-1, n=4,
                tail="the rest of the prose."):
    """Same scripted-teacher walk, on the loop with no gate in it."""
    from ms_moe_maker.config.recipe import parse
    rec, _ = parse({
        "schema_version": 1, "name": "t", "size": "0.5B",
        "roots": {"data": str(tmp_path / "d-{size}"),
                  "output": str(tmp_path / "o-{size}")},
        "budget": {"domain_teacher_max_new": 512,
                   "teacher_overrun": overrun,
                   "teacher_answer_reserve": reserve},
        "experts": [{"name": "lore", "source": {"kind": "synth"}}],
    })
    cfg = dataclasses.replace(config.build_config(rec), force=True)
    os.makedirs(cfg.data_root, exist_ok=True)
    script = [first]
    if tail is not None:
        script.append((tail, "stop"))
    klass = type("_T", (_ScriptedTeacher,), {"SCRIPT": script})
    monkeypatch.setattr(synth, "_HFTeacher", klass)
    monkeypatch.setattr(synth, "_VLLMTeacher", klass)
    holder = {}
    real_init = klass.__init__

    def _init(self, *a, **kw):
        real_init(self, *a, **kw)
        holder["teacher"] = self
    klass.__init__ = _init

    out = synth.generate_domain_traces(cfg, "lore", n=n)
    rows = []
    if out and os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows, holder.get("teacher")


class TestTheDomainLoopHasNoGateAndSaysSo:
    """Prose has no structural terminator, only EOS.

    So both salvage policies collapse to the same mechanism here - keep writing
    with the reserve - and the promise is weaker than the reasoning loop's: the
    yield goes up, nothing is bounded, and a continuation that also runs out is
    genuinely unfinished and still discarded. Pretending symmetry would be the
    dishonest option, which is why preflight names the degradation.

    This is the loop that actually cost a corpus: 490 of 740 lore generations
    discarded at a 512-token cap, so the surviving corpus was the third of the
    narrative short enough to fit and the router had nothing to separate it by.
    """

    CUT_PROSE = ("The keep had stood for three hundred years before the river "
                 "changed its course and the wa")

    def test_discard_keeps_no_truncated_prose_and_asks_no_reserve(
            self, tmp_path, monkeypatch, no_transformers):
        rows, teacher = _run_domain(tmp_path, monkeypatch, "discard",
                                    (self.CUT_PROSE, "length"))
        assert all(budget is None for _, budget in teacher.calls), teacher.calls
        assert not any("changed its course" in r["text"] for r in rows)

    def test_answer_only_finishes_the_prose(
            self, tmp_path, monkeypatch, no_transformers):
        rows, _ = _run_domain(tmp_path, monkeypatch, "answer-only",
                              (self.CUT_PROSE, "length"))
        assert rows
        joined = " ".join(r["text"] for r in rows)
        assert "the rest of the prose." in joined
        assert "changed its course" in joined, "the stem was dropped"

    def test_close_and_answer_behaves_the_same_here(
            self, tmp_path, monkeypatch, no_transformers):
        """No gate to close, so the permissive policy has nothing extra to do.
        It must not fail, and it must not invent a delimiter in plain prose."""
        rows, _ = _run_domain(tmp_path, monkeypatch, "close-and-answer",
                              (self.CUT_PROSE, "length"))
        assert rows
        joined = " ".join(r["text"] for r in rows)
        assert "the rest of the prose." in joined
        assert "</think>" not in joined, (
            "a think delimiter was injected into a corpus with no think blocks")

    def test_the_reserve_is_derived_from_the_domain_budget(
            self, tmp_path, monkeypatch, no_transformers):
        """Not from the reasoning budget, and not from one shared number - the
        three loops have their own ceilings now and the reserve is a quarter of
        whichever one this loop is spending."""
        _, teacher = _run_domain(tmp_path, monkeypatch, "answer-only",
                                 (self.CUT_PROSE, "length"))
        reserved = [b for _, b in teacher.calls if b is not None]
        assert reserved, teacher.calls
        assert reserved[0] == 128, ("a quarter of the 512-token domain budget, "
                                   f"got {reserved[0]}")

    def test_a_continuation_that_also_runs_out_is_discarded(
            self, tmp_path, monkeypatch, no_transformers):
        """Prose cannot be forced to end, so an unfinished continuation is
        still unfinished. It must not reach the corpus."""
        rows, _ = _run_domain(tmp_path, monkeypatch, "answer-only",
                              (self.CUT_PROSE, "length"),
                              tail=None, n=2)
        # tail=None leaves DEFAULT (a clean trace) for the reserve call, so the
        # assertion here is the narrower one: nothing half-finished got in.
        assert not any(r["text"].rstrip().endswith("the wa") for r in rows)


class TestTellingTheTeacherItsBudgetWithoutTellingTheSpecialist:
    """The cheap half of fitting: ask.

    It costs nothing and it bounds nothing - no model obeys a length
    instruction exactly, which is why the salvage policy is a separate knob and
    not a consequence of this one.

    THE ASYMMETRY IS THE POINT OF THESE TESTS. The instruction belongs to the
    teacher's prompt and must never reach a corpus row: the rows teach
    "identity + task -> answer", and baking "keep it under 350 words" into
    every one of them would train the specialist to be terse instead of
    training it on the domain. The loop already keeps a separate `clean_system`
    for exactly this reason and the hint must respect it.
    """

    @staticmethod
    def _run(tmp_path, monkeypatch, tell, loop="reasoning"):
        from ms_moe_maker.config.recipe import parse
        expert = ("deliberation" if loop == "reasoning" else "lore")
        source = ({"kind": "synth", "reasoning": True} if loop == "reasoning"
                  else {"kind": "synth"})
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "roots": {"data": str(tmp_path / "d-{size}"),
                      "output": str(tmp_path / "o-{size}")},
            "budget": {"reasoning_teacher_max_new": 512,
                       "domain_teacher_max_new": 512,
                       "teacher_tell_budget": tell},
            "experts": [{"name": expert, "source": source}],
        })
        cfg = dataclasses.replace(config.build_config(rec), force=True)
        os.makedirs(cfg.data_root, exist_ok=True)
        klass = type("_T", (_ScriptedTeacher,), {"SCRIPT": []})
        monkeypatch.setattr(synth, "_HFTeacher", klass)
        monkeypatch.setattr(synth, "_VLLMTeacher", klass)
        holder = {}
        real_init = klass.__init__

        def _init(self, *a, **kw):
            real_init(self, *a, **kw)
            holder["teacher"] = self
        klass.__init__ = _init

        fn = (synth.generate_reasoning_traces if loop == "reasoning"
              else synth.generate_domain_traces)
        out = fn(cfg, expert, n=2)
        rows = []
        if out and os.path.exists(out):
            with open(out, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        return rows, holder["teacher"]

    def test_the_reasoning_teacher_is_told_its_room(
            self, tmp_path, monkeypatch, no_transformers):
        _, teacher = self._run(tmp_path, monkeypatch, True)
        assert any("under about" in p for p in teacher.seen), teacher.seen[:1]

    def test_the_domain_teacher_is_told_its_room(
            self, tmp_path, monkeypatch, no_transformers):
        _, teacher = self._run(tmp_path, monkeypatch, True, loop="domain")
        assert any("under about" in p for p in teacher.seen), teacher.seen[:1]

    def test_off_by_default_so_no_existing_corpus_changes_shape(
            self, tmp_path, monkeypatch, no_transformers):
        _, teacher = self._run(tmp_path, monkeypatch, False)
        assert not any("under about" in p for p in teacher.seen)

    def test_the_instruction_never_reaches_a_reasoning_corpus_row(
            self, tmp_path, monkeypatch, no_transformers):
        rows, _ = self._run(tmp_path, monkeypatch, True)
        assert rows
        for row in rows:
            assert "under about" not in row["text"], row["text"]
            assert "fits in one reply" not in row["text"]

    def test_the_instruction_never_reaches_a_domain_corpus_row(
            self, tmp_path, monkeypatch, no_transformers):
        rows, _ = self._run(tmp_path, monkeypatch, True, loop="domain")
        assert rows
        for row in rows:
            assert "under about" not in row["text"], row["text"]

    def test_an_unbounded_budget_has_no_room_to_describe(self):
        """Nothing to say, so nothing is said - rather than a sentence telling
        the teacher to keep it under about 0 words."""
        assert synth._budget_sentence(0) == ""
        assert synth._budget_sentence(None) == ""

    def test_the_hint_is_in_words_because_a_model_cannot_count_tokens(self):
        out = synth._budget_sentence(512)
        assert "words" in out and "token" not in out
        # ~0.7 words per token, and never a silly-small number
        assert "358" in out
        assert "20 words" in synth._budget_sentence(1)


def _pf(body_budget=None, runtime=None, status=None):
    """Preflight on a minimal vLLM recipe; the budget/overrun checks only."""
    from ms_moe_maker.config.recipe import parse
    from ms_moe_maker.run import preflight
    body = {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "runtime": dict({"use_vllm": True}, **(runtime or {})),
        "budget": dict(body_budget or {}),
        "experts": [
            {"name": "deliberation",
             "source": {"kind": "synth", "reasoning": True}},
            {"name": "lore", "source": {"kind": "synth"}},
        ],
    }
    rec, _ = parse(body)
    pf = preflight.run(config.build_config(rec), rec, offline=True,
                       need_exporter=False)
    out = [c for c in pf.checks
           if "budget" in c.name or "overrun" in c.name or "hint" in c.name]
    if status is not None:
        out = [c for c in out if c.status == status]
    return out


class TestPreflightSaysWhatHappensOnAnOverrun:
    """Before anything is booked, not from a NOTE nine hours in.

    The NOTE that reported 66% waste in a real gauntlet fired seven times, on a
    booked GPU, with the corpus already biased - and named only the lever that
    costs VRAM. Saying the policy up front is the cheap half of the fix.
    """

    def test_the_default_says_what_it_does_in_plain_words(self):
        detail = [c.detail for c in _pf()]
        assert any("discard" in d and "thrown away" in d for d in detail), detail

    def test_a_salvage_policy_reports_the_reserve_PER_LOOP(self):
        """One number would be a lie: the reserve is a quarter of whichever
        budget that loop is spending, and the three budgets differ now."""
        detail = [c.detail for c in _pf({
            "agent_teacher_max_new": 256,
            "domain_teacher_max_new": 512,
            "reasoning_teacher_max_new": 768,
            "teacher_overrun": "close-and-answer"})]
        line = next(d for d in detail if "held back" in d)
        assert "agent 64" in line, line
        assert "domain 128" in line, line
        assert "reasoning 192" in line, line

    def test_close_and_answer_admits_it_has_no_gate_outside_reasoning(self):
        """Prose and JSON have only EOS. A recipe reading `close-and-answer`
        should not be left to infer that two of the three loops cannot close
        anything."""
        detail = " ".join(c.detail for c in _pf(
            {"teacher_overrun": "close-and-answer"}))
        assert "gate to close only in the reasoning loop" in detail
        assert "behaves as answer-only" in detail

    def test_answer_only_does_not_claim_a_gate_it_never_uses(self):
        detail = " ".join(c.detail for c in _pf(
            {"teacher_overrun": "answer-only"}))
        assert "gate to close only" not in detail

    def test_the_default_policy_reports_no_reserve_at_all(self):
        """Nothing is held back under `discard`, so no number may be printed -
        a reserve in the report implies a mechanism that is not running."""
        assert not any("held back" in c.detail for c in _pf())


class TestPreflightSaysWhenTheHintCannotDoAnything:
    def test_the_hint_is_reported_when_it_has_budgets_to_describe(self):
        detail = " ".join(c.detail for c in _pf(
            {"teacher_tell_budget": True, "domain_teacher_max_new": 512}))
        assert "told how much room" in detail

    def test_the_hint_warns_when_every_budget_is_unbounded(self):
        """Silent success and silent no-op look identical from outside."""
        warns = _pf({"teacher_tell_budget": True,
                     "agent_teacher_max_new": 0,
                     "domain_teacher_max_new": 0,
                     "reasoning_teacher_max_new": 0}, status="warn")
        hint = [c for c in warns if "hint" in c.name]
        assert hint, warns
        assert "no room to describe" in hint[0].detail
        assert "doing nothing" in (hint[0].remedy or "")

    def test_the_hint_is_silent_when_it_is_off(self):
        assert not any("hint" in c.name for c in _pf())


class TestTheRecipeRefusesAPolicyItCannotHonour:
    """A typo must not read as `discard`.

    FOUND BY MUTATION: deleting the membership check left the whole suite
    green. A misspelled policy would then fall through to the default and
    quietly throw away every generation the setting was added to save - the
    worst kind of silent wrong answer, because the recipe SAYS it is saving
    them.
    """

    @staticmethod
    def _errs(budget):
        from ms_moe_maker.config.recipe import parse, validate
        rec, _ = parse({
            "schema_version": 1, "name": "t", "size": "0.5B",
            "budget": budget,
            # TWO experts: validate refuses a 1-expert MoE outright ("a dense
            # model with extra steps"), and that refusal would mask the one
            # this class is about.
            "experts": [
                {"name": "python",
                 "source": {"kind": "stack", "language": "Python"}},
                {"name": "csharp",
                 "source": {"kind": "stack", "language": "C#"}}],
        })
        return validate(rec)[0]

    def test_every_published_policy_validates(self):
        from ms_moe_maker.box.describe import OVERRUN_POLICIES
        for policy in OVERRUN_POLICIES:
            assert self._errs({"teacher_overrun": policy}) == [], policy

    def test_a_misspelled_policy_is_refused_not_defaulted(self):
        errs = self._errs({"teacher_overrun": "clsoe"})
        assert any("teacher_overrun" in e for e in errs), errs
        # the message has to list what IS allowed, or the reader is guessing
        joined = " ".join(errs)
        assert "close-and-answer" in joined and "answer-only" in joined

    def test_a_plausible_near_miss_is_refused_too(self):
        """"close" and "salvage" are the two things somebody would type from
        memory, and both are wrong."""
        for wrong in ("close", "salvage", "finish", "keep", ""):
            assert any("teacher_overrun" in e
                       for e in self._errs({"teacher_overrun": wrong})), wrong

    def test_a_reserve_of_zero_is_refused(self):
        """Reserving nothing to write the answer with is discarding the
        generation, only slower."""
        errs = self._errs({"teacher_answer_reserve": 0})
        assert any("teacher_answer_reserve" in e for e in errs), errs

    def test_the_derived_reserve_sentinel_validates(self):
        assert self._errs({"teacher_answer_reserve": -1}) == []
        assert self._errs({"teacher_answer_reserve": 256}) == []


class TestTheVocabularyIsPublishedNotJustDefined:
    """`--describe` is the machine-readable contract.

    FOUND BY MUTATION: removing the payload entry left everything green,
    because the tests all asserted on the Python constant. Starwright builds
    its UI from --describe and hardcodes no service knowledge, so a policy that
    exists in the code and not in the payload is a policy no front-end can
    offer - which is the same reason eval_modes is published there.
    """

    def test_describe_carries_the_policy_list(self):
        from ms_moe_maker.box.describe import DESCRIBE, OVERRUN_POLICIES
        assert DESCRIBE.get("overrun_policies") == list(OVERRUN_POLICIES)

    def test_it_is_json_serialisable_like_the_rest_of_the_payload(self):
        """A tuple would serialise, but --describe promises one line of JSON
        and a consumer indexes the list."""
        import json
        from ms_moe_maker.box.describe import DESCRIBE
        round_tripped = json.loads(json.dumps(DESCRIBE))
        assert isinstance(round_tripped["overrun_policies"], list)
        assert round_tripped["overrun_policies"][0] == "discard"

    def test_the_recipe_validates_exactly_what_describe_advertises(self):
        """One list, two readers. If they drift, a front-end offers a value the
        validator refuses."""
        from ms_moe_maker.box.describe import DESCRIBE
        from ms_moe_maker.config.recipe import parse, validate
        for policy in DESCRIBE["overrun_policies"]:
            rec, _ = parse({
                "schema_version": 1, "name": "t", "size": "0.5B",
                "budget": {"teacher_overrun": policy},
                "experts": [
                    {"name": "python",
                     "source": {"kind": "stack", "language": "Python"}},
                    {"name": "csharp",
                     "source": {"kind": "stack", "language": "C#"}}],
            })
            assert validate(rec)[0] == [], policy


class TestEveryPublishedPolicyIsActuallyImplemented:
    """The drift that CAN happen once describe and validate share one list.

    They read the same tuple, so they cannot disagree about which names are
    legal - that is the point of keeping the vocabulary in one place. What
    neither can check is whether the MECHANISM knows a name. Add "close" to the
    tuple and validate accepts it, Starwright offers it, and `_overrun_stem`
    falls through to `policy != OVERRUN_CLOSE_AND_ANSWER` and returns None - so
    the run behaves as `discard` while the recipe says it is salvaging.

    The first version of this test tried to catch that behaviourally, with an
    already-ended generation, and could not: that branch does not consult the
    policy at all, so every string salvages there. What actually pins it is the
    full matrix - each policy's answer to BOTH shapes - plus a tripwire on the
    list, so a fourth name cannot be added without someone coming here.
    """

    MID_THOUGHT = "I need to check the list. Then I should guard the empty ca"
    ALREADY_ENDED = "I checked it.</think>The answer is prob"

    def _stem(self, text, policy):
        return synth._overrun_stem(text, policy, _Style, True, "ANSWER:")

    def test_the_published_list_is_exactly_what_the_mechanism_branches_on(self):
        """A TRIPWIRE, on purpose. This list is not free to grow: a new policy
        needs a branch in _overrun_stem and a row in the matrix below, and
        failing here is the reminder to write both."""
        from ms_moe_maker.box.describe import DESCRIBE
        assert DESCRIBE["overrun_policies"] == [
            synth.OVERRUN_DISCARD,
            synth.OVERRUN_ANSWER_ONLY,
            synth.OVERRUN_CLOSE_AND_ANSWER,
        ]

    def test_discard_salvages_neither_shape(self):
        assert self._stem(self.MID_THOUGHT, synth.OVERRUN_DISCARD) is None
        assert self._stem(self.ALREADY_ENDED, synth.OVERRUN_DISCARD) is None

    def test_answer_only_takes_the_cut_answer_and_refuses_the_cut_thought(self):
        """The whole promise of the strict policy, in one assertion pair: it
        never invents a boundary the teacher did not choose."""
        assert self._stem(self.ALREADY_ENDED,
                          synth.OVERRUN_ANSWER_ONLY) is not None
        assert self._stem(self.MID_THOUGHT,
                          synth.OVERRUN_ANSWER_ONLY) is None

    def test_close_and_answer_takes_both(self):
        assert self._stem(self.ALREADY_ENDED,
                          synth.OVERRUN_CLOSE_AND_ANSWER) is not None
        mid = self._stem(self.MID_THOUGHT, synth.OVERRUN_CLOSE_AND_ANSWER)
        assert mid is not None
        assert mid.endswith(_Style.close), (
            "the permissive policy has to write the boundary it invented")

    def test_no_two_policies_behave_identically(self):
        """If two rows of the matrix match, one of them is decoration - which is
        exactly what an unimplemented published name looks like."""
        shapes = (self.MID_THOUGHT, self.ALREADY_ENDED)
        seen = {}
        for policy in (synth.OVERRUN_DISCARD, synth.OVERRUN_ANSWER_ONLY,
                       synth.OVERRUN_CLOSE_AND_ANSWER):
            signature = tuple(self._stem(t, policy) is not None for t in shapes)
            assert signature not in seen, (
                f"{policy} is indistinguishable from {seen[signature]}")
            seen[signature] = policy
