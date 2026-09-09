"""Which budget each generator actually generates against.

THIS FILE EXISTS BECAUSE A CONTRACT TEST IS NOT A WIRING TEST, and that gap
has now cost this repo twice: once when the vLLM lever was asserted against
`translate()`'s env dict while `build_config` never read it, and once when the
agent loop and the domain loop shared `teacher_max_new` by omission rather
than by decision.

A test that the registry contains "agent_teacher_max_new" proves a name
exists. A test that reads `inspect.getsource` proves a line was typed. Neither
proves a generator ever sees the number. So these tests call the loops, catch
the teacher at construction, and assert the budget the teacher is HOLDING -
and assert it differs from the general fallback, so reverting the wiring to
`config.teacher_max_new` fails here instead of passing quietly and biasing a
corpus nine hours into a build.
"""

import dataclasses
import sys
import types

import pytest

from ms_moe_maker.config import pipeline as config
from ms_moe_maker.data import synth
from ms_moe_maker.eval import harness


class _Caught(Exception):
    """Raised by the fake teacher so the loop stops at construction."""

    def __init__(self, kwargs):
        super().__init__("caught")
        self.kwargs = kwargs


def _recorder():
    """A stand-in teacher that records its kwargs and aborts the loop."""

    class _Fake:
        def __init__(self, cfg, model=None, max_new_tokens=None, **rest):
            raise _Caught({"model": model, "max_new_tokens": max_new_tokens})

    return _Fake


def _cfg(tmp_path, **over):
    """A REAL PipelineConfig with every budget set to a different number.

    Built rather than faked: a hand-rolled namespace has to guess which
    attributes each loop touches on its way to the teacher (hf_home,
    base_safe, num_agent_samples, ...), and a guess that comes up short fails
    the test for a reason that has nothing to do with budgets. Going through
    build_config also means this test exercises the resolution path a real run
    takes, so a fallback that stops working shows up here too.

    The four numbers are DISTINCT on purpose: if any two matched, a loop
    reading the wrong knob would still pass and the assertion would be
    decoration.
    """
    from ms_moe_maker.config.recipe import parse
    rec, _ = parse({
        "schema_version": 1, "name": "t", "size": "0.5B",
        "roots": {"data": str(tmp_path / "data-{size}"),
                  "output": str(tmp_path / "run-{size}")},
        "budget": {"teacher_max_new": 111,
                   "agent_teacher_max_new": 222,
                   "domain_teacher_max_new": 333,
                   "reasoning_teacher_max_new": 444},
        "experts": [
            {"name": "deliberation",
             "source": {"kind": "synth", "reasoning": True}},
            {"name": "lore", "source": {"kind": "synth"}},
        ],
    })
    # PipelineConfig is FROZEN, which is the right shape for a thing whose
    # every field is fingerprinted into build_id - a config you can edit
    # after the stamp is a build that cannot describe itself.
    return dataclasses.replace(config.build_config(rec), force=True, **over)


def _budget_of(loop, tmp_path, monkeypatch, use_vllm, **kw):
    """Run `loop` far enough to build a teacher; return the budget it got."""
    monkeypatch.setattr(synth, "_HFTeacher", _recorder())
    monkeypatch.setattr(synth, "_VLLMTeacher", _recorder())
    cfg = _cfg(tmp_path, use_vllm=use_vllm)
    with pytest.raises(_Caught) as caught:
        loop(cfg, **kw)
    return caught.value.kwargs["max_new_tokens"]


@pytest.mark.parametrize("use_vllm", [False, True], ids=["hf", "vllm"])
class TestEachLoopGeneratesAgainstItsOwnBudget:
    """Three loops, three budgets, and no loop may reach for another's.

    The agent prompt carries a whole tool surface and answers with one JSON
    object. A domain prompt is a question and answers with a page of prose. A
    reasoning prompt answers with a think block AND an answer. One number
    cannot be right for all three, and the run that proved it capped the
    shared knob at 512 for the tool calls: 66% of domain generations hit the
    cap, and because a truncated generation is DISCARDED rather than trimmed,
    that corpus silently became the third of its material short enough to fit.
    The eval called it `NO ROUTER SIGNAL ... the fix is upstream` without
    being able to say upstream of what.
    """

    def test_the_agent_loop_uses_agent_teacher_max_new(self, tmp_path,
                                                       monkeypatch, use_vllm,
                                                       no_transformers):
        got = _budget_of(synth.generate_agent_traces, tmp_path, monkeypatch,
                         use_vllm, expert_name="agentcore", n=4)
        assert got == 222, got
        assert got != 111, "fell back to the shared teacher_max_new"

    def test_the_domain_loop_uses_domain_teacher_max_new(self, tmp_path,
                                                         monkeypatch, use_vllm,
                                                         no_transformers):
        got = _budget_of(synth.generate_domain_traces, tmp_path, monkeypatch,
                         use_vllm, expert_name="lore", n=4)
        assert got == 333, got
        assert got != 111, "fell back to the shared teacher_max_new"

    def test_the_reasoning_loop_uses_reasoning_teacher_max_new(
            self, tmp_path, monkeypatch, use_vllm, no_transformers):
        got = _budget_of(synth.generate_reasoning_traces, tmp_path, monkeypatch,
                         use_vllm, expert_name="deliberation", n=4)
        assert got == 444, got
        assert got != 111, "fell back to the shared teacher_max_new"


class TestTheFallbackKeepsOldRecipesMeaningTheSameThing:
    """Splitting a knob must not move a recipe that only set the old one."""

    @staticmethod
    def _built(blocks):
        from ms_moe_maker.config.recipe import parse
        body = {"schema_version": 1, "name": "t", "size": "0.5B",
                "experts": [{"name": "python",
                             "source": {"kind": "stack", "language": "Python"}}]}
        body.update(blocks)
        rec, _ = parse(body)
        return config.build_config(rec)

    def test_setting_only_the_general_knob_moves_all_three(self):
        cfg = self._built({"budget": {"teacher_max_new": 777}})
        assert cfg.teacher_max_new == 777
        assert cfg.agent_teacher_max_new == 777
        assert cfg.domain_teacher_max_new == 777

    def test_a_loop_knob_overrides_the_fallback_without_touching_the_others(self):
        cfg = self._built({"budget": {"teacher_max_new": 777,
                                      "domain_teacher_max_new": 4096}})
        assert cfg.domain_teacher_max_new == 4096
        assert cfg.agent_teacher_max_new == 777
        assert cfg.teacher_max_new == 777

    def test_an_untouched_recipe_keeps_the_shipped_defaults(self):
        cfg = self._built({})
        assert cfg.teacher_max_new == 512
        assert cfg.agent_teacher_max_new == 512
        assert cfg.domain_teacher_max_new == 512
        assert cfg.reasoning_teacher_max_new == 1024

    def test_unbounded_survives_build_config_as_zero(self):
        """`_knob`'s sentinel test is `< 0`, which is the entire reason 0 could
        be given a meaning at all. If it ever becomes `<= 0`, unbounded
        silently becomes the default and nothing says so."""
        cfg = self._built({"budget": {"domain_teacher_max_new": 0,
                                      "reasoning_teacher_max_new": 0}})
        assert cfg.domain_teacher_max_new == 0
        assert cfg.reasoning_teacher_max_new == 0

    def test_unbounded_on_the_fallback_reaches_the_loops_that_inherit_it(self):
        cfg = self._built({"budget": {"teacher_max_new": 0}})
        assert cfg.agent_teacher_max_new == 0
        assert cfg.domain_teacher_max_new == 0


class TestTheAdviceNamesTheKnobThatWouldHelp:
    """The NOTE fired seven times in a real build naming the wrong knob.

    It said "Raise budget.teacher_max_new" from inside the domain loop, which
    was true only because both loops read it. After the split, following that
    advice would raise the agent loop's ceiling and leave the starved loop
    exactly where it was - and the person reading it has no way to know.
    """

    @staticmethod
    def _source(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_agent_note_names_the_agent_knob(self):
        src = self._source(synth.generate_agent_traces)
        assert "budget.agent_teacher_max_new" in src
        assert "Raise budget.teacher_max_new" not in src

    def test_the_domain_note_names_the_domain_knob(self):
        src = self._source(synth.generate_domain_traces)
        assert "budget.domain_teacher_max_new" in src
        assert "Raise budget.teacher_max_new" not in src

    def test_the_reasoning_note_names_the_reasoning_knob(self):
        src = self._source(synth.generate_reasoning_traces)
        assert "budget.reasoning_teacher_max_new" in src

    def test_every_note_still_says_how_to_spell_unbounded(self):
        """The advice is the only place the 0 spelling is discoverable from a
        running build, and it was advice the validator used to refuse."""
        for fn in (synth.generate_agent_traces, synth.generate_domain_traces,
                   synth.generate_reasoning_traces):
            assert "0 = as far" in self._source(fn), fn.__name__
            assert "engine window allows" in self._source(fn), fn.__name__


class TestTheEvalResolvesUnboundedBeforeAnythingReadsIt:
    """`0` must not survive into the generation loop, and the reason is sharp.

    Six places downstream read this as a real budget. `model.generate` emits
    nothing against 0. `_truncate_reference` leaves the reference at full
    length while the generation is empty, so BLEU's brevity penalty scores the
    empty string against a whole file. And the cut-off counter asks
    `generated >= budget`, which is true of EVERY generation against 0 - a run
    would report sixteen of sixteen samples "cut off by the token budget"
    having produced not one token, which reads exactly like a broken model.

    The real run that motivated unbounded reported 16/16 cut off for five
    separate experts at a budget of 1024. A sentinel leaking through here
    would produce the same sentence for the opposite reason, and nothing in
    the report could tell the two apart.
    """

    class _Window:
        def __init__(self, window):
            self.config = type("C", (), {"max_position_embeddings": window})()

    class _LyingTokenizer:
        model_max_length = 1000000000000000019884624838656

    def test_an_explicit_budget_is_returned_untouched(self):
        """0 is the ONLY value this function is allowed to change."""
        for asked in (1, 256, 1024, 32000):
            assert harness._eval_budget(asked, 1024, self._Window(32768)) == asked

    def test_zero_becomes_the_window_less_the_prompt_cap(self):
        got = harness._eval_budget(0, 1024, self._Window(32768))
        assert got == 31744

    def test_zero_never_survives_as_zero(self):
        """The assertion the five downstream readers actually depend on."""
        for window in (512, 4096, 32768, 131072):
            for cap in (128, 1024, 4096):
                got = harness._eval_budget(0, cap, self._Window(window))
                assert got > 0, (window, cap, got)

    def test_a_prompt_cap_past_the_window_still_yields_a_usable_budget(self):
        assert harness._eval_budget(0, 99999, self._Window(4096)) >= 64

    def test_a_tokenizer_claiming_a_trillion_is_not_believed_here_either(self):
        """The guard lives in one place now; this proves the eval reaches it."""
        got = harness._eval_budget(0, 1024, None, self._LyingTokenizer())
        assert got == 4096 - 1024

    def test_junk_does_not_crash_the_eval_before_it_starts(self):
        assert harness._eval_budget(None, 1024, self._Window(4096)) == 256

    def test_the_recipe_spelling_reaches_the_resolver_as_zero(self):
        """End to end on the sentinel: recipe 0 -> config 0 -> a real budget.

        The two halves are deliberately tested together. eval_max_new_tokens
        passing 0 through is only correct BECAUSE something downstream
        resolves it, and _eval_budget resolving 0 is only reachable BECAUSE
        the sentinel is passed through. Either one alone is a bug.
        """
        cfg = types.SimpleNamespace(reasoning_experts=("deliberation",),
                                    reasoning_open="<think>",
                                    reasoning_close="</think>")
        passed_through = config.eval_max_new_tokens(cfg, 0)
        assert passed_through == 0
        resolved = harness._eval_budget(passed_through, 1024,
                                        self._Window(32768))
        assert resolved == 31744


class TestTheEvalActuallyCallsItsOwnResolver:
    """The seam is only worth having if the caller goes through it.

    HONEST DISCLOSURE: these are SOURCE checks, not behavioural ones, and a
    source check is the weaker instrument - it proves a line was typed, not
    that it runs. `eval_generation` cannot be called here because it needs
    torch, which this suite deliberately does not require (see
    test_no_leaked_imports for why that property is defended). The repo
    already uses inspect.getsource for exactly this situation in
    test_preflight_generator.

    They are here anyway because the alternative is worse. Extracting
    `_eval_budget` for testability created the precise gap this repo has been
    bitten by twice: a helper with thorough tests that nothing calls. The
    tests above would all pass with the call deleted from eval_generation, and
    a budget of 0 would then reach model.generate and produce a report saying
    every sample was cut off having generated nothing.
    """

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_eval_generation_routes_its_budget_through_the_resolver(self):
        assert "_eval_budget(" in self._src(harness.eval_generation)

    def test_eval_generation_does_not_resolve_the_window_itself(self):
        """ONE SEAM. A second copy of the window arithmetic is how the guard
        against a tokeniser claiming a trillion tokens ends up living in two
        places and being maintained in one."""
        src = self._src(harness.eval_generation)
        assert "unbounded_budget" not in src
        assert "max_position_embeddings" not in src

    def test_the_resolver_is_the_only_route_from_the_eval_to_the_window(self):
        import inspect
        mod = inspect.getsource(harness)
        assert mod.count("unbounded_budget") == 1, (
            "the eval reaches the shared window guard from exactly one place")

    def test_the_eval_never_asks_a_model_how_big_its_window_is(self):
        """NO SECOND COPY OF THE ARITHMETIC, not just no second call.

        This assertion is here because the first version of it was weaker than
        the property it claimed. It counted `unbounded_budget` calls, which a
        mutation that added an independent `max_position_embeddings` read
        walked straight past - a second window computation, maintained
        nowhere, in the module whose whole point was to stop having one.

        The window is asked for in exactly one place in this package
        (config.model_window) because the sanitising is the substance: a
        tokeniser will cheerfully report a window of 10**30, and a budget
        derived from that asks for a trillion tokens. Any raw read here is a
        copy of that guard, and copies of guards rot separately.
        """
        import inspect
        mod = inspect.getsource(harness)
        for raw in ("max_position_embeddings", "model_max_length"):
            assert raw not in mod, (
                f"{raw} read directly in the eval - go through "
                f"config.model_window so the sentinel guard stays in one place")


def _pf_lines(body, status=None):
    """Run preflight on a recipe body; return the budget-related checks."""
    from ms_moe_maker.config.recipe import parse
    from ms_moe_maker.run import preflight
    rec, _ = parse(body)
    pf = preflight.run(config.build_config(rec), rec, offline=True,
                       need_exporter=False)
    out = [c for c in pf.checks if "budget" in c.name]
    if status is not None:
        out = [c for c in out if c.status == status]
    return out


def _body(**over):
    """A minimal vLLM recipe with a reasoning expert."""
    body = {
        "schema_version": 1, "name": "t", "size": "0.5B",
        "runtime": {"use_vllm": True},
        "budget": {},
        "experts": [
            {"name": "deliberation",
             "source": {"kind": "synth", "reasoning": True}},
            {"name": "lore", "source": {"kind": "synth"}},
        ],
    }
    for key, value in over.items():
        body.setdefault(key, {}).update(value)
    return body


class TestPreflightSaysWhichCeilingEachLoopWillWriteUnder:
    """The answer used to be discoverable only from a running build.

    The domain loop hit its cap on 66% of generations in a real gauntlet. The
    NOTE that said so fired seven times - hours in, on a booked GPU, with the
    corpus already biased, because a generation cut at the cap is DISCARDED
    rather than trimmed. Three numbers on one line before anything is booked
    is the whole fix, and `domain 512` sitting next to `reasoning 4096` is
    legible as a mistake in a way that neither number is alone.
    """

    def test_all_three_budgets_are_reported_before_the_build(self):
        detail = _pf_lines(_body(budget={"agent_teacher_max_new": 512,
                                         "domain_teacher_max_new": 4096,
                                         "reasoning_teacher_max_new": 8192}))
        line = next(c.detail for c in detail if "agent" in c.detail)
        assert "agent 512" in line
        assert "domain 4096" in line
        assert "reasoning 8192" in line

    def test_unbounded_is_reported_as_a_word_not_a_zero(self):
        """`reasoning 0` reads as "generates nothing", which is the opposite."""
        line = next(c.detail for c in _pf_lines(
            _body(budget={"reasoning_teacher_max_new": 0}))
            if "agent" in c.detail)
        assert "reasoning unbounded" in line
        assert "reasoning 0" not in line

    def test_the_eval_budget_is_reported_too(self):
        lines = [c.detail for c in _pf_lines(_body(eval={"max_new_tokens": 2048}))]
        assert any("2048 tokens per sample" in d for d in lines)

    def test_the_automatic_eval_budget_says_it_is_automatic(self):
        lines = [c.detail for c in _pf_lines(_body())]
        assert any("automatic" in d for d in lines), lines


class TestPreflightCatchesTheUnboundedDowngrade:
    """Unbounded on vLLM is NOT "as much as it takes".

    It is `max_model_len` - runtime.vllm_max_len - minus the prompt. So a
    recipe carrying `reasoning_teacher_max_new: 4096` next to the default
    `vllm_max_len: 4096` gets LESS room by asking for unbounded than by naming
    the number, because the prompt comes out of the same window. Silently, and
    in the reverse direction from what the setting reads like.

    The warning fires on NOT HAVING TOUCHED the knob that decides it, not on
    the window being small - somebody who set the ceiling deliberately has
    already thought about this and gets the arithmetic as information instead.
    That distinction can only be asked of the recipe: a config cannot tell
    "set to 4096" from "defaulted to 4096".
    """

    def test_unbounded_with_an_unset_window_warns(self):
        warns = _pf_lines(_body(budget={"reasoning_teacher_max_new": 0}),
                          status="warn")
        assert warns, "unbounded against a default window said nothing"
        assert "default nobody chose" in warns[0].detail
        assert "vllm_max_len" in (warns[0].remedy or "")

    def test_a_deliberate_window_gets_the_arithmetic_without_the_scolding(self):
        body = _body(budget={"reasoning_teacher_max_new": 0},
                     runtime={"vllm_max_len": 32768})
        assert _pf_lines(body, status="warn") == []
        passes = [c.detail for c in _pf_lines(body, status="pass")]
        assert any("32768" in d and "minus the prompt" in d for d in passes), passes

    def test_a_bounded_budget_says_nothing_about_windows_at_all(self):
        """No unbounded budget, no warning - this must not fire on every run."""
        body = _body(budget={"reasoning_teacher_max_new": 4096})
        assert _pf_lines(body, status="warn") == []
        assert not any("minus the prompt" in c.detail for c in _pf_lines(body))

    def test_the_warning_names_every_budget_that_asked(self):
        warns = _pf_lines(_body(budget={"domain_teacher_max_new": 0,
                                        "reasoning_teacher_max_new": 0}),
                          status="warn")
        assert warns
        assert "domain_teacher_max_new" in warns[0].detail
        assert "reasoning_teacher_max_new" in warns[0].detail
        assert "agent_teacher_max_new" not in warns[0].detail

    def test_the_transformers_path_is_not_warned_about_a_vllm_window(self):
        body = _body(budget={"reasoning_teacher_max_new": 0},
                     runtime={"use_vllm": False})
        assert _pf_lines(body, status="warn") == []


class TestPreflightSaysWhatUnboundedEvalCosts:
    """Accepting it silently is how it gets left on for a shakedown."""

    def test_unbounded_eval_warns_with_the_cost(self):
        warns = _pf_lines(_body(eval={"max_new_tokens": 0}), status="warn")
        assert warns
        assert "unbounded" in warns[0].detail
        assert "linear" in (warns[0].remedy or "")

    def test_a_named_eval_budget_does_not_warn(self):
        assert _pf_lines(_body(eval={"max_new_tokens": 1024}),
                         status="warn") == []

    def test_the_cost_is_counted_not_called_expensive(self):
        """"Expensive" is an adjective; the recipe already knows the number.

        Every sample is generated once per expert and once more against the
        MoE, so the multiplier is knowable before anything is booked - and it
        is the difference between a setting somebody tries and one somebody
        regrets. A 0.5B with a 32k window unbounded is tens of thousands of
        tokens per generation; the same run capped at 1024 took under an hour.
        """
        body = _body(eval={"max_new_tokens": 0, "num_samples": 16})
        remedy = _pf_lines(body, status="warn")[0].remedy or ""
        # 16 samples x 2 experts x 2 surfaces
        assert "64 generations" in remedy, remedy
        assert "16 samples" in remedy and "2 experts" in remedy

    def test_the_count_scales_with_the_experts_it_will_actually_run(self):
        small = _pf_lines(_body(eval={"max_new_tokens": 0, "num_samples": 4}),
                          status="warn")[0].remedy
        assert "16 generations" in small, small
