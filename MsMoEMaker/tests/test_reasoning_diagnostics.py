"""Diagnostics that were firing on their own denominator.

`reasoned` has a documented "never asked" value of -1, and three separate
readers already handled it correctly: the quality table prints '-' instead of
0.00, the row flag skips "does not reliably reason", and the wrong-tag-style
caveat leaves the row out of its average. All three were unreachable, because
the run's reasoning style was handed to EVERY expert - so on any run that
reasons at all, every expert was measured for a think block it was never asked
to write, scored 0.00, and the sentinel never appeared.

What that cost on a real nine-expert run with one reasoning expert: eight rows
of 0.00 each stamped "does not reliably reason", and a caveat that averaged
0.38 with seventeen zeros to announce "almost nothing emitted a think block ...
or the WRONG TAG STYLE" - printed three lines under a row whose 0.38 proves the
tags parse fine. The alarm fired by construction on every realistic run, which
is the same as it never firing.
"""

import types

from ms_moe_maker.eval.harness import (style_for_domain,
                                       wrong_tag_style_caveat)


STYLE = types.SimpleNamespace(open="<think>", close="</think>")


def _row(reasoned):
    return types.SimpleNamespace(reasoned=reasoned)


def _the_real_run(never_asked=-1.0):
    """Chad's gauntlet: nine experts, one of them asked to reason.

    `never_asked` is the knob this whole file turns on. -1.0 is what the
    sentinel is for; 0.0 is what the code used to produce, and the tests below
    show the two produce opposite reports from identical model behaviour.
    """
    rows = {"deliberation": _row(0.38)}
    for name in ("agentcore", "csharp", "docs", "lore", "notes",
                 "powershell", "python", "shell"):
        rows[name] = _row(never_asked)
        rows[f"moe/{name}"] = _row(never_asked)
    rows["moe/deliberation"] = _row(0.0)
    return rows


class TestOnlyTheRowsThatWereAskedGetMeasured:
    """The style is chosen per DOMAIN, not per run."""

    def test_a_reasoning_expert_is_scored_against_the_style(self):
        assert style_for_domain("deliberation", STYLE, {"deliberation"}) is STYLE

    def test_an_expert_never_asked_to_reason_gets_no_style(self):
        """None here is what produces the -1 sentinel downstream."""
        assert style_for_domain("notes", STYLE, {"deliberation"}) is None

    def test_the_moe_surface_follows_the_domain_not_the_model(self):
        """moe/deliberation IS asked - the gap between it and deliberation's
        own row is a finding, not noise: discipline present in the specialist
        and lost at the stitch. moe/notes is not asked."""
        assert style_for_domain("deliberation", STYLE, {"deliberation"}) is STYLE
        assert style_for_domain("notes", STYLE, {"deliberation"}) is None

    def test_a_run_with_no_reasoning_experts_asks_nobody(self):
        for domain in ("deliberation", "notes", "python"):
            assert style_for_domain(domain, STYLE, set()) is None
            assert style_for_domain(domain, STYLE, None) is None

    def test_no_style_configured_means_no_measurement_anywhere(self):
        assert style_for_domain("deliberation", None, {"deliberation"}) is None


class TestTheWrongTagStyleAlarmStoppedFiringOnItself:
    """An alarm that fires on every realistic run is not an alarm."""

    def test_the_real_run_no_longer_trips_it(self):
        """One expert reasoning at 0.38 is not "almost nothing emitted a think
        block" - it is one expert reasoning at 0.38."""
        assert wrong_tag_style_caveat(_the_real_run(), STYLE) is None

    def test_the_old_denominator_would_still_trip_it(self):
        """THE REGRESSION GUARD, and the reason this file exists.

        Identical model behaviour, one difference: the rows that were never
        asked carry 0.00 instead of the sentinel. That alone flips the report
        from silence to a wrong-tag-style alarm, which is proof the finding was
        a property of the denominator and not of anything the model wrote.
        """
        caveat = wrong_tag_style_caveat(_the_real_run(never_asked=0.0), STYLE)
        assert caveat is not None
        assert "WRONG TAG STYLE" in caveat

    def test_it_still_fires_when_the_rows_that_were_asked_are_all_low(self):
        """The alarm has to survive being made quieter, or the fix broke it."""
        rows = {"deliberation": _row(0.0), "moe/deliberation": _row(0.02)}
        for name in ("notes", "python"):
            rows[name] = _row(-1.0)
        caveat = wrong_tag_style_caveat(rows, STYLE)
        assert caveat is not None
        assert "<think>" in caveat and "reasoning.yaml" in caveat

    def test_it_says_how_many_rows_it_is_speaking_about(self):
        """The denominator travels with the average - the same rule the rest of
        this report already follows."""
        rows = {"deliberation": _row(0.0), "moe/deliberation": _row(0.0),
                "notes": _row(-1.0)}
        assert "2 row(s)" in wrong_tag_style_caveat(rows, STYLE)

    def test_no_style_means_nothing_to_be_wrong_about(self):
        assert wrong_tag_style_caveat(_the_real_run(never_asked=0.0), None) is None

    def test_a_run_where_nobody_was_asked_says_nothing(self):
        rows = {"notes": _row(-1.0), "python": _row(-1.0)}
        assert wrong_tag_style_caveat(rows, STYLE) is None

    def test_a_healthy_reasoning_run_says_nothing(self):
        rows = {"deliberation": _row(0.94), "moe/deliberation": _row(0.88)}
        assert wrong_tag_style_caveat(rows, STYLE) is None


class TestTheQualityLoopActuallyChoosesPerDomain:
    """HONEST DISCLOSURE: source checks. run_eval needs torch, which this
    suite deliberately does not require. They are here because the tests above
    would all pass with the per-domain choice deleted from the call sites - the
    exact gap that made these diagnostics wrong in the first place."""

    @staticmethod
    def _src():
        import inspect
        from ms_moe_maker.eval import harness
        return inspect.getsource(harness.run_eval)

    def test_both_surfaces_choose_their_style_per_domain(self):
        src = self._src()
        assert src.count("reasoning_style=_style_for(expert_name)") == 2, (
            "the specialist loop and the MoE loop must both choose per domain")

    def test_the_runs_style_is_not_handed_to_the_quality_loop_wholesale(self):
        """`reasoning_style=reasoning_style` in the quality loop IS the bug."""
        assert "reasoning_style=reasoning_style)" not in self._src()

    def test_the_caveat_is_reached_through_the_extracted_function(self):
        assert "wrong_tag_style_caveat(report.stages, reasoning_style)" in self._src()
