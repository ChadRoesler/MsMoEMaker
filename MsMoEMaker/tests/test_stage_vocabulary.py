"""Pin the published stage vocabulary against the pipeline that runs.

`--describe` now answers "what does a build actually do, and what does each
step leave on disk". That answer is only worth publishing if it cannot drift
from the code, and the way it drifts is not subtle: someone adds a stage to
LABELS and the payload silently keeps describing an eight-stage pipeline.

SO THE CENTRAL TEST HERE COMPARES describe() TO plan()'s REAL OUTPUT, not to a
literal in this file. plan() is what runner.py and cli/build.py call - a fixture
built from the same list describe() reads would agree with itself and could
never fail. The rule this suite keeps learning: a contract test proves two ends
agree, never that either one is connected.

The optional flags get the same treatment from the other direction. A minimal
build has to emit every non-optional row and an expanded one has to emit them
all, so both claims are checked against plan() rather than asserted.
"""
import json

import pytest

from ms_moe_maker.run import stages as st


EXPERTS = ["python", "powershell", "bestiary"]


def expand(rows, experts):
    """describe()'s ids with the templated row expanded, in order."""
    out = []
    for row in rows:
        if row["parameter"]:
            out.extend(row["id"].format(**{row["parameter"]: e})
                       for e in experts)
        else:
            out.append(row["id"])
    return out


# -- the wiring: the published order is the order that runs ------------------

class TestThePublishedOrderIsReal:

    def test_a_maximal_build_walks_the_published_rows_in_order(self):
        """Every row, expanded, exactly as plan() emits them.

        The one assertion that makes publishing the order honest. abliterate on,
        a generated expert present, gates possible - so no row has an excuse to
        be absent and the sequence has to match position for position.
        """
        planned = [sid for sid, _ in st.plan(EXPERTS, synth=("bestiary",),
                                             gates=True, abliterate=True)]
        assert planned == expand(st.describe(), EXPERTS), (
            "the stage vocabulary --describe publishes is not the order a build "
            "runs in. A front-end drawing a plan from it would show the wrong "
            "shape, and would say '3 of 9' about a different nine.")

    def test_the_labels_match_too_not_just_the_ids(self):
        """plan() returns (id, label) pairs; both halves are published."""
        planned = dict(st.plan(EXPERTS, synth=("bestiary",),
                               gates=True, abliterate=True))
        for row in st.describe():
            if row["parameter"]:
                for expert in EXPERTS:
                    sid = row["id"].format(**{row["parameter"]: expert})
                    assert planned[sid] == row["label"].format(
                        **{row["parameter"]: expert})
            else:
                assert planned[row["id"]] == row["label"], (
                    f"{row['id']}: --describe publishes a different sentence "
                    f"than the one the run writes into its manifest.")

    @pytest.mark.parametrize("experts,synth,gates,abliterate", [
        (["solo"], (), False, False),
        (["solo"], (), True, False),
        (["a", "b"], (), True, False),
        (["a", "b"], ("b",), False, True),
    ])
    def test_every_required_row_appears_in_every_build(self, experts, synth,
                                                       gates, abliterate):
        """`optional: False` is a promise. This is where it gets kept.

        A row nobody can skip must appear whatever the recipe asks for -
        otherwise a viewer drawing the plan reports a stage as pending forever
        and the run never closes it.
        """
        planned = {sid for sid, _ in st.plan(experts, synth=synth, gates=gates,
                                             abliterate=abliterate)}
        for row in st.describe():
            if row["optional"] or row["parameter"]:
                continue
            assert row["id"] in planned, (
                f"{row['id']} is published as required and this build does not "
                f"run it; it should be optional: True.")

    def test_nothing_is_marked_optional_that_a_minimal_build_still_runs(self):
        """The flag has to be wrong in the other direction too, or it is noise.

        Without this, marking every row optional would pass the test above and
        make the flag meaningless.
        """
        minimal = {sid for sid, _ in st.plan(["solo"], synth=(), gates=False,
                                             abliterate=False)}
        for row in st.describe():
            if not row["optional"]:
                continue
            assert row["id"] not in minimal, (
                f"{row['id']} is published as optional but the smallest "
                f"possible build runs it anyway.")

    def test_a_build_can_emit_nothing_the_vocabulary_cannot_name(self):
        """The reverse-coverage check: no stage runs that --describe omits."""
        for experts, synth, gates, abl in [
                (EXPERTS, ("bestiary",), True, True),
                (["solo"], (), False, False),
                (["a", "b"], ("a", "b"), True, False)]:
            planned = {sid for sid, _ in st.plan(experts, synth=synth,
                                                 gates=gates, abliterate=abl)}
            unpublished = planned - set(expand(st.describe(), experts))
            assert not unpublished, (
                f"{sorted(unpublished)} run in a build and are absent from "
                f"--describe. A consumer would meet a stage it has no name, "
                f"label or artifact for.")


# -- coverage of the constants the payload is built from --------------------

class TestNothingIsLeftBehind:

    def test_every_labelled_stage_is_published(self):
        """Adding a stage to LABELS and not to describe() is the drift."""
        published = {row["id"] for row in st.describe()}
        missing = set(st.LABELS) - published
        assert not missing, (
            f"{sorted(missing)} have labels and are not in the published "
            f"vocabulary. LABELS is where a new stage gets named first, so "
            f"this is the one that catches it.")

    def test_every_artifact_directory_is_published(self):
        """The half a viewer needs most: which directory proves a stage ran."""
        published = {row["id"]: row["artifact"] for row in st.describe()}
        for stage_id, directory in st.ARTIFACTS.items():
            assert published.get(stage_id) == directory, (
                f"{stage_id} writes {directory!r} and --describe says "
                f"{published.get(stage_id)!r}. A reader confirming a manifest "
                f"against the disk would look in the wrong place.")

    def test_stages_with_no_artifact_say_so_rather_than_being_absent(self):
        """None is an answer. Omitting the key would make it a guess."""
        rows = {row["id"]: row for row in st.describe()}
        for stage_id in st.LABELS:
            if stage_id in st.ARTIFACTS:
                continue
            assert "artifact" in rows[stage_id]
            assert rows[stage_id]["artifact"] is None, (
                f"{stage_id} leaves no directory of its own; publishing one "
                f"would send a reader looking for something that never exists.")


# -- the row shape, which is the part a consumer writes code against --------

class TestTheRowShape:

    KEYS = {"id", "label", "artifact", "optional", "parameter"}

    def test_every_row_carries_every_key(self):
        """Uniform rows, so no consumer has to branch on which fields exist.

        `kinds` shipped as bare strings once and a front-end built a form off it
        with no summary to render. Same failure, smaller: a row missing
        `optional` reads as required.
        """
        for row in st.describe():
            assert set(row) == self.KEYS, (
                f"{row.get('id')!r} publishes {sorted(set(row))}, not "
                f"{sorted(self.KEYS)}.")

    def test_exactly_one_row_is_templated(self):
        parametrised = [r for r in st.describe() if r["parameter"]]
        assert [r["id"] for r in parametrised] == ["finetune.{expert}"]

    def test_the_template_is_expandable_everywhere_it_appears(self):
        """id, label and artifact must all take the same substitution.

        Publishing a template a consumer cannot expand is worse than publishing
        nothing: it looks usable.
        """
        row, = [r for r in st.describe() if r["parameter"]]
        arg = {row["parameter"]: "python"}
        assert row["id"].format(**arg) == st.finetune_id("python")
        assert row["label"].format(**arg) == st.finetune_label("python")
        assert row["artifact"].format(**arg) == st.artifact_for(
            st.finetune_id("python")), (
            "the published artifact template does not expand to the directory "
            "the run actually writes for that expert.")

    def test_the_parameter_names_the_brace_it_substitutes(self):
        """`parameter: "expert"` and `{expert}` are one fact, published twice."""
        row, = [r for r in st.describe() if r["parameter"]]
        assert "{" + row["parameter"] + "}" in row["id"]
        assert "{" + row["parameter"] + "}" in row["artifact"]

    def test_untemplated_rows_hold_no_braces_to_expand(self):
        for row in st.describe():
            if row["parameter"]:
                continue
            for field in ("id", "label", "artifact"):
                assert "{" not in (row[field] or ""), (
                    f"{row['id']}: {field} looks templated and publishes no "
                    f"parameter, so nobody can fill it in.")

    def test_describe_hands_back_a_fresh_list_each_time(self):
        """A consumer mutating the payload must not edit the vocabulary."""
        first = st.describe()
        first[0]["label"] = "vandalised"
        first.pop()
        assert st.describe()[0]["label"] != "vandalised"
        assert len(st.describe()) == len(first) + 1


# -- the contract surface: what a stranger's tooling actually reads ---------

class TestTheDescribePayload:

    def test_the_cli_publishes_the_vocabulary(self):
        from ms_moe_maker.__main__ import DESCRIBE
        assert DESCRIBE["stages"] == st.describe()

    def test_the_payload_survives_json(self):
        """--describe is one line of JSON or it is nothing.

        Same check `overrun_policies` carries: a tuple, a Path or a None-keyed
        dict passes an equality assertion in-process and breaks the contract on
        the wire.
        """
        from ms_moe_maker.__main__ import DESCRIBE
        assert json.loads(json.dumps(DESCRIBE))["stages"] == st.describe()

    def test_describe_still_answers_when_the_pipeline_cannot_be_imported(self,
                                                                        monkeypatch):
        """The one moment the contract most has to hold is a broken install.

        run/__init__.py imports the builder, so `from .run import stages` is not
        the free stdlib import it looks like. This proves the degradation is
        real rather than commented.
        """
        import sys
        import ms_moe_maker.__main__ as m
        monkeypatch.setitem(sys.modules, "ms_moe_maker.run", None)
        assert m._box_stages() == []

    def test_a_degraded_payload_is_still_valid_json_with_every_other_key(self,
                                                                        monkeypatch):
        import sys
        import ms_moe_maker.__main__ as m
        monkeypatch.setitem(sys.modules, "ms_moe_maker.run", None)
        payload = {**m.DESCRIBE, "stages": m._box_stages()}
        round_tripped = json.loads(json.dumps(payload))
        assert round_tripped["stages"] == []
        assert round_tripped["commands"] and round_tripped["events"]
