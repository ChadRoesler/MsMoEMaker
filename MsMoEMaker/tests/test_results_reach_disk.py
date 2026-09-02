"""A measurement that never reaches disk did not happen.

THE BUG THIS PINS, and it is a class, not an instance. `save_eval_report` and
`eval_from_manifest` are a matched pair - one writes the JSON the other reads -
and for the entire life of the file NEITHER HAD A CALLER outside this suite.
They round-tripped in tests and nowhere else. So `ms-moe-maker eval` ran the
whole measurement, printed a beautiful table, and persisted nothing: forty
minutes of generation lived in a terminal scrollback until the window closed.

And because the only exercise they ever got was two tests that populate
`stages` and nothing else, the writer could silently drop `routing` - the
enrichment table, the JS divergence, the gate confidence, the think-block
segmentation, the headline claim the command exists to make - and the suite
stayed green over the loss. A test that never sets a field cannot notice the
field is gone. THAT is the shape of the hole, and it is the fifth time in this
codebase that data has been present and not surfaced.

So there are two guards here, and they are different guards:

  * EVERY DATA FIELD ROUND-TRIPS, enumerated from the dataclass rather than
    listed by hand - so the next field added to EvalReport is covered the day
    it is added, whether or not anyone remembers this file.
  * THE WRITERS HAVE CALLERS. A perfect writer nobody calls is indistinguish-
    able, from the user's chair, from no writer at all.
"""
import ast
import dataclasses
from pathlib import Path

import pytest

from ms_moe_maker.eval import harness as ev
from ms_moe_maker.eval.harness import (EVAL_REPORT_NAME, EvalReport, EvalResult,
                                       eval_from_manifest, save_eval_report)
from ms_moe_maker.train import experts as ex

PKG = Path(ev.__file__).parent.parent


def _full_report():
    """An EvalReport with EVERY field carrying a distinctive non-default value.

    Distinctive on purpose: a field that round-trips as its default is
    indistinguishable from a field that was dropped, which is precisely how
    `routing` hid.
    """
    r = EvalReport(
        ok=True,
        message="a message",
        dead_experts=["rust"],
        undiscriminating=["markdown"],
        caveats=["measured on three rows"],
        unmeasured=["csharp: no compiler"],
        experts={"status": "ok", "divergence": {"python": 0.42},
                 "cross_loss": {"python": {"markdown": 3.1}},
                 "findings": ["two experts are nearly identical"]},
        routing={"experts": {"python": {"own_share": 0.7, "enrichment": 2.1}},
                 "mean_js_bits": 0.44, "mean_enrichment": 2.1,
                 "p_value": 0.031, "moe_layers": 12, "top_k": 2,
                 "mean_gate_confidence": 0.33, "uniform_confidence": 0.5,
                 "think_segments": {"python": {"verdict": "relay",
                                               "swing": 0.21}}},
        build_id="abc123def456",
        generated=1234567890.0,
    )
    r.stages["python"] = EvalResult(
        expert_name="python", domain="py", exact_match=0.9, rouge1=0.8,
        bleu=0.7, avg_length=64.0, scored_samples=3, attempted_samples=20,
        reasoned=0.55, capped_generations=2, status="done", note="a note")
    return r


# -- guard one: nothing is dropped on the way to disk ----------------------

def test_every_field_of_the_report_survives_the_round_trip(tmp_path):
    """Enumerated from the dataclass, so a NEW field is covered on arrival.

    Listing the fields by hand is how `routing` was lost: the hand-written
    list was written before routing existed and never revisited.
    """
    before = _full_report()
    save_eval_report(before, tmp_path / EVAL_REPORT_NAME,
                     build_id=before.build_id)
    after = eval_from_manifest(tmp_path)

    missing = []
    for f in dataclasses.fields(EvalReport):
        got = getattr(after, f.name)
        if not got:
            missing.append(f.name)
    assert not missing, (
        f"these fields did not survive save -> load: {missing}. A field the "
        f"writer forgets is a measurement the user paid GPU time for and "
        f"cannot read back.")


def test_the_routing_table_is_the_same_table_on_the_other_side(tmp_path):
    """The specific instance, kept alongside the general guard.

    `routing` is not just another field - it is the enrichment table, the JS
    divergence and the think-block segmentation, which is to say it is the
    ANSWER. It was absent from the writer entirely.
    """
    before = _full_report()
    save_eval_report(before, tmp_path / EVAL_REPORT_NAME)
    after = eval_from_manifest(tmp_path)
    assert after.routing == before.routing
    assert after.routing["think_segments"]["python"]["verdict"] == "relay", (
        "the think-block segmentation is nested two deep and is exactly the "
        "kind of thing a hand-written serializer flattens away")


def test_provenance_is_written_and_an_unstamped_report_reads_as_unknown(
        tmp_path):
    """build_id absent must read as UNKNOWN, never as agreement.

    A reader comparing this against the manifest's build_id is asking "did
    this eval measure THIS build". An empty string answers "I do not know",
    and the one answer it must never accidentally give is "yes".
    """
    save_eval_report(_full_report(), tmp_path / EVAL_REPORT_NAME,
                     build_id="deadbeef1234")
    stamped = eval_from_manifest(tmp_path)
    assert stamped.build_id == "deadbeef1234"
    assert stamped.generated > 0, "an undated claim has no age"

    save_eval_report(_full_report(), tmp_path / EVAL_REPORT_NAME)
    plain = eval_from_manifest(tmp_path)
    assert plain.build_id == "", "unknown provenance must be empty, not a guess"


def test_a_garbled_generated_stamp_does_not_take_the_whole_report_down(
        tmp_path):
    """Provenance is metadata about the claim, not the claim.

    A report whose timestamp is a string is still a report full of real
    numbers, and refusing to load it would lose the measurement to protect a
    field nobody scores on.
    """
    import json
    p = tmp_path / EVAL_REPORT_NAME
    save_eval_report(_full_report(), p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["generated"] = "tuesday"
    p.write_text(json.dumps(raw), encoding="utf-8")

    back = eval_from_manifest(p.parent)
    assert back.generated == 0.0
    assert back.routing, "the numbers must survive a bad timestamp"


# -- guard two: the writers are actually wired up --------------------------

def _names_used(path: Path):
    """Every identifier and attribute name used in a file, docstrings out.

    Docstrings are stripped for the same reason test_eval_memory strips them:
    a guard that matches the prose EXPLAINING the guard is not a guard, and
    this file's own comments name both symbols repeatedly.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            node.value.value = ""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.name.rsplit(".", 1)[-1])
    return out


def _package_files(exclude: str):
    for p in sorted(PKG.rglob("*.py")):
        if p.name != exclude and "__pycache__" not in p.parts:
            yield p


@pytest.mark.parametrize("symbol, defined_in", [
    ("save_eval_report", "harness.py"),
    ("GATE_REPORT_NAME", "experts.py"),
])
def test_the_artifact_writers_have_a_caller(symbol, defined_in):
    """Built-and-never-called is this codebase's recurring disease.

    `save_eval_report` had no caller for the life of the file. `EvalSidecar` -
    a complete append-only JSONL writer with a header, a footer and a five-word
    verdict vocabulary - still has none, and seren-theatre carries a complete
    READER for it: two implementations of one wire format, a contract test
    pinning them together, green forever, and not one byte has ever travelled
    between them. A contract test proves the two ends agree. It cannot prove
    either end is connected to anything.

    So: if the package defines a thing whose only purpose is to put a result
    on disk, some other module in the package has to reach for it.
    """
    users = [p.relative_to(PKG).as_posix()
             for p in _package_files(exclude=defined_in)
             if symbol in _names_used(p)]
    assert users, (
        f"{symbol} is defined in {defined_in} and referenced nowhere else in "
        f"the package. A writer nobody calls is, from the user's chair, "
        f"indistinguishable from no writer at all.")


def test_the_two_filenames_are_named_once_each():
    """The literal must live beside the shape it describes, not at call sites.

    seren-theatre finds these files by NAME, which makes each string a wire
    format between two packages. A second spelling of it anywhere is a drift
    waiting to happen, and the drift would present as "the viewer shows no
    eval results" - which reads exactly like "there was no eval".
    """
    assert EVAL_REPORT_NAME == "eval_report.json"
    assert ex.GATE_REPORT_NAME == "gate_experts.json"

    for p in _package_files(exclude="harness.py"):
        src = p.read_text(encoding="utf-8")
        assert '"eval_report.json"' not in src, (
            f"{p.name} hardcodes the eval report filename; import "
            f"EVAL_REPORT_NAME instead")
    for p in _package_files(exclude="experts.py"):
        src = p.read_text(encoding="utf-8")
        assert '"gate_experts.json"' not in src, (
            f"{p.name} hardcodes the gate report filename; import "
            f"GATE_REPORT_NAME instead")
