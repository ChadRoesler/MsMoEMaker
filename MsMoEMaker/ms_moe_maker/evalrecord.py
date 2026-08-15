"""The eval sidecar - writing half. Un-black-boxing the black box.

WHY THIS IS NOT IN THE MANIFEST.

The manifest is a small file rewritten on every stage transition. Eval detail
is the opposite shape: dozens to hundreds of records, each carrying a whole
generation, written steadily over minutes. Putting them in the manifest would
mean rewriting a growing document hundreds of times, and would make the file
that answers "where is this run" expensive to read.

So eval detail is a SIDECAR, and the manifest points at it: the eval stage's
`artifact` field carries this file's name. One small authoritative status file,
one append-only stream of evidence beside it.

APPEND-ONLY, AND FLUSHED PER RECORD, ON PURPOSE. That is what makes the eval
watchable while it runs rather than a verdict that lands at the end. A reader
tailing this file sees question 7 of 30 as it happens - which is the entire
point of the feature. Never rewrite a line; never seek backwards.

────────────────────────────────────────────────────────────────────────────
THE VERDICT VOCABULARY, AND THE ONE DISTINCTION THAT MATTERS

    pass          measured, met the bar
    fail          measured, did not meet the bar
    unmeasurable  COULD NOT BE MEASURED. Not a failure.
    error         the harness itself broke on this item
    skipped       deliberately not run

`unmeasurable` exists because of a real result. An eval run reported C# 0/10
and the honest reading of that number is "this model cannot write C#". The
model was fine. The harness was shelling out to `csc`/`mcs`, neither of which
was installed, and a missing compiler was being recorded as ten wrong answers.

A score that folds "we could not check" into "it got it wrong" is not a
measurement, it is a confidently wrong claim about a model - and a confidently
wrong claim is the one failure mode this whole viewer exists to prevent. So
the denominator is MEASURED, never TOTAL, and the two are kept apart at the
format level where nobody can quietly average them together later.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pinned against seren_theatre.evalrecord by SerenTheatre's
# tests/test_eval_contract.py - the same bargain as the run manifest. Two
# implementations of one wire format is the normal cost of a protocol; drift
# is the cost of that cost, and the contract test is where it gets paid.
SCHEMA_VERSION = 1

SIDECAR_PREFIX = "eval-"
SIDECAR_SUFFIX = ".jsonl"

# Line kinds. Every line is a JSON object carrying "kind", so a reader can
# stream the file without knowing how many of each to expect, and so a future
# kind can be added without any existing reader mis-parsing it.
KIND_HEADER = "header"
KIND_RECORD = "record"
KIND_FOOTER = "footer"
KINDS = (KIND_HEADER, KIND_RECORD, KIND_FOOTER)

PASS = "pass"
FAIL = "fail"
UNMEASURABLE = "unmeasurable"
ERROR = "error"
SKIPPED = "skipped"

VERDICTS = (PASS, FAIL, UNMEASURABLE, ERROR, SKIPPED)

# THE LOAD-BEARING TUPLE. Only these two are a measurement, so only these two
# form the denominator of a score. Adding UNMEASURABLE here would recreate the
# C# 0/10 result exactly; the contract test asserts it is absent.
MEASURED = (PASS, FAIL)
UNMEASURED = (UNMEASURABLE, ERROR, SKIPPED)


def sidecar_name(eval_id: str) -> str:
    """The filename for an eval run. Goes in the manifest stage's `artifact`."""
    return f"{SIDECAR_PREFIX}{eval_id}{SIDECAR_SUFFIX}"


def score(counts: Dict[str, int]) -> Optional[float]:
    """Fraction of MEASURED items that passed, or None if nothing was measured.

    None rather than 0.0, and the distinction is the whole reason this
    function exists rather than being written inline at three call sites. A
    suite where every item was unmeasurable scored 0.0 under the old reading,
    which is indistinguishable from a suite where the model got everything
    wrong. One of those is a broken harness and the other is a broken model.
    """
    measured = sum(counts.get(v, 0) for v in MEASURED)
    if not measured:
        return None
    return counts.get(PASS, 0) / measured


class EvalSidecar:
    """Append-only writer. Use as a context manager so the footer always lands.

        with EvalSidecar(run_dir, "smoke-1", suite="code_fluency") as sc:
            sc.record(item_id="posh-03", prompt=p, generation=g,
                      verdict=PASS, validator="parse")

    The footer is what tells a reader the run ENDED rather than died. A file
    with no footer and no recent writes is a crashed eval, and a viewer should
    say so instead of showing a spinner forever - the same stall reasoning the
    run manifest uses.
    """

    def __init__(self, run_dir: Path, eval_id: str, *, suite: str = "",
                 rung: str = "", model: str = "", total: Optional[int] = None):
        self.run_dir = Path(run_dir)
        self.eval_id = str(eval_id)
        self.path = self.run_dir / sidecar_name(self.eval_id)
        self.suite = suite
        self.rung = rung
        self.model = model
        self.total = total
        self.counts: Dict[str, int] = {}
        self._seq = 0
        self._fh = None

    # -- lifecycle --
    def open(self) -> "EvalSidecar":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Line-buffered text append. encoding and newline are both explicit:
        # a generation can contain anything a model emits, and the default
        # locale codec on Windows would raise on the first non-cp1252
        # character - mid-eval, after the GPU time was already spent.
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        self._emit({
            "kind": KIND_HEADER,
            "schema_version": SCHEMA_VERSION,
            "eval_id": self.eval_id,
            "suite": self.suite,
            "rung": self.rung,
            "model": self.model,
            "total": self.total,
            "started": time.time(),
        })
        return self

    def __enter__(self) -> "EvalSidecar":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        # ok=False on an exception, so a crashed eval is not silently footed as
        # a clean finish. Returning False re-raises, which is what we want.
        self.close(ok=exc_type is None)
        return False

    def close(self, ok: Optional[bool] = None) -> None:
        if self._fh is None:
            return
        self._emit({
            "kind": KIND_FOOTER,
            "ended": time.time(),
            "counts": dict(self.counts),
            "measured": sum(self.counts.get(v, 0) for v in MEASURED),
            "score": score(self.counts),
            "ok": ok,
        })
        self._fh.close()
        self._fh = None

    # -- the one write --
    def record(self, *, item_id: str, verdict: str, prompt: str = "",
               generation: str = "", validator: str = "", reason: str = "",
               expected: Any = None, language: str = "",
               elapsed: Optional[float] = None,
               extra: Optional[Dict[str, Any]] = None) -> None:
        """One eval item, with everything needed to argue with the verdict.

        `reason` is not optional in spirit. For a fail it is what the validator
        objected to; for an unmeasurable it is WHY - "no C# compiler on PATH"
        is the sentence that turns a mysterious zero into a five-minute fix.
        """
        if self._fh is None:
            self.open()
        self._seq += 1
        self.counts[verdict] = self.counts.get(verdict, 0) + 1
        row: Dict[str, Any] = {
            "kind": KIND_RECORD,
            "seq": self._seq,
            "item_id": str(item_id),
            "suite": self.suite,
            "language": language,
            "prompt": prompt,
            "generation": generation,
            "validator": validator,
            "verdict": verdict,
            "reason": reason,
            "expected": expected,
            "elapsed": elapsed,
            "ts": time.time(),
        }
        if extra:
            # Namespaced so a leaf's own fields can never collide with a future
            # top-level key and change the meaning of an existing one.
            row["extra"] = extra
        self._emit(row)

    def _emit(self, obj: Dict[str, Any]) -> None:
        assert self._fh is not None
        # ensure_ascii=False keeps generations readable in the file; a person
        # opening this in an editor is a first-class use, not an afterthought.
        # default=str so an unexpected object never kills an eval mid-run -
        # losing fidelity on one field beats losing the run.
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        # Flush per line: this is what makes the file watchable live. Without
        # it a reader sees nothing until the buffer happens to fill, and the
        # streaming property - the whole reason for JSON Lines - is lost.
        self._fh.flush()
