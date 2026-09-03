#!/usr/bin/env python3
"""Repair a synth corpus written by the broken reasoning split.

WHAT IT REPAIRS, and why the rows are worth repairing rather than regenerating.

Every row is `<think>{think}</think>\\n{answer}` — the shape the generator
writes. Two things went wrong upstream, and both left the ORIGINAL text intact
inside the row, just filed under the wrong heading:

  shape A   think ends with a stray `</think>` and nothing follows it.
            The corpus row reads `…</think></think>\\nanswer`.
            The stored answer was already the right answer. Drop the tag.

  shape B   think ends with a stray `</think>` and the REAL ANSWER follows it,
            because the teacher wrote a full answer and then, obeying the
            `ANSWER:` instruction, a one-line summary — and the marker split
            cut at the summary. The stored "answer" is that summary; the real
            one is sitting in the think half.
            Promote it. Drop the summary.

So nothing has to be re-generated: one row of the sample had an entire Python
function inside `think` and the single word `calculate_average` as its answer,
and the function is still right there.

DRY RUN BY DEFAULT. It rewrites a corpus somebody paid GPU-hours for; it prints
what it would do and writes nothing unless you pass --write, and even then the
original is kept as `<name>.bak`.

    python3 tools/repair_reasoning_corpus.py gauntlet-data/0.5B/*.jsonl*
    python3 tools/repair_reasoning_corpus.py --write  gauntlet-data/0.5B/x.jsonl

Rows it cannot make sense of are LEFT ALONE and counted, never guessed at.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

OPEN, CLOSE = "<think>", "</think>"
MARKER = "ANSWER:"
# The doubled stop every loop used to write: a Qwen template closes the turn
# and the generator appended eos on top of it.
DOUBLE_EOS = re.compile(r"(<\|im_end\|>)\s*\1\s*$")


def repair(text: str):
    """(new_text, what_changed). `what_changed` is "" when the row was fine."""
    notes = []

    fixed = DOUBLE_EOS.sub(r"\1", text)
    if fixed != text:
        notes.append("double-eos")
        text = fixed

    # WHERE THE STRAY TAG ACTUALLY IS, and getting this backwards cost a
    # round: it is not inside the think half, it is the SECOND closer.
    #
    # The generator wrote `<think>{think}</think>` around a `think` that
    # already ended in a closer, so a broken row carries two (or more):
    #
    #   shape A   <think> reasoning </think></think> answer
    #   shape B   <think> reasoning </think> REAL ANSWER </think> summary
    #
    # So the FIRST closer ends the true reasoning, the LAST is the one the
    # re-emit added, and whatever sits between them is the answer the marker
    # split buried. One closer means the row is fine.
    start = text.find(OPEN)
    if start == -1:
        return text, ",".join(notes)
    body_start = start + len(OPEN)
    first = text.find(CLOSE, body_start)
    last = text.rfind(CLOSE)
    if first == -1 or first == last:
        return text, ",".join(notes)

    reasoning = text[body_start:first].strip()
    buried = text[first + len(CLOSE):last].strip()
    tail = text[last + len(CLOSE):]

    if buried:
        # shape B: the real answer was filed under think, and `tail` holds
        # the one-line summary the marker split promoted. Keep the answer.
        answer = buried
        notes.append("promoted-buried-answer")
    else:
        # shape A: only the tag was misplaced.
        answer = tail.strip()
        notes.append("dropped-stray-tag")

    # A trailing ANSWER: summary is the teacher obeying an instruction we
    # should not have given it. It is not part of the answer.
    cut = answer.upper().find(MARKER)
    if cut > 0:
        answer = answer[:cut].strip()
        notes.append("dropped-summary")
    elif cut == 0:
        answer = answer[len(MARKER):].strip()

    # THE STOP TOKEN STAYS ON THE END. It is part of the row, not part of the
    # answer - and on shape B the answer came out of the MIDDLE of the row,
    # so it does not carry one. Pull it off whichever half has it and put it
    # back last, exactly once.
    stop = re.search(r"((?:<\|im_end\|>|</s>)\s*)$", text)
    suffix = stop.group(1) if stop else ""
    if suffix:
        answer = re.sub(r"(?:<\|im_end\|>|</s>)\s*$", "", answer).rstrip()

    if not reasoning or not answer:
        # Refuse rather than write half a row. Counted, never guessed at.
        return None, "unrepairable"

    return (f"{text[:start]}{OPEN}{reasoning}{CLOSE}\n{answer}{suffix}",
            ",".join(notes))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".jsonl or .jsonl.partial corpora")
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite (keeps <name>.bak)")
    ap.add_argument("--show", type=int, default=1,
                    help="print N repaired rows per file (default 1)")
    a = ap.parse_args(argv)

    worst = 0
    for path in a.paths:
        if not os.path.isfile(path):
            print(f"  ! {path}: not a file")
            continue
        rows, counts, shown, out = 0, {}, 0, []
        broken = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                try:
                    doc = json.loads(line)
                    text = doc["text"]
                except (ValueError, KeyError):
                    counts["unparseable"] = counts.get("unparseable", 0) + 1
                    out.append(line)
                    continue
                new, note = repair(text)
                if new is None:
                    counts["unrepairable"] = counts.get("unrepairable", 0) + 1
                    out.append(line)
                    continue
                for n in filter(None, note.split(",")):
                    counts[n] = counts.get(n, 0) + 1
                if note:
                    broken += 1
                    if shown < a.show:
                        shown += 1
                        print(f"\n--- {path} row {rows} [{note}] "
                              f"---\n{new[:600]}\n")
                doc["text"] = new
                out.append(json.dumps(doc, ensure_ascii=False) + "\n")

        pct = 100.0 * broken / max(rows, 1)
        print(f"  {path}: {rows} rows, {broken} touched ({pct:.0f}%)  "
              + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        worst = max(worst, broken)

        if a.write and broken:
            os.replace(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(out)
            print(f"    wrote {path} (original kept at {path}.bak)")
        elif broken:
            print("    dry run - pass --write to rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
