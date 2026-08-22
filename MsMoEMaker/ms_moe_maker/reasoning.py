"""Reasoning tag styles and model families — a living table, not a release.

WHY THIS IS A FILE AND NOT A DICT. A wrong tag style is a SILENT WRONG ANSWER,
not a crash: the splitter finds no delimiters, reports "did not reason", and
the whole `<think>` block gets scored as if it were the answer. So the day a
new model family ships a new delimiter, every number quietly goes wrong, and
waiting on a release to fix that is the wrong shape. Somebody should be able to
drop a yaml on the box and carry on.

Layers, later wins — the same ones the defaults table uses, because two files
that layer differently is two things to explain and two things to get wrong:

    1. FLOOR             below. ONE style. Not a copy of the shipped table -
                         a panic minimum, so a missing or broken file can
                         never take a build down.
    2. packaged          ms_moe_maker/reasoning.yaml. THE table.
    3. user              ~/.msmoe/reasoning.yaml (or $MSMOE_REASONING).
    4. explicit          a path, for CI and for reproducing someone's run.

Merged BY NAME, never replaced: adding one family must not cost you the other
four.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import defaults as _defaults

PACKAGED_NAME = "reasoning.yaml"
USER_ENV = "MSMOE_REASONING"
USER_PATH = os.path.join("~", ".msmoe", "reasoning.yaml")


@dataclass(frozen=True)
class ReasoningStyle:
    """One convention for separating a thinking trace from an answer."""
    name: str
    open: str
    close: str
    interwoven: bool = False   # reasoning interleaves with tool calls


@dataclass(frozen=True)
class ReasoningFamily:
    name: str
    hints: Tuple[str, ...]   # model-id fragments, matched loosely (see _norm)
    style: str               # a key into the style table


# The panic minimum. Plain `<think>`/`</think>` is what a build falls back to
# when there is no readable table at all, and it is also the overwhelmingly
# common convention, so the failure mode of a missing file is "still mostly
# right" rather than "silently scoring think blocks as answers".
FLOOR_STYLES: Dict[str, ReasoningStyle] = {
    "xml": ReasoningStyle("Standard XML Style", "<think>", "</think>"),
}


def _norm(text: str) -> str:
    """Lowercase, alphanumerics only.

    POSTEL, APPLIED TO MODEL NAMES. The table is written the way a vendor
    writes a model - "Llama 3.1", "DeepSeek V4", "Qwen2.5-Math" - and it is
    matched against ids written the way a hub writes them:
    "meta-llama/Llama-3.1-8B-Instruct". Requiring those to agree on spaces,
    dots and hyphens makes the table a trivia quiz about punctuation. Strip
    them from both sides and the person writing the yaml can write what they
    see on the model card.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _styles_from(doc: Dict[str, Any], into: Dict[str, ReasoningStyle],
                 by_name: Dict[str, str], where: str) -> List[str]:
    warns: List[str] = []
    for entry in (doc.get("TagStyles") or []):
        if not isinstance(entry, dict):
            warns.append(f"{where}: a TagStyles entry is not a mapping - IGNORED")
            continue
        label = str(entry.get("TagStyleName") or "").strip()
        key = str(entry.get("Key") or _slug(label)).strip()
        opening = entry.get("OpeningTag")
        closing = entry.get("ClosingTag")
        if not key or not opening or not closing:
            warns.append(
                f"{where}: tag style {label or key or '(unnamed)'!r} needs a "
                f"Key/TagStyleName, an OpeningTag and a ClosingTag - IGNORED")
            continue
        into[key] = ReasoningStyle(
            name=label or key, open=str(opening), close=str(closing),
            interwoven=bool(entry.get("Interwoven", False)))
        if label:
            by_name[_norm(label)] = key
    return warns


def _families_from(doc: Dict[str, Any], into: Dict[str, ReasoningFamily],
                   styles: Dict[str, ReasoningStyle], by_name: Dict[str, str],
                   where: str) -> List[str]:
    warns: List[str] = []
    for entry in (doc.get("Families") or []):
        if not isinstance(entry, dict):
            warns.append(f"{where}: a Families entry is not a mapping - IGNORED")
            continue
        label = str(entry.get("FamilyName") or "").strip()
        key = str(entry.get("Key") or _slug(label)).strip()
        raw = entry.get("Models") or entry.get("Hints") or []
        if isinstance(raw, str):
            raw = [raw]
        hints = tuple(_norm(h) for h in raw if _norm(h))
        pref = str(entry.get("PreferredStyle") or "").strip()
        # PreferredStyle may name a Key or a TagStyleName. Accepting both is
        # the difference between a table you can write and one you can only
        # copy.
        style = pref if pref in styles else by_name.get(_norm(pref), "")
        if not key or not hints or not style:
            warns.append(
                f"{where}: family {label or key or '(unnamed)'!r} needs a "
                f"FamilyName, at least one model, and a PreferredStyle that "
                f"names a known tag style - IGNORED")
            continue
        into[key] = ReasoningFamily(name=label or key, hints=hints, style=style)
    return warns


def load(explicit: Optional[str] = None, include_user: bool = True):
    """(styles, families, warnings). Floor first, then every readable layer."""
    styles: Dict[str, ReasoningStyle] = dict(FLOOR_STYLES)
    by_name: Dict[str, str] = {_norm(v.name): k for k, v in styles.items()}
    families: Dict[str, ReasoningFamily] = {}
    warns: List[str] = []

    for label, path in _defaults.layers_for(PACKAGED_NAME, USER_ENV,
                                            USER_PATH, explicit):
        if label == "user" and not include_user:
            continue
        doc = _defaults.read_yaml(path)
        if doc is None:
            if label == "explicit":
                warns.append(
                    f"reasoning table {path} could not be read - IGNORED. "
                    f"Tag styles fall back to the packaged table, which is "
                    f"probably not what you asked for.")
            continue
        # MERGE BY NAME, NEVER REPLACE. A box that adds one family must not
        # lose the other four.
        warns += _styles_from(doc, styles, by_name, path)
        warns += _families_from(doc, families, styles, by_name, path)
    return styles, families, warns


def style_for_base(base: str, kind: str = "auto",
                   styles: Optional[Dict[str, ReasoningStyle]] = None,
                   families: Optional[Dict[str, ReasoningFamily]] = None) -> str:
    """The style KEY a base model uses, or '' for a non-reasoning base.

    `kind` is the recipe's `base_kind`: nonreasoning short-circuits, reasoning
    falls back to plain xml when the id matches no family, auto only sniffs.

    THE LONGEST HINT WINS, so the answer does not depend on dict order. A table
    someone else edits should not change meaning because of where they put
    their entry.
    """
    if kind == "nonreasoning":
        return ""
    if styles is None or families is None:
        styles, families, _ = load()
    needle = _norm(base)
    best_key, best_len = "", 0
    for fam in families.values():
        for h in fam.hints:
            if h and h in needle and len(h) > best_len:
                best_key, best_len = fam.style, len(h)
    if best_key:
        return best_key
    return "xml" if kind == "reasoning" else ""


def split(text: str, style: Optional[ReasoningStyle]) -> Tuple[str, str, bool]:
    """(think, answer, reasoned). ONE splitter, used to write AND to read.

    There were two: eval's, which scored the answer, and data's, which
    validated a generated trace. They agreed until they didn't - the
    generator's fallback style differed from the reader's - and a scorer that
    splits differently from the writer is measuring a different artifact than
    the one on disk.

    `interwoven` is honoured here, and until now it was loaded and never read.
    An agentic model emits many think blocks around tool calls; taking the
    first close tag and calling the rest the answer leaves later think blocks
    IN the thing being scored. Strip every block; the answer is what is left.

    A model that did not reason is a FINDING, not a crash: with no clean split
    the whole text is the answer and `reasoned` is False.
    """
    if style is None or not text:
        return "", text, False

    if style.interwoven:
        parts, thoughts, i = [], [], 0
        while True:
            a = text.find(style.open, i)
            if a == -1:
                parts.append(text[i:])
                break
            b = text.find(style.close, a + len(style.open))
            if b == -1:
                parts.append(text[i:])
                break
            parts.append(text[i:a])
            thoughts.append(text[a + len(style.open):b].strip())
            i = b + len(style.close)
        answer = "".join(parts).strip()
        think = "\n".join(t for t in thoughts if t).strip()
        if not thoughts:
            return "", text, False
        return think, (answer or text), bool(think)

    open_i = text.find(style.open)
    if open_i == -1:
        return "", text, False
    after = text[open_i + len(style.open):]
    close_i = after.find(style.close)
    if close_i == -1:
        return "", text, False
    think = after[:close_i].strip()
    answer = after[close_i + len(style.close):].strip()
    return think, (answer or text), bool(think)
