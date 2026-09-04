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
    2. packaged          ms_moe_maker/assets/reasoning.yaml. THE table.
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
    # Plain-language marker for teachers that emit NO delimiter (R1-distill
    # rambles prose). The generator prompts for it and parses on it; the
    # delimiter `open`/`close` above is still what a finished specialist learns
    # and what eval reads back.
    answer_marker: str = "ANSWER:"
    # WHICH FILE SAID SO. Three layers merge into this table - the packaged
    # asset, ~/.msmoe/reasoning.yaml, and $MSMOE_REASONING - and a wrong tag
    # style is a silent wrong answer, so "where is this coming from" is a
    # question somebody WILL ask at the worst moment. Answering it costs a
    # string. Empty means the built-in floor.
    source: str = ""


@dataclass(frozen=True)
class ReasoningFamily:
    name: str
    hints: Tuple[str, ...]   # model-id fragments, matched loosely (see _norm)
    style: str               # a key into the style table
    source: str = ""         # the layer that supplied it; see ReasoningStyle


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
            interwoven=bool(entry.get("Interwoven", False)),
            answer_marker=str(entry.get("AnswerMarker") or "ANSWER:"),
            source=where)
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
        into[key] = ReasoningFamily(name=label or key, hints=hints,
                                    style=style, source=where)
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


def describe():
    """The reasoning registry as data - for `--describe` and Backstage's form.

    Same contract as corpus.describe() and validators.describe(), and here for
    the same reason their docstrings give: a consumer builds its picker from
    THIS rather than from a hardcoded list, so a family somebody added on their
    own box shows up without the viewer having heard of it.

    That argument is stronger here than for either of the others. The corpus
    kinds and the validators are extensible in principle; this table is the one
    users are actively TOLD to extend - "drop a yaml on the box and carry on" -
    because a model family shipping a new delimiter must not have to wait for a
    release. A picker that cannot see the user layer is showing the packaged
    table to someone who has already moved past it.

    Reports what the BOX has, user layer included (`load()` defaults to
    include_user=True), not what the source ships. Warnings are not folded in:
    they belong to load_errors(), matching the other two registries.
    """
    styles, families, _ = load()
    return {
        "styles": [{"key": k, "name": v.name, "open": v.open,
                    "close": v.close, "interwoven": v.interwoven}
                   for k, v in sorted(styles.items())],
        "families": [{"key": k, "name": v.name, "style": v.style}
                     for k, v in sorted(families.items())],
    }


def load_errors():
    """Problems reading the layered table. Empty is the healthy answer.

    Not an exception path: a malformed user file degrades to the packaged
    table (and, in the worst case, to the one-style FLOOR) rather than taking a
    build down - so the only way anyone finds out is if somebody asks.
    """
    _, _, warns = load()
    return list(warns)


def explain_base(base: str, kind: str = "auto",
                 styles: Optional[Dict[str, ReasoningStyle]] = None,
                 families: Optional[Dict[str, ReasoningFamily]] = None
                 ) -> Dict[str, Any]:
    """Which style a base model uses, AND WHY. The answer with its receipts.

    style_for_base returns the key alone, which is all a build needs and
    exactly nothing when somebody is staring at a trace wondering where its
    delimiters came from. Three layers merge into this table and a wrong tag
    style is a silent wrong answer, so the provenance is worth carrying:
    which family matched, on which hint, out of which file.

    Keys: style ('' for none), family, hint, source (the file that
    defined the matching FAMILY), style_source (the file that defined the
    tags), why.
    """
    out: Dict[str, Any] = {"style": "", "family": "", "hint": "",
                           "source": "", "style_source": "",
                           "why": ""}
    if kind == "nonreasoning":
        out["why"] = "base_kind is nonreasoning, so nothing reasons"
        return out
    if styles is None or families is None:
        styles, families, _ = load()

    # THE LONGEST HINT WINS, so the answer does not depend on dict order. A
    # table someone else edits should not change meaning because of where
    # they put their entry - and it is what makes "r1distillqwen" beat
    # "deepseek" on an id that contains both.
    best: Optional[ReasoningFamily] = None
    best_hint = ""
    needle = _norm(base)
    for fam in families.values():
        for h in fam.hints:
            if h and h in needle and len(h) > len(best_hint):
                best, best_hint = fam, h
    if best is not None:
        # TWO FILES CAN BE INVOLVED and the useful one is the FAMILY's: it
        # holds the rule that matched this model id. A user table that adds a
        # family pointing at a packaged style is the ordinary case, and
        # "where did this come from" means "which file decided", not "which
        # file spelled <think>". Both are returned; nobody has to guess.
        style_obj = styles.get(best.style)
        out.update(style=best.style, family=best.name, hint=best_hint,
                   source=best.source or (style_obj.source if style_obj
                                          else ""),
                   style_source=(style_obj.source if style_obj else ""),
                   why=f"{base} matched family {best.name!r} on hint "
                       f"{best_hint!r}")
        return out

    if kind == "reasoning":
        out.update(style="xml",
                   why=f"{base} matched no family, and base_kind is "
                       f"reasoning - falling back to plain xml")
        return out
    out["why"] = (f"{base} matched no family and base_kind is {kind!r}, so "
                  f"this base is treated as non-reasoning")
    return out


def style_for_base(base: str, kind: str = "auto",
                   styles: Optional[Dict[str, ReasoningStyle]] = None,
                   families: Optional[Dict[str, ReasoningFamily]] = None) -> str:
    """The style KEY a base model uses, or '' for a non-reasoning base.

    A thin read of explain_base, so the answer a BUILD acts on and the answer
    a PERSON is shown can never be two different answers.
    """
    return explain_base(base, kind, styles, families)["style"]


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

    # THE OPENER IS OFTEN IN THE PROMPT, NOT IN THE COMPLETION, and this
    # branch is what that costs when it is missing.
    #
    # DeepSeek-R1 and its distills put "<think>\n" at the END of the
    # generation prompt: the model wakes up already inside the block and
    # emits ONLY the closer. Requiring the pair therefore reports "did not
    # reason" for the most reasoning-shaped output there is, and every
    # caller downstream believes it - the generator falls back to a marker
    # prompt, the marker cuts at the wrong seam, and the corpus ends up
    # with the model's real answer filed under `think`.
    #
    # A lone closer is not ambiguous. Nothing else in a completion ends a
    # block that was never opened here, so everything before it is the
    # thinking and everything after it is the answer. Reading it that way
    # is the lenient half of strict-train/lenient-infer; we still WRITE the
    # pair.
    if open_i == -1:
        lone = text.find(style.close)
        if lone == -1:
            return "", text, False
        think = text[:lone].strip()
        answer = text[lone + len(style.close):].strip()
        # Both halves must exist. A completion that is nothing but a closing
        # tag is not a reasoned answer, it is a malformed one.
        if not think or not answer:
            return "", text, False
        return think, answer, True

    after = text[open_i + len(style.open):]
    close_i = after.find(style.close)
    if close_i == -1:
        return "", text, False
    think = after[:close_i].strip()
    answer = after[close_i + len(style.close):].strip()
    return think, (answer or text), bool(think)
