"""Freezing a recipe so it builds the same thing on somebody else's box.

THE PROBLEM, IN ONE SENTENCE. A recipe is mostly holes. Nearly every knob is a
sentinel - `-1`, `-1.0`, `"auto"`, `""` - meaning "you decide", and
`build_config` decides using the defaults of whichever ms-moe-maker is running.
So handing a friend `recipe.yaml` hands them YOUR INTENTIONS and THEIR
DEFAULTS. Same file, different model, no error, no warning. The failure mode is
that it looks like it worked, which is the only failure mode this project
really fears.

WHAT STAMPING DOES. Resolve the recipe here, then write every resolved value
back into the recipe as an explicit key. The holes are filled with the answers
this box gave, so the far box has nothing left to decide.

WHAT IS STAMPED, AND WHAT DELIBERATELY IS NOT

  stamped      fields with `derived_from is None` in the knob glossary - a
               recipe key or a default supplied them directly. 35 of them.
  NOT stamped  DERIVED fields, and this is the load-bearing exclusion. A value
               computed from other values has to be RECOMPUTED over there, not
               frozen. Stamping `collect_token_target` would leave it fighting
               the `collect_headroom` it is supposed to follow, and the recipe
               has no key to write it into anyway.
  cannot       16 fields that are in the fingerprint - so they change what the
               build produces - and have NO recipe key at all: three read
               environment variables, twelve are literals in `build_config`,
               and one is a CLI flag. See knobs.UNPINNABLE.

Stamp the inputs; let the outputs fall out. That is also exactly why the
build_id round-trips: the inputs are pinned, so the outputs re-derive to the
same numbers, and `verify` below checks that rather than assuming it.

ON THE SIXTEEN WE CANNOT PIN. Pretending is not an option and neither is
silence. The bundle records the whole resolved fingerprint, and import diffs it
against what the far box resolves - so an unpinnable field that differs is
reported BY NAME instead of discovered later as a model that came out wrong.
Same bargain as the `defaults_files` sha256: the divergence is not prevented,
it is made impossible to miss.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.knobs import UNPINNABLE, pinnable

# Top-level recipe keys, in the order a person reads them. Anything not named
# here keeps its own order after these - an unknown key is somebody else's
# extension or a newer schema, and dropping it would be the worst possible way
# to help them share a recipe.
BLOCK_ORDER = ("schema_version", "name", "base", "base_kind", "size",
               "template", "tier", "experts", "tools_expert",
               "reasoning_expert", "budget", "moe", "router", "corpus",
               "eval", "gates", "runtime", "abliterate", "smoke", "roots")


def stamp(raw: Dict[str, Any], config) -> Tuple[Dict[str, Any],
                                                Set[Tuple[str, str]]]:
    """Fill a raw recipe dict's holes from a resolved config.

    Operates on the RAW DICT the author wrote rather than on a Recipe object,
    because the difference between "they typed this" and "the default supplied
    it" survives only in the raw text. A Recipe has already collapsed the two -
    every sentinel is a value by then - and that distinction is the whole
    reason the output can mark which lines were authored.

    NEVER OVERWRITES AN AUTHORED KEY. If the key is present, it stays, whatever
    it says. A stamp that could change what somebody wrote would be an
    editorial pass rather than a fill-in-the-blanks, and the one thing worse
    than a recipe that builds differently elsewhere is one that builds
    differently HERE after being exported.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    stamped: Set[Tuple[str, str]] = set()
    for field, path in sorted(pinnable().items()):
        block, key = path.split(".", 1)
        section = out.get(block)
        if not isinstance(section, dict):
            section = {} if section is None else section
            if not isinstance(section, dict):
                # Somebody wrote `budget: 5`. Not our business to repair, and
                # validate will say so far more clearly than a stamp could.
                continue
            out[block] = section
        if key in section:
            continue                    # authored. Leave it alone.
        section[key] = _plain(getattr(config, field))
        stamped.add((block, key))
    return out, stamped


def _plain(value: Any) -> Any:
    """A YAML-safe copy. Tuples become lists; nothing else is touched.

    Deliberately NOT a coercion layer. If a value cannot be represented in
    YAML that is a fact worth failing on rather than papering over, because the
    silent alternative is a stamped recipe that reloads as something else.
    """
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def render(raw: Dict[str, Any], stamped: Set[Tuple[str, str]],
           header: Optional[List[str]] = None) -> str:
    """The stamped recipe as YAML, with the filled-in lines MARKED.

    WHY THE COMMENTS ARE THE POINT AND NOT DECORATION. A fully resolved recipe
    is ninety-odd explicit keys where the author wrote six. If everything is
    explicit then nothing is emphasised, and "here are the knobs I actually
    chose - change these" is exactly the information a person needs when
    handed someone else's recipe. It is also the information stamping
    destroys, unless it is written back down.

    So every stamped line ends in `# default`, and the authored lines are the
    ones with nothing after them. The reader's eye finds six bare lines in a
    field of annotated ones without being told how.

    Emitted block by block rather than in one dump so the comments can be
    attached by key. Every pinnable path is exactly `block.key` - two levels,
    checked by a test - so a line at indent two carrying a key we stamped is
    unambiguous.
    """
    import yaml

    lines: List[str] = list(header or [])
    keys = [k for k in BLOCK_ORDER if k in raw]
    keys += [k for k in raw if k not in BLOCK_ORDER]

    for block in keys:
        value = raw[block]
        text = yaml.safe_dump({block: value}, sort_keys=False,
                              default_flow_style=False,
                              allow_unicode=True).rstrip("\n")
        marked = {key for blk, key in stamped if blk == block}
        for line in text.split("\n"):
            if marked and line.startswith("  ") and not line.startswith("   "):
                name = line[2:].split(":", 1)[0].strip()
                if name in marked:
                    line = f"{line}    # default"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def unpinnable_snapshot(config) -> Dict[str, Any]:
    """The sixteen values a recipe cannot express, as this box resolved them.

    Recorded so the far side can DIFF them. It is the honest half of the
    feature: these will not travel in the recipe, and the person receiving it
    is entitled to know which ones their box disagrees about rather than
    finding out from a model that came out wrong.
    """
    return {name: _plain(getattr(config, name, None))
            for name in sorted(UNPINNABLE)}


def diff_fingerprints(theirs: Dict[str, Any],
                      ours: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Field-by-field disagreement between two resolved fingerprints.

    THIS IS THE WHOLE SAFETY NET. Stamping handles the 35 fields a recipe can
    express; this catches everything, including the sixteen it cannot and any
    field a future version adds before anyone teaches the glossary about it.
    A bundle that will build something different says so, by name, before a GPU
    is warm.

    Missing on either side is reported as its own kind, not silently skipped -
    a field one box has and the other does not is a version difference, which
    is exactly the thing worth knowing.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(theirs) | set(ours)):
        if name not in theirs:
            out[name] = {"theirs": None, "ours": ours[name],
                         "why": "this field did not exist when the bundle was made"}
        elif name not in ours:
            out[name] = {"theirs": theirs[name], "ours": None,
                         "why": "this box does not have this field"}
        elif theirs[name] != ours[name]:
            out[name] = {"theirs": theirs[name], "ours": ours[name],
                         "why": UNPINNABLE.get(name, "")
                                or "a recipe key that resolved differently"}
    return out


def verify(stamped_text: str, expected: Dict[str, Any],
           *, defaults_path: Optional[str] = None) -> Dict[str, Any]:
    """Re-resolve a stamped recipe and prove it produces the same build.

    THE ACCEPTANCE TEST, RUN AT EXPORT TIME RATHER THAN ONLY IN CI. A stamp
    that missed a field produces a recipe which builds something else, and the
    person who finds out is the one it was given to. So the exporter loads its
    own output back, resolves it, and compares - and a bundle that does not
    round-trip is not written.

    Note what this can and cannot see. On THIS box the unpinnable sixteen
    resolve identically either way, so they always agree here - which is
    correct: they are not what this checks. This checks that the 35 stampable
    fields and everything derived from them survived the round trip.
    """
    import tempfile
    import os
    from ..config.pipeline import build_config, build_fingerprint
    from ..config.recipe import load

    handle, path = tempfile.mkstemp(suffix=".yaml", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(stamped_text)
        reloaded, _ = load(path, defaults_path=defaults_path)
        got = build_fingerprint(build_config(reloaded))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return diff_fingerprints(expected, got)
