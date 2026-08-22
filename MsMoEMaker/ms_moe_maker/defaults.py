"""Layered defaults — the floor, as data instead of as constants.

WHY THIS EXISTS. Every default in this tool used to live in Python, which
means changing one for a whole box is a code edit and a release. That is the
wrong shape for the case this tool is actually for: someone sets a machine up
for someone else, picks sensible values for THAT hardware, and hands it over.
The alternative was "every recipe must carry these eleven lines or you wasted
an evening", which is friction we refuse to accept as the cost of doing
business.

So defaults are a FILE. Layers, later wins:

    1. FLOOR              in this module. Cannot be missing, so the tool
                          always runs even with no data files at all. It is a
                          PANIC MINIMUM, not the real table.
    2. packaged           ms_moe_maker/defaults.yaml — what we ship. THE table.
    3. user               ~/.msmoe/defaults.yaml (or $MSMOE_DEFAULTS) — the
                          box. This is the file you edit for someone else.
    4. explicit           --defaults <path>, for CI and for reproducing a run
                          that was not made on your machine.
    5. the recipe         always wins. (Handled by recipe.parse, not here.)

WHAT DOES *NOT* BELONG HERE. Preferences, not correctness. The test: if
getting it wrong makes a WORSE BUILD, it is a knob and it belongs in a
defaults file. If getting it wrong makes a WRONG MEASUREMENT, it does not —
TRAIN_SPLIT_SHARE, ROUTER_DOC_MARGIN, the router's quota arithmetic, the
top-1 refusal and the supported-architecture list stay in Python, because a
yaml that can quietly falsify a number is a footgun wearing a knob's clothes.

A DEFAULTS FILE IS A RECIPE WITH NO EXPERTS. Same schema, same dataclasses,
same `-1` sentinels, same typo warnings. There is no second format to keep in
sync, which is the whole point.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

# The panic minimum. Two values, because a build that reaches the tools expert
# with no name and no teacher cannot proceed at all — everything else has a
# working default in the dataclasses already. The packaged defaults.yaml
# carries these same two, and a test asserts the two do not disagree.
FLOOR: Dict[str, Any] = {
    "tools_expert": {
        "name": "agentcore",
        "teacher": "Qwen/Qwen2.5-7B-Instruct",
    },
}

PACKAGED_NAME = "defaults.yaml"
USER_ENV = "MSMOE_DEFAULTS"
USER_PATH = os.path.join("~", ".msmoe", "defaults.yaml")

# Top-level keys a defaults file may set. Deliberately NOT every recipe key:
# `experts`, `name` and `template` describe one build, not a box, and a
# defaults file that could inject experts would make "which experts am I
# training" unanswerable from the recipe alone.
# SAYING WHAT A THING IS, VERSUS ASKING FOR ONE. These blocks supply CONTENT
# for a feature the recipe has to opt into; they are handed to `parse` through
# the `defaults=` channel and are NOT merged into the recipe dict, because
# merging them would make their mere presence the request.
#
# It bit immediately: with `tools_expert:` in defaults.yaml and a raw merge,
# every recipe on the box grew an `agentcore` expert nobody asked for - a
# whole extra specialist, its teacher, and its GPU hours, injected by a file
# the recipe never mentions. A defaults file configures the answer; only a
# recipe asks the question.
CONTENT_ONLY: frozenset = frozenset({"tools_expert"})

# THE BOX ITSELF. What a tier IS, and which checkpoint a size maps to, are
# statements about a MACHINE - so a recipe may name a tier but must never
# redefine one, or the same recipe would describe different hardware depending
# on who ran it. These reach config through the same `defaults=` channel and
# are never merged into the recipe dict.
#
# Unlike CONTENT_ONLY they are ALWAYS in play - there is no opting in to having
# hardware - so provenance reports them unconditionally.
BOX_ONLY: frozenset = frozenset({"tiers", "models"})

# Everything the recipe merge must leave alone.
NOT_MERGED: frozenset = CONTENT_ONLY | BOX_ONLY

ALLOWED: frozenset = frozenset({
    "base", "base_kind", "size", "budget", "moe", "gates", "runtime", "roots",
    "corpus", "router", "eval", "smoke", "tools_expert",
    # box-only (see BOX_ONLY): not recipe keys at all
    "tiers", "models",
})


def read_yaml(path: str) -> Optional[Dict[str, Any]]:
    """Parse one layer file. `{}` = nothing set; None = not there / not usable.

    AN ALL-COMMENTS FILE IS NOT A BROKEN FILE. yaml.safe_load returns None for
    a document that is entirely comments, and treating that as "unreadable"
    made `init --defaults-template` produce a file whose very next use warned
    that it could not be read - the on-ramp tripping over its own first step,
    which is the same shape as the bug the init/validate round-trip test was
    written to catch. Empty means empty.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:
        return None
    if data is None:
        return {}
    return data if isinstance(data, dict) else None


def _packaged_path() -> str:
    return packaged_path(PACKAGED_NAME)


def _user_path() -> str:
    return user_path(USER_ENV, USER_PATH)


def packaged_path(name: str) -> str:
    """A data file that ships inside the package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def user_path(env_var: str, rel: str) -> str:
    """The box's copy of a data file: $ENV wins, else ~/.msmoe/<name>."""
    return os.path.expanduser(os.environ.get(env_var) or rel)


def layers_for(packaged_name: str, env_var: str, user_rel: str,
               explicit: Optional[str] = None) -> List[Tuple[str, str]]:
    """(label, path) per layer above the floor, lowest priority first.

    Generalised so the reasoning table gets the same layering as the defaults
    table for free. Two files that layer differently is two things to explain
    and two things to get wrong.
    """
    layers = [("packaged", packaged_path(packaged_name)),
              ("user", user_path(env_var, user_rel))]
    if explicit:
        layers.append(("explicit", os.path.expanduser(explicit)))
    return layers


def layer_paths(explicit: Optional[str] = None) -> List[Tuple[str, str]]:
    """(label, path) for every layer above the floor, lowest priority first."""
    return layers_for(PACKAGED_NAME, USER_ENV, USER_PATH, explicit)


def _merge(base: Dict[str, Any], over: Dict[str, Any], label: str,
           prov: Dict[str, str], prefix: str = "") -> Dict[str, Any]:
    """One level of block-wise merge, recording where each leaf came from.

    Blocks merge key by key; anything else replaces. `-1` is dropped rather
    than written, because `-1` has meant "you decide" since the first recipe
    and the layer below is now who decides.
    """
    out = dict(base)
    for k, v in over.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            # Recurse even when the layer below has no block there, so
            # provenance is always recorded per LEAF. A line saying "budget
            # came from somewhere" is not an answer to "why is target_steps
            # 500".
            below = out.get(k) if isinstance(out.get(k), dict) else {}
            out[k] = _merge(below, v, label, prov, prefix=f"{key}.")
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == -1:
            continue  # "you decide" — let the layer below stand
        out[k] = copy.deepcopy(v)
        prov[key] = label
    return out


def resolve(explicit: Optional[str] = None,
            include_user: bool = True) -> Tuple[Dict[str, Any], Dict[str, str],
                                                List[str]]:
    """Merge every layer below the recipe.

    Returns (defaults, provenance, warnings). `provenance` maps a dotted key
    to the label of the layer that last set it — which is what turns layered
    config from spooky action at a distance into one printable line.

    `include_user=False` skips the machine-specific layer. Tests and CI want
    that: a unit test whose result depends on whoever's laptop it runs on is
    not a unit test.
    """
    prov: Dict[str, str] = {}
    merged = _merge({}, copy.deepcopy(FLOOR), "floor", prov)
    warnings: List[str] = []

    for label, path in layer_paths(explicit):
        if label == "user" and not include_user:
            continue
        data = read_yaml(path)
        if data is None:
            if label == "explicit":
                warnings.append(
                    f"--defaults {path} could not be read - IGNORED. The build "
                    f"will use the packaged defaults, which is probably not "
                    f"what you asked for.")
            continue
        unknown = [k for k in data if k not in ALLOWED]
        for k in unknown:
            warnings.append(
                f"{path}: {k!r} is not a defaults key - IGNORED. A defaults "
                f"file sets the BOX, not the build; `experts`, `name` and "
                f"`template` belong to a recipe.")
        clean = {k: v for k, v in data.items() if k in ALLOWED}
        merged = _merge(merged, clean, path, prov)

    return merged, prov, warnings


def file_digests(explicit: Optional[str] = None,
                 include_user: bool = True) -> Dict[str, str]:
    """{path: sha256[:12]} for every defaults file that actually contributed.

    Recorded in the run manifest. A build id says two runs differ; this says
    WHICH file on which box was different, which is the question somebody
    actually has at 2am when their run and yours disagree.

    Files that are not there simply do not appear - absence is not a hash.
    """
    import hashlib
    out: Dict[str, str] = {}
    for label, path in layer_paths(explicit):
        if label == "user" and not include_user:
            continue
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                out[path] = hashlib.sha256(fh.read()).hexdigest()[:12]
        except OSError:
            continue
    return out


def apply_to(recipe_data: Dict[str, Any],
             defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Lay a recipe over resolved defaults. THE RECIPE ALWAYS WINS.

    Merged as raw dicts, before parse, so there is exactly one parser, one set
    of typo warnings and one validate for both files.
    """
    prov: Dict[str, str] = {}
    base = {k: v for k, v in copy.deepcopy(defaults).items()
            if k not in NOT_MERGED}
    return _merge(base, copy.deepcopy(recipe_data), "recipe", prov)


def describe(prov: Dict[str, str], warnings: List[str]) -> List[str]:
    """Human-readable provenance lines, deepest layer last.

    Layered config without provenance is how config debugging becomes a
    seance. This is the antidote and it costs four lines.
    """
    lines = [f"{k} <- {v}" for k, v in sorted(prov.items()) if v != "floor"]
    return lines + list(warnings)
