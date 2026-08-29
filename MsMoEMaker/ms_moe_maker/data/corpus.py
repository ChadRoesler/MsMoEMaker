"""Corpus source kinds - the registry that makes a Ms.MoE not-about-code.

WHAT WAS WRONG WITH THE TUPLE.

`("hf", "stack", "synth")` appeared as a literal in three places: the Source
docstring, the validator, and the --describe payload. Adding a kind meant
finding all three, and MISSING one meant a recipe that validated and then
failed at build time, or validated and was advertised as unsupported. A closed
list written down three times is a closed list that will be wrong.

More importantly it was closed at all. Everything between the corpus and the
GGUF is domain-blind - tokenise, finetune, stitch, router-train, export and
smoke have no idea whether the text is PowerShell or the Monster Manual. The
ONLY thing in this whole package that assumed "code" was the set of ways to
get text, and one of those (`stack`) is code-specific while the other two
already are not. `hf` is any HuggingFace dataset plus a field name; it would
have built a D&D lore expert the day it was written.

So the generality was already there and a hardcoded tuple was standing in
front of it. This is the registry.

────────────────────────────────────────────────────────────────────────────
REGISTERING YOUR OWN

Two ways, and the second is the point:

    from ms_moe_maker.corpus import register, Kind
    register(Kind(name="local", requires=("path",),
                  summary="read text files off this box"))

or, from another distribution entirely, an entry point:

    [project.entry-points."ms_moe_maker.corpus_kinds"]
    obsidian = "my_pkg.kinds:OBSIDIAN"

The entry point is what makes this open to someone who is not us. A person
building a Ms.MoE to drill pharmacology should not have to send a PR to this
repo to describe where their notes live.

VALIDATION IS DECLARATIVE ON PURPOSE. A Kind names its required fields rather
than shipping a validate() callback, because validation has to run on a laptop
with `ms-moe-maker validate` and no GPU, no torch and no network - the laptop
promise. Declared requirements can be checked by reading them. Arbitrary code
cannot, and would eventually import something.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Fields on Source that every kind may set. A kind declares which of these it
# REQUIRES; anything else is optional and simply ignored by kinds that do not
# care. Kept as data so `--describe` can report the whole schema without the
# caller knowing any kind names.
KNOWN_FIELDS = ("repo", "split", "text_field", "language", "max_shards",
                "teacher", "generator", "examples", "path", "glob", "ref",
                "subdir")


@dataclass(frozen=True)
class Kind:
    """One way of getting an expert's training text."""

    name: str
    summary: str = ""
    # Fields on `source:` that MUST be present. Checked by reading, never by
    # calling - see the laptop promise above.
    requires: Tuple[str, ...] = ()
    # Does this kind produce text by GENERATING it rather than fetching it?
    # stages.plan uses this to decide whether a build gets a data.synth stage,
    # which used to be the hardcoded check `if "agentcore" in experts` - a test
    # that worked for exactly as long as there was one generated expert in the
    # world.
    generated: bool = False
    # Purely advisory, surfaced by validate as a warning rather than an error.
    notes: str = ""


_REGISTRY: Dict[str, Kind] = {}


def register(kind: Kind, *, replace: bool = False) -> Kind:
    """Add a kind. Refuses to shadow an existing one unless asked.

    Silently replacing would let a plugin change what `hf` means for every
    recipe on the box, and the recipe that broke would be someone else's.
    """
    if kind.name in _REGISTRY and not replace:
        raise ValueError(
            f"corpus kind {kind.name!r} is already registered. Pass "
            f"replace=True if you genuinely mean to redefine it - note that "
            f"this changes the meaning of every recipe on this machine that "
            f"uses that kind, including ones you did not write.")
    _REGISTRY[kind.name] = kind
    return kind


def get(name: str) -> Optional[Kind]:
    _load_entry_points()
    return _REGISTRY.get(name)


def names() -> List[str]:
    _load_entry_points()
    return sorted(_REGISTRY)


def all_kinds() -> List[Kind]:
    _load_entry_points()
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def generated_kinds() -> List[str]:
    return [k.name for k in all_kinds() if k.generated]


def describe() -> List[Dict[str, object]]:
    """The registry as data, for --describe and for Backstage's craft form.

    Backstage builds its source picker from THIS rather than from a hardcoded
    list in the viewer, so a kind registered by a plugin shows up in the form
    without the viewer having heard of it.
    """
    return [{"name": k.name, "summary": k.summary,
             "requires": list(k.requires), "generated": k.generated,
             "notes": k.notes}
            for k in all_kinds()]


# ── the built-ins ───────────────────────────────────────────────────────────
# `hf` and `local` are domain-neutral. `stack` is the code-specific one, and it
# is one entry among several rather than a third of the world.

register(Kind(
    name="hf",
    summary="a dedicated HuggingFace dataset",
    requires=("repo",),
    notes="Use this for anything a general corpus is thin on. PowerShell runs "
          "~141 files/shard in stack-v3: 80 shards (~45 GB) yielded 11,627. A "
          "language-partitioned dataset gave 140,000 in a 392 MB download. "
          "~500x. Domain-neutral - `repo` plus `text_field` is equally a lore "
          "corpus, a statute corpus or a pile of lecture notes."))

register(Kind(
    name="stack",
    summary="scan the general CODE corpus for a language name",
    requires=("language",),
    notes="The one code-specific kind. Cheap for Python (~18,000/shard), "
          "useless for anything rare. `language` must be spelled EXACTLY as "
          "the corpus spells it - an inexact match is a silent zero for an "
          "unrelated-looking reason."))

register(Kind(
    name="synth",
    summary="generate it with a teacher model and a rejection-sampling validator",
    requires=("teacher",),
    generated=True,
    notes="For a domain no corpus exists to scrape. This is how the MCP-trace "
          "expert is built, and it is equally how you would build one on "
          "encounter design or worked exam answers."))

register(Kind(
    name="gh",
    summary="files out of a public GitHub repository",
    requires=("repo",),
    notes="`repo` is owner/name. `glob` picks the files (default **/*.md), "
          "`ref` picks a branch or tag (default: the repo's default branch), "
          "`subdir` narrows to one directory. Fetched as a single tarball "
          "from codeload, not a git clone - one request, no git binary, and "
          "no .git history to download. The corpus that is documentation, a "
          "wiki, or a specific project's source rather than a slice of a "
          "general code dump: use this when the text you want IS a particular "
          "repository, and `stack` when you want a language in general."))

register(Kind(
    name="local",
    summary="text already on this box",
    requires=("path",),
    notes="The offline / private-corpus path: your own notes, a scraped wiki, "
          "a folder of PDFs already converted to text. Nothing leaves the "
          "machine, which for some corpora is the whole requirement."))


# ── third-party kinds ───────────────────────────────────────────────────────

_ENTRY_POINT_GROUP = "ms_moe_maker.corpus_kinds"
_loaded = False


def _load_entry_points() -> None:
    """Pull in kinds published by other distributions. Once, and never fatally.

    A broken plugin must not take down `ms-moe-maker validate`. It is reported
    through the normal warning channel by the caller if they ask for
    load_errors(); what it may never do is stop a person validating a recipe
    that does not use it.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        found = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - <3.10 signature
        found = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return
    for entry in found:
        try:
            obj = entry.load()
            for kind in (obj if isinstance(obj, (list, tuple)) else [obj]):
                if isinstance(kind, Kind) and kind.name not in _REGISTRY:
                    _REGISTRY[kind.name] = kind
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERRORS.append(f"{entry.name}: {exc}")


_LOAD_ERRORS: List[str] = []


def load_errors() -> List[str]:
    _load_entry_points()
    return list(_LOAD_ERRORS)


def check(kind_name: str, source) -> Tuple[List[str], List[str]]:
    """Validate one `source:` block against its kind. Returns (errors, warnings).

    Kept here rather than in recipe.py so that adding a kind adds its rules in
    the same edit - the old arrangement had the kind list in one file and its
    per-kind requirements in an elif chain in another, which is two places to
    forget.
    """
    errs: List[str] = []
    warns: List[str] = []
    kind = get(kind_name)
    if kind is None:
        available = ", ".join(names())
        errs.append(f"source.kind {kind_name!r} is not registered. Available: "
                    f"{available}. A distribution can add more by publishing a "
                    f"{_ENTRY_POINT_GROUP} entry point.")
        return errs, warns

    for required in kind.requires:
        if not getattr(source, required, None):
            errs.append(f"source.kind={kind.name} needs a {required}")
    return errs, warns
