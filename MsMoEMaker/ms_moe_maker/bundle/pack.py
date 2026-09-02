"""The zip you hand somebody. Writing it, and reading one safely.

    RecipeName.zip
    |- recipe.yaml      the stamped recipe - every default filled in
    |- bundle.json      what this is, what it built, and what it CANNOT pin
    `- data/            optional; synth corpora, one directory per expert

WHY bundle.json IS NOT OPTIONAL DECORATION. The recipe carries the 35 knobs a
recipe can express. It cannot carry the sixteen that have no key, it cannot
carry the fingerprint of a build that has already run, and it cannot carry the
sentence "this produced 2.14x enrichment on my box" - which is the actual gift.
The manifest is where the claim lives, and where the far side gets something
to DIFF instead of something to trust.

────────────────────────────────────────────────────────────────────────────
READING ONE IS THE DANGEROUS DIRECTION, AND IT IS DANGEROUS TWICE.

A zip is somebody else's archive, and `extractall` has written files outside
its destination since the format was invented: an entry named `../../.bashrc`,
or an absolute path, or a symlink pointing somewhere useful. `_safe_members`
refuses all three by inspecting names BEFORE anything is written, and the
extractor joins-and-resolves rather than trusting the name it just approved.

The second danger is not the zip at all - it is what is inside it. A recipe is
a document that names an `eval.script`, and the harness runs that script with
the interpreter. **A recipe from someone else can execute arbitrary code by
design.** That is fine between friends and it is not fine silently, so
`read()` reports `executes` naming any such field, and any surface that offers
an import button is expected to put it in front of a person rather than in a
log. Nothing here decides for them; it refuses to let them not know.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1
MANIFEST_NAME = "bundle.json"
RECIPE_NAME = "recipe.yaml"
DATA_DIR = "data"
NOTES_NAME = "NOTES.md"

# A corpus is a dataset and a recipe is a document. Anything claiming to be a
# recipe and weighing more than this is not one, and reading it into memory to
# find that out is the mistake.
MAX_RECIPE_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class UnreadableBundle(Exception):
    """The archive is not a bundle, or is one that must not be opened."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """Every member, or a refusal. Checked BEFORE a single byte is written.

    Three refusals, and each has cost somebody a machine at some point:
    absolute paths, `..` traversal, and symlinks (a link is a write primitive
    that the name alone looks innocent for). Refusing the whole archive rather
    than skipping the bad entry, because an archive containing one of these is
    not a bundle with a flaw - it is not a bundle.
    """
    out: List[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
            raise UnreadableBundle(
                f"{info.filename!r} is an absolute path. A bundle names files "
                f"relative to itself; this one is trying to choose where it "
                f"lands.")
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if ".." in parts:
            raise UnreadableBundle(
                f"{info.filename!r} climbs out of the bundle with `..`. "
                f"Nothing legitimate needs that.")
        # The high four bits of external_attr are the unix file type; 0xA is a
        # symlink. A link entry writes a pointer, and the pointer can aim
        # anywhere at all.
        if (info.external_attr >> 16) & 0xF000 == 0xA000:
            raise UnreadableBundle(
                f"{info.filename!r} is a symlink. A bundle carries files, not "
                f"pointers to files somewhere on your disk.")
        out.append(info)
    return out


def _executes(recipe_text: str) -> List[str]:
    """Fields in this recipe that cause somebody else's code to run.

    Reported, never blocked. `eval.script` is a documented feature and the
    honest position is that a recipe is executable content - the failure this
    prevents is a person clicking Import without being told, not a person
    deciding to trust a friend.
    """
    found: List[str] = []
    try:
        import yaml
        data = yaml.safe_load(recipe_text) or {}
    except Exception:                                   # noqa: BLE001
        return found
    if not isinstance(data, dict):
        return found
    for block, key in (("eval", "script"), ("smoke", "script"),
                       ("gguf", "smoke_script")):
        section = data.get(block)
        if isinstance(section, dict) and section.get(key):
            found.append(f"{block}.{key} = {section[key]!r}")
    return found


def write(dest: Path, *, recipe_text: str, meta: Dict[str, Any],
          data_dirs: Iterable[Tuple[str, Path]] = (),
          notes: str = "") -> Path:
    """Write a bundle. Returns the path.

    DEFLATED, and the corpora are the reason: a synth corpus is jsonl, which is
    the most compressible thing in this project by a wide margin. A stored zip
    of one is an insult to whoever has to download it.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["schema_version"] = SCHEMA_VERSION
    payload["recipe_sha256"] = _sha256(recipe_text.encode("utf-8"))
    files: List[Dict[str, Any]] = []

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(RECIPE_NAME, recipe_text)
        if notes:
            zf.writestr(NOTES_NAME, notes)
        for label, directory in data_dirs:
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(directory).as_posix()
                arc = posixpath.join(DATA_DIR, label, rel)
                zf.write(path, arc)
                files.append({"expert": label, "path": rel,
                              "bytes": path.stat().st_size})
        payload["data"] = files
        payload["data_bytes"] = sum(f["bytes"] for f in files)
        zf.writestr(MANIFEST_NAME, json.dumps(payload, indent=2,
                                              sort_keys=True) + "\n")
    return dest


def read(path: Path) -> Dict[str, Any]:
    """Open a bundle and report what is in it. Extracts NOTHING.

    Reading and extracting are separate on purpose. A person deciding whether
    to accept a bundle needs to see the recipe, the claim and the warnings
    first - and a design where you must write it to disk to find out what it
    is has the consent backwards.
    """
    path = Path(path)
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnreadableBundle(f"{path.name}: not a readable zip ({exc})") from exc
    with zf:
        members = _safe_members(zf)
        names = {m.filename for m in members}
        if RECIPE_NAME not in names:
            raise UnreadableBundle(
                f"{path.name} has no {RECIPE_NAME}, so it is an archive rather "
                f"than a bundle. Nothing here knows what to build from it.")
        info = zf.getinfo(RECIPE_NAME)
        if info.file_size > MAX_RECIPE_BYTES:
            raise UnreadableBundle(
                f"{RECIPE_NAME} is {info.file_size} bytes. A recipe is a "
                f"document; this is a dataset wearing its name.")
        recipe_text = zf.read(RECIPE_NAME).decode("utf-8", errors="replace")

        meta: Dict[str, Any] = {}
        meta_error = ""
        if MANIFEST_NAME in names:
            if zf.getinfo(MANIFEST_NAME).file_size > MAX_MANIFEST_BYTES:
                meta_error = f"{MANIFEST_NAME} is implausibly large; ignored."
            else:
                try:
                    loaded = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
                    meta = loaded if isinstance(loaded, dict) else {}
                    if not isinstance(loaded, dict):
                        meta_error = f"{MANIFEST_NAME} is not an object."
                except ValueError as exc:
                    meta_error = f"{MANIFEST_NAME} is not valid JSON ({exc})."
        else:
            # NOT AN ERROR. A hand-rolled zip with a recipe in it is a
            # perfectly good thing for somebody to make, and refusing it would
            # make the format a gate rather than a convenience.
            meta_error = (f"no {MANIFEST_NAME}: this bundle makes no claim "
                          f"about what it built, so there is nothing to "
                          f"compare against.")

        notes = ""
        if NOTES_NAME in names:
            notes = zf.read(NOTES_NAME).decode("utf-8", errors="replace")

        data = sorted({m.filename.split("/")[1] for m in members
                       if m.filename.startswith(DATA_DIR + "/")
                       and len(m.filename.split("/")) > 2})

        version = meta.get("schema_version")
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise UnreadableBundle(
                f"{path.name} is schema_version {version}; this ms-moe-maker "
                f"understands {SCHEMA_VERSION}. Upgrade rather than being "
                f"shown a guess about somebody else\'s build.")

        return {"path": str(path), "recipe": recipe_text, "meta": meta,
                "meta_error": meta_error, "notes": notes,
                "data_experts": data,
                "bytes": sum(m.file_size for m in members),
                # See the module docstring. This goes in front of a person.
                "executes": _executes(recipe_text)}


def extract(path: Path, dest: Path) -> List[str]:
    """Unpack a bundle into `dest`. Names are re-checked, then re-derived.

    The approved name is not reused as a path. Every entry is joined onto the
    resolved destination and the RESULT is checked to still be inside it -
    belt and braces, because the cost of being wrong here is a file written
    somewhere on somebody\'s disk and the cost of the second check is a
    string comparison.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in _safe_members(zf):
            if info.is_dir():
                continue
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise UnreadableBundle(
                    f"{info.filename!r} resolves outside the destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            written.append(info.filename)
    return written
