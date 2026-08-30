"""The run manifest - the ONLY interface between ms-moe-maker and seren-theatre.

Neither package imports the other. Neither package may. `ms-moe-maker` is a build
pipeline that knows nothing about Seren, and `seren-theatre` is a viewer that
works perfectly on a directory no pipeline ever touched. What connects them is
a document on disk, in the run directory, that one writes and the other reads.

WHY A MANIFEST AND NOT MORE SCRAPING. Theatre can already read a run by
pattern-matching artifact names on disk - `qwen_coder_*`, a directory called
`fraunkenstein_moe_untrained`, a `.gguf` next to a `.smoketest.txt`. That works,
it needs no cooperation from anything, and it must keep working, because "a
stage is a directory" is the whole reason Theatre requires nothing. But it
couples the viewer to the pipeline's internal file NAMES. Rename one artifact
during the decomposition and Theatre reports "no specialists" for a completely
healthy run - and says it with total confidence, which is the worst way for a
dashboard to be wrong.

So: the manifest is authoritative when present, scraping is the fallback when
it is not. An instrumented run becomes exact. An uninstrumented directory keeps
working exactly as before. Nobody is forced to instrument anything.

This is deliberately the same shape as SerenObservatory reading
~/.seren/services/*.json: the reporter reports what it was TOLD, it does not go
probing. A node with no manifests looks empty, and that is a true reading of
what the node declared. Same bargain here, with the same honesty requirement -
which is why `stale()` exists below. A manifest whose process died mid-stage
must be readable AS a manifest whose process died mid-stage, not as a run that
is still going.

FORWARD COMPATIBILITY. `schema_version` is checked by the reader and a version
it does not understand is reported as unreadable rather than parsed optimism-
ally. Unknown KEYS inside a known version are preserved and ignored - strict on
write, lenient on read, the family's Postel bargain. That is what lets ms-moe-maker
add a field without every installed Theatre needing an upgrade first.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# The document this module reads and writes. Bump ONLY for a breaking change;
# additive fields do not need it, because the reader ignores what it does not
# know.
SCHEMA_VERSION = 1

# The filename, at the root of a RUN directory (one rung). Dot-prefixed so it
# never collides with a model artifact and never gets mistaken for one by a
# glob that is looking for directories.
# NOT "msmoemaker-run.json". This file describes a RUN THAT PRODUCES A Ms.MoE,
# so it is named after the OUTPUT, not after the tool that wrote it - the same
# rule that leaves the `msmoe_{size}` run directories and the recipe's `name:`
# field alone while the package, module and CLI all became ms-moe-maker.
#
# It is also a PUBLISHED WIRE FORMAT. seren-theatre implements an independent
# reader for it and never imports this package, so renaming this constant on
# one side silently blinds the other: the viewer finds no manifest, falls back
# to glob-scraping, and the stage ladder just quietly stops appearing. That
# happened during the rename and nothing caught it, because the contract test
# that exists to catch exactly this was keyed on a directory name that changed
# in the same sweep.
MANIFEST_NAME = "msmoe-run.json"

# How long a "running" stage may go without a heartbeat before a reader should
# treat the manifest as abandoned rather than live. Generous on purpose: a
# 14B expert can spend a long time inside one stage with nothing to report,
# and calling a slow run dead is its own kind of lie.
STALE_AFTER_SECONDS = 15 * 60


# -- stage status ------------------------------------------------------------
# A closed vocabulary, because the viewer paints from it. Adding a status is a
# schema change; sending one that is not here is a bug the reader will surface
# rather than render.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"     # already present on disk; the pipeline's _done() fired
FAILED = "failed"
REFUSED = "refused"     # never attempted - see levers.py and the refusal list
# Completed without the full result the stage exists for: no llama.cpp means
# the export stage reports "converted nothing" rather than "done" - a viewer
# must not paint a GGUF that was never attempted. The runner is no longer
# allowed to overwrite a terminal status with `done` (see Runner._finish_current).
WARNED = "warned"

STATUSES = (PENDING, RUNNING, DONE, SKIPPED, FAILED, REFUSED, WARNED)

# Statuses that mean "this will not change again without a new run".
TERMINAL = (DONE, SKIPPED, FAILED, REFUSED, WARNED)


@dataclass
class Stage:
    """One step of a build, as reported by whoever is doing the building."""

    id: str                       # stable, machine-readable: "finetune.python"
    label: str                    # human: "Fine-tune python specialist"
    status: str = PENDING
    started: Optional[float] = None
    ended: Optional[float] = None
    # Path RELATIVE to the run directory. Relative on purpose: a manifest has
    # to survive the directory being moved, copied to another box, or read
    # through a mount with a different prefix - which is exactly what Theatre
    # does when it watches a stage over a network share.
    artifact: Optional[str] = None
    note: Optional[str] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started is None:
            return None
        return (self.ended or time.time()) - self.started


@dataclass
class Manifest:
    """What a run is, and how far along it is."""

    schema_version: int = SCHEMA_VERSION
    recipe_id: str = ""
    name: str = ""
    size: str = ""
    base: str = ""
    experts: List[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    # Set once the run reaches a terminal state, so a reader can tell "finished
    # a while ago" from "died a while ago" without guessing from timestamps.
    finished: Optional[float] = None
    ok: Optional[bool] = None
    stages: List[Stage] = field(default_factory=list)
    # Recipe fields the runner could not honour. Carried IN the manifest, not
    # just printed to a log, because the person reading the dashboard six hours
    # later is exactly who needs to know a lever was ignored. See levers.py.
    refusals: List[str] = field(default_factory=list)
    # WHAT THIS RUN ACTUALLY BUILT, not what its recipe said.
    #
    # `recipe_id` excludes runtime and has never identified a build; once the
    # defaults layer arrived, the values that decide what gets trained can
    # legitimately live in a file the recipe never mentions. These three make a
    # resumed run answerable:
    #
    #   build_id       digest of the resolved config (config.build_id)
    #   resolved       the fingerprint itself, so a drifted resume can say
    #                  WHICH field moved instead of only that one did
    #   defaults_files {path: sha256[:12]} for each file that contributed
    #
    # Additive, so no schema bump: this format has an independent reader in
    # seren-theatre and unknown keys already fall through to `extra`.
    build_id: str = ""
    resolved: Dict[str, Any] = field(default_factory=dict)
    defaults_files: Dict[str, str] = field(default_factory=dict)
    # Anything a future ms-moe-maker adds that this reader does not know about.
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- queries the viewer wants -------------------------------------------

    def stage(self, stage_id: str) -> Optional[Stage]:
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    @property
    def running(self) -> Optional[Stage]:
        for s in self.stages:
            if s.status == RUNNING:
                return s
        return None

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.stages if s.status in (DONE, SKIPPED))

    def stale(self, now: Optional[float] = None,
              after: float = STALE_AFTER_SECONDS) -> bool:
        """True if this manifest claims to be running but has gone quiet.

        The honest reading of an abandoned manifest. A process that is killed -
        OOM, a closed SSH session, a box reboot - never gets to write a
        terminal status, so its last word is "running" forever. A viewer that
        believes that shows a spinner for a run that died last Tuesday.
        """
        if self.finished is not None or self.running is None:
            return False
        return ((now or time.time()) - self.updated) > after


# -- writing -----------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then replace.

    Theatre polls this file every few seconds while the pipeline is writing it.
    A plain open-truncate-write has a window where the reader gets a half
    document and reports a broken run that is fine - so the reader would need
    retry logic to paper over a race the writer could simply not create.
    os.replace is atomic on POSIX and on Windows for same-volume replaces,
    which is why the temp file goes in the TARGET directory rather than /tmp.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write(run_dir: Path, manifest: Manifest) -> Path:
    """Persist a manifest into a run directory. Returns the path written."""
    manifest.updated = time.time()
    payload = {
        "schema_version": manifest.schema_version,
        "recipe_id": manifest.recipe_id,
        "name": manifest.name,
        "size": manifest.size,
        "base": manifest.base,
        "experts": list(manifest.experts),
        "started": manifest.started,
        "updated": manifest.updated,
        "finished": manifest.finished,
        "ok": manifest.ok,
        "refusals": list(manifest.refusals),
        "build_id": manifest.build_id,
        "resolved": dict(manifest.resolved),
        "defaults_files": dict(manifest.defaults_files),
        "stages": [asdict(s) for s in manifest.stages],
    }
    payload.update(manifest.extra)
    target = Path(run_dir) / MANIFEST_NAME
    _atomic_write(target, json.dumps(payload, indent=2) + "\n")
    return target


# -- reading -----------------------------------------------------------------

class UnreadableManifest(Exception):
    """The file is there and this reader cannot honestly interpret it."""


def read(run_dir: Path) -> Optional[Manifest]:
    """Load a manifest from a run directory.

    Returns None if there simply isn't one - that is not an error, it is an
    uninstrumented directory, which is a supported and normal thing to watch.

    Raises UnreadableManifest if a file EXISTS but cannot be trusted: corrupt
    JSON, a schema version from the future, a top-level shape that isn't an
    object. The distinction matters. "No manifest" means fall back to scraping
    and lose nothing. "Bad manifest" means something wrote a file we cannot
    read, and quietly scraping past it would hide a real problem behind a
    display that looks fine.
    """
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnreadableManifest(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise UnreadableManifest(f"{path}: top level is not an object")

    version = raw.get("schema_version")
    if not isinstance(version, int):
        raise UnreadableManifest(f"{path}: no usable schema_version")
    if version > SCHEMA_VERSION:
        raise UnreadableManifest(
            f"{path}: schema_version {version} is newer than this reader "
            f"understands ({SCHEMA_VERSION}). Upgrade the reader rather than "
            f"guessing at fields it has never seen.")

    known = {"build_id", "resolved", "defaults_files",
             "schema_version", "recipe_id", "name", "size", "base", "experts",
             "started", "updated", "finished", "ok", "refusals", "stages"}

    stages: List[Stage] = []
    for entry in raw.get("stages") or []:
        if not isinstance(entry, dict) or "id" not in entry:
            continue    # lenient: one malformed stage does not sink the run
        status = entry.get("status", PENDING)
        stages.append(Stage(
            id=str(entry["id"]),
            label=str(entry.get("label") or entry["id"]),
            # An unrecognised status is reported as-is rather than coerced. A
            # viewer that silently renders an unknown state as "pending" is
            # inventing a reading.
            status=status if status in STATUSES else str(status),
            started=entry.get("started"),
            ended=entry.get("ended"),
            artifact=entry.get("artifact"),
            note=entry.get("note"),
        ))

    return Manifest(
        schema_version=version,
        recipe_id=str(raw.get("recipe_id") or ""),
        name=str(raw.get("name") or ""),
        size=str(raw.get("size") or ""),
        base=str(raw.get("base") or ""),
        experts=[str(e) for e in (raw.get("experts") or [])],
        started=raw.get("started") or 0.0,
        updated=raw.get("updated") or 0.0,
        finished=raw.get("finished"),
        ok=raw.get("ok"),
        stages=stages,
        refusals=[str(r) for r in (raw.get("refusals") or [])],
        build_id=str(raw.get("build_id") or ""),
        resolved=raw.get("resolved") if isinstance(raw.get("resolved"), dict) else {},
        defaults_files=(raw.get("defaults_files")
                        if isinstance(raw.get("defaults_files"), dict) else {}),
        extra={k: v for k, v in raw.items() if k not in known},
    )
