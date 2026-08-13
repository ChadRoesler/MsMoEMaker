"""Fork the pipeline, follow it, and report what it is doing.

WRAP-THEN-CARVE, and why this file is a subprocess driver rather than an import.

`fraunkenstein_universal.py` is 2483 lines that work. Inside them are three
rungs' worth of hard-won knowledge about Jetson allocators, GGUF loader limits,
tqdm bars that lie, and an OOM that took a box down for measuring the wrong
thing. Rewriting that as a first move risks all of it to gain a file layout.
So the first move is to put a stable CLI, an event stream and a manifest in
front of it, and carve behind those. The contract is the product; the internals
can move for a year without anyone downstream noticing.

FORK, NEVER IMPORT. Non-negotiable, and not only for tidiness:

  * importing the pipeline executes ~800 lines of configuration, prints a
    banner, and imports torch - so `ms-moe validate` would cost a CUDA context.
  * PYTORCH_CUDA_ALLOC_CONF must be set BEFORE torch initialises or it does
    nothing at all. In-process, our own import graph decides whether we win
    that race. In a child, the environment is simply correct at exec time.
    This is the lever measured at 106.6 GB reserved versus 8.3 GB for the same
    weights; losing it silently is not an option.
  * a build that OOMs should kill a subprocess, not the thing reporting on it.

It is also the same rule stagehand follows one level up - a stagehand is not on
stage - and it means the automated path runs the literal command a human would
type, so the hand-run path can never rot from having no users.

READING THE CHILD. Milestone regexes are lifted from seren_theatre/sources.py,
which already parses these logs and has already been burned by them: the
`_LABELLED` check exists because "579" appears identically under every expert
and is a weight-loading bar, not a step count, and the first parser written
against this log reported 579/579 at 2398it/s. Those patterns are quoted here,
not re-derived. When the carve replaces prints with structured output, this
module gets simpler and sources.py stops needing them at all.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import manifest as mf
from . import stages as st
from .events import Events
from .levers import Translation, resolved_roots

# -- what the child says, and what it means ----------------------------------
# Quoted from the pipeline's own prints. Each maps a line to (stage_id, status).

_SKIP = re.compile(r"^\[skip\] (.+?) already present at (.+)$")
# The code-corpus phase, which _SKIP alone does NOT cover. Three separate
# messages, all quoted from the pipeline, all previously invisible:
#
#   "[skip] code datasets already built:"  - the SUMMARY early-out, taken when
#       every language is already on disk. Different wording from _done()'s
#       per-language line, so it matched nothing.
#   "Shard scan will hunt only: ..." / "N shards available; ..."  - the START
#       of a fresh scan. On a first run with no cached corpora this is the
#       single longest phase of the build - 45 GB of shards - and the stage sat
#       at `pending` for all of it, which is the worst possible time for the
#       viewer to have nothing to say.
#   "Scanned N repos across M shard(s)."  - the end of it.
_SKIP_ALL_CODE = re.compile(r"^\[skip\] code datasets already built")
_CODE_START = re.compile(r"^Shard scan will hunt only:|shards available; will pull")
_CODE_DONE = re.compile(r"^Scanned (\d+) repos across (\d+) shard")
_FINETUNE = re.compile(r"^Fine-tuning (\S+?)\.\.\.")
_SAVED = re.compile(r"Dense specialist saved to (.+)$")
_STITCH = re.compile(r"Stitching (\d+) experts")
_SKELETON = re.compile(r"MoE skeleton saved")
_ROUTER = re.compile(r"router-only training: ([\d,]+) trainable")
_ALIVE = re.compile(r"Agent MoE is ALIVE")
_GGUF_START = re.compile(r"^Exporting GGUF")
_GGUF_OK = re.compile(r"converted OK \(([\d.]+) GB\)")
_SMOKE_OK = re.compile(r"smoke test PASSED")
_AGENT_READY = re.compile(r"Agent dataset ready: (\d+) samples")
_CFG = re.compile(r"^\[cfg\] (.+)$")

# Lines that mean the run is in trouble. Surfaced as warnings on the event
# stream AND pinned onto the current stage's note, because a warning that
# scrolls past in a six-hour log is a warning nobody sees.
_WARNINGS = (
    (re.compile(r"\*\*\* DISAGREES WITH THE ENV"),
     "dense_layers env is being ignored - a skeleton already exists"),
    (re.compile(r"is SHORT of the token budget"),
     "an expert is short of its token budget"),
    (re.compile(r"ALLOCATOR BALLOON"), "allocator ballooning"),
    (re.compile(r"did not finish in \d+s"), "smoke test hung"),
    (re.compile(r"CUDA out of memory|NV_ERR_NO_MEMORY"), "CUDA out of memory"),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback in the child"),
)

# The one child failure that is ALWAYS an environment answer rather than a bug.
# Reported with the interpreter that was actually used, because "No module
# named 'torch'" on a box that definitely has torch is a genuinely confusing
# thing to read at 2am.
_MISSING_MODULE = re.compile(
    r"ModuleNotFoundError: No module named '([^']+)'")

# The pipeline's `_done()` messages name the artifact in prose. Map its words
# back onto stage ids so a resumed run reports SKIPPED instead of looking like
# it never happened.
_SKIP_WHAT = {
    "MoE skeleton": st.STITCH,
    "trained MoE (final)": st.ROUTER,
    "GGUF export": st.EXPORT_GGUF,
    "agent dataset": st.DATA_AGENT,
    "MCP agent traces": st.DATA_AGENT,
}


class Runner:
    """Drives one build and keeps the manifest honest while it runs."""

    def __init__(self, recipe: Any, pipeline: Path, translation: Translation,
                 events: Events, cwd: Optional[Path] = None,
                 dryrun: bool = False, python: Optional[str] = None) -> None:
        self.recipe = recipe
        self.pipeline = Path(pipeline)
        self.translation = translation
        self.ev = events
        self.cwd = Path(cwd or self.pipeline.parent)
        self.dryrun = dryrun
        # WHICH INTERPRETER RUNS THE PIPELINE. Defaults to ours, and that
        # default was a bug for exactly as long as it was the only option.
        #
        # The entire argument for ms-moe being a separate package is that it is
        # small enough to live anywhere - `ms-moe validate` on a laptop, the CLI
        # inside seren-theatre's venv - while the TRAINER lives in whatever fat
        # venv has torch. Forking is what makes that possible. Hardcoding
        # sys.executable silently collapsed the two back into one, so the
        # pipeline was launched with the viewer's interpreter and died on
        # `import torch` - which is not a missing dependency, it is the right
        # dependency in the other venv.
        self.python = python or sys.executable

        roots = resolved_roots(recipe.size, dryrun)
        self.run_dir = self.cwd / roots["output"]
        self.data_root = self.cwd / roots["data"]

        experts = [e.name for e in recipe.experts]
        self.manifest = mf.Manifest(
            recipe_id=_recipe_id(recipe),
            name=recipe.name,
            size=recipe.size,
            base=recipe.base,
            experts=experts,
            refusals=list(translation.refusals),
            stages=[mf.Stage(id=sid, label=label)
                    for sid, label in st.plan(experts)],
        )
        self._current: Optional[str] = None
        self._missing_module: Optional[str] = None

    # -- manifest bookkeeping ----------------------------------------------

    def _flush(self) -> None:
        """Persist. Cheap, atomic, and the only way a watcher learns anything.

        Called after every state change rather than on a timer: the manifest is
        the heartbeat, and `Manifest.stale()` reads `updated` to decide whether
        a run died. A lazy writer would make a healthy slow stage look dead.
        """
        try:
            mf.write(self.run_dir, self.manifest)
        except OSError as exc:
            # Never let reporting kill the build it is reporting on.
            self.ev.warning(f"could not write the run manifest: {exc}")

    def _set(self, stage_id: str, status: str, **kw: Any) -> None:
        stage = self.manifest.stage(stage_id)
        if stage is None:                   # an expert the plan didn't predict
            stage = mf.Stage(id=stage_id, label=st.label_for(stage_id))
            self.manifest.stages.append(stage)
        now = time.time()
        if status == mf.RUNNING and stage.started is None:
            stage.started = now
        if status in mf.TERMINAL:
            stage.ended = now
            if stage.started is None:
                stage.started = now
        stage.status = status
        for key, value in kw.items():
            setattr(stage, key, value)
        if status == mf.RUNNING:
            self._current = stage_id
        self.ev.stage(stage_id, status, label=stage.label,
                      elapsed=round(stage.elapsed or 0.0, 1))
        self._flush()

    def _finish_current(self, status: str = mf.DONE, **kw: Any) -> None:
        if self._current:
            # APPEND to a note rather than replacing it. Losing what the stage
            # had already learned is a real loss: a GGUF stage that recorded
            # "converted 3.78 GB - smoke test pending" and then died would have
            # that replaced by a bare "child exited 1", throwing away the one
            # fact that says how far it actually got. Caught by the fake
            # pipeline; the two facts are both true and both wanted.
            note = kw.get("note")
            if note:
                stage = self.manifest.stage(self._current)
                existing = stage.note if stage else None
                if existing and note not in existing:
                    kw["note"] = f"{existing}; {note}"
            self._set(self._current, status, **kw)
            self._current = None

    # -- the run ------------------------------------------------------------

    def run(self) -> int:
        env = dict(os.environ)
        env.update(self.translation.env)
        if self.dryrun:
            env["FRAUNK_DRYRUN"] = "1"
        # Line-buffer the child so we see its output as it happens rather than
        # in 8 KiB gulps. Without this a stage boundary can arrive minutes
        # after the child crossed it, and the dashboard lags reality.
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [self.python, str(self.pipeline.name)]
        self.ev.started(recipe_id=self.manifest.recipe_id,
                        name=self.manifest.name, size=self.manifest.size,
                        experts=self.manifest.experts,
                        run_dir=str(self.run_dir), data_root=str(self.data_root),
                        command=" ".join(cmd), cwd=str(self.cwd),
                        env_applied=self.translation.env,
                        agreed=self.translation.agreed)
        self.ev.say(f"→ {' '.join(cmd)}   (cwd {self.cwd})")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._set(st.PREFLIGHT, mf.RUNNING)

        proc = subprocess.Popen(
            cmd, cwd=str(self.cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace")

        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                # Pass the child's own words through untouched. A wrapper that
                # swallows the log it is summarising makes debugging strictly
                # harder than not having the wrapper.
                self.ev.say(line)
                self._consume(line)
        finally:
            code = proc.wait()

        ok = code == 0
        if not ok:
            hint = ""
            if self._missing_module:
                hint = (f"; the pipeline was run with {self.python} and could "
                        f"not import {self._missing_module!r} - that is an "
                        f"environment answer, not a missing dependency. Point "
                        f"--python at the venv that has it.")
                self.ev.error(stage="environment", message=hint.lstrip("; "))
                self.ev.say("")
                self.ev.say(f"   The pipeline ran under: {self.python}")
                self.ev.say(f"   It could not import:    {self._missing_module}")
                self.ev.say("   ms-moe is deliberately small and does NOT ship "
                            "the trainer's dependencies.")
                self.ev.say("   Point it at the training venv:")
                self.ev.say("     ms-moe build recipe.yaml "
                            "--python /path/to/train-venv/bin/python")
            self._finish_current(mf.FAILED, note=f"child exited {code}{hint}")
            self.ev.error(stage=self._current or "run",
                          message=f"pipeline exited {code}")
        else:
            self._finish_current(mf.DONE)
            for stage in self.manifest.stages:
                if stage.status == mf.PENDING:
                    # Honest: the run finished without ever reaching this.
                    # Almost always means the artifact already existed and the
                    # child said nothing, but we did not SEE that, so we do not
                    # claim it.
                    stage.note = stage.note or "never reported by the pipeline"

        self.manifest.finished = time.time()
        self.manifest.ok = ok
        self._flush()
        self.ev.done(ok=ok, exit_code=code, run_dir=str(self.run_dir),
                     stages_done=self.manifest.done_count,
                     stages_total=len(self.manifest.stages),
                     refusals=len(self.manifest.refusals))
        return code

    # -- line -> meaning ----------------------------------------------------

    def _consume(self, line: str) -> None:
        m = _MISSING_MODULE.search(line)
        if m:
            self._missing_module = m.group(1)

        for pattern, message in _WARNINGS:
            if pattern.search(line):
                self.ev.warning(message)
                if self._current:
                    stage = self.manifest.stage(self._current)
                    if stage:
                        stage.note = message
                        self._flush()

        m = _CFG.match(line)
        if m:
            self.ev.progress(st.PREFLIGHT, cfg=m.group(1))
            return

        if _SKIP_ALL_CODE.match(line):
            self._finish_current()
            self._set(st.DATA_CODE, mf.SKIPPED,
                      note="every language already on disk")
            return

        if _CODE_START.search(line):
            # Only if it has not already finished - the pipeline prints the
            # shard-count line even on a partial resume.
            stage = self.manifest.stage(st.DATA_CODE)
            if stage is None or stage.status not in mf.TERMINAL:
                self._finish_current()
                self._set(st.DATA_CODE, mf.RUNNING,
                          note="scanning shards")
            return

        m = _CODE_DONE.match(line)
        if m:
            self._set(st.DATA_CODE, mf.DONE,
                      note=f"scanned {m.group(1)} repos across "
                           f"{m.group(2)} shard(s)")
            self._current = None
            return

        m = _SKIP.match(line)
        if m:
            what, path = m.group(1), m.group(2)
            stage_id = _skip_to_stage(what)
            if stage_id:
                self._set(stage_id, mf.SKIPPED,
                          artifact=_relative(path, self.run_dir),
                          note="already present on disk")
            return

        m = _FINETUNE.match(line)
        if m:
            self._finish_current()
            self._set(st.finetune_id(m.group(1)), mf.RUNNING)
            return

        m = _SAVED.search(line)
        if m:
            self._finish_current(artifact=_relative(m.group(1), self.run_dir))
            return

        if _AGENT_READY.search(line):
            self._finish_current()
            self._set(st.DATA_AGENT, mf.DONE)
            return

        if _STITCH.search(line):
            self._finish_current()
            self._set(st.STITCH, mf.RUNNING)
            return

        if _SKELETON.search(line):
            self._set(st.STITCH, mf.DONE,
                      artifact=st.artifact_for(st.STITCH))
            self._current = None
            return

        if _ROUTER.search(line):
            self._finish_current()
            self._set(st.ROUTER, mf.RUNNING)
            return

        if _ALIVE.search(line):
            self._set(st.ROUTER, mf.DONE, artifact=st.artifact_for(st.ROUTER))
            self._current = None
            return

        if _GGUF_START.search(line):
            self._finish_current()
            self._set(st.EXPORT_GGUF, mf.RUNNING)
            return

        m = _GGUF_OK.search(line)
        if m:
            # CONVERTED IS NOT PROVEN, and this is the one place the pipeline
            # has actually been burned: a GGUF that converts and then hangs its
            # smoke test would be treated as finished forever. So conversion
            # only records the artifact and a note - the stage stays RUNNING
            # until the smoke test says otherwise.
            stage = self.manifest.stage(st.EXPORT_GGUF)
            if stage:
                stage.note = f"converted {m.group(1)} GB - smoke test pending"
                self._flush()
            self.ev.progress(st.EXPORT_GGUF, converted_gb=float(m.group(1)))
            return

        if _SMOKE_OK.search(line):
            self._set(st.EXPORT_GGUF, mf.DONE,
                      note="converted and smoke-tested")
            self._current = None
            return


def _skip_to_stage(what: str) -> Optional[str]:
    """Map a `_done()` prose label onto a stage id.

    The per-language datasets say "<Language> dataset", which is one stage
    (data.code) covering several artifacts, so it is matched by suffix rather
    than listed.
    """
    if what in _SKIP_WHAT:
        return _SKIP_WHAT[what]
    if what.endswith(" dataset"):
        return st.DATA_CODE
    return None


def _relative(path: str, root: Path) -> str:
    """Store artifact paths relative to the run dir - see manifest.Stage.

    Falling back to the ABSOLUTE path was the original behaviour and it was
    wrong: manifest.Stage.artifact promises a relative path precisely so the
    manifest survives the run directory being moved, copied to another box, or
    read through a mount with a different prefix - which is exactly what
    Theatre does when it watches a stage over a share. One absolute path in
    there and that guarantee is gone, silently, for that entry.

    So an un-relativisable path degrades to its BASENAME, which is the part
    that was ever meaningful relative to the run directory. Caught by
    tests/test_runner.py the first time the fake pipeline ran.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except (ValueError, OSError):
        return Path(str(path).rstrip("/\\")).name or str(path)


def _recipe_id(recipe: Any) -> str:
    getter = getattr(recipe, "recipe_id", None)
    if callable(getter):
        return str(getter())
    return str(getter or "")
