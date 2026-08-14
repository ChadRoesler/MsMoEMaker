"""JSON Lines events - the machine-readable twin of the human output.

The Starwright contract, reused: human prose on stderr, one JSON object per
line on stdout when --json is on. Two channels, never interleaved, so a
consumer can parse stdout without a heuristic for "is this line prose".

WHY STDERR FOR THE PROSE. The other direction - prose on stdout, events on a
side channel - means anything piping the tool has to know about the side
channel. Putting the machine stream on stdout makes `ms-moe-maker build r.yaml --json
| jq` work with no ceremony, which is the whole point of having it.

ONE RULE, and it is the one that gets broken: flush every line. A consumer
following a build is reading a pipe, and Python block-buffers a pipe by
default. Without the flush, a run that takes six hours emits its first event
when the buffer fills or the process exits, and a dashboard watching it shows
nothing at all for hours while everything is completely fine. That failure
looks exactly like a hang.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO


class Events:
    """Emitter. Inert unless --json was passed, so call sites need no branch."""

    def __init__(self, enabled: bool = False, stream: TextIO | None = None,
                 prose: TextIO | None = None) -> None:
        self.enabled = enabled
        self._out = stream or sys.stdout
        self._prose = prose or sys.stderr

    def emit(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        # default=str so a Path or a dataclass never turns a progress report
        # into a crash. An event stream that can kill the build it is
        # describing is worse than no event stream.
        self._out.write(json.dumps({"event": kind, **fields}, default=str) + "\n")
        self._out.flush()

    def say(self, message: str) -> None:
        """Human line. Always stderr, never the event stream."""
        self._prose.write(message + "\n")
        self._prose.flush()

    # -- the vocabulary, as methods so typos are import errors not silence ---

    def started(self, **kw: Any) -> None:
        self.emit("started", **kw)

    def stage(self, id: str, status: str, **kw: Any) -> None:
        self.emit("stage", id=id, status=status, **kw)

    def progress(self, id: str, **kw: Any) -> None:
        self.emit("progress", id=id, **kw)

    def refused(self, reasons: list[str]) -> None:
        self.emit("refused", reasons=reasons, count=len(reasons))

    def warning(self, message: str) -> None:
        self.emit("warning", message=message)

    def error(self, stage: str, message: str) -> None:
        self.emit("error", stage=stage, message=message)

    def done(self, ok: bool, **kw: Any) -> None:
        self.emit("done", ok=ok, **kw)
