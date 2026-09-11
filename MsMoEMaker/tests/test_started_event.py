"""The `started` event is a published contract, so pin it to a real build.

WHY THIS EVENT AND NOT THE OTHERS. The manifest deliberately carries no
absolute path - `Stage.artifact` is relative precisely so a run directory can be
moved, copied to another box, or read through a mount with a different prefix.
That portability is worth keeping, and it leaves exactly one place where a build
states where it is actually running: this event.

So it is the only bridge between "the process I launched" and "the directory on
disk", and seren-theatre crosses it. A reader should not have to run a build and
read the output to learn the keys, hence `started_fields` in `--describe`.

THE ASSERTION IS AGAINST A REAL RUN, not against a literal in this file. A list
compared to a copy of itself agrees by construction and cannot fail; the point
is to catch a kwarg being renamed at the call site while the published contract
goes on claiming the old name.
"""
from __future__ import annotations

import json

import pytest

from ms_moe_maker.box.describe import DESCRIBE, STARTED_FIELDS
from ms_moe_maker.config.levers import Translation
from ms_moe_maker.run.events import Events
from ms_moe_maker.run.runner import Runner

from test_runner import FakeRecipe, lab                      # noqa: F401


def started_event(lab, capsys, experts=("python", "csharp")):
    """The real `started` object a dry-run build emits."""
    recipe = FakeRecipe(list(experts))
    Runner(recipe, lab, Translation(), Events(enabled=True),
           cwd=lab.parent, dryrun=True).run()
    for line in capsys.readouterr().out.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "started":
            return row
    pytest.fail("the build emitted no `started` event at all")


class TestThePublishedFieldsAreTheEmittedFields:

    def test_every_published_field_is_actually_emitted(self, lab, capsys):
        """The one with teeth. A renamed kwarg fails here, not at a consumer."""
        row = started_event(lab, capsys)
        missing = [f for f in STARTED_FIELDS if f not in row]
        assert not missing, (
            f"--describe publishes {missing} on the `started` event and a real "
            f"build does not emit them. A reader written against the contract "
            f"would find None where it was promised a path.")

    def test_nothing_load_bearing_is_emitted_without_being_published(self,
                                                                    lab, capsys):
        """The other direction, bounded to what a reader would want.

        `event` is the envelope, not payload. Anything else a build states about
        WHERE and UNDER WHAT it is running belongs in the contract, or a
        consumer has to discover it by reading our source.
        """
        row = started_event(lab, capsys)
        extra = set(row) - set(STARTED_FIELDS) - {"event"}
        assert not extra, (
            f"the `started` event carries {sorted(extra)} that --describe does "
            f"not publish. Add them to STARTED_FIELDS or stop emitting them; "
            f"an unpublished field is one a consumer can only find by reading "
            f"this package's source.")

    def test_the_run_directory_is_absolute(self, lab, capsys):
        """The whole reason a consumer crosses this bridge.

        A relative run_dir would be interpreted against the READER's cwd, which
        is a different box in the case this event exists to serve.
        """
        import os
        row = started_event(lab, capsys)
        assert os.path.isabs(row["run_dir"]), (
            f"run_dir {row['run_dir']!r} is relative; a reader on another box "
            f"would resolve it against its own working directory.")

    def test_the_payload_survives_json(self, lab, capsys):
        """It is emitted as JSON Lines, so every value has to round-trip."""
        row = started_event(lab, capsys)
        assert json.loads(json.dumps(row)) == row

    def test_describe_publishes_the_same_tuple(self):
        assert DESCRIBE["started_fields"] == list(STARTED_FIELDS)

    def test_started_is_a_declared_event_kind(self):
        """The envelope and the payload are published by the same card."""
        assert "started" in DESCRIBE["events"]
