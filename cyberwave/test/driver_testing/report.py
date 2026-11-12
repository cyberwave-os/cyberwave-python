"""The report of a stress run: ``report.json``, ``events.jsonl`` and ``joints.csv``.

``report.json`` holds the outcome, the configuration, every phase (passed,
failed or skipped, with metrics, error or reason) and whatever the caller adds
(the resolved plan, the twin profile, the scene). ``events.jsonl`` is the full
trace: every measured sample, sent target and command, and motion status, with
local monotonic and UTC receipt times. ``joints.csv`` flattens the measured
joints and the transmitted targets for plotting; a blank field in a target row
is an axis that target did not address, not zero.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def report_paths(path: Path) -> tuple[Path, str]:
    """``--report`` as (directory, report file name): a ``.json`` path or a directory."""
    path = Path(path)
    if path.suffix == ".json":
        return path.parent, path.name
    return path, "report.json"


def jsonable_config(args: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_report(
    path: Path,
    *,
    outcome: str,
    error: str | None,
    config: Mapping[str, Any],
    phases: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    **extra: Any,
) -> Path:
    """Write the three files; returns the directory they are in."""
    directory, name = report_paths(path)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "outcome": outcome,
        "error": error,
        "config": dict(config),
        "phases": list(phases),
        **extra,
    }
    (directory / name).write_text(json.dumps(report, indent=2, default=str) + "\n")
    with (directory / "events.jsonl").open("w") as output:
        for event in events:
            output.write(json.dumps(event, default=str) + "\n")
    with (directory / "joints.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["monotonic_s", "kind", *joint_names])
        for event in events:
            if event["kind"] in ("joints", "target"):
                values = event.get(
                    "positions", event["payload"].get("positions", event["payload"])
                )
                writer.writerow(
                    [
                        event["monotonic_s"],
                        event["kind"],
                        *(values.get(joint, "") for joint in joint_names),
                    ]
                )
    return directory
