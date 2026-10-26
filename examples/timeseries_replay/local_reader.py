"""
Local rows — reads a long-layout Parquet folder with its recipe (contract section 3).

Used by player.py --from-folder in place of the dataset API. Kept minimal:
Parquet only, long layout, signal templates over path fields, scale/offset applied.
"""

from __future__ import annotations

import fnmatch
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_ANNOTATION = {"type": "generic", "unit": "", "scale": 1.0, "offset": 0.0, "hidden": False}
EPOCH_UNITS = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


def pattern_to_regex(pattern: str) -> re.Pattern:
    """`{name}` captures one segment, `*` matches inside a segment, `**/` any leading segments."""
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "{":
            j = pattern.index("}", i)
            out += f"(?P<{pattern[i + 1:j]}>[^/]*)"
            i = j + 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(f"^{out}$")


def thing_value(fmt: str, column: str, thing: str) -> str | None:
    """Column value that renders `thing` with `fmt` (default "{column}={value}"), or None."""
    prefix, _, suffix = fmt.format(column=column, value="\0").partition("\0")
    if len(thing) >= len(prefix) + len(suffix) and thing.startswith(prefix) and thing.endswith(suffix):
        return thing[len(prefix) : len(thing) - len(suffix)]
    return None


class LocalRows:
    """Rows of the requested channels straight from the folder; never truncated."""

    def __init__(self, folder: str, recipe_path: str, channel_ids) -> None:
        self.folder = Path(folder)
        self.recipe = json.loads(Path(recipe_path).read_text())
        if self.recipe.get("layout") != "long" or self.recipe["files"].get("format") != "parquet":
            raise SystemExit("--from-folder supports layout 'long' with Parquet files only")
        self.time_column = self.recipe["time"]["column"]
        self.timezone = self.recipe["time"].get("timezone", "UTC")
        self.time_unit = self.recipe["time"].get("unit", "s")
        self.value_column = self.recipe["value"]["column"]
        thing = self.recipe.get("thing") or {}
        self.thing_columns = thing.get("columns") or []
        self.thing_format = thing.get("format") or "{column}={value}"
        self.files = self._scan_files()
        self.wanted: dict[str, dict[str, str]] = {}  # signal -> {thing: channel_id}
        self.units: dict[str, str] = {}
        for channel_id in channel_ids:
            signal, thing = self._split_channel(channel_id)
            self.wanted.setdefault(signal, {})[thing] = channel_id
            self.units[channel_id] = self.annotation(signal)["unit"]

    def _scan_files(self) -> dict[str, list[Path]]:
        regex = pattern_to_regex(self.recipe["path_pattern"]) if self.recipe.get("path_pattern") else None
        files: dict[str, list[Path]] = {}
        for path in sorted(self.folder.glob(self.recipe["files"]["glob"])):
            fields = {}
            if regex is not None:
                match = regex.match(path.relative_to(self.folder).as_posix())
                if match is None:
                    print(f"warning: {path} does not match path_pattern, skipped")
                    continue
                fields = match.groupdict()
            try:
                signal = self.recipe["signal"].format(**fields)
            except KeyError as exc:
                raise SystemExit(f"signal template field {exc} is not a path field (columns are not supported here)")
            files.setdefault(signal, []).append(path)
        return files

    def _split_channel(self, channel_id: str) -> tuple[str, str]:
        # channel_id is the signal, or "signal/thing"; the longest known signal wins.
        for signal in sorted(self.files, key=len, reverse=True):
            if channel_id == signal:
                return signal, ""
            if channel_id.startswith(signal + "/"):
                return signal, channel_id[len(signal) + 1 :]
        raise SystemExit(f"channel {channel_id}: no data file for its signal under {self.folder}")

    def annotation(self, signal: str) -> dict:
        ann = dict(DEFAULT_ANNOTATION, label=signal.rsplit("/", 1)[-1])
        for pattern, values in (self.recipe.get("signals") or {}).items():
            if fnmatch.fnmatchcase(signal, pattern):
                ann.update(values)
        return ann

    def fetch(self, start: float, end: float) -> tuple[list[tuple[float, str, float]], bool]:
        """Rows (epoch seconds, channel_id, value) of [start, end) sorted by time; never truncated."""
        rows: list[tuple[float, str, float]] = []
        for signal, wanted in self.wanted.items():
            ann = self.annotation(signal)
            for path in self.files[signal]:
                rows += self._read(path, start, end, wanted, ann)
        rows.sort()
        return rows, False

    def _read(self, path: Path, start: float, end: float, wanted: dict, ann: dict) -> list:
        schema = pq.read_schema(path)
        time_type = schema.field(self.time_column).type
        expr = (pc.field(self.time_column) >= self._literal(start, time_type)) & (
            pc.field(self.time_column) < self._literal(end, time_type)
        )
        columns = [self.time_column, self.value_column]
        column = next((c for c in self.thing_columns if c in schema.names), None)
        if column is None:
            if "" not in wanted:
                return []
            channel_of = {None: wanted[""]}
        else:
            channel_of = {}
            for thing, channel_id in wanted.items():
                value = thing_value(self.thing_format, column, thing)
                if value is not None:
                    channel_of[value] = channel_id
            if not channel_of:
                return []
            expr = expr & pc.field(column).cast(pa.string()).isin(list(channel_of))
            columns.append(column)
        table = pq.read_table(path, columns=columns, filters=expr)
        if table.num_rows == 0:
            return []
        frame = table.to_pandas()
        frame["value"] = pd.to_numeric(frame[self.value_column], errors="coerce") * ann["scale"] + ann["offset"]
        frame = frame[frame["value"].notna()]
        epoch = self._epoch(frame[self.time_column], time_type)
        things = frame[column].astype(str) if column else [None] * len(frame)
        # Plain floats: numpy scalars (e.g. int64 after an integer scale) are not JSON serializable.
        return [(float(t), channel_of[thing], float(v)) for t, thing, v in zip(epoch, things, frame["value"])]

    def _literal(self, epoch: float, time_type):
        if not pa.types.is_timestamp(time_type):
            return epoch * EPOCH_UNITS[self.time_unit]
        if time_type.tz is None:
            return datetime.fromtimestamp(epoch, ZoneInfo(self.timezone)).replace(tzinfo=None)
        return pa.scalar(datetime.fromtimestamp(epoch, ZoneInfo("UTC")), type=time_type)

    def _epoch(self, values: pd.Series, time_type) -> pd.Series:
        if pa.types.is_timestamp(time_type):
            times = pd.to_datetime(values)
            if times.dt.tz is None:
                times = times.dt.tz_localize(self.timezone)
        else:
            times = pd.to_datetime(values, unit=self.time_unit, utc=True)
        return (times.dt.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
