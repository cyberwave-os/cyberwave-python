"""On-disk format for recorded data sessions.

A recording is a directory containing one binary file per channel and a
``manifest.json`` that describes the session.

Directory layout::

    my_recording/
    ├── manifest.json
    ├── frames__default.bin
    ├── depth__realsense_left.bin
    └── joint_states__default.bin

Each ``.bin`` file is a sequence of *entries*.  Every entry is self-describing::

    [ts: 8 bytes LE float64]
    [payload_len: 4 bytes LE uint32]
    [payload: payload_len bytes]

``manifest.json`` schema::

    {
      "version": 1,
      "channels": ["frames/default", "depth/realsense_left"],
      "start_ts": 1711234567.123,
      "end_ts": 1711234597.456,
      "sample_count": 1500,
      "metadata": {}
    }

Version 1 is the initial format.  Readers MUST reject manifests with an
unsupported version.  Writers MUST always set ``version`` to the current
:data:`FORMAT_VERSION`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

FORMAT_VERSION = 1

_TS_FMT = "<d"  # float64 little-endian
_TS_SIZE = struct.calcsize(_TS_FMT)
_LEN_FMT = "<I"  # uint32 little-endian
_LEN_SIZE = struct.calcsize(_LEN_FMT)
_ENTRY_HEADER_SIZE = _TS_SIZE + _LEN_SIZE


@dataclass(slots=True)
class RecordingManifest:
    """Metadata written to ``manifest.json`` on session close."""

    version: int = FORMAT_VERSION
    channels: list[str] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "channels": self.channels,
                "start_ts": self.start_ts,
                "end_ts": self.end_ts,
                "sample_count": self.sample_count,
                "metadata": self.metadata,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> RecordingManifest:
        data = json.loads(text)
        version = data.get("version", 0)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported recording format version {version}; "
                f"expected {FORMAT_VERSION}"
            )
        return cls(
            version=version,
            channels=data.get("channels", []),
            start_ts=data.get("start_ts", 0.0),
            end_ts=data.get("end_ts", 0.0),
            sample_count=data.get("sample_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class RecordingEntry:
    """A single entry read back from a ``.bin`` file."""

    channel: str
    timestamp: float
    payload: bytes


def channel_to_filename(channel: str) -> str:
    """Convert a channel name to a safe filename for the ``.bin`` file.

    ``/`` is replaced with ``__`` to match the filesystem backend convention.
    """
    safe = channel.replace("/", "__")
    safe = safe.replace("\x00", "")
    safe = safe.replace("..", "_dotdot_")
    if not safe or safe.strip(".") == "":
        safe = "_empty_"
    return f"{safe}.bin"


def filename_to_channel(filename: str) -> str:
    """Reverse :func:`channel_to_filename` — recover the original channel name.

    Strips the ``.bin`` suffix and replaces ``__`` with ``/``.
    """
    stem = filename
    if stem.endswith(".bin"):
        stem = stem[:-4]
    return stem.replace("__", "/")


def encode_entry(timestamp: float, payload: bytes) -> bytes:
    """Encode a single recording entry to bytes."""
    return (
        struct.pack(_TS_FMT, timestamp) + struct.pack(_LEN_FMT, len(payload)) + payload
    )


def iter_entries(path: Path, channel: str) -> Iterator[RecordingEntry]:
    """Yield :class:`RecordingEntry` objects from a ``.bin`` file."""
    data = path.read_bytes()
    offset = 0
    total = len(data)
    while offset < total:
        if offset + _ENTRY_HEADER_SIZE > total:
            break
        (ts,) = struct.unpack_from(_TS_FMT, data, offset)
        offset += _TS_SIZE
        (payload_len,) = struct.unpack_from(_LEN_FMT, data, offset)
        offset += _LEN_SIZE
        if offset + payload_len > total:
            break
        payload = data[offset : offset + payload_len]
        offset += payload_len
        yield RecordingEntry(channel=channel, timestamp=ts, payload=payload)
