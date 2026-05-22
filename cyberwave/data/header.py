"""Wire format encoding/decoding for the Cyberwave data layer.

Every Zenoh sample (and every filesystem blob) uses one framing::

    ┌──────────────────┬───────────┬───────────┬─────────────────────┬─────────────────┐
    │ header_len (u32) │ ts (f64)  │ seq (i64) │ header JSON (UTF-8) │ payload (bytes) │
    │  4 bytes, LE     │ 8 bytes   │ 8 bytes   │ variable length     │ variable length │
    └──────────────────┴───────────┴───────────┴─────────────────────┴─────────────────┘

*  ``header_len`` covers ``ts`` + ``seq`` + JSON (i.e. everything *between*
   the length prefix and the payload).
*  ``ts`` and ``seq`` are binary-packed per-sample fields — no JSON
   parsing needed to read the acquisition timestamp or detect drops.
*  The JSON portion carries static channel metadata (content_type, shape,
   dtype, channel-specific fields).  It is identical across samples on the
   same channel, which enables the :class:`HeaderTemplate` optimisation.
*  The round-trip is deterministic: ``decode(encode(hdr, payload))``
   returns the same header and payload bytes.

The :class:`HeaderTemplate` pre-encodes the JSON portion once and splices
only ``ts``/``seq`` on each :meth:`~HeaderTemplate.pack` call — suitable
for drivers publishing at 60 fps+ with sub-microsecond header overhead.
"""

from __future__ import annotations

import itertools
import json
import struct
import time
from dataclasses import dataclass
from typing import Any

from .exceptions import WireFormatError

_HEADER_LEN_FMT = "<I"  # 4-byte LE uint32 — total header length
_TS_SEQ_FMT = "<dq"  # 8-byte LE float64 (ts) + 8-byte LE int64 (seq)
_HEADER_LEN_SIZE = struct.calcsize(_HEADER_LEN_FMT)  # 4
_TS_SEQ_SIZE = struct.calcsize(_TS_SEQ_FMT)  # 16
_MAX_HEADER_BYTES = 64 * 1024

# Canonical content-type strings used in wire headers.
# Other SDKs (C++, etc.) must produce identical strings for cross-SDK compat.
CONTENT_TYPE_NUMPY = "numpy/ndarray"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_BYTES = "application/octet-stream"


@dataclass(slots=True)
class HeaderMeta:
    """Decoded header metadata for a single sample."""

    content_type: str
    ts: float
    seq: int = 0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    metadata: dict[str, Any] | None = None


# ── Encode / Decode (stateless, one-shot) ────────────────────────────


def encode(header: HeaderMeta, payload: bytes) -> bytes:
    """Encode *header* + *payload* into the canonical wire format.

    This is the stateless one-shot encoder.  For repeated publishes on the
    same channel, prefer :class:`HeaderTemplate` which caches the JSON
    portion.
    """
    json_dict: dict[str, Any] = {"content_type": header.content_type}
    if header.shape is not None:
        json_dict["shape"] = list(header.shape)
    if header.dtype is not None:
        json_dict["dtype"] = header.dtype
    if header.metadata:
        json_dict.update(header.metadata)
    json_bytes = json.dumps(json_dict, separators=(",", ":")).encode()
    total_header_len = _TS_SEQ_SIZE + len(json_bytes)
    if total_header_len > _MAX_HEADER_BYTES:
        raise WireFormatError(
            f"Header exceeds {_MAX_HEADER_BYTES} bytes ({total_header_len})."
        )
    return (
        struct.pack(_HEADER_LEN_FMT, total_header_len)
        + struct.pack(_TS_SEQ_FMT, header.ts, header.seq)
        + json_bytes
        + payload
    )


def decode(raw: bytes) -> tuple[HeaderMeta, bytes]:
    """Split a wire-format sample into ``(HeaderMeta, payload_bytes)``.

    Raises:
        WireFormatError: If *raw* is too short or the embedded header
            length is inconsistent.
    """
    min_size = _HEADER_LEN_SIZE + _TS_SEQ_SIZE
    if len(raw) < min_size:
        raise WireFormatError(
            f"Frame too short: expected at least {min_size} bytes, got {len(raw)}."
        )

    (total_header_len,) = struct.unpack_from(_HEADER_LEN_FMT, raw, 0)
    if total_header_len > _MAX_HEADER_BYTES:
        raise WireFormatError(
            f"Header length {total_header_len} exceeds maximum ({_MAX_HEADER_BYTES})."
        )
    header_end = _HEADER_LEN_SIZE + total_header_len
    if header_end > len(raw):
        raise WireFormatError(
            f"Frame truncated: header declares {total_header_len} bytes but "
            f"only {len(raw) - _HEADER_LEN_SIZE} available."
        )

    ts, seq = struct.unpack_from(_TS_SEQ_FMT, raw, _HEADER_LEN_SIZE)
    json_start = _HEADER_LEN_SIZE + _TS_SEQ_SIZE
    json_bytes = raw[json_start:header_end]

    try:
        json_dict = json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WireFormatError(f"Invalid header JSON: {exc}") from exc

    if not isinstance(json_dict, dict):
        raise WireFormatError(
            f"Header must be a JSON object, got {type(json_dict).__name__}."
        )
    if "content_type" not in json_dict:
        raise WireFormatError("Header missing required field 'content_type'.")

    # Copy before popping so the original parsed object is not mutated;
    # cheap for small headers and safe against future introspection uses.
    json_dict = dict(json_dict)
    content_type: str = json_dict.pop("content_type")
    shape_raw = json_dict.pop("shape", None)
    shape = tuple(shape_raw) if shape_raw is not None else None
    dtype: str | None = json_dict.pop("dtype", None)
    remaining_meta = json_dict if json_dict else None

    payload = raw[header_end:]
    return (
        HeaderMeta(
            content_type=content_type,
            ts=ts,
            seq=seq,
            shape=shape,
            dtype=dtype,
            metadata=remaining_meta,
        ),
        payload,
    )


# ── HeaderTemplate (cached encoder for hot paths) ────────────────────


class HeaderTemplate:
    """Pre-compiled header for repeated publishing on the same channel.

    Encodes the static JSON (content_type, shape, dtype, metadata) once
    during ``__init__``.  On each :meth:`pack` call, only ``ts`` and ``seq``
    are packed as binary — no JSON serialisation, no string formatting.

    Typical per-sample overhead of :meth:`pack` is **< 500 ns**.
    """

    __slots__ = (
        "_cached_json_bytes",
        "_cached_header_len_packed",
        "_seq_counter",
        "content_type",
        "shape",
        "dtype",
    )

    def __init__(
        self,
        content_type: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content_type = content_type
        self.shape = shape
        self.dtype = dtype
        self._seq_counter = itertools.count()

        json_dict: dict[str, Any] = {"content_type": content_type}
        if shape is not None:
            json_dict["shape"] = list(shape)
        if dtype is not None:
            json_dict["dtype"] = dtype
        if metadata:
            json_dict.update(metadata)
        self._cached_json_bytes: bytes = json.dumps(
            json_dict, separators=(",", ":")
        ).encode()
        cached_header_len = _TS_SEQ_SIZE + len(self._cached_json_bytes)
        if cached_header_len > _MAX_HEADER_BYTES:
            raise WireFormatError(
                f"Header exceeds {_MAX_HEADER_BYTES} bytes "
                f"({cached_header_len})."
            )
        self._cached_header_len_packed: bytes = struct.pack(
            _HEADER_LEN_FMT, cached_header_len
        )

    def pack(self, payload: bytes, *, ts: float | None = None) -> bytes:
        """Combine the cached header with *payload*.

        Per-sample cost: one ``struct.pack`` (16 bytes) + one ``b"".join``.
        """
        if ts is None:
            ts = time.time()
        seq = next(self._seq_counter)
        return b"".join((
            self._cached_header_len_packed,
            struct.pack(_TS_SEQ_FMT, ts, seq),
            self._cached_json_bytes,
            payload,
        ))

    @property
    def seq(self) -> int:
        """Approximate next sequence number (for monitoring; not exact under concurrency).

        Parses the internal ``itertools.count`` repr.  Not on the hot path.
        """
        r = repr(self._seq_counter)
        try:
            return int(r.split("(")[1].rstrip(")"))
        except (IndexError, ValueError):
            return 0
