"""Persistent offline queue for the Zenoh-MQTT bridge.

When the MQTT connection is down, outbound messages are appended to an
append-only log file on disk.  On reconnect the queue is drained in FIFO
order.

File format (one record per message)::

    <4-byte big-endian payload length><payload bytes>

The queue rotates to a new segment file when the current one exceeds a
threshold.  Old segments are deleted once fully drained and the total
queue size exceeds ``max_bytes``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER_STRUCT = struct.Struct(">I")  # 4-byte big-endian unsigned int
_SEGMENT_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB per segment file


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """A message waiting for MQTT delivery."""

    mqtt_topic: str
    payload: bytes
    qos: int = 1
    enqueued_at: float = 0.0


@dataclass
class OfflineQueue:
    """File-backed FIFO queue for offline buffering.

    Thread-safe: multiple threads may enqueue concurrently, but only one
    thread should call :meth:`drain` at a time (enforced internally).
    """

    queue_dir: str
    max_bytes: int = 50 * 1024 * 1024

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _drain_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _total_bytes: int = field(default=0, init=False, repr=False)
    _write_segment: Path | None = field(default=None, init=False, repr=False)
    _write_fp: io.BufferedWriter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.queue_dir, exist_ok=True)
        self._total_bytes = self._scan_size()

    @property
    def size_bytes(self) -> int:
        return self._total_bytes

    @property
    def is_empty(self) -> bool:
        return self._total_bytes <= 0

    def enqueue(self, msg: QueuedMessage) -> None:
        """Append *msg* to the queue."""
        record = self._encode(msg)
        with self._lock:
            self._ensure_write_segment()
            if self._write_fp is None:
                logger.error(
                    "Queue: no write segment available; message will be dropped"
                )
                return
            self._write_fp.write(record)
            self._write_fp.flush()
            self._total_bytes += len(record)
            if self._total_bytes > self.max_bytes:
                self._evict_oldest_unlocked()

    def drain(self, batch_size: int = 64) -> list[QueuedMessage]:
        """Pop up to *batch_size* messages from the front of the queue.

        Returns an empty list when the queue is empty.
        """
        with self._drain_lock:
            return self._drain_batch(batch_size)

    def close(self) -> None:
        with self._lock:
            if self._write_fp is not None:
                try:
                    self._write_fp.close()
                except Exception:
                    pass
                self._write_fp = None

    # ── Internal helpers ─────────────────────────────────────────────

    def _segment_paths(self) -> list[Path]:
        """Return segment files sorted oldest-first."""
        base = Path(self.queue_dir)
        return sorted(base.glob("seg_*.bin"))

    def _scan_size(self) -> int:
        total = 0
        for seg in self._segment_paths():
            try:
                total += seg.stat().st_size
            except OSError:
                pass
        return total

    def _ensure_write_segment(self) -> None:
        if self._write_fp is not None:
            try:
                pos = self._write_fp.tell()
            except Exception:
                pos = _SEGMENT_MAX_BYTES + 1
            if pos < _SEGMENT_MAX_BYTES:
                return
            self._write_fp.close()
            self._write_fp = None

        name = f"seg_{int(time.time() * 1_000_000):020d}.bin"
        self._write_segment = Path(self.queue_dir) / name
        self._write_fp = open(self._write_segment, "ab")  # noqa: SIM115

    def _evict_oldest_unlocked(self) -> None:
        """Delete oldest segments until under budget (caller holds _lock)."""
        segments = self._segment_paths()
        for seg in segments:
            if self._total_bytes <= self.max_bytes:
                break
            if self._write_segment is not None and seg == self._write_segment:
                continue
            try:
                sz = seg.stat().st_size
                seg.unlink()
                self._total_bytes -= sz
                logger.warning(
                    "Bridge queue: evicted segment %s (%d bytes) to stay under budget",
                    seg.name,
                    sz,
                )
            except OSError:
                pass

    def _drain_batch(self, batch_size: int) -> list[QueuedMessage]:
        messages: list[QueuedMessage] = []
        segments = self._segment_paths()

        for seg in segments:
            if len(messages) >= batch_size:
                break
            is_active = self._write_segment is not None and seg == self._write_segment
            msgs_before = len(messages)
            try:
                with open(seg, "rb") as fp:
                    while len(messages) < batch_size:
                        msg = self._read_one(fp)
                        if msg is None:
                            break
                        messages.append(msg)
            except OSError:
                continue

            if not is_active:
                try:
                    sz = seg.stat().st_size
                    seg.unlink()
                    with self._lock:
                        self._total_bytes -= sz
                except OSError:
                    pass
            else:
                # Rewrite the active segment, stripping only the records that
                # were consumed from *this* segment (not the global batch total).
                consumed = len(messages) - msgs_before
                self._rewrite_active_segment(seg, consumed)

        return messages

    def _rewrite_active_segment(self, seg: Path, consumed: int) -> None:
        """Rewrite the active segment, stripping already-drained records."""
        with self._lock:
            if self._write_fp is not None:
                self._write_fp.close()
                self._write_fp = None

            remaining = b""
            try:
                with open(seg, "rb") as fp:
                    skipped = 0
                    while skipped < consumed:
                        header = fp.read(_HEADER_STRUCT.size)
                        if len(header) < _HEADER_STRUCT.size:
                            break
                        (length,) = _HEADER_STRUCT.unpack(header)
                        fp.read(length)
                        skipped += 1
                    remaining = fp.read()
            except OSError:
                pass

            try:
                with open(seg, "wb") as fp:
                    fp.write(remaining)
                self._total_bytes = self._scan_size()
            except OSError:
                pass

            self._write_segment = seg
            self._write_fp = open(seg, "ab")  # noqa: SIM115

    # ── Wire format ──────────────────────────────────────────────────

    @staticmethod
    def _encode(msg: QueuedMessage) -> bytes:
        envelope = json.dumps(
            {
                "topic": msg.mqtt_topic,
                "qos": msg.qos,
                "ts": msg.enqueued_at or time.time(),
            },
            separators=(",", ":"),
        ).encode()
        # envelope + NUL + payload
        body = envelope + b"\x00" + msg.payload
        return _HEADER_STRUCT.pack(len(body)) + body

    @staticmethod
    def _read_one(fp: io.BufferedReader) -> QueuedMessage | None:
        header = fp.read(_HEADER_STRUCT.size)
        if len(header) < _HEADER_STRUCT.size:
            return None
        (length,) = _HEADER_STRUCT.unpack(header)
        body = fp.read(length)
        if len(body) < length:
            return None
        nul_idx = body.index(b"\x00")
        envelope = json.loads(body[:nul_idx])
        payload = body[nul_idx + 1 :]
        return QueuedMessage(
            mqtt_topic=envelope["topic"],
            payload=payload,
            qos=envelope.get("qos", 1),
            enqueued_at=envelope.get("ts", 0.0),
        )
