"""Record and replay debug utilities for the data layer.

``record()`` captures live samples to disk; ``replay()`` reads them back and
re-publishes through the data bus, triggering the same hooks as live data.

Quick start::

    from cyberwave.data import get_backend
    from cyberwave.data.recording import record, replay

    backend = get_backend()

    # Record
    with record(backend, ["frames/default", "depth"], "/tmp/session1"):
        ...  # samples flow through the backend

    # Replay
    replay(backend, "/tmp/session1", speed=1.0)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .backend import DataBackend, Sample, Subscription
from .exceptions import RecordingError
from .recording_format import (
    RecordingEntry,
    RecordingManifest,
    channel_to_filename,
    encode_entry,
    iter_entries,
)

logger = logging.getLogger(__name__)


class RecordingSession:
    """Active recording.  Use as a context manager or call :meth:`stop` manually.

    Subscribes to the requested channels via the backend and writes every
    incoming raw sample to per-channel ``.bin`` files.  A ``manifest.json``
    is written when the session is stopped.
    """

    def __init__(
        self,
        backend: DataBackend,
        channels: list[str],
        path: Path,
        *,
        max_samples: int | None = None,
        max_duration_s: float | None = None,
    ) -> None:
        self._backend = backend
        self._channels = list(channels)
        self._path = path
        self._max_samples = max_samples
        self._max_duration_s = max_duration_s

        self._lock = threading.Lock()
        self._sample_count = 0
        self._start_ts: float | None = None
        self._end_ts: float = 0.0
        self._stopped = False
        self._subscriptions: list[Subscription] = []
        self._file_handles: dict[str, Any] = {}
        self._channel_set: set[str] = set()

        self._path.mkdir(parents=True, exist_ok=True)
        self._start()

    def _start(self) -> None:
        self._start_ts = time.time()
        for ch in self._channels:
            fname = channel_to_filename(ch)
            fpath = self._path / fname
            fh = open(fpath, "wb")  # noqa: SIM115
            self._file_handles[ch] = fh
            self._channel_set.add(ch)

            sub = self._backend.subscribe(
                ch,
                self._make_callback(ch),
                policy="fifo",
            )
            self._subscriptions.append(sub)

    def _make_callback(self, channel: str) -> Any:
        def _on_sample(sample: Sample) -> None:
            with self._lock:
                if self._stopped:
                    return
                if (
                    self._max_samples is not None
                    and self._sample_count >= self._max_samples
                ):
                    self._do_stop()
                    return
                if (
                    self._max_duration_s is not None
                    and self._start_ts is not None
                    and (time.time() - self._start_ts) >= self._max_duration_s
                ):
                    self._do_stop()
                    return

                fh = self._file_handles.get(channel)
                if fh is None or fh.closed:
                    return

                ts = sample.timestamp
                entry_bytes = encode_entry(ts, sample.payload)
                fh.write(entry_bytes)
                fh.flush()
                self._sample_count += 1
                self._end_ts = max(self._end_ts, ts)

        return _on_sample

    def _do_stop(self) -> RecordingManifest:
        if self._stopped:
            return self._build_manifest()
        self._stopped = True

        for sub in self._subscriptions:
            sub.close()
        self._subscriptions.clear()

        for fh in self._file_handles.values():
            if not fh.closed:
                fh.close()
        self._file_handles.clear()

        manifest = self._build_manifest()
        manifest_path = self._path / "manifest.json"
        manifest_path.write_text(manifest.to_json())
        return manifest

    def _build_manifest(self) -> RecordingManifest:
        return RecordingManifest(
            channels=list(self._channel_set),
            start_ts=self._start_ts or 0.0,
            end_ts=self._end_ts,
            sample_count=self._sample_count,
        )

    def stop(self) -> RecordingManifest:
        """Stop the recording and write the manifest.

        Returns the :class:`~.recording_format.RecordingManifest` describing
        the session.  Safe to call multiple times.
        """
        with self._lock:
            return self._do_stop()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def __enter__(self) -> RecordingSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def record(
    backend: DataBackend,
    channels: list[str],
    path: str | Path,
    *,
    max_samples: int | None = None,
    max_duration_s: float | None = None,
) -> RecordingSession:
    """Start recording samples from *channels* to disk.

    Returns a :class:`RecordingSession` that can be used as a context manager.

    Args:
        backend: The data backend to subscribe through.
        channels: Channel names to record (e.g. ``["frames/default", "depth"]``).
        path: Directory where the recording will be stored.
        max_samples: Stop automatically after this many samples.
        max_duration_s: Stop automatically after this many seconds.
    """
    if not channels:
        raise RecordingError("At least one channel must be specified for recording")
    return RecordingSession(
        backend,
        channels,
        Path(path),
        max_samples=max_samples,
        max_duration_s=max_duration_s,
    )


def replay(
    backend: DataBackend,
    path: str | Path,
    *,
    speed: float = 1.0,
    loop: bool = False,
    channels: list[str] | None = None,
) -> ReplayResult:
    """Replay recorded samples by publishing them to *backend*.

    Preserves inter-sample timing scaled by *speed*.  ``speed=0`` means
    instantaneous (no sleeping).  Replayed samples trigger the same
    subscriber hooks as live data.

    Args:
        backend: The data backend to publish through.
        path: Directory containing the recording (with ``manifest.json``).
        speed: Playback speed multiplier.  ``1.0`` = real-time, ``2.0`` = 2x,
            ``0`` = as fast as possible.
        loop: If ``True``, replay repeats until interrupted.
        channels: Subset of recorded channels to replay.  ``None`` = all.

    Returns:
        A :class:`ReplayResult` with summary statistics.

    Raises:
        RecordingError: If the recording path or manifest is invalid.
    """
    rec_path = Path(path)
    manifest_path = rec_path / "manifest.json"
    if not manifest_path.exists():
        raise RecordingError(f"No manifest.json found in {rec_path}")

    manifest = RecordingManifest.from_json(manifest_path.read_text())

    target_channels = channels if channels is not None else manifest.channels
    if not target_channels:
        raise RecordingError("No channels to replay")

    available = set(manifest.channels)
    missing = set(target_channels) - available
    if missing:
        raise RecordingError(
            f"Channels not found in recording: {sorted(missing)}.  "
            f"Available: {sorted(available)}"
        )

    total_published = 0
    pass_count = 0

    while True:
        pass_count += 1
        entries = _load_timeline(rec_path, target_channels)
        if not entries:
            break

        entries.sort(key=lambda e: e.timestamp)

        prev_ts: float | None = None
        for entry in entries:
            if speed > 0 and prev_ts is not None:
                delta = entry.timestamp - prev_ts
                if delta > 0:
                    time.sleep(delta / speed)
            prev_ts = entry.timestamp

            backend.publish(entry.channel, entry.payload)
            total_published += 1

        if not loop:
            break

    return ReplayResult(
        samples_published=total_published,
        passes=pass_count,
        channels=list(target_channels),
    )


class ReplayResult:
    """Summary returned by :func:`replay`."""

    __slots__ = ("samples_published", "passes", "channels")

    def __init__(
        self, *, samples_published: int, passes: int, channels: list[str]
    ) -> None:
        self.samples_published = samples_published
        self.passes = passes
        self.channels = channels

    def __repr__(self) -> str:
        return (
            f"ReplayResult(samples_published={self.samples_published}, "
            f"passes={self.passes}, channels={self.channels!r})"
        )


def _load_timeline(rec_path: Path, channels: list[str]) -> list[RecordingEntry]:
    """Read all entries for the given channels into a flat list."""
    entries: list[RecordingEntry] = []
    for ch in channels:
        fname = channel_to_filename(ch)
        fpath = rec_path / fname
        if not fpath.exists():
            logger.warning("Channel file missing: %s", fpath)
            continue
        entries.extend(iter_entries(fpath, ch))
    return entries
