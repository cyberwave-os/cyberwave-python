"""Filesystem-based DataBackend — stdlib-only fallback.

Each channel maps to a directory under *base_dir*.  ``publish()`` writes
timestamped ``.bin`` files and maintains an atomic ``latest.bin`` pointer.
``subscribe()`` uses a polling thread to detect new files.  A configurable
ring-buffer size prevents unbounded disk growth.

This backend is intentionally simple and has **no external dependencies**.  It
is not the primary data path — Zenoh is.  Use it for minimal environments
where Zenoh cannot be installed, or for testing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .backend import DataBackend, Sample, Subscription
from .exceptions import PublishError


class _FileWatcher(threading.Thread):
    """Background thread that polls a channel directory for new samples."""

    def __init__(
        self,
        *,
        channel: str,
        channel_dir: Path,
        callback: Callable[[Sample], None],
        poll_interval: float,
        policy: str,
    ) -> None:
        super().__init__(daemon=True)
        self._channel = channel
        self._channel_dir = channel_dir
        self._callback = callback
        self._poll_interval = poll_interval
        self._policy = policy
        self._stop_event = threading.Event()
        self._seen: set[str] = set()
        self._init_seen()

    def _init_seen(self) -> None:
        """Mark existing files so we only deliver *new* samples."""
        if self._channel_dir.exists():
            for p in self._channel_dir.glob("[0-9]*.bin"):
                self._seen.add(p.name)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._poll()
            self._stop_event.wait(self._poll_interval)

    def _poll(self) -> None:
        if not self._channel_dir.exists():
            return
        new_files = sorted(
            p for p in self._channel_dir.glob("[0-9]*.bin") if p.name not in self._seen
        )
        if not new_files:
            return

        if self._policy == "latest":
            targets = new_files[-1:]
        else:
            targets = new_files

        for path in targets:
            try:
                payload = path.read_bytes()
            except OSError:
                continue
            meta_path = path.with_suffix(".meta.json")
            metadata = None
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            ts_ns = int(path.stem)
            self._callback(
                Sample(
                    channel=self._channel,
                    payload=payload,
                    timestamp=ts_ns / 1e9,
                    metadata=metadata,
                )
            )

        for path in new_files:
            self._seen.add(path.name)

    def stop(self) -> None:
        self._stop_event.set()


class FileSubscription(Subscription):
    """Subscription handle for the filesystem backend."""

    def __init__(self, watcher: _FileWatcher) -> None:
        self._watcher = watcher
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._watcher.stop()


class FilesystemBackend(DataBackend):
    """Filesystem-backed data bus.

    Args:
        base_dir: Root directory for channel data.  Defaults to
            ``$CYBERWAVE_DATA_DIR`` or ``/tmp/cyberwave_data``.
        ring_buffer_size: Max samples to keep per channel.
        poll_interval_s: Polling interval for subscriber watchers.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        ring_buffer_size: int = 100,
        poll_interval_s: float = 0.05,
    ) -> None:
        resolved = base_dir or os.environ.get(
            "CYBERWAVE_DATA_DIR", "/tmp/cyberwave_data"
        )
        self._base_dir = Path(resolved)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._ring_buffer_size = ring_buffer_size
        self._poll_interval = poll_interval_s
        self._lock = threading.Lock()
        self._watchers: list[_FileWatcher] = []
        self._closed = False

    # -- DataBackend implementation -------------------------------------------

    def publish(
        self,
        channel: str,
        payload: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        channel_dir = self._base_dir / _safe_channel_name(channel)
        channel_dir.mkdir(parents=True, exist_ok=True)

        ts = time.time_ns()
        sample_path = channel_dir / f"{ts}.bin"
        latest_path = channel_dir / "latest.bin"

        try:
            tmp_sample = channel_dir / f".sample_{ts}.tmp"
            tmp_sample.write_bytes(payload)
            tmp_sample.replace(sample_path)

            tmp = channel_dir / f".latest_{ts}.tmp"
            tmp.write_bytes(payload)
            tmp.replace(latest_path)

            if metadata:
                meta_path = channel_dir / f"{ts}.meta.json"
                meta_path.write_text(json.dumps(metadata))
        except OSError as exc:
            raise PublishError(
                f"Failed to write to channel '{channel}': {exc}"
            ) from exc

        self._prune(channel_dir)

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Sample], None],
        *,
        policy: str = "latest",
    ) -> Subscription:
        self._validate_policy(policy)
        channel_dir = self._base_dir / _safe_channel_name(channel)
        channel_dir.mkdir(parents=True, exist_ok=True)

        watcher = _FileWatcher(
            channel=channel,
            channel_dir=channel_dir,
            callback=callback,
            poll_interval=self._poll_interval,
            policy=policy,
        )
        watcher.start()
        with self._lock:
            self._watchers.append(watcher)
        return FileSubscription(watcher)

    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
    ) -> Sample | None:
        latest_path = self._base_dir / _safe_channel_name(channel) / "latest.bin"
        if not latest_path.exists():
            return None
        try:
            return Sample(
                channel=channel,
                payload=latest_path.read_bytes(),
            )
        except OSError:
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for w in self._watchers:
                w.stop()
            self._watchers.clear()

    # -- internal helpers -----------------------------------------------------

    def _prune(self, channel_dir: Path) -> None:
        bins = sorted(
            p for p in channel_dir.glob("[0-9]*.bin") if p.name != "latest.bin"
        )
        while len(bins) > self._ring_buffer_size:
            oldest = bins.pop(0)
            oldest.unlink(missing_ok=True)
            meta = oldest.with_suffix(".meta.json")
            meta.unlink(missing_ok=True)


def _safe_channel_name(channel: str) -> str:
    """Sanitise a channel name so it maps to a single, safe directory name.

    Replaces ``/`` with ``__`` and neutralises path-traversal components.

    Order is intentional: ``/`` is replaced first so that a literal ``..``
    in a channel segment (e.g. ``"foo/../bar"``) is caught by the second
    substitution *after* the slash has already been collapsed to ``__``.
    """
    name = channel.replace("/", "__")
    name = name.replace("\x00", "")
    name = name.replace("..", "_dotdot_")
    if not name or name.strip(".") == "":
        name = "_empty_"
    return name
