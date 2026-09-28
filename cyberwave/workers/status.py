"""Startup milestones the worker publishes for its supervisor.

A worker container reaching Docker's ``running`` state is not a worker that can
accept work: the process still has to import worker modules (downloading models
on a cold device), connect to MQTT, subscribe, and warm up. Edge-core watches
this file to tell the two apart.

A file rather than a bus, because the phases worth reporting most (``booting``,
``connecting_mqtt``) happen before any bus session exists. Edge-core mounts the
directory on tmpfs, so these writes never reach flash.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKER_STATUS_PATH_ENV = "CYBERWAVE_WORKER_STATUS_PATH"
DEFAULT_WORKER_STATUS_PATH = "/run/cyberwave/worker-status.json"

# Mirrored by ``WORKER_STARTUP_PHASES`` in edge-core and ``_PHASE_MESSAGES`` in
# the backend. Three separate deployables, so duplicated rather than imported.
PHASE_BOOTING = "booting"
PHASE_CONNECTING_MQTT = "connecting_mqtt"
PHASE_LOADING_MODELS = "loading_models"
PHASE_SUBSCRIBED = "subscribed"
PHASE_WARMING_UP = "warming_up"
PHASE_READY = "ready"

STATUS_RECORD_VERSION = 1
"""Bumped only for an incompatible shape; readers ignore unknown keys."""


class WorkerStatusReporter:
    """Writes the worker's startup phase to a file its supervisor polls.

    Best-effort throughout: a worker must never fail to start because nobody
    can be told about it.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        resolved = str(
            path
            or os.environ.get(WORKER_STATUS_PATH_ENV)
            or DEFAULT_WORKER_STATUS_PATH
        )
        self._path = Path(resolved)
        # The supervisor owns the directory and mounts it in. Creating it here
        # would produce a file only this container can see, leaving the
        # supervisor waiting on a path that never updates.
        self._enabled = self._path.parent.is_dir()
        self._boot_id = uuid.uuid4().hex
        self._started_at = time.time()
        self._seq = 0
        if not self._enabled:
            # Warning, not debug: the supervisor's only fallback is "running
            # means started", which silently reproduces the behaviour this
            # file exists to replace. A misconfigured mount has to be visible
            # in the worker's own logs, because nothing downstream reports it.
            logger.warning(
                "Worker status directory %s is not mounted; "
                "startup phases will not be reported",
                self._path.parent,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    def publish(
        self,
        phase: str,
        *,
        detail: str | None = None,
        models_total: int | None = None,
        models_ready: int | None = None,
    ) -> None:
        """Record *phase* as the current startup milestone.

        The model counts drive a progress bar through warm-up, which is
        otherwise the longest stretch with nothing moving on screen.
        """
        if not self._enabled:
            return
        self._seq += 1
        record: dict[str, Any] = {
            "version": STATUS_RECORD_VERSION,
            "phase": phase,
            "seq": self._seq,
            "pid": os.getpid(),
            "boot_id": self._boot_id,
            "started_at": self._started_at,
            "updated_at": time.time(),
        }
        if detail:
            record["detail"] = detail
        if models_total is not None:
            record["models_total"] = models_total
        if models_ready is not None:
            record["models_ready"] = models_ready
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        tmp_path: str | None = None
        try:
            # Same directory as the target, so ``os.replace`` stays within one
            # filesystem and a concurrent reader never sees a partial record.
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=f".{self._path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._path)
            tmp_path = None
        except Exception:
            # One warning naming the consequence, with the traceback kept at
            # debug. A write that failed once usually keeps failing (read-only
            # mount, wrong owner), so reporting is disabled here rather than
            # logging once per phase — which makes this the only notice an
            # operator gets.
            logger.warning(
                "Could not publish worker status to %s (phase=%s); startup "
                "phases will not be reported for this worker",
                self._path,
                record.get("phase"),
            )
            logger.debug("Worker status write failed", exc_info=True)
            self._enabled = False
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


_reporter: WorkerStatusReporter | None = None


def get_status_reporter() -> WorkerStatusReporter:
    """Process-wide reporter, so milestones published from the client and the
    runtime share one file, boot id and sequence."""
    global _reporter
    if _reporter is None:
        _reporter = WorkerStatusReporter()
    return _reporter


def reset_status_reporter() -> None:
    """Drop the singleton. For tests that redirect the status path."""
    global _reporter
    _reporter = None
