"""WorkerRuntime — process-level entrypoint that loads modules and dispatches hooks."""

from __future__ import annotations

import builtins
import logging
import os
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cyberwave.data.keys import build_key
from cyberwave.exceptions import CyberwaveError
from cyberwave.workers.context import HookContext
from cyberwave.workers.hooks import HookRegistration, HookRegistry, SynchronizedGroup
from cyberwave.workers.loader import load_workers

if TYPE_CHECKING:
    from cyberwave.client import Cyberwave

logger = logging.getLogger(__name__)

DEFAULT_WORKERS_DIR = "/app/workers"
FALLBACK_WORKERS_DIR = os.path.expanduser("~/.cyberwave/workers")


class WorkerRuntime:
    """Manages the lifecycle of a worker process.

    Relationship: **one edge device -> one worker container -> one runtime
    -> many modules**.
    """

    def __init__(self, cw_client: Cyberwave) -> None:
        self._cw = cw_client
        self._registry: HookRegistry = cw_client._hook_registry
        self._subscriptions: list[Any] = []
        self._dispatch_threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────

    def load(self, workers_dir: str | Path | None = None) -> int:
        """Load worker modules from disk."""
        if workers_dir is None:
            env_dir = os.environ.get("CYBERWAVE_WORKERS_DIR")
            if env_dir:
                workers_dir = env_dir
            elif Path(DEFAULT_WORKERS_DIR).is_dir():
                workers_dir = DEFAULT_WORKERS_DIR
            else:
                workers_dir = FALLBACK_WORKERS_DIR

        return load_workers(workers_dir, cw_instance=self._cw)

    def start(self) -> None:
        """Wire registered hooks to data-layer subscriptions."""
        for hook in self._registry.hooks:
            self._subscribe_hook(hook)
            logger.info(
                "Activated hook: @cw.on_%s(%s) -> %s",
                hook.hook_type,
                hook.twin_uuid[:8] + "..." if hook.twin_uuid else "<none>",
                hook.callback.__name__,
            )

        for group in self._registry.synchronized_groups:
            self._subscribe_synchronized_group(group)
            logger.info(
                "Activated synchronized hook: @cw.on_synchronized(%s) -> %s",
                group.twin_uuid[:8] + "..." if group.twin_uuid else "<none>",
                group.callback.__name__,
            )

        total = len(self._registry.hooks) + len(self._registry.synchronized_groups)
        logger.info("Worker runtime started with %d hook(s)", total)

    def run(self) -> None:
        """Block until :meth:`stop` is called or a signal is received."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # signal.signal() only works from the main thread; gracefully
            # degrade when called from a non-main thread (e.g. tests).
            pass
        logger.info("Worker runtime running. Press Ctrl+C to stop.")
        self._stop_event.wait()

    def stop(self) -> None:
        """Stop the runtime: unsubscribe all hooks and release resources."""
        logger.info("Stopping worker runtime...")
        self._stop_event.set()

        for sub in self._subscriptions:
            try:
                sub.close()
            except Exception:
                logger.exception("Error closing subscription")
        self._subscriptions.clear()

        for t in self._dispatch_threads:
            t.join(timeout=5.0)
        self._dispatch_threads.clear()

        if getattr(builtins, "cw", None) is self._cw:
            delattr(builtins, "cw")

    # ── internals ────────────────────────────────────────────────

    def _build_context(self, hook: HookRegistration, sample: Any) -> HookContext:
        parts = hook.channel.rsplit("/", 1)
        sensor_name = parts[1] if len(parts) > 1 else "default"
        return HookContext(
            timestamp=getattr(sample, "timestamp", 0.0),
            channel=hook.channel,
            sensor_name=sensor_name,
            twin_uuid=hook.twin_uuid,
            metadata=getattr(sample, "metadata", None) or {},
        )

    def _subscribe_hook(self, hook: HookRegistration) -> None:
        """Create a data-layer subscription that dispatches to *hook*.

        Each hook gets a dedicated dispatch thread with a single-slot
        buffer.  When the data bus delivers a sample faster than the
        hook can process it, the newest sample silently replaces the
        previous one (drop-oldest).  This keeps the hook working on
        the most recent data without unbounded queue growth.
        """
        ready = threading.Event()
        slot: list[Any] = [None]

        def on_sample(sample: Any) -> None:
            slot[0] = sample
            ready.set()

        def dispatch_loop() -> None:
            while not self._stop_event.is_set():
                if not ready.wait(timeout=1.0):
                    continue
                ready.clear()
                sample = slot[0]
                if sample is None:
                    continue
                ctx = self._build_context(hook, sample)
                try:
                    hook.callback(sample.payload, ctx)
                except Exception:
                    logger.exception(
                        "Error in hook %s for channel %s",
                        hook.callback.__name__,
                        hook.channel,
                    )

        data_bus = self._get_data_bus()
        if data_bus is not None:
            try:
                key = self._build_key_for_hook(hook, data_bus)
                sub = data_bus.backend.subscribe(key, on_sample)
                self._subscriptions.append(sub)
                t = threading.Thread(
                    target=dispatch_loop,
                    name=f"cw-hook-{hook.callback.__name__}",
                    daemon=True,
                )
                t.start()
                self._dispatch_threads.append(t)
            except Exception:
                logger.exception(
                    "Failed to subscribe hook '%s' to channel '%s'",
                    hook.callback.__name__,
                    hook.channel,
                )
        else:
            logger.warning(
                "Data backend not available. Hook '%s' on channel '%s' "
                "will not receive samples. Set CYBERWAVE_DATA_BACKEND to "
                "enable the data layer.",
                hook.callback.__name__,
                hook.channel,
            )

    # ── multi-channel synchronized dispatch ──────────────────────

    def _subscribe_synchronized_group(self, group: SynchronizedGroup) -> None:
        """Create subscriptions for a multi-channel synchronized hook.

        Subscribes to each channel independently via the data bus.
        Maintains a shared ``latest_samples`` buffer protected by a lock.
        On every incoming sample the buffer is updated and the alignment
        check runs: when all channels have a sample within
        ``tolerance_ms`` of each other the callback fires with a
        ``dict[str, Sample]`` snapshot and a :class:`HookContext`.
        """
        data_bus = self._get_data_bus()
        if data_bus is None:
            logger.warning(
                "Skipping synchronized hook '%s' — no data backend.",
                group.callback.__name__,
            )
            return

        channels = list(group.channels)
        tolerance_s: float = group.tolerance_ms / 1000.0
        latest_samples: dict[str, Any] = {}
        lock = threading.Lock()

        def _check_and_fire() -> None:
            if len(latest_samples) < len(channels):
                return
            timestamps = [getattr(s, "timestamp", 0.0) for s in latest_samples.values()]
            if max(timestamps) - min(timestamps) <= tolerance_s:
                ctx = HookContext(
                    timestamp=max(timestamps),
                    channel=",".join(channels),
                    twin_uuid=group.twin_uuid,
                    metadata={"synchronized_channels": list(channels)},
                )
                try:
                    group.callback(dict(latest_samples), ctx)
                except Exception:
                    logger.exception(
                        "Error in synchronized hook %s",
                        group.callback.__name__,
                    )

        for ch in channels:

            def _make_on_sample(channel_name: str):  # noqa: E301
                def on_sample(sample: Any) -> None:
                    with lock:
                        latest_samples[channel_name] = sample
                        _check_and_fire()

                return on_sample

            try:
                ch_parts = ch.split("/", 1)
                key = build_key(
                    group.twin_uuid,
                    ch_parts[0],
                    ch_parts[1] if len(ch_parts) > 1 else None,
                    prefix=data_bus.key_prefix,
                )
                sub = data_bus.backend.subscribe(key, _make_on_sample(ch))
                self._subscriptions.append(sub)
            except Exception:
                logger.exception(
                    "Failed to subscribe synchronized channel '%s' for hook '%s'",
                    ch,
                    group.callback.__name__,
                )

    def _build_key_for_hook(self, hook: HookRegistration, data_bus: Any) -> str:
        """Build the Zenoh key expression for a hook registration.

        Hook channels use the compound ``"base/sensor"`` format (e.g.
        ``"frames/front"``).  Split on the first ``"/"`` to separate the
        base channel from the optional sensor qualifier before calling
        :func:`~cyberwave.data.keys.build_key`.
        """
        parts = hook.channel.split("/", 1)
        base_ch = parts[0]
        sensor = parts[1] if len(parts) > 1 else None
        return build_key(hook.twin_uuid, base_ch, sensor, prefix=data_bus.key_prefix)

    def _get_data_bus(self) -> Any | None:
        """Return the data bus if available, ``None`` otherwise.

        Returns ``None`` when the data backend dependency is not
        installed or the backend cannot be constructed.  Configuration
        errors (e.g. missing ``CYBERWAVE_TWIN_UUID``) are surfaced at
        WARNING level so they are visible in normal operation.
        """
        try:
            return self._cw.data
        except ImportError:
            return None
        except CyberwaveError as exc:
            logger.warning("Data bus configuration error: %s", exc)
            return None
        except Exception:
            logger.debug("Could not initialise data bus", exc_info=True)
            return None

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info("Received signal %s, stopping...", signum)
        self.stop()
