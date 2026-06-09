"""WorkerRuntime — process-level entrypoint that loads modules and dispatches hooks."""

from __future__ import annotations

import builtins
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cyberwave.data.exceptions import ChannelError
from cyberwave.data.keys import build_key, build_wildcard, parse_key
from cyberwave.exceptions import CyberwaveError
from cyberwave.workers.constants import MONITOR_STATS_KEY
from cyberwave.workers.context import HookContext
from cyberwave.workers.decode import decode_sample_payload, extract_wire_metadata
from cyberwave.workers.hooks import (
    SENSOR_BEARING_CHANNELS,
    WILDCARD_SENSOR,
    HookRegistration,
    HookRegistry,
    ScheduleRegistration,
    SynchronizedGroup,
)
from cyberwave.workers.loader import load_workers

if TYPE_CHECKING:
    from cyberwave.client import Cyberwave

logger = logging.getLogger(__name__)

DEFAULT_WORKERS_DIR = "/app/workers"
FALLBACK_WORKERS_DIR = os.path.expanduser("~/.cyberwave/workers")

MONITOR_PUBLISH_INTERVAL_S = 2.0
SCHEDULE_POLL_INTERVAL_S = 1.0
HOOK_ERROR_ALERT_COOLDOWN_S = 60.0


@dataclass(frozen=True)
class _ScheduleRegistration:
    module_name: str
    node_uuid: str
    cron: str
    timezone: str
    callback: Any
    callback_style: str
    options: dict[str, Any]


def _hook_min_interval_seconds(hook: HookRegistration) -> float:
    """Return the minimum dispatch interval enforced by ``hook.options['fps']``.

    ``@cw.on_frame(..., fps=N)`` stores ``fps`` in
    :attr:`HookRegistration.options`. The dispatcher consults this helper
    once per hook to derive a wall-clock floor (``1 / fps`` seconds)
    between callback invocations; samples arriving faster than that are
    counted as drops and skipped. ``0.0`` means "no throttling" and is
    returned for missing, non-numeric, boolean, or non-positive values
    so the gate stays a no-op when ``fps`` was never set.
    """
    fps = hook.options.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int | float) or fps <= 0:
        return 0.0
    return 1.0 / float(fps)


def _previous_cron_fire(cron: str, local_now: datetime) -> datetime | None:
    """Return the most recent cron "fire time" at or before *local_now*.

    Returns ``None`` when the expression is unparseable or croniter is
    missing. Uses ``second_at_beginning=True`` so 6-field expressions
    (``second minute hour dom month dow``) are interpreted with seconds
    as the leading field — matching the cloud's
    :mod:`src.lib.schedule_utils` and Quartz convention. This is what
    makes sub-minute schedules work on edge.

    ``croniter.get_prev`` is strict-less-than, so the candidate tick
    that lands exactly on ``local_now`` is treated as "future" and only
    picked up by the next poll. With the 1 s edge poll cadence that's
    at worst a 1 s delay, which is well inside the 5 s minimum sub-
    minute interval enforced by the cloud.
    """
    try:
        from croniter import CroniterBadCronError, croniter
    except ImportError:
        logger.warning(
            "Schedule trigger requires croniter. Install cyberwave[schedule] "
            "or add croniter to the worker image."
        )
        return None
    try:
        itr = croniter(cron, local_now, second_at_beginning=True)
        return itr.get_prev(datetime)
    except (CroniterBadCronError, ValueError):
        return None


def _cron_matches(cron: str, local_now: datetime) -> bool:
    """Back-compat shim: ``True`` when the most recent cron tick falls
    in the same wall-clock minute as ``local_now``.

    Kept for tests and external callers that still mock or call this
    helper directly. The runtime itself now uses
    :func:`_previous_cron_fire` so it can fire sub-minute schedules
    multiple times per minute.
    """
    prev_fire = _previous_cron_fire(cron, local_now)
    if prev_fire is None:
        return False
    return prev_fire.replace(second=0, microsecond=0) == local_now.replace(
        second=0, microsecond=0
    )


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
        self._worker_modules: list[object] = []
        self._schedule_registrations: list[_ScheduleRegistration] = []
        self._schedule_thread: threading.Thread | None = None
        self._schedule_lock = threading.Lock()
        # Per-registration last cron tick that successfully fired.
        # Keyed by ``"{module}:{node_uuid}"``. Used for instant-level
        # dedup so a single tick is dispatched at most once even when
        # the 1 s poll wakes multiple times between ticks; replaces the
        # old minute-string key that prevented sub-minute schedules
        # from firing more than once per minute.
        self._schedule_last_fire: dict[str, datetime] = {}
        self._schedule_running: set[str] = set()
        self._schedule_run_threads: list[threading.Thread] = []

        # Per-hook metrics: {hook_callback_name: {"frames": int, "drops": int}}
        self._hook_stats_lock = threading.Lock()
        self._hook_stats: dict[str, dict[str, int]] = {}
        self._stats_thread: threading.Thread | None = None

        # Per-hook error alert cooldown: {alert_key: last_alert_monotonic_time}
        self._hook_error_alert_times: dict[str, float] = {}
        self._hook_error_alert_lock = threading.Lock()

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

        self._worker_modules.clear()
        loaded = load_workers(
            workers_dir, cw_instance=self._cw, loaded_modules=self._worker_modules
        )
        self._schedule_registrations = self._collect_schedule_registrations()
        return loaded

    def start(self) -> None:
        """Warm up loaded models, then wire hooks to data-layer subscriptions.

        Models are warmed up before any hook can call ``predict()`` so
        backends that are not thread-safe (e.g. whisper.cpp) never see
        concurrent inference during worker startup.
        """
        self._warm_up_models()

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
            if group.twin_channels:
                twin_summary = ", ".join(
                    f"{lbl}={tu[:8]}..." for lbl, tu, _ch in group.twin_channels
                )
                logger.info(
                    "Activated cross-twin synchronized hook: "
                    "@cw.on_synchronized(%s) -> %s",
                    twin_summary,
                    group.callback.__name__,
                )
            else:
                logger.info(
                    "Activated synchronized hook: @cw.on_synchronized(%s) -> %s",
                    group.twin_uuid[:8] + "..." if group.twin_uuid else "<none>",
                    group.callback.__name__,
                )

        total = (
            len(self._registry.hooks)
            + len(self._registry.synchronized_groups)
            + len(self._schedule_registrations)
        )
        logger.info("Worker runtime started with %d hook(s)", total)

        self._start_schedule_dispatcher()
        self._start_stats_publisher()

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
        """Stop the runtime: flush pending work, unsubscribe hooks, disconnect data bus."""
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

        if self._schedule_thread is not None:
            self._schedule_thread.join(timeout=3.0)
            self._schedule_thread = None

        self._join_schedule_run_threads()

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=3.0)
            self._stats_thread = None

        try:
            self._cw.disconnect()
        except Exception:
            logger.debug("Error disconnecting Cyberwave client on shutdown", exc_info=True)

        if getattr(builtins, "cw", None) is self._cw:
            delattr(builtins, "cw")

    # ── internals ────────────────────────────────────────────────

    def _collect_schedule_registrations(self) -> list[_ScheduleRegistration]:
        registrations: list[_ScheduleRegistration] = []
        for module in self._worker_modules:
            callback = getattr(module, "run", None)
            if not callable(callback):
                continue
            manifest = getattr(module, "SCHEDULE_TRIGGERS", None)
            if not isinstance(manifest, list):
                continue
            module_name = getattr(module, "__name__", "<worker>")
            for entry in manifest:
                if not isinstance(entry, dict):
                    logger.warning(
                        "Ignoring invalid schedule manifest entry in %s: %r",
                        module_name,
                        entry,
                    )
                    continue
                node_uuid = str(entry.get("node_uuid") or "")
                cron = str(entry.get("cron") or "")
                timezone_name = str(entry.get("timezone") or "UTC")
                if not node_uuid or not cron:
                    logger.warning(
                        "Ignoring incomplete schedule manifest entry in %s: %r",
                        module_name,
                        entry,
                    )
                    continue
                registrations.append(
                    _ScheduleRegistration(
                        module_name=module_name,
                        node_uuid=node_uuid,
                        cron=cron,
                        timezone=timezone_name,
                        callback=callback,
                        callback_style="client",
                        options={},
                    )
                )
        for registration in self._registry.schedule_hooks:
            registrations.append(self._schedule_hook_to_runtime_registration(registration))
        return registrations

    @staticmethod
    def _schedule_hook_to_runtime_registration(
        registration: ScheduleRegistration,
    ) -> _ScheduleRegistration:
        callback_name = getattr(registration.callback, "__qualname__", "schedule")
        return _ScheduleRegistration(
            module_name=getattr(registration.callback, "__module__", "<worker>"),
            node_uuid=callback_name,
            cron=registration.cron,
            timezone=registration.timezone,
            callback=registration.callback,
            callback_style="context",
            options=dict(registration.options),
        )

    def _start_schedule_dispatcher(self) -> None:
        if not self._schedule_registrations or self._schedule_thread is not None:
            return

        def loop() -> None:
            while not self._stop_event.is_set():
                self._dispatch_due_schedules()
                self._stop_event.wait(SCHEDULE_POLL_INTERVAL_S)

        self._schedule_thread = threading.Thread(
            target=loop,
            name="cw-schedule-dispatcher",
            daemon=True,
        )
        self._schedule_thread.start()

    def _dispatch_due_schedules(self, now: datetime | None = None) -> None:
        now = now or datetime.now(tz=ZoneInfo("UTC"))
        for registration in self._schedule_registrations:
            try:
                local_now = now.astimezone(ZoneInfo(registration.timezone))
            except ZoneInfoNotFoundError:
                logger.warning(
                    "Skipping schedule %s from %s with unknown timezone %r",
                    registration.node_uuid,
                    registration.module_name,
                    registration.timezone,
                )
                continue
            registration_key = f"{registration.module_name}:{registration.node_uuid}"
            prev_fire = _previous_cron_fire(registration.cron, local_now)
            if prev_fire is None:
                continue
            with self._schedule_lock:
                last_fire = self._schedule_last_fire.get(registration_key)
                # Instant-level dedup. The first time we see a
                # registration we anchor at ``local_now`` so we only
                # fire for ticks strictly after worker startup — this
                # mirrors the cloud's ``reference = created_at`` seed
                # and avoids replaying a tick that already passed
                # before the worker came online. Subsequent polls fire
                # whenever a strictly newer cron instant has elapsed.
                if last_fire is None:
                    self._schedule_last_fire[registration_key] = local_now
                    continue
                if prev_fire <= last_fire:
                    continue
                if registration_key in self._schedule_running:
                    logger.warning(
                        "Skipping overlapping scheduled workflow run for %s",
                        registration_key,
                    )
                    continue
                self._schedule_last_fire[registration_key] = prev_fire
                self._schedule_running.add(registration_key)

            def run_registration(
                reg: _ScheduleRegistration = registration,
                key: str = registration_key,
                scheduled_at: datetime = local_now,
            ) -> None:
                try:
                    if reg.callback_style == "context":
                        reg.callback(
                            HookContext(
                                timestamp=scheduled_at.timestamp(),
                                channel="schedule",
                                twin_uuid="",
                                metadata={
                                    "cron": reg.cron,
                                    "timezone": reg.timezone,
                                    **reg.options,
                                },
                            )
                        )
                    else:
                        reg.callback(self._cw)
                except Exception as exc:
                    logger.exception(
                        "Scheduled workflow run failed for %s from %s",
                        reg.node_uuid,
                        reg.module_name,
                    )
                    schedule_twin = getattr(
                        self._cw.config, "twin_uuid", None
                    ) or ""
                    self._publish_hook_error_alert(
                        twin_uuid=schedule_twin,
                        hook_name=f"schedule:{reg.node_uuid}",
                        channel="schedule",
                        error=exc,
                    )
                finally:
                    with self._schedule_lock:
                        self._schedule_running.discard(key)

            t = threading.Thread(
                target=run_registration,
                name=f"cw-schedule-{registration.node_uuid[:8]}",
                daemon=True,
            )
            t.start()
            with self._schedule_lock:
                self._prune_schedule_run_threads_locked()
                self._schedule_run_threads.append(t)

    def _join_schedule_run_threads(self) -> None:
        with self._schedule_lock:
            threads = list(self._schedule_run_threads)
        for t in threads:
            t.join(timeout=5.0)
        with self._schedule_lock:
            self._prune_schedule_run_threads_locked()
            if self._schedule_run_threads:
                logger.warning(
                    "Stopping runtime with %d scheduled workflow run(s) still active",
                    len(self._schedule_run_threads),
                )

    def _prune_schedule_run_threads_locked(self) -> None:
        self._schedule_run_threads = [
            t for t in self._schedule_run_threads if t.is_alive()
        ]

    def _warm_up_models(self) -> None:
        """Run warm-up inference on all loaded models to eliminate cold-start latency."""
        models_mgr = getattr(self._cw, "models", None)
        if models_mgr is None:
            return
        loaded_models: dict[str, Any] = getattr(models_mgr, "_loaded", {})
        if not loaded_models:
            return
        logger.info(
            "Warming up %d loaded model(s) (startup inference only — not a wake-word trigger)...",
            len(loaded_models),
        )
        for model in loaded_models.values():
            warm_up_fn = getattr(model, "warm_up", None)
            if warm_up_fn is not None:
                try:
                    warm_up_fn()
                except Exception:
                    logger.warning(
                        "Model warm-up failed for %s", getattr(model, "name", "?"), exc_info=True,
                    )

    def _publish_hook_error_alert(
        self,
        *,
        twin_uuid: str,
        hook_name: str,
        channel: str,
        error: Exception,
    ) -> None:
        """Send a ``worker_runtime_error`` alert when a hook raises an exception.

        Rate-limited per hook: at most one alert every
        :data:`HOOK_ERROR_ALERT_COOLDOWN_S` seconds to avoid flooding the
        alert feed when a hook fails on every incoming sample.
        """
        if not twin_uuid:
            return

        alert_key = f"{twin_uuid}:{hook_name}:{channel}"
        now = time.monotonic()
        with self._hook_error_alert_lock:
            last = self._hook_error_alert_times.get(alert_key)
            if last is not None and now - last < HOOK_ERROR_ALERT_COOLDOWN_S:
                return
            self._hook_error_alert_times[alert_key] = now

        error_msg = str(error)[:500]
        try:
            self._cw.publish_alert(
                twin_uuid,
                f"Worker runtime error in {hook_name}",
                description=(
                    f"Hook '{hook_name}' on channel '{channel}' raised "
                    f"{type(error).__name__}: {error_msg}"
                ),
                alert_type="worker_runtime_error",
                severity="error",
                category="technical",
            )
        except Exception:
            logger.debug(
                "Could not send worker_runtime_error alert for hook %s: %s",
                hook_name,
                error_msg,
                exc_info=True,
            )

    def _build_context(
        self,
        hook: HookRegistration,
        sample: Any,
        *,
        wire_ts: float | None = None,
        wire_metadata: dict[str, Any] | None = None,
    ) -> HookContext:
        # Prefer the sensor name carried by the sample (wire key) so that
        # wildcard hooks see the *actual* sensor that published the frame
        # (``color_camera``, ``depth_camera``, …) instead of the hook's
        # abstract channel.  Fall back to the hook's declared sensor for
        # specific-sensor hooks; fall back to ``"default"`` for bare or
        # wildcard channels — matching both the pre-existing runtime
        # behavior and :class:`HookContext`'s dataclass default, so
        # non-sensor hooks (``on_imu``, ``on_joint_states``, …) keep
        # their ``ctx.sensor_name == "default"`` contract.
        sensor_name = self._sensor_name_from_sample(sample)
        if sensor_name is None:
            sensor_name = (
                hook.sensor_name
                if hook.sensor_name and hook.sensor_name != WILDCARD_SENSOR
                else "default"
            )
        ts = wire_ts if wire_ts is not None else getattr(sample, "timestamp", 0.0)
        # Merge wire header metadata (sample_rate_hz, channels, encoding, etc.)
        # with any metadata already on the Sample object.
        metadata = getattr(sample, "metadata", None) or {}
        if wire_metadata:
            merged = dict(wire_metadata)
            merged.update(metadata)
            metadata = merged
        return HookContext(
            timestamp=ts,
            channel=hook.channel,
            sensor_name=sensor_name,
            twin_uuid=hook.twin_uuid,
            metadata=metadata,
        )

    @staticmethod
    def _sensor_name_from_sample(sample: Any) -> str | None:
        """Extract the sensor name from ``sample.channel`` when it is a
        full wire key.

        Zenoh delivers samples with ``Sample.channel`` set to the actual
        published key (``cw/<twin>/data/frames/color_camera``); this helper
        parses that key and returns the sensor segment.  Returns ``None``
        if the channel is a hook-level name (``frames``, ``frames/front``)
        — those are not parseable as wire keys.
        """
        channel = getattr(sample, "channel", "") or ""
        if not channel or "/data/" not in channel:
            return None
        try:
            return parse_key(channel).sensor_name
        except ChannelError:
            return None

    def _subscribe_hook(self, hook: HookRegistration) -> None:
        """Create a data-layer subscription that dispatches to *hook*.

        Each hook gets a dedicated dispatch thread with a single-slot
        buffer.  When the data bus delivers a sample faster than the
        hook can process it, the newest sample silently replaces the
        previous one (drop-oldest).  This keeps the hook working on
        the most recent data without unbounded queue growth.

        ``mqtt`` hooks (registered via :meth:`HookRegistry.on_mqtt`)
        bypass the data bus entirely and subscribe directly through
        ``client.mqtt.subscribe`` because the corresponding twin
        subtopic is not bridged into Zenoh by default. This keeps the
        ``@cw.on_mqtt`` decorator usable on any twin subtopic without
        having to extend the bridge configuration first.
        """
        if hook.hook_type == "mqtt":
            self._subscribe_mqtt_hook(hook)
            return
        ready = threading.Event()
        slot: list[Any] = [None]
        hook_name = hook.callback.__name__
        drop_counter = [0]

        with self._hook_stats_lock:
            self._hook_stats[hook_name] = {"frames": 0, "drops": 0}

        def on_sample(sample: Any) -> None:
            if slot[0] is not None:
                drop_counter[0] += 1
            slot[0] = sample
            ready.set()

        fps_min_interval = _hook_min_interval_seconds(hook)

        def dispatch_loop() -> None:
            hint = hook.content_hint
            frames_processed = 0
            # Monotonic timestamp of the last dispatched sample. Used to
            # enforce ``fps=`` on frame hooks: samples that arrive faster
            # than ``1 / fps`` are counted as drops and skipped without
            # invoking the user callback. ``None`` means "never dispatched"
            # — the first sample after start always passes the gate.
            last_dispatched_at: float | None = None
            while not self._stop_event.is_set():
                if not ready.wait(timeout=1.0):
                    continue
                ready.clear()
                sample = slot[0]
                slot[0] = None
                if sample is None:
                    continue
                if fps_min_interval > 0.0:
                    now_monotonic = time.monotonic()
                    if (
                        last_dispatched_at is not None
                        and now_monotonic - last_dispatched_at < fps_min_interval
                    ):
                        drop_counter[0] += 1
                        with self._hook_stats_lock:
                            entry = self._hook_stats.get(hook_name)
                            if entry is not None:
                                entry["drops"] = drop_counter[0]
                        continue
                    last_dispatched_at = now_monotonic
                decoded_data, wire_ts = decode_sample_payload(sample, content_hint=hint)
                wire_meta = extract_wire_metadata(sample)
                ctx = self._build_context(hook, sample, wire_ts=wire_ts, wire_metadata=wire_meta)
                try:
                    hook.callback(decoded_data, ctx)
                    frames_processed += 1
                    with self._hook_stats_lock:
                        entry = self._hook_stats.get(hook_name)
                        if entry is not None:
                            entry["frames"] = frames_processed
                            entry["drops"] = drop_counter[0]
                    if frames_processed % 100 == 0:
                        logger.info(
                            "Hook %s: processed %d frames",
                            hook.callback.__name__,
                            frames_processed,
                        )
                except Exception as exc:
                    logger.exception(
                        "Error in hook %s for channel %s",
                        hook.callback.__name__,
                        hook.channel,
                    )
                    self._publish_hook_error_alert(
                        twin_uuid=hook.twin_uuid,
                        hook_name=hook.callback.__name__,
                        channel=hook.channel,
                        error=exc,
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

    def _subscribe_mqtt_hook(self, hook: HookRegistration) -> None:
        """Subscribe a ``@cw.on_mqtt`` hook directly via the MQTT client.

        Mirrors the alert-trigger flow in spirit (the SDK owns the
        subscription lifecycle so the worker module stays declarative)
        but uses the raw MQTT broker because user-defined twin
        subtopics are not part of the Zenoh-MQTT bridge inbound
        defaults. Hooks register interest, the runtime resolves the
        full prefixed topic, and dispatch invokes
        ``hook.callback(payload, topic, ctx)``.
        """
        subtopic = str(hook.options.get("subtopic") or "").strip()
        qos = int(hook.options.get("qos") or 0)
        if not subtopic:
            logger.warning(
                "Skipping MQTT hook '%s': missing subtopic option",
                hook.callback.__name__,
            )
            return
        if not hook.twin_uuid:
            logger.warning(
                "Skipping MQTT hook '%s': missing twin_uuid",
                hook.callback.__name__,
            )
            return

        try:
            mqtt_client = self._cw.mqtt
        except Exception:
            logger.exception(
                "Skipping MQTT hook '%s': MQTT client unavailable on Cyberwave instance",
                hook.callback.__name__,
            )
            return

        if not getattr(mqtt_client, "connected", False):
            try:
                mqtt_client.connect()
            except Exception:
                logger.exception(
                    "Failed to connect MQTT client for hook '%s'",
                    hook.callback.__name__,
                )
                return

        prefix = getattr(mqtt_client, "topic_prefix", "") or ""
        full_topic = f"{prefix}cyberwave/twin/{hook.twin_uuid}/{subtopic}"
        hook_name = hook.callback.__name__

        with self._hook_stats_lock:
            self._hook_stats[hook_name] = {"frames": 0, "drops": 0}

        def on_message(payload: Any) -> None:
            try:
                ctx = HookContext(
                    timestamp=time.time(),
                    channel=hook.channel,
                    twin_uuid=hook.twin_uuid,
                    metadata={"subtopic": subtopic, "topic": full_topic, "qos": qos},
                )
                hook.callback(payload, full_topic, ctx)
                with self._hook_stats_lock:
                    entry = self._hook_stats.get(hook_name)
                    if entry is not None:
                        entry["frames"] += 1
            except Exception as exc:
                logger.exception(
                    "Error in MQTT hook %s for topic %s",
                    hook_name,
                    full_topic,
                )
                self._publish_hook_error_alert(
                    twin_uuid=hook.twin_uuid,
                    hook_name=hook_name,
                    channel=f"mqtt/{subtopic}",
                    error=exc,
                )

        try:
            mqtt_client.subscribe(full_topic, on_message, qos=qos)
            logger.info(
                "Subscribed MQTT hook '%s' to topic %s (qos=%d)",
                hook_name,
                full_topic,
                qos,
            )
        except Exception:
            logger.exception(
                "Failed to subscribe MQTT hook '%s' to topic %s",
                hook_name,
                full_topic,
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

        Supports two modes:

        * **Single-twin** — ``group.twin_channels`` is empty; all
          channels are scoped to ``group.twin_uuid``.
        * **Cross-twin** — ``group.twin_channels`` contains
          ``(label, twin_uuid, channel)`` triples; each entry subscribes
          to a potentially different twin.
        """
        data_bus = self._get_data_bus()
        if data_bus is None:
            logger.warning(
                "Skipping synchronized hook '%s' — no data backend.",
                group.callback.__name__,
            )
            return

        labels = list(group.channels)
        tolerance_s: float = group.tolerance_ms / 1000.0
        latest_samples: dict[str, Any] = {}
        lock = threading.Lock()

        is_cross_twin = bool(group.twin_channels)
        twin_uuids: list[str] = (
            sorted({tc[1] for tc in group.twin_channels}) if is_cross_twin else []
        )

        def _check_and_fire() -> None:
            if len(latest_samples) < len(labels):
                return
            timestamps = [getattr(s, "timestamp", 0.0) for s in latest_samples.values()]
            if max(timestamps) - min(timestamps) <= tolerance_s:
                meta: dict[str, Any] = {"synchronized_channels": list(labels)}
                if is_cross_twin:
                    meta["twin_uuids"] = twin_uuids
                ctx = HookContext(
                    timestamp=max(timestamps),
                    channel=",".join(labels),
                    twin_uuid=group.twin_uuid,
                    metadata=meta,
                )
                try:
                    group.callback(dict(latest_samples), ctx)
                except Exception as exc:
                    logger.exception(
                        "Error in synchronized hook %s",
                        group.callback.__name__,
                    )
                    self._publish_hook_error_alert(
                        twin_uuid=group.twin_uuid,
                        hook_name=group.callback.__name__,
                        channel=",".join(labels),
                        error=exc,
                    )

        def _key_for_sync_channel(ch: str, twin_uuid: str) -> str:
            """Build a Zenoh key for a synchronized channel spec.

            ``"frames"`` → ``cw/<twin>/data/frames/**`` (wildcard — matches
            any sensor the driver publishes, keeps single-sensor twins
            working without the author knowing the sensor name).

            ``"frames/*"`` → same as above (explicit wildcard).

            ``"frames/front"`` → ``cw/<twin>/data/frames/front`` (exact).

            ``"joint_states"`` → ``cw/<twin>/data/joint_states`` (exact —
            channel is not sensor-bearing so no wildcard is added).
            """
            ch_parts = ch.split("/", 1)
            root = ch_parts[0]
            sensor = ch_parts[1] if len(ch_parts) > 1 else None
            wants_wildcard = sensor in (None, "*", "**") and root in SENSOR_BEARING_CHANNELS
            if wants_wildcard:
                return build_wildcard(twin_uuid, root, prefix=data_bus.key_prefix)
            return build_key(
                twin_uuid,
                root,
                sensor,
                prefix=data_bus.key_prefix,
            )

        if is_cross_twin:
            for label, tc_twin_uuid, tc_channel in group.twin_channels:

                def _make_on_sample(sample_label: str):  # noqa: E301
                    def on_sample(sample: Any) -> None:
                        with lock:
                            latest_samples[sample_label] = sample
                            _check_and_fire()

                    return on_sample

                try:
                    key = _key_for_sync_channel(tc_channel, tc_twin_uuid)
                    sub = data_bus.backend.subscribe(key, _make_on_sample(label))
                    self._subscriptions.append(sub)
                except Exception:
                    logger.exception(
                        "Failed to subscribe cross-twin channel '%s' (twin %s) "
                        "for hook '%s'",
                        tc_channel,
                        tc_twin_uuid,
                        group.callback.__name__,
                    )
        else:
            for ch in labels:

                def _make_on_sample(channel_name: str):  # noqa: E301
                    def on_sample(sample: Any) -> None:
                        with lock:
                            latest_samples[channel_name] = sample
                            _check_and_fire()

                    return on_sample

                try:
                    key = _key_for_sync_channel(ch, group.twin_uuid)
                    sub = data_bus.backend.subscribe(key, _make_on_sample(ch))
                    self._subscriptions.append(sub)
                except Exception:
                    logger.exception(
                        "Failed to subscribe synchronized channel '%s' for hook '%s'",
                        ch,
                        group.callback.__name__,
                    )

    # ── stats publisher ────────────────────────────────────────────

    def _start_stats_publisher(self) -> None:
        """Spawn a daemon thread that periodically publishes runtime stats."""
        data_bus = self._get_data_bus()
        if data_bus is None:
            logger.debug("No data bus — monitor stats publisher disabled.")
            return

        def _publish_loop() -> None:
            while not self._stop_event.is_set():
                self._stop_event.wait(MONITOR_PUBLISH_INTERVAL_S)
                if self._stop_event.is_set():
                    break
                try:
                    self._publish_stats_snapshot(data_bus)
                except Exception:
                    logger.debug("Failed to publish monitor stats", exc_info=True)

        t = threading.Thread(target=_publish_loop, name="cw-stats-publisher", daemon=True)
        t.start()
        self._stats_thread = t

    def _publish_stats_snapshot(self, data_bus: Any) -> None:
        """Collect and publish a single stats snapshot."""
        # Hook stats.
        with self._hook_stats_lock:
            hooks_snap = {k: dict(v) for k, v in self._hook_stats.items()}

        # Zenoh backend transport counters.
        transport_stats = data_bus.stats()

        # Model inference stats.
        model_stats = []
        models_mgr = getattr(self._cw, "models", None)
        loaded_models: dict[str, Any] = getattr(models_mgr, "_loaded", {}) if models_mgr else {}
        for model in loaded_models.values():
            fn = getattr(model, "inference_stats", None)
            if fn is not None:
                model_stats.append(fn())

        backend_connected = getattr(data_bus.backend, "is_connected", None)

        snapshot = {
            "ts": time.time(),
            "hooks": hooks_snap,
            "transport": transport_stats,
            "models": model_stats,
            "zenoh_connected": backend_connected if backend_connected is not None else True,
        }
        payload = json.dumps(snapshot, separators=(",", ":")).encode()
        data_bus.backend.publish(MONITOR_STATS_KEY, payload)

    def _build_key_for_hook(self, hook: HookRegistration, data_bus: Any) -> str:
        """Build the Zenoh key expression for a hook registration.

        Three cases:

        * **Wildcard sensor hook** (``@cw.on_frame(twin)`` → channel
          ``"frames"``, sensor ``"*"``): build a ``cw/<twin>/data/frames/**``
          wildcard so the hook matches whatever sensor name the driver
          actually publishes under.  Drivers take that name from the
          twin's asset (e.g. ``color_camera``, ``depth_camera``) — not
          from a hard-coded ``"default"``.
        * **Specific sensor hook** (``@cw.on_frame(twin, sensor="front")``
          → channel ``"frames/front"``): build the exact key
          ``cw/<twin>/data/frames/front``.
        * **Sensor-less hook** (``@cw.on_imu(twin)`` → channel ``"imu"``,
          sensor ``""``): build the exact key ``cw/<twin>/data/imu``.
        """
        if hook.is_wildcard_sensor:
            return build_wildcard(
                hook.twin_uuid, hook.channel, prefix=data_bus.key_prefix
            )

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
