"""Lightweight health check for edge data streaming."""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


#: Type alias for the optional callback that augments each ``edge_health``
#: payload with host-level metrics (memory %, CPU temp, watchdog layers,
#: ...).  The callable must be cheap (executed every ``interval`` seconds
#: on the publisher thread) and must not raise — exceptions are caught and
#: logged at debug.  See ``cyberwave-edge-core`` for the canonical
#: implementation that wraps the host ``SystemResourceMonitor`` and
#: ``ProcessWatchdog``.
HostMetricsProvider = Callable[[], Dict[str, Any]]


#: Type alias for the optional callback that supplies dynamic per-stream
#: ``stream_config`` blocks on every heartbeat.  Use this rather than
#: :meth:`EdgeHealthCheck.register_stream_config` when the driver needs
#: to publish values that are only known post-negotiation (e.g. a cv2
#: camera whose configured ``fps=30`` may be clamped to 15 by the V4L2
#: stack).  Returning ``{stream_id: config}`` ovewrites whatever was
#: previously registered for that ``stream_id`` for the next heartbeat
#: only; calling :meth:`register_stream_config` directly remains the
#: right tool for truly static configs.
StreamConfigProvider = Callable[[], Mapping[str, Mapping[str, Any]]]


#: Recognised values for ``stream_config.kind``.  The discriminated union
#: lets a single ``edge_health`` heartbeat describe a multi-sensor edge
#: (e.g. a RealSense d455 publishing both ``rgb-0`` and ``depth-0``, or
#: a robot with a paired microphone and lidar) without needing a new
#: top-level field per sensor kind.  The frontend uses this discriminator
#: to render the right unit (``fps`` for camera, ``Hz`` for lidar,
#: ``kHz · channels`` for audio) without falling back to asset-spec
#: lookups.
KNOWN_STREAM_CONFIG_KINDS = frozenset({"camera", "audio", "lidar", "imu"})


#: Required fields per kind, enforced by :meth:`_validate_stream_config`.
#: Mirrors the frontend's ``StreamConfig`` discriminated union in
#: ``cyberwave-frontend/hooks/useMQTTEdgeHealth.ts``: when a driver
#: claims a kind, it must supply the fields the dashboard needs to
#: render meaningfully.  Without this validation we would happily ship
#: wire payloads that pass Pydantic ``extra="allow"`` on the backend
#: but render as empty rows on the frontend.
#:
#: Unknown kinds (forward-compat for future sensor types) bypass field
#: validation entirely — they only need to carry ``kind`` itself.
_STREAM_CONFIG_REQUIRED_FIELDS: Dict[str, frozenset] = {
    "camera": frozenset({"source", "resolution", "fps"}),
    "audio": frozenset({"source", "sample_rate_hz", "channels"}),
    "lidar": frozenset({"source", "scan_rate_hz"}),
    "imu": frozenset({"source", "rate_hz"}),
}


def _validate_stream_config(stream_id: str, config: Mapping[str, Any]) -> None:
    """Enforce the per-kind required-field contract.

    Raises ``ValueError`` when a known kind is missing required fields.
    Unknown kinds pass through so a future sensor type can ship from a
    driver before its required-field shape is canonicalised here.
    """
    kind = config.get("kind")
    required = _STREAM_CONFIG_REQUIRED_FIELDS.get(kind) if kind else None
    if required is None:
        return
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f"stream_config for stream_id={stream_id!r} kind={kind!r} "
            f"is missing required fields: {missing}.  See "
            f"_STREAM_CONFIG_REQUIRED_FIELDS in cyberwave/edge/health.py "
            f"for the per-kind contract."
        )


class EdgeHealthCheck:
    """Publishes edge health status on a background thread.

    Simple, self-contained health monitoring without requiring BaseEdgeNode.
    Designed for data streaming scripts that need basic health reporting.

    Example:
        health = EdgeHealthCheck(
            mqtt_client=cw.mqtt,
            twin_uuids=["camera_uuid", "robot_uuid"],
        )
        health.start()

        # Update stats when frames are sent
        health.update_frame_count()

        health.stop()

    The optional ``host_metrics_provider`` is invoked once per publish
    cycle.  Whatever dict it returns is merged into the payload before
    publishing — typical keys are ``host_memory_percent``,
    ``host_memory_available_mb``, ``cpu_temp_c``, ``consecutive_critical``
    and ``watchdog_layers``.  Provider exceptions are swallowed: the
    publisher must keep heartbeats flowing even when the host monitor
    is malfunctioning.
    """

    def __init__(
        self,
        mqtt_client: Any,
        twin_uuids: List[str],
        edge_id: Optional[str] = None,
        stale_timeout: int = 60,
        interval: int = 5,
        host_metrics_provider: Optional[HostMetricsProvider] = None,
        stream_config_provider: Optional[StreamConfigProvider] = None,
    ):
        """Initialize health publisher.

        Args:
            mqtt_client: MQTT client with publish() and topic_prefix
            twin_uuids: List of twin UUIDs to publish health to
            edge_id: Unique edge device ID (default: first twin UUID)
            stale_timeout: Seconds before stream is considered stale
            interval: Publish interval in seconds
            host_metrics_provider: Optional zero-arg callable returning a
                dict of host-level metrics merged into every payload.
                Only edge-core (running on the host) should pass one;
                driver containers see their container's ``/proc``, not
                the host's, so they must leave this ``None``.
            stream_config_provider: Optional zero-arg callable returning
                ``{stream_id: stream_config}`` on every heartbeat.  Use
                this when the config carries runtime-negotiated values
                (post-V4L2 ``actual_fps``, post-handshake codec, ...)
                that are only known after the streamer has started.
                Static configs known at construction time are simpler
                to wire via :meth:`register_stream_config`; the two
                interoperate (provider keys override registered keys
                on collision, so the dynamic value wins).
        """
        self.mqtt_client = mqtt_client
        self.twin_uuids = twin_uuids
        self.edge_id = edge_id or twin_uuids[0]  # Default to first twin UUID
        self.stale_timeout = stale_timeout
        self.interval = interval
        self.host_metrics_provider = host_metrics_provider
        self.stream_config_provider = stream_config_provider

        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Per-stream static config registered by drivers (see
        # ``register_stream_config``).  Read on every publish cycle, written
        # at driver startup; guarded by a lock so ``get_health_data()`` on
        # the publisher thread can never observe a half-mutated dict.
        self._stream_configs: Dict[str, Dict[str, Any]] = {}
        self._stream_configs_lock = threading.Lock()


    def register_stream_config(
        self, stream_id: str, config: Mapping[str, Any]
    ) -> None:
        """Attach a typed static config block to one entry of ``streams``.

        Drivers call this once at startup with the runtime-negotiated
        config for the stream they publish (camera resolution / fps /
        source, lidar scan rate, audio sample rate, ...).  The publisher
        merges the registered block into the matching ``streams[stream_id]``
        entry on every heartbeat under the ``stream_config`` key, and
        emits one ``streams[…]`` entry per registered ``stream_id`` —
        multi-stream publishers (RealSense RGB + depth from one driver,
        paired mic + camera) work by registering one config per stream
        they advertise.

        ``config`` MUST carry a ``kind`` discriminator (``"camera"``,
        ``"audio"``, ``"lidar"``, ``"imu"``) and the kind-specific
        required fields enumerated in ``_STREAM_CONFIG_REQUIRED_FIELDS``
        so the dashboard can render per-stream units without consulting
        the asset spec.  Unknown kinds are accepted (the schema is
        intentionally additive), but a debug log is emitted to flag
        accidental typos and required-field validation is skipped.

        Idempotent across redundant calls; the latest call for a given
        ``stream_id`` wins.  Safe to call from any thread.

        For configs whose values are only known after streamer startup
        (post-V4L2 ``actual_fps``, post-handshake codec, ...) prefer the
        ``stream_config_provider`` constructor argument so the publisher
        re-reads on every heartbeat instead of replaying a stale
        snapshot.

        Args:
            stream_id: The stream key inside ``streams[…]`` this config
                describes.  Pick a stable identifier (``"rgb-0"``,
                ``"depth-0"``, ``"mic-front"``, ...) that survives
                driver restarts; the dashboard groups history by it.
            config: A mapping carrying ``kind`` plus kind-specific
                required fields.  Credentials must already be masked by
                the caller; the publisher does not redact.

        Raises:
            ValueError: stream_id is empty, or a known-kind config is
                missing required fields.
            TypeError: config is not a mapping.
        """
        if not stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        kind = config.get("kind")
        if kind not in KNOWN_STREAM_CONFIG_KINDS:
            # Additive schema — accept unknown kinds rather than refusing
            # them, but flag the typo at debug level.  The frontend
            # ignores unknown kinds and falls back to the asset spec.
            logger.debug(
                "register_stream_config: unrecognised kind=%r for stream_id=%r "
                "(known: %s)",
                kind,
                stream_id,
                sorted(KNOWN_STREAM_CONFIG_KINDS),
            )
        else:
            _validate_stream_config(stream_id, config)
        with self._stream_configs_lock:
            self._stream_configs[stream_id] = dict(config)


    def unregister_stream_config(self, stream_id: str) -> None:
        """Remove a previously registered ``stream_config``.

        Symmetric with :meth:`register_stream_config`; used when a driver
        tears down a stream so the dashboard stops showing its config
        before the next heartbeat retracts the stream itself.  No-op if
        ``stream_id`` was never registered.
        """
        with self._stream_configs_lock:
            self._stream_configs.pop(stream_id, None)


    def update_frame_count(self):
        """Update frame count and timestamp (call this when sending frames)."""
        self.frame_count += 1
        self.last_frame_time = time.time()
    

    def start(self):
        """Start publishing health in background thread."""
        if self._thread and self._thread.is_alive():
            return  # Already running
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        logger.info(f"🟢 Health publisher started (interval={self.interval}s)")
    

    def stop(self):
        """Stop publishing health."""
        if not self._thread:
            return
        
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("🛑 Health publisher stopped")
    

    def get_health_data(self) -> Dict[str, Any]:
        """Build health data for current stream state.

        Output shape (additive over the legacy format):

        ``streams[id]`` carries per-stream live metrics (fps,
        frames_sent, is_stale, ...) plus an optional ``stream_config``
        block when the driver has registered one via
        :meth:`register_stream_config` or supplied one via the
        ``stream_config_provider`` callback.  ``stream_config`` is a
        discriminated union keyed by ``kind`` (camera / audio / lidar /
        imu) — see the module docstring.

        Multi-stream publishers register one config per stream id; the
        ``streams`` map then carries one entry per registered id.  The
        live counters (``frame_count``, ``last_frame_time``) are tracked
        at the ``EdgeHealthCheck`` instance level and shared across
        every entry — useful when paired streams (RGB + depth from one
        driver) advance at the same cadence, but inadequate for drivers
        with truly independent per-stream live metrics.  Such drivers
        should instantiate one ``EdgeHealthCheck`` per stream.

        The top-level ``camera_config`` field is a deprecated alias
        kept populated for one release for out-of-tree consumers.  When
        a camera-kind ``stream_config`` is registered, it is mirrored
        into ``camera_config``; otherwise the field stays ``None``.
        New code must read ``streams[id].stream_config`` instead.
        """
        now = time.time()
        uptime = now - self.start_time
        fps = self.frame_count / uptime if uptime > 0 else 0.0
        time_since_last = now - self.last_frame_time
        is_stale = time_since_last > self.stale_timeout

        effective_configs = self._collect_effective_stream_configs()
        stream_ids = sorted(effective_configs) if effective_configs else ["stream"]

        def _build_stream_entry(stream_id: str) -> Dict[str, Any]:
            entry: Dict[str, Any] = {
                "camera_id": stream_id,
                "connection_state": "disconnected" if is_stale else "connected",
                "ice_connection_state": "connected" if self.frame_count > 0 else "new",
                "frames_sent": self.frame_count,
                "last_frame_ts": self.last_frame_time,
                "fps": round(fps, 2),
                "uptime_seconds": round(uptime, 1),
                "restart_count": 0,
                "is_stale": is_stale,
                "is_healthy": not is_stale,
            }
            cfg = effective_configs.get(stream_id)
            if cfg is not None:
                entry["stream_config"] = cfg
            return entry

        streams = {sid: _build_stream_entry(sid) for sid in stream_ids}
        stream_count = len(streams)

        return {
            "streams": streams,
            "stream_count": stream_count,
            "healthy_streams": 0 if is_stale else stream_count,
            "camera_config": self._derive_legacy_camera_config(effective_configs),
        }


    def _collect_effective_stream_configs(self) -> Dict[str, Dict[str, Any]]:
        """Merge registered static configs with the dynamic provider's snapshot.

        Registered configs (set via :meth:`register_stream_config`) form
        the baseline.  When ``stream_config_provider`` is set, its
        return value is merged on top per stream id; on key collision
        within a stream the provider wins because the dynamic path is
        the runtime truth (e.g. ``actual_fps`` after V4L2 negotiation).

        A provider that raises, returns the wrong shape, or yields an
        invalid kind-specific payload is silently ignored for that
        cycle — the heartbeat must keep flowing even when a misbehaving
        provider would otherwise mark the edge offline.
        """
        with self._stream_configs_lock:
            base = {
                stream_id: dict(cfg)
                for stream_id, cfg in self._stream_configs.items()
            }

        if self.stream_config_provider is None:
            return base

        try:
            dynamic = self.stream_config_provider()
        except Exception as exc:
            logger.debug("Stream config provider raised: %s", exc)
            return base

        if not isinstance(dynamic, Mapping):
            return base

        for stream_id, cfg in dynamic.items():
            if not isinstance(stream_id, str) or not stream_id:
                continue
            if not isinstance(cfg, Mapping):
                continue
            kind = cfg.get("kind")
            if kind in KNOWN_STREAM_CONFIG_KINDS:
                try:
                    _validate_stream_config(stream_id, cfg)
                except ValueError as exc:
                    # A misbehaving provider must not break the cycle; we
                    # surface the issue at debug and fall back to whatever
                    # static config was registered for this stream id.
                    logger.debug("Stream config provider validation: %s", exc)
                    continue
            merged = dict(base.get(stream_id, {}))
            merged.update(cfg)
            base[stream_id] = merged
        return base


    @staticmethod
    def _derive_legacy_camera_config(
        registered_configs: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Mirror the first camera-kind ``stream_config`` into the legacy slot.

        Only the fields the frontend's ``EdgeCameraConfig`` interface
        declares are forwarded; extras are dropped so the shim does not
        accidentally leak new fields under an old field name.  Iteration
        is sorted by ``stream_id`` so multi-camera devices (e.g.
        RealSense ``depth-0`` + ``rgb-0``) yield a deterministic winner
        across heartbeats — important because the dashboard otherwise
        shows the camera identity flapping every 5 s.

        Returns ``None`` when no camera-kind config is registered, which
        preserves the historical wire shape for non-camera publishers
        (lidar bridges, microphone publishers) that never populated this
        field.
        """
        for stream_id in sorted(registered_configs):
            cfg = registered_configs[stream_id]
            if cfg.get("kind") != "camera":
                continue
            shim: Dict[str, Any] = {
                "camera_id": stream_id,
                "camera_type": cfg.get("camera_type", ""),
                "source": cfg.get("source", ""),
                "fps": cfg.get("fps", 0),
                "resolution": cfg.get("resolution", ""),
                "enabled": True,
            }
            return shim
        return None
    

    def _collect_host_metrics(self) -> Dict[str, Any]:
        """Call the optional ``host_metrics_provider`` and absorb failures.

        Returns an empty dict when no provider is registered or when the
        provider raises — the publisher MUST keep heartbeats flowing even
        when the host monitor is misbehaving, otherwise a single bug in
        the resource monitor would silently mark the edge offline.
        """
        if self.host_metrics_provider is None:
            return {}
        try:
            metrics = self.host_metrics_provider()
        except Exception as exc:
            logger.debug("Host metrics provider raised: %s", exc)
            return {}
        return metrics if isinstance(metrics, dict) else {}

    def _publish_loop(self):
        """Background thread loop for publishing health."""
        while not self._stop_event.is_set():
            try:
                # Build health data
                health_data = self.get_health_data()
                host_metrics = self._collect_host_metrics()

                # Build complete payload
                now = time.time()
                base_payload = {
                    "type": "edge_health",
                    "timestamp": now,
                    "edge_id": self.edge_id,
                    "uptime_seconds": round(now - self.start_time, 1),
                    **health_data,  # Include streams, stream_count, etc.
                    **host_metrics,  # Optional host_memory_percent, cpu_temp_c, ...
                }

                # Publish to each twin UUID
                prefix = getattr(self.mqtt_client, "topic_prefix", None) or ""
                for twin_uuid in self.twin_uuids:
                    if not twin_uuid:
                        continue
                    twin_uuid_str = str(twin_uuid)
                    payload = dict(base_payload, twin_uuid=twin_uuid_str)
                    topic = f"{prefix}cyberwave/twin/{twin_uuid_str}/edge_health"

                    try:
                        # Pass dict so MQTT client can add session_id (matches other messages)
                        self.mqtt_client.publish(topic, payload, qos=0)
                    except Exception as e:
                        logger.debug("Edge health publish failed: %s", e)

            except Exception as e:
                logger.warning(f"⚠️ Health publish error: {e}")

            # Wait for next interval
            self._stop_event.wait(self.interval)
