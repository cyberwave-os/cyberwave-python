"""Camera drivers on :class:`~cyberwave.driver.BaseDriver`: open, stream, recover, publish.

:class:`CameraStreamMixin` runs every camera a driver has, each on its own:

- **Sources**: one :class:`~cyberwave.driver.sensors.camera_source.CameraSourceConfig`
  per camera (:attr:`~CameraStreamMixin.CAMERAS`), or a single camera from
  :attr:`~CameraStreamMixin.CAMERA_SOURCE` or twin metadata: OpenCV/V4L2 (USB,
  ``/dev/v4l/by-id``, RTSP/HTTP decoded by OpenCV), Intel RealSense (color +
  depth), or pre-encoded H.264 forwarded without re-encoding (a stream URL, or
  frames pushed with :meth:`~CameraStreamMixin.push_h264`). A driver with one
  camera falls back to an auto-detected camera when the configured one fails
  to open, unless the configuration names one physical device.
- **WebRTC**: one stream per camera, kept alive with the SDK's reconnect loop;
  ``start_video`` / ``stop_video`` apply to every camera, or to the one whose
  sensor the command names (:class:`~cyberwave.driver.sensors.video.VideoStreamMixin`).
- **Isolation**: a camera that fails to open is retried with backoff, raises
  its own alert and reports through :meth:`~CameraStreamMixin.on_camera_status`;
  the driver and the other cameras keep running.
- **Frame sync**: every frame carries video sync data when the
  ``cyberwave_video_sync`` extension is installed; set
  :attr:`~CameraStreamMixin.VIDEO_SYNC_REQUIRED` to refuse to start without it.
- **Local bus**: raw frames on ``frames/<sensor>`` and depth on
  ``depth/<sensor>`` through :class:`~cyberwave.driver.ZenohPublisherMixin`.
- **Frame pipeline** before encoding: the fail-closed frame filter
  (:mod:`~cyberwave.driver.sensors.frame_filter`), worker overlays
  (:mod:`~cyberwave.driver.sensors.overlays`), then :meth:`~CameraStreamMixin.on_camera_frame`.

:class:`BaseCameraDriver` wires the mixin into every lifecycle hook, so a
complete driver is::

    from cyberwave.driver import BaseCameraDriver

    class MyCameraDriver(BaseCameraDriver):
        def on_camera_frame(self, frame, camera):
            ...  # optional: edit the BGR frame in place before it is encoded

    if __name__ == "__main__":
        MyCameraDriver.main()

Twin metadata read by the frame pipeline (``CYBERWAVE_METADATA_*``):

- ``frame_filter_enabled``: substitute ``frames/filtered`` worker frames; black
  when stale. Needs the Zenoh data bus, an SDK-encoded source and a single
  camera (the channel names no sensor), or the driver refuses to start.
- ``frame_filter_freshness_ms``: max age of a substituted frame (default 200).

Environment: ``CYBERWAVE_DETECTION_OVERLAYS`` (default ``true``) draws raw
``detections/**`` boxes on the camera whose ``sensor`` the payload names (or
on the driver's only camera when it names none); ``CYBERWAVE_FRAME_ENCODING``
(``raw``/``jpeg``) and ``CYBERWAVE_FRAME_JPEG_QUALITY`` shape the Zenoh frame payload.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from ...sensor.gst_sources import redact_credentials
from ..base import BaseDriver
from ..cloud.alerts import (
    AlertCode,
    create_camera_failure_alert,
    create_config_error_alert,
)
from ..transports.zenoh_publisher import ZenohFramePublisher, ZenohPublisherMixin
from . import camera_source as _camera_source
from .camera_source import (
    CameraConfigurationError,
    CameraHardwareError,
    CameraSourceConfig,
    env_flag,
    env_int,
    list_cameras,
    load_twin_json,
    parse_camera_id,
    resolve_usb_serial,
)
from .frame_filter import FRESHNESS_MS, FrameFilter
from .overlays import (
    DetectionCache,
    OverlayCache,
    draw_detections,
    draw_overlay,
    parse_detections_payload,
    validate_overlay_payload,
)
from .video import VideoStreamMixin

logger = logging.getLogger(__name__)

#: Exit code for "the camera hardware is not there", distinct from config errors.
HARDWARE_CONNECTION_EXIT_CODE = 66

_FILTERED_FRAME_CHANNEL_FALLBACK = "frames/filtered"
_FRAME_OVERLAY_CHANNEL_FALLBACK = "frames/overlay"


class VideoSyncRequiredError(RuntimeError):
    """:attr:`CameraStreamMixin.VIDEO_SYNC_REQUIRED` is set but frame sync is unavailable."""


class CameraStatus(str, Enum):
    """State of one camera, reported through :meth:`CameraStreamMixin.on_camera_status`."""

    STARTING = "starting"
    STREAMING = "streaming"
    RETRYING = "retrying"
    STOPPED = "stopped"
    FAILED = "failed"


class CameraStream:
    """One camera of a driver: its source, WebRTC stream and state.

    Read it from :attr:`CameraStreamMixin.cameras`; the mixin owns its lifecycle.
    """

    def __init__(self, name: str, config: CameraSourceConfig, *, primary: bool) -> None:
        #: Sensor name: the WebRTC ``sensor`` and the ``frames/<name>`` channel.
        self.name = name
        #: The source; replaced by the detected device after a fallback.
        self.config = config
        #: The first camera; it alone takes the sensor-less frame filter and
        #: workflow overlays.
        self.primary = primary
        self.status = CameraStatus.STOPPED
        #: The active WebRTC streamer, or ``None``.
        self.streamer: Any = None
        self._frame_publisher: ZenohFramePublisher | None = None
        self._depth_publisher: ZenohFramePublisher | None = None
        self._detection_cache: DetectionCache | None = None
        self._overlay_cache: OverlayCache | None = None

    def push_h264(self, data: bytes, *, capture_time: float | None = None) -> bool:
        """See :meth:`CameraStreamMixin.push_h264`."""
        push = getattr(self.streamer, "push", None)
        if push is None:
            return False
        return bool(push(data, capture_time=capture_time))

    def __repr__(self) -> str:
        return f"CameraStream({self.name!r}, {self.config.describe()}, {self.status.value})"


def _data_channels() -> tuple[str, str]:
    try:
        from cyberwave.data import FILTERED_FRAME_CHANNEL, FRAME_OVERLAY_CHANNEL
    except ImportError:
        return _FILTERED_FRAME_CHANNEL_FALLBACK, _FRAME_OVERLAY_CHANNEL_FALLBACK
    return FILTERED_FRAME_CHANNEL, FRAME_OVERLAY_CHANNEL


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.0f", name, raw, default)
        return default


def _payload_sensor(payload: bytes) -> str | None:
    """The ``sensor`` a detections payload names, if any."""
    try:
        sensor = json.loads(payload).get("sensor")
    except (ValueError, TypeError, AttributeError):
        return None
    return str(sensor) if sensor else None


def _is_sync_unavailable(exc: BaseException) -> bool:
    """A pre-encoded stream refused to start without the frame-sync probe."""
    module = sys.modules.get("cyberwave.sensor.camera_gstreamer")
    error = getattr(module, "VideoSyncUnavailableError", None)
    return error is not None and isinstance(exc, error)


def _command_sensor(payload: Mapping[str, Any]) -> str | None:
    """The sensor a video command targets, if it names one."""
    data = payload.get("data")
    sensor = data.get("sensor") if isinstance(data, dict) else None
    sensor = sensor or payload.get("sensor")
    return str(sensor) if sensor else None


class CameraStreamMixin(VideoStreamMixin, ZenohPublisherMixin):
    """Open a driver's cameras, stream each over WebRTC, recover them, publish their frames.

    Compose it with any :class:`~cyberwave.driver.BaseDriver`, including
    :class:`~cyberwave.driver.ros2.BaseROS2Driver`, and it runs itself: once the
    driver is ACTIVE, :meth:`on_start_monitoring` configures the cameras,
    subscribes ``start_video`` / ``stop_video``, opens every stream and keeps
    them connected until shutdown, alongside the base class's own monitoring::

        class MyArmDriver(CameraStreamMixin, BaseDriver):
            CAMERAS = [
                CameraSourceConfig.from_url("rtsp://10.0.0.5/wrist", camera_name="wrist_camera"),
                CameraSourceConfig(camera_id="/dev/v4l/by-id/usb-base", camera_name="base_camera"),
                CameraSourceConfig.push(camera_name="overhead_camera"),
            ]

            async def on_connect_to_device(self) -> None:
                arm.on_h264_frame(lambda f: self.push_h264(f, camera="overhead_camera"))

    A single camera can be declared with :attr:`CAMERA_SOURCE`; without either,
    the one camera comes from twin metadata. Each camera fails and recovers on
    its own: see :meth:`on_camera_status` and :attr:`cameras`.

    :class:`BaseCameraDriver` calls the same steps earlier, from the lifecycle
    hooks, so a missing camera fails the driver before it reports ACTIVE:
    :meth:`configure_cameras`, :meth:`start_cameras_async`,
    :meth:`stop_cameras_async`. All are idempotent.
    """

    #: Refuse to start unless every streamed frame carries video sync data.
    VIDEO_SYNC_REQUIRED: bool = False
    #: Every camera of the driver, one :class:`CameraSourceConfig` each, each
    #: with a distinct ``camera_name`` (the twin's sensor id). Empty: one
    #: camera, from :attr:`CAMERA_SOURCE` or twin metadata.
    CAMERAS: Sequence[CameraSourceConfig] = ()
    #: The driver's single camera; ``None`` resolves it from twin metadata. See
    #: :meth:`CameraSourceConfig.push` and :meth:`CameraSourceConfig.from_url`.
    CAMERA_SOURCE: CameraSourceConfig | None = None
    #: Video toggles come from live (``tele``) and editor (``edit``) views and
    #: from scripts that set no ``source_type``: honour them all.
    VIDEO_COMMAND_SOURCE_TYPES = None
    #: Stopping the video must not mark a robot driver inactive.
    VIDEO_COMMANDS_DRIVE_LIFECYCLE = False
    #: Seconds to wait for a pre-encoded source's first keyframe.
    GST_KEYFRAME_TIMEOUT_S: float = 15.0
    #: First retry delay when a camera does not start (doubles, max 60 s).
    CAMERA_START_RETRY_S: float = 5.0

    _camera_streams: dict[str, CameraStream] | None = None
    _camera_state_ready: bool = False
    _camera_io_open: bool = False
    _camera_subscriptions: list[Any]
    _frame_filter: FrameFilter
    _timeref_stop: threading.Event
    _timeref_thread: threading.Thread | None

    # ── Hooks ────────────────────────────────────────────────────────────────

    def camera_sources(self) -> list[CameraSourceConfig]:
        """Every camera to open: :attr:`CAMERAS`, else :meth:`camera_source_config`."""
        if self.CAMERAS:
            return list(self.CAMERAS)
        return [self.camera_source_config()]

    def camera_source_config(self) -> CameraSourceConfig:
        """The single camera: :attr:`CAMERA_SOURCE`, else twin metadata and sensors."""
        declared = self.CAMERA_SOURCE
        if declared is not None:
            return declared
        data = load_twin_json()
        sensors = (data.get("capabilities") or {}).get("sensors")
        twin = getattr(self, "_twin", None)
        if not sensors and twin is not None:
            sensors = (getattr(twin, "capabilities", None) or {}).get("sensors")
        return CameraSourceConfig.from_sensors(sensors or [])

    def on_camera_frame(self, frame: np.ndarray, camera: str) -> None:
        """Edit *camera*'s *frame* (BGR, in place) after the filter and overlays, before encoding.

        Zenoh subscribers still receive the raw capture. Not called for
        pre-encoded sources. Default: no-op.
        """

    def on_camera_status(self, camera: str, status: CameraStatus, reason: str | None) -> None:
        """Called whenever *camera* changes state; *reason* explains ``retrying`` / ``failed``.

        A ``retrying`` camera recovers on its own; ``failed`` means one that
        cannot stream (e.g. frame sync required but unavailable). Default: no-op.
        """

    def frame_filter_enabled(self) -> bool:
        """Substitute worker-filtered frames (``metadata.frame_filter_enabled``)."""
        return env_flag("CYBERWAVE_METADATA_FRAME_FILTER_ENABLED")

    def detection_overlays_enabled(self) -> bool:
        """Draw raw ``detections/**`` boxes (``CYBERWAVE_DETECTION_OVERLAYS``, default on)."""
        return env_flag("CYBERWAVE_DETECTION_OVERLAYS", default=True)

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def cameras(self) -> Mapping[str, CameraStream]:
        """The driver's cameras by sensor name, in declaration order (empty before configuring)."""
        return MappingProxyType(self._camera_streams or {})

    @property
    def camera_config(self) -> CameraSourceConfig:
        """The first camera's source."""
        return self._primary_camera().config

    @property
    def camera_streamer(self) -> Any:
        """The first camera's WebRTC streamer, or ``None``."""
        streams = self._camera_streams
        if not streams:
            return None
        return next(iter(streams.values())).streamer

    def push_h264(
        self, data: bytes, *, capture_time: float | None = None, camera: str | None = None
    ) -> bool:
        """Feed pre-encoded H.264 to a push camera (``CameraSourceConfig.push``). Any thread, never blocks.

        For video that already arrives encoded inside the driver's process,
        e.g. the frames a robot's own WebRTC session delivers: they are muxed
        into the Cyberwave stream without decoding. *capture_time*
        (``time.time()`` seconds) is when the frame was captured, and lands in
        its video sync data. *camera* names the push camera; it may be omitted when
        the driver has only one. Returns ``False`` while that stream is not open.

        Raises:
            ValueError: *camera* is unknown, or omitted with several push cameras.
        """
        streams = self._camera_streams
        if not streams:
            return False
        if camera is not None:
            target = streams.get(camera)
            if target is None:
                raise ValueError(f"no camera named {camera!r}; cameras: {', '.join(streams)}")
        else:
            pushers = [s for s in streams.values() if s.config.gst_source == "appsrc"]
            if len(pushers) > 1:
                raise ValueError(
                    "several push cameras: name one with push_h264(..., camera=...): "
                    + ", ".join(s.name for s in pushers)
                )
            target = pushers[0] if pushers else next(iter(streams.values()))
        return target.push_h264(data, capture_time=capture_time)

    # ── Lifecycle helpers ────────────────────────────────────────────────────

    def configure_cameras(self) -> list[CameraSourceConfig]:
        """Resolve every camera source and check frame sync. Call from ``on_configure``."""
        self._init_camera_state()
        if self._camera_streams:
            return [stream.config for stream in self._camera_streams.values()]
        try:
            configs = self.camera_sources()
        except CameraHardwareError as exc:
            self._raise_camera_alert(None, "configured", str(exc))
            raise
        except CameraConfigurationError as exc:
            raise self._config_error("camera_source", str(exc)) from exc
        if not configs:
            raise self._config_error("cameras", "the driver declares no camera")
        names = self._camera_names(configs)
        if len(configs) > 1 and self.frame_filter_enabled():
            raise self._config_error(
                "frame_filter_enabled",
                "the frame filter channel carries one camera's frames, and this driver "
                "has several: the others would stream unfiltered",
            )
        for config in configs:
            self._check_camera_config(config)
        self._camera_streams = {
            name: CameraStream(name, config, primary=index == 0)
            for index, (name, config) in enumerate(zip(names, configs))
        }
        for stream in self._camera_streams.values():
            self._log_camera(stream)
        return configs

    async def start_cameras_async(self) -> None:
        """Open the data bus, frame pipeline and every camera's stream. Call from ``on_activate``.

        Cameras that fail are left to the retry in :meth:`on_start_monitoring`;
        when none starts, the first failure is raised.
        """
        self._init_camera_state()
        if not self._camera_streams:
            self.configure_cameras()
        self._open_camera_io()
        failures: list[tuple[CameraStream, Exception]] = []
        for stream in self._camera_streams_list():
            if stream.streamer is not None:
                continue
            try:
                await self._start_camera(stream)
            except CameraHardwareError as exc:
                failures.append((stream, exc))
        none_started = len(failures) == len(self._camera_streams_list())
        for stream, error in failures:
            status = CameraStatus.FAILED if none_started else CameraStatus.RETRYING
            self._set_camera_status(stream, status, str(error))
        if failures and none_started:
            raise failures[0][1]

    async def on_start_monitoring(self) -> None:
        """Run the cameras for the driver's ACTIVE phase, next to the base monitoring.

        Configures, subscribes video commands and opens what a lifecycle hook
        has not already opened (:class:`BaseCameraDriver`), keeps each camera
        connected (retrying the ones that fail) until shutdown, then tears down.
        """
        # The driver base's own monitoring (it waits for shutdown by default).
        await self._run_cameras(super().on_start_monitoring)  # type: ignore[misc]

    async def _run_cameras(self, alongside: Callable[[], Awaitable[None]]) -> None:
        if not self._camera_streams:
            self.configure_cameras()
        await self.register_video_commands_async()
        try:
            self._open_camera_io()
            await asyncio.gather(
                *(self._run_camera(stream) for stream in self._camera_streams_list()),
                alongside(),
            )
        finally:
            await self.stop_cameras_async()

    async def _run_camera(self, stream: CameraStream) -> None:
        """Open *stream* (retrying with backoff), then keep it connected until shutdown.

        Never raises: a camera must not end its driver or the other cameras.
        """
        shutdown: asyncio.Event = self._shutdown  # type: ignore[attr-defined]
        delay = self.CAMERA_START_RETRY_S
        try:
            while stream.streamer is None and not shutdown.is_set():
                try:
                    await self._start_camera(stream)
                except (CameraConfigurationError, VideoSyncRequiredError) as exc:
                    reason = redact_credentials(str(exc))
                    logger.error("Camera '%s' cannot stream: %s", stream.name, reason)
                    self._set_camera_status(stream, CameraStatus.FAILED, reason)
                    return
                except Exception as exc:
                    reason = redact_credentials(str(exc))
                    logger.warning(
                        "Camera '%s' did not start (%s); retrying in %.0fs", stream.name, reason, delay
                    )
                    self._set_camera_status(stream, CameraStatus.RETRYING, reason)
                    if not isinstance(exc, CameraHardwareError):  # those already raised the alert
                        self._raise_camera_alert(stream, stream.name, reason)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(shutdown.wait(), timeout=delay)
                    delay = min(delay * 2, 60.0)
            streamer = stream.streamer
            if streamer is None:
                return
            # start() connected already; arm the reconnect monitor for later drops.
            streamer._should_reconnect = streamer.auto_reconnect
            await streamer.run_with_auto_reconnect(stop_event=shutdown, subscribe_to_commands=False)
        except Exception as exc:
            logger.exception("Camera '%s' stopped unexpectedly", stream.name)
            self._set_camera_status(stream, CameraStatus.FAILED, str(exc))

    async def stop_cameras_async(self) -> None:
        """Stop every stream, subscription and publisher. Idempotent."""
        self._init_camera_state()
        for sub in self._camera_subscriptions:
            try:
                sub.close()
            except Exception:
                logger.debug("Subscription close failed", exc_info=True)
        self._camera_subscriptions = []
        self._close_zenoh_bus()
        self._camera_io_open = False
        self._streaming_active = False
        for stream in self._camera_streams_list():
            stream._frame_publisher = stream._depth_publisher = None
            streamer, stream.streamer = stream.streamer, None
            if streamer is not None:
                await self._close_streamer(streamer, stream.name)
            if stream.status is not CameraStatus.FAILED:
                self._set_camera_status(stream, CameraStatus.STOPPED, None)
        self._timeref_stop.set()
        if self._timeref_thread is not None:
            self._timeref_thread.join(timeout=2.0)
            self._timeref_thread = None

    # ── VideoStreamMixin hooks ───────────────────────────────────────────────

    def _video_command_needed(self, command: str) -> bool:
        # The streamers' handlers are idempotent, per camera.
        return True

    async def _on_start_video_async(self) -> None:
        payload = self._video_command_payload
        data = payload.get("data")
        recording = data.get("recording", True) if isinstance(data, dict) else None
        if recording is None:
            recording = payload.get("recording", True)

        async def _start(streamer: Any) -> None:
            streamer._should_record = recording
            await self._delegate(streamer._handle_start_command)

        await self._for_command_targets(_start)

    async def _on_stop_video_async(self) -> None:
        async def _stop(streamer: Any) -> None:
            await self._delegate(streamer._handle_stop_command)

        await self._for_command_targets(_stop)

    async def _for_command_targets(self, action: Callable[[Any], Awaitable[None]]) -> None:
        """Run *action* on the commanded camera, or every open one; raise if any failed."""
        sensor = _command_sensor(self._video_command_payload)
        streams = self._camera_streams or {}
        if sensor is not None and len(streams) > 1:
            if sensor not in streams:
                raise RuntimeError(f"no camera named {sensor!r}")
            targets = [streams[sensor]]
        else:
            targets = list(streams.values())
        open_targets = [stream for stream in targets if stream.streamer is not None]
        if not open_targets:
            raise RuntimeError("camera stream is not open")
        errors: list[str] = []
        for stream in open_targets:
            try:
                await action(stream.streamer)
            except Exception as exc:
                errors.append(f"{stream.name}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    # ── Internals ────────────────────────────────────────────────────────────

    def _init_camera_state(self) -> None:
        if self._camera_state_ready:
            return
        self._camera_state_ready = True
        self._camera_streams = {}
        self._camera_subscriptions = []
        self._frame_filter = FrameFilter(channel=None)
        self._timeref_stop = threading.Event()
        self._timeref_thread = None
        # Zenoh attributes exist even when the bus is never opened.
        self._zenoh_data_bus = None
        self._zenoh_frame_slot = None
        self._zenoh_thread = None
        self._zenoh_extra_publishers = []

    def _camera_streams_list(self) -> list[CameraStream]:
        return list((self._camera_streams or {}).values())

    def _primary_camera(self) -> CameraStream:
        streams = self._camera_streams
        if not streams:
            raise RuntimeError("cameras accessed before configure_cameras()")
        return next(iter(streams.values()))

    def _camera_names(self, configs: Sequence[CameraSourceConfig]) -> list[str]:
        if len(configs) == 1:
            return [self._sensor_name(configs[0]) or "default"]
        names = [config.camera_name for config in configs]
        if not all(names):
            raise self._config_error(
                "cameras", "with several cameras, each needs a camera_name (its sensor id)"
            )
        duplicates = sorted({name for name in names if names.count(name) > 1 and name})
        if duplicates:
            raise self._config_error(
                "cameras", f"camera names must be distinct; repeated: {', '.join(duplicates)}"
            )
        return [str(name) for name in names]

    def _check_camera_config(self, config: CameraSourceConfig) -> None:
        if config.is_pre_encoded and not _camera_source.gstreamer_available():
            raise self._config_error(
                "gst_source",
                "pre-encoded H.264 sources need GStreamer (PyGObject and the GStreamer "
                "libraries), which this environment lacks; use an image with GStreamer "
                "(e.g. the -gstreamer driver image)",
            )
        self._check_video_sync(config)
        if self.frame_filter_enabled() and config.is_pre_encoded:
            raise self._config_error(
                "frame_filter_enabled",
                "the frame filter replaces frames before encoding, but a pre-encoded "
                "H.264 source is forwarded as-is; use an OpenCV/RealSense source or "
                "disable the filter",
            )

    def _log_camera(self, stream: CameraStream) -> None:
        config = stream.config
        logger.info(
            "Camera '%s': %s (resolution=%s, fps=%d, pinned=%s)",
            stream.name,
            config.describe(),
            "x".join(map(str, config.resolution)) if config.resolution else "default",
            config.fps,
            config.pinned,
        )
        if isinstance(config.camera_id, str) and config.camera_id.startswith("/dev/"):
            if not os.path.exists(config.camera_id):
                logger.warning(
                    "Configured camera device '%s' does not exist inside the container; %s",
                    config.camera_id,
                    "the driver will not substitute another camera"
                    if config.pinned or len(self._camera_streams or {}) > 1
                    else "will try auto-detected cameras if it fails to open",
                )

    def _set_camera_status(
        self, stream: CameraStream, status: CameraStatus, reason: str | None
    ) -> None:
        if status is stream.status and status is not CameraStatus.RETRYING:
            return
        stream.status = status
        try:
            self.on_camera_status(stream.name, status, reason)
        except Exception:
            logger.exception("on_camera_status failed")

    @staticmethod
    async def _delegate(handler: Callable[..., Any]) -> None:
        """Run a streamer command handler; raise when it reports an error."""
        outcome: dict[str, str] = {}

        def _callback(status: str, message: str) -> None:
            outcome["status"], outcome["message"] = status, message

        await handler(_callback)
        if outcome.get("status") == "error":
            raise RuntimeError(outcome.get("message", "video command failed"))

    def _check_video_sync(self, config: CameraSourceConfig) -> None:
        sync_module: Any
        try:
            import cyberwave_video_sync as sync_module
        except ImportError:
            sync_module = None
        needed = "attach_gst_probe" if config.is_pre_encoded else "install"
        available = sync_module is not None and hasattr(sync_module, needed)
        disabled = os.getenv("CYBERWAVE_SYNC_ENABLED", "1") != "1" and not config.is_pre_encoded
        if available and not disabled:
            logger.info("Video sync data enabled (cyberwave_video_sync.%s)", needed)
            return
        reason = (
            "CYBERWAVE_SYNC_ENABLED disables it"
            if available
            else f"cyberwave_video_sync.{needed} is not installed"
        )
        if self.VIDEO_SYNC_REQUIRED:
            raise VideoSyncRequiredError(
                f"This driver streams every frame with video sync data, but {reason}."
            )
        logger.warning("Frames will stream without video sync data: %s", reason)

    def _open_camera_io(self) -> None:
        """Time reference, data bus and frame pipeline: once for all cameras."""
        if self._camera_io_open:
            return
        self._start_time_reference_keepalive()
        self._open_camera_bus()
        self._open_frame_pipeline()
        self._camera_io_open = True

    def _start_time_reference_keepalive(self) -> None:
        """Refresh the client's shared ``TimeReference`` at ~100 Hz.

        Camera tracks stamp frames from the client's ``TimeReference``, whose
        ``read()`` returns the last ``update()``. Robot drivers refresh it from
        their control loop; a camera-only driver has none, so without this every
        frame would carry the same frozen timestamp.
        """
        time_reference = getattr(self._cw, "time_reference", None)  # type: ignore[attr-defined]
        if time_reference is None or self._timeref_thread is not None:
            return
        self._timeref_stop.clear()

        def _run() -> None:
            while not self._timeref_stop.is_set():
                time_reference.update()
                self._timeref_stop.wait(0.01)

        self._timeref_thread = threading.Thread(target=_run, daemon=True, name="timeref-update")
        self._timeref_thread.start()

    def _sensor_name(self, config: CameraSourceConfig) -> str | None:
        if config.camera_name:
            return config.camera_name
        name = getattr(getattr(self, "_twin", None), "default_camera_name", None)
        return str(name) if name and name != "default" else None

    def _open_camera_bus(self) -> None:
        streams = self._camera_streams_list()
        primary = streams[0]
        required = False
        if os.getenv("CYBERWAVE_DATA_BACKEND"):
            try:
                from cyberwave.data.config import BackendConfig

                required = BackendConfig().publish_mode == "zenoh_only"
            except ImportError:
                logger.info("cyberwave.data is not available; Zenoh publishing disabled")
        if primary.name == "default":
            logger.warning(
                "Twin %s declares no camera sensor; publishing on the legacy "
                "'frames/default' channel. Declare a sensor in the asset schema so "
                "workers can pin to it.",
                self.twin_uuid,  # type: ignore[attr-defined]
            )
        encoding = os.getenv("CYBERWAVE_FRAME_ENCODING", "raw").lower()
        jpeg_quality = env_int("CYBERWAVE_FRAME_JPEG_QUALITY", 90)
        self._init_zenoh_bus(
            twin_uuid=self.twin_uuid,  # type: ignore[attr-defined]
            cw_client=self._cw,  # type: ignore[attr-defined]
            frame_channel=f"frames/{primary.name}",
            frame_metadata={"fps": primary.config.fps},
            frame_encoding=encoding,
            jpeg_quality=jpeg_quality,
            required=required,
        )
        if self._zenoh_data_bus is None:
            return
        for stream in streams:
            if not stream.primary:
                stream._frame_publisher = self.open_zenoh_frame_channel(
                    f"frames/{stream.name}",
                    metadata={"fps": stream.config.fps},
                    encoding=encoding,
                    jpeg_quality=jpeg_quality,
                )
            if stream.config.is_depth:
                # Depth stays raw uint16: JPEG would corrupt millimetre values.
                stream._depth_publisher = self.open_zenoh_frame_channel(
                    f"depth/{stream.config.depth_sensor_name}",
                    metadata={"fps": stream.config.depth_fps},
                )

    def _open_frame_pipeline(self) -> None:
        filtered_channel, overlay_channel = _data_channels()
        bus = self._zenoh_data_bus
        if self.frame_filter_enabled():
            if bus is None:
                # Without the bus no anonymised frame can arrive, and the raw
                # frame would be encoded: the opposite of the opt-in. Fail closed.
                raise self._config_error(
                    "frame_filter_enabled",
                    "the frame filter needs the Zenoh data bus to receive filtered "
                    "frames; set CYBERWAVE_DATA_BACKEND=zenoh (and install "
                    "cyberwave[zenoh]) or disable the filter",
                )
            self._frame_filter = FrameFilter(
                channel=filtered_channel,
                freshness_ms=_env_float(
                    "CYBERWAVE_METADATA_FRAME_FILTER_FRESHNESS_MS", FRESHNESS_MS
                ),
            )
            try:
                self._camera_subscriptions.append(
                    bus.subscribe(filtered_channel, self._frame_filter.store_processed)
                )
                logger.info(
                    "Frame filter subscribed to '%s' (freshness=%.0f ms)",
                    filtered_channel,
                    self._frame_filter.freshness_s * 1000.0,
                )
            except Exception:
                # Stay fail-closed: apply() keeps returning black frames.
                logger.error(
                    "Frame-filter subscription failed; streaming black frames (fail-closed)",
                    exc_info=True,
                )
        if bus is None:
            return
        drawable = [
            stream
            for stream in self._camera_streams_list()
            if not stream.config.is_depth and not stream.config.is_pre_encoded
        ]
        if drawable:
            self._subscribe_overlays(bus, overlay_channel, drawable)

    def _subscribe_overlays(
        self, bus: Any, overlay_channel: str, streams: list[CameraStream]
    ) -> None:
        by_name = {stream.name: stream for stream in streams}
        primary = next((stream for stream in streams if stream.primary), None)
        if self.detection_overlays_enabled():
            try:
                from cyberwave.data.keys import build_wildcard

                key = build_wildcard(
                    twin_uuid=self.twin_uuid,  # type: ignore[attr-defined]
                    channel="detections",
                    prefix=bus.key_prefix,
                )
                for stream in streams:
                    stream._detection_cache = DetectionCache()

                # The driver's one camera, if it has one: even a pre-encoded
                # second camera makes an unnamed box ambiguous.
                every = self._camera_streams_list()
                only = streams[0] if len(every) == 1 and streams[0] is every[0] else None

                def _on_detections(sample: Any) -> None:
                    # Keys are ``detections/<runtime>``; the camera is the
                    # payload's ``sensor``. A missing or unknown name (tracked
                    # results say "default") goes to the driver's only camera,
                    # or nowhere: never on a guessed one.
                    sensor = _payload_sensor(sample.payload)
                    target = by_name.get(sensor) if sensor else None
                    if target is None:
                        target = only
                    if target is None or target._detection_cache is None:
                        return
                    parsed = parse_detections_payload(sample.payload)
                    if parsed is not None:
                        target._detection_cache.update(*parsed)

                self._camera_subscriptions.append(
                    bus.backend.subscribe(key, _on_detections, policy="latest")
                )
                logger.info("Detection overlays subscribed to '%s'", key)
            except Exception:
                logger.warning("Detection overlay subscription failed", exc_info=True)
        if primary is None:
            return
        # The workflow overlay channel names no sensor: it annotates the first camera.
        try:
            overlay = OverlayCache()

            def _on_overlay(decoded: object) -> None:
                parsed = validate_overlay_payload(decoded)
                if parsed is not None:
                    overlay.update(parsed)

            self._camera_subscriptions.append(bus.subscribe(overlay_channel, _on_overlay))
            primary._overlay_cache = overlay
            logger.info("Workflow overlays subscribed to '%s'", overlay_channel)
        except Exception:
            logger.warning("Workflow overlay subscription failed", exc_info=True)

    def _publish_frame(self, stream: CameraStream, frame: np.ndarray) -> None:
        if stream.primary:
            self.zenoh_publish_frame(frame)
        elif stream._frame_publisher is not None:
            stream._frame_publisher.put(frame)

    def _publishes_frames(self, stream: CameraStream) -> bool:
        if stream.primary:
            return self._zenoh_frame_slot is not None
        return stream._frame_publisher is not None

    def _on_camera_frame(self, stream: CameraStream, frame: np.ndarray, _frame_count: int) -> None:
        """Frame callback for SDK-encoded sources: publish raw, then edit in place."""
        overlay = stream._overlay_cache.snapshot() if stream._overlay_cache is not None else None
        detections = (
            stream._detection_cache.snapshot()
            if stream._detection_cache is not None and overlay is None
            else None
        )
        filtered = stream.primary and self._frame_filter.enabled
        custom = type(self).on_camera_frame is not CameraStreamMixin.on_camera_frame
        if not (filtered or overlay or detections or custom):
            self._publish_frame(stream, frame)
            return
        # Workers on the bus always see the raw capture.
        if self._publishes_frames(stream):
            self._publish_frame(stream, frame.copy())
        if filtered:
            replacement = self._frame_filter.apply(frame)
            if replacement is not None:
                np.copyto(frame, replacement)
        if overlay is not None:
            draw_overlay(frame, overlay)
        elif detections is not None:
            boxes, det_w, det_h = detections
            draw_detections(frame, boxes, det_w, det_h)
        if custom:
            self.on_camera_frame(frame, stream.name)

    def _on_depth_frame(self, stream: CameraStream, depth: np.ndarray, _frame_count: int) -> None:
        if stream._depth_publisher is not None:
            stream._depth_publisher.put(depth)

    def _on_decoded_frame(self, stream: CameraStream, frame: np.ndarray, _frame_count: int) -> None:
        self._publish_frame(stream, frame)

    def _build_streamer(self, stream: CameraStream, config: CameraSourceConfig) -> Any:
        client = self._cw  # type: ignore[attr-defined]
        if config.is_pre_encoded:
            from cyberwave.sensor.camera_gstreamer import GStreamerH264CameraStreamer

            width, height = config.resolution or (1280, 720)
            return GStreamerH264CameraStreamer(
                client.mqtt,
                source=config.gst_source or "",
                width=width,
                height=height,
                fps=config.fps,
                require_sync=self.VIDEO_SYNC_REQUIRED,
                keyframe_timeout=self.GST_KEYFRAME_TIMEOUT_S,
                frame_callback=functools.partial(self._on_decoded_frame, stream),
                twin_uuid=self.twin_uuid,  # type: ignore[attr-defined]
                camera_name=stream.name,
                time_reference=getattr(client, "time_reference", None),
            )
        kwargs: dict[str, Any] = {
            "twin_uuid": self.twin_uuid,  # type: ignore[attr-defined]
            "camera_type": config.backend,
            "camera_id": config.camera_id,
            "fps": config.fps,
            "camera_name": stream.name,
            "frame_callback": functools.partial(self._on_camera_frame, stream),
        }
        if config.resolution is not None:
            kwargs["resolution"] = config.resolution
        if config.is_depth:
            kwargs.update(
                enable_depth=True,
                depth_fps=config.depth_fps,
                depth_callback=(
                    functools.partial(self._on_depth_frame, stream)
                    if stream._depth_publisher
                    else None
                ),
                serial_number=config.serial_number,
            )
        return client._build_video_streamer(**kwargs)

    async def _start_camera(self, stream: CameraStream) -> None:
        """Open *stream*; raises :class:`CameraHardwareError` when it cannot."""
        self._set_camera_status(stream, CameraStatus.STARTING, None)
        try:
            await self._start_stream_with_fallback(stream)
        except CameraHardwareError as exc:
            config = stream.config
            self._raise_camera_alert(
                stream,
                redact_credentials(str(config.serial_number or config.camera_id)),
                redact_credentials(str(exc)),
            )
            raise
        if stream.streamer is None:  # shutdown arrived while opening
            return
        self._resolve_camera_alert(stream)
        self._streaming_active = True
        self._set_camera_status(stream, CameraStatus.STREAMING, None)
        # Tell the UI the stream is live, so its toggle reflects it on load.
        self._publish_video_status("start_video", "ok")

    async def _open_stream(self, stream: CameraStream, config: CameraSourceConfig) -> None:
        streamer = self._build_streamer(stream, config)
        # Published before start(): a push-mode source only sees its first
        # keyframe once push_h264() can reach it.
        stream.streamer = streamer
        try:
            await streamer.start()
        except BaseException:
            if stream.streamer is streamer:
                stream.streamer = None
            # A failed start can leave the source running (a GStreamer pipeline's
            # supervisor reconnects until stopped): every retry would add one.
            await self._close_streamer(streamer, stream.name)
            raise

    @staticmethod
    async def _close_streamer(streamer: Any, name: str) -> None:
        """Stop *streamer* and its source. Never raises: cleanup must not hide the cause."""
        try:
            close = getattr(streamer, "close", None)
            if close is not None and asyncio.iscoroutinefunction(close):
                await close()
            else:
                await streamer.stop()
        except Exception:
            logger.warning("Failed while stopping camera '%s'", name, exc_info=True)

    async def _start_stream_with_fallback(self, stream: CameraStream) -> None:
        # A USB serial is looked up at each open: the camera may have moved port.
        config = resolve_usb_serial(stream.config)
        # With several cameras, "the first detected device" may be another one's.
        may_fall_back = not config.pinned and len(self._camera_streams or {}) == 1
        logger.info("Starting camera '%s' (%s)...", stream.name, config.describe())
        try:
            await self._open_stream(stream, config)
            return
        except (CameraConfigurationError, VideoSyncRequiredError):
            raise
        except Exception as exc:
            if self._shutdown.is_set():  # type: ignore[attr-defined]
                return
            if _is_sync_unavailable(exc):
                raise VideoSyncRequiredError(str(exc)) from exc
            if not may_fall_back:
                device = config.serial_number or config.gst_source or config.camera_id
                raise CameraHardwareError(
                    f"Camera '{stream.name}' "
                    f"('{redact_credentials(str(device))}') "
                    "could not be opened. Not falling back to auto-detect: it names a "
                    "specific camera, so a substitute would stream different footage "
                    "under this twin. Check the value and that no other driver holds "
                    "the device."
                ) from exc
            logger.exception(
                "Camera stream failed on '%s'; trying an auto-detected camera",
                redact_credentials(str(config.camera_id)),
            )
            first_error = exc
        cv2_cameras, realsense_cameras = await asyncio.to_thread(list_cameras)
        candidates = realsense_cameras if config.is_depth else cv2_cameras
        if not candidates:
            raise CameraHardwareError(
                "No camera hardware available for configured device "
                f"'{redact_credentials(str(config.camera_id))}'"
            ) from first_error
        fallback = (
            replace(config, serial_number=candidates[0])
            if config.is_depth
            else replace(config, camera_id=parse_camera_id(candidates[0]))
        )
        logger.info("Retrying with auto-detected camera '%s'", candidates[0])
        try:
            await self._open_stream(stream, fallback)
        except Exception as exc:
            raise CameraHardwareError(
                f"Camera hardware unavailable: could not start fallback '{candidates[0]}'"
            ) from exc
        stream.config = fallback

    def _alert_component(self, stream: CameraStream | None) -> str:
        # One camera keeps the v1 component; several get one alert each.
        if stream is None or len(self._camera_streams or {}) <= 1:
            return "camera"
        return f"camera/{stream.name}"

    def _raise_camera_alert(self, stream: CameraStream | None, camera_id: str, reason: str) -> None:
        try:
            self.raise_alert(  # type: ignore[attr-defined]
                create_camera_failure_alert(
                    component=self._alert_component(stream), camera_id=camera_id, reason=reason
                )
            )
        except Exception:
            logger.debug("Could not raise camera alert", exc_info=True)

    def _resolve_camera_alert(self, stream: CameraStream) -> None:
        try:
            self.resolve_alert(self._alert_component(stream), AlertCode.CAMERA_FAILURE)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Could not resolve camera alert", exc_info=True)

    def _config_error(self, field: str, reason: str) -> CameraConfigurationError:
        try:
            self.raise_alert(  # type: ignore[attr-defined]
                create_config_error_alert(
                    component="camera",
                    resource_type="Camera metadata",
                    resource_id=field,
                    details={"reason": reason},
                )
            )
        except Exception:
            logger.debug("Could not raise config alert", exc_info=True)
        return CameraConfigurationError(f"{field}: {reason}")


class BaseCameraDriver(CameraStreamMixin, BaseDriver):
    """A complete camera driver: subclass it, optionally override a hook, call :meth:`main`.

    The camera serves whatever asset the twin uses, so :attr:`REGISTRY_ID` may
    stay empty: it then comes from the twin JSON Edge Core mounts. The driver
    does not register a driver schema on the twin (:attr:`auto_register_interface`
    is ``False``), so it can replace another camera driver on an existing twin.
    """

    auto_register_interface = False
    #: The WebRTC streamer publishes ``edge_health`` with the stream's config.
    EDGE_HEALTH_ENABLED = False
    #: The video is the device: start/stop video moves the lifecycle, as in v1.
    VIDEO_COMMANDS_DRIVE_LIFECYCLE = True

    @property
    def registry_id(self) -> str:
        key = type(self)._class_registry_id()
        if key:
            return key
        asset = load_twin_json().get("asset") or {}
        return str(asset.get("registry_id") or os.getenv("CYBERWAVE_METADATA_ASSET_KEY") or "camera")

    @classmethod
    def create(cls) -> BaseCameraDriver:
        return cls()

    async def on_configure(self) -> None:
        self.configure_cameras()

    async def on_connect_to_device(self) -> None:
        """The cameras open in :meth:`on_activate`, with their streams (and fallback)."""

    async def on_register_callbacks(self) -> None:
        await self.register_video_commands_async()

    async def on_activate(self) -> None:
        await self.start_cameras_async()

    async def on_shutdown(self) -> None:
        await self.stop_cameras_async()

    @classmethod
    def main(cls) -> None:
        """Process entrypoint: run the driver and map failures to exit codes.

        Exits with :data:`HARDWARE_CONNECTION_EXIT_CODE` when no camera can
        open, ``1`` on any other failure.
        """
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S UTC",
        )
        logging.Formatter.converter = time.gmtime
        try:
            asyncio.run(cls.create_and_run_async())
        except CameraHardwareError as exc:
            logger.error("Exiting: camera hardware unavailable: %s", exc)
            sys.exit(HARDWARE_CONNECTION_EXIT_CODE)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("Camera driver failed")
            sys.exit(1)
