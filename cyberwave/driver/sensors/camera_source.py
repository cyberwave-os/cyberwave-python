"""Which camera a driver opens, resolved from twin metadata.

Edge Core hands a camera driver its twin as JSON (``CYBERWAVE_TWIN_JSON_FILE``)
and the driver image's entrypoint flattens ``metadata.*`` into
``CYBERWAVE_METADATA_*`` environment variables. :meth:`CameraSourceConfig.from_env`
turns those into one :class:`CameraSourceConfig`: the backend to use
(OpenCV/V4L2, RealSense or a pre-encoded GStreamer pipeline), the device to
open, its resolution and rates, and the sensor names to publish under.

Twin metadata read (all optional):

=========================  ===================================================
``video_device``           Index, ``/dev/video*`` or ``/dev/v4l/by-*`` path, or a
                           stream URL (``rtsp://``, ``http://``). Default ``0``.
``serial_number``          Hardware serial. UVC cameras resolve it to their
                           ``/dev/v4l/by-id`` node; RealSense pins it.
``is_depth_camera``        Overrides the depth mode derived from the sensors.
``resolution``             ``"WxH"`` capture mode (e.g. ``"256x192"``).
``fps`` / ``depth_fps``    Color / depth frame rates. Default 30.
``gst_source``             ``gst-launch`` fragment that produces H.264 (e.g.
                           ``"v4l2src device=/dev/video2 ! video/x-h264"``).
                           Streams it without re-encoding.
``h264_passthrough``       ``true`` with a stream URL as ``video_device``
                           (``rtsp://``, ``rtp://``, ``udp://``, ``tcp://``): mux the
                           camera's own H.264 instead of decoding and re-encoding it.
=========================  ===================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from ...sensor.gst_sources import PUSH_SOURCE, h264_source_for_url, redact_credentials

logger = logging.getLogger(__name__)

V4L_BY_ID_DIR = "/dev/v4l/by-id"
V4L_BY_PATH_DIR = "/dev/v4l/by-path"

#: Sensor ``type`` values that mean "this twin has a depth stream". Assets spell
#: it ``depth`` in ``capabilities`` and ``depth_camera`` in ``universal_schema``.
DEPTH_SENSOR_TYPES = frozenset({"depth", "depth_camera"})

BACKEND_CV2 = "cv2"
BACKEND_REALSENSE = "realsense"
BACKEND_GSTREAMER = "gstreamer"

_TRUE = ("1", "true", "yes", "on")


class CameraHardwareError(RuntimeError):
    """The configured camera cannot be opened and no substitute is allowed."""


class CameraConfigurationError(RuntimeError):
    """The camera configuration cannot be honoured safely in this environment."""


def realsense_available() -> bool:
    """True when ``pyrealsense2`` imports cleanly.

    Catches every exception: a wheel without its native library raises
    ``OSError``, not ``ImportError``.
    """
    try:
        import pyrealsense2  # type: ignore[import-not-found,import-untyped]  # noqa: F401
    except Exception:
        return False
    return True


def gstreamer_available() -> bool:
    """True when PyGObject and the GStreamer 1.0 introspection data load."""
    try:
        import gi  # type: ignore[import-not-found,import-untyped,unused-ignore]

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401
    except Exception:
        return False
    return True


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``1/true/yes/on``)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE


def env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable, warning on bad values."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r (must be > 0); using %d", name, raw, default)
        return default
    return value


def parse_camera_id(video_device: str) -> int | str:
    """Numeric values are camera indices; paths and URLs pass through.

    Never call this on a hardware serial: serials are often all digits, and
    ``int()`` would strip leading zeros.
    """
    try:
        return int(video_device)
    except ValueError:
        return video_device


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse ``"WxH"`` into ``(width, height)``; ``None`` when unset or malformed.

    Some sensors (e.g. entry-level thermal cameras) do not serve the default
    640x480, and V4L2 then silently picks another mode. Pinning a native mode
    avoids that.
    """
    if not value:
        return None
    raw = value.strip().lower().replace(" ", "")
    width_s, sep, height_s = raw.partition("x")
    if not sep:
        logger.warning("Ignoring resolution %r: expected 'WxH' (e.g. '256x192')", value)
        return None
    try:
        width, height = int(width_s), int(height_s)
    except ValueError:
        logger.warning("Ignoring resolution %r: width/height must be integers", value)
        return None
    if width <= 0 or height <= 0:
        logger.warning("Ignoring resolution %r: width and height must be positive", value)
        return None
    return width, height


def sensors_declare_depth(sensors: list[Any]) -> bool:
    """True when any declared sensor is a depth stream."""
    return any(
        isinstance(sensor, dict) and sensor.get("type") in DEPTH_SENSOR_TYPES
        for sensor in sensors
    )


def is_stable_device_identifier(camera_id: int | str) -> bool:
    """True when *camera_id* names one physical device, so no substitute is allowed.

    Indices and ``/dev/videoN`` are positional (they move on replug);
    ``/dev/v4l/by-id``/``by-path`` entries, URLs and serials are not.
    """
    if not isinstance(camera_id, str):
        return False
    if camera_id.startswith((V4L_BY_ID_DIR, V4L_BY_PATH_DIR)):
        return True
    return not camera_id.startswith("/dev/")


def resolve_uvc_serial(serial: str, by_id_dir: str | None = None) -> str | None:
    """``/dev/v4l/by-id`` path of the UVC camera with *serial*, or ``None``.

    Matches the delimited serial field (``_<serial>-video-indexN``), so
    ``ABC123`` does not match ``ABC1234``. Prefers ``video-index0``, the
    capture node.
    """
    by_id_dir = by_id_dir or V4L_BY_ID_DIR
    try:
        entries = sorted(os.listdir(by_id_dir))
    except OSError:
        return None
    pattern = re.compile(rf"_{re.escape(serial)}-video-index\d+$")
    matches = [entry for entry in entries if pattern.search(entry)]
    for entry in matches:
        if entry.endswith("video-index0"):
            return os.path.join(by_id_dir, entry)
    return os.path.join(by_id_dir, matches[0]) if matches else None


def list_cameras() -> tuple[list[str], list[str]]:
    """``(opencv_indices, realsense_serials)`` currently present on the host."""
    cv2_cameras: list[str] = []
    realsense_cameras: list[str] = []
    try:
        import cv2

        for index in range(10):
            cap = cv2.VideoCapture(index)
            try:
                if cap.isOpened():
                    cv2_cameras.append(str(index))
            finally:
                cap.release()
    except Exception:
        logger.debug("Failed to enumerate OpenCV cameras", exc_info=True)
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found,import-untyped]

        for device in rs.context().devices:
            serial = device.get_info(rs.camera_info.serial_number)
            realsense_cameras.append(serial or device.get_info(rs.camera_info.name))
    except Exception:
        logger.debug("Failed to enumerate RealSense devices", exc_info=True)
    return cv2_cameras, realsense_cameras


def _metadata(name: str) -> str | None:
    value = os.getenv(f"CYBERWAVE_METADATA_{name}")
    return value.strip() if value is not None and value.strip() else None


def load_twin_json(path: str | None = None) -> dict[str, Any]:
    """Read the twin JSON Edge Core mounts (``CYBERWAVE_TWIN_JSON_FILE``); ``{}`` if absent."""
    path = path or os.getenv("CYBERWAVE_TWIN_JSON_FILE")
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read twin JSON file at %s", path)
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class CameraSourceConfig:
    """Everything needed to open one camera and name its streams.

    Build it with :meth:`usb` (a USB/V4L2 camera), :meth:`from_url` (a stream
    already on the network: RTSP, RTP over UDP, MPEG-TS, or a URL OpenCV
    decodes), :meth:`push` (H.264 the driver hands over) or :meth:`from_env`
    (twin metadata), and adjust with :func:`dataclasses.replace`. Set it as
    ``CAMERA_SOURCE`` or in ``CAMERAS`` on a
    :class:`~cyberwave.driver.CameraStreamMixin` driver.
    """

    backend: str = BACKEND_CV2
    camera_id: int | str = 0
    serial_number: str | None = None
    #: The configuration names one physical device: never fall back to another.
    pinned: bool = False
    resolution: tuple[int, int] | None = None
    fps: int = 30
    depth_fps: int = 30
    #: WebRTC sensor id and ``frames/<name>`` Zenoh channel; ``None`` uses the
    #: twin's first sensor.
    camera_name: str | None = None
    depth_sensor_name: str = "default"
    #: ``gst-launch`` fragment producing H.264 (``backend == "gstreamer"``).
    gst_source: str | None = None
    sensors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_depth(self) -> bool:
        return self.backend == BACKEND_REALSENSE

    @property
    def is_pre_encoded(self) -> bool:
        return self.backend == BACKEND_GSTREAMER

    def describe(self) -> str:
        if self.is_pre_encoded:
            return f"gstreamer[{redact_credentials(self.gst_source or '')}]"
        device = self.serial_number or self.camera_id
        return f"{self.backend}[{redact_credentials(str(device))}]"

    @classmethod
    def usb(
        cls,
        device: int | str = 0,
        camera_name: str | None = None,
        *,
        serial_number: str | None = None,
        resolution: tuple[int, int] | None = None,
        fps: int = 30,
    ) -> CameraSourceConfig:
        """A USB (UVC/V4L2) camera, captured and encoded by the SDK.

        *device* is an index, ``/dev/videoN``, or a ``/dev/v4l/by-id/...`` /
        ``by-path`` link. Indices and ``/dev/videoN`` move when cameras are
        replugged; prefer a ``by-id`` link or *serial_number*, the camera's UVC
        serial, which is looked up under ``/dev/v4l/by-id`` each time the
        camera opens (so an absent camera is retried, not a startup error).
        Either one pins the camera: the driver never substitutes another.
        *resolution* (``(width, height)``) picks a capture mode the camera
        supports.
        """
        camera_id = parse_camera_id(device) if isinstance(device, str) else device
        return cls(
            backend=BACKEND_CV2,
            camera_id=camera_id,
            serial_number=serial_number,
            pinned=serial_number is not None or is_stable_device_identifier(camera_id),
            resolution=resolution,
            camera_name=camera_name,
            fps=fps,
        )

    @classmethod
    def push(cls, camera_name: str | None = None, *, fps: int = 30) -> CameraSourceConfig:
        """H.264 the driver feeds with ``push_h264()``, forwarded without re-encoding.

        For video that already arrives encoded inside the driver, e.g. a
        robot's own WebRTC session. *fps* paces the WebRTC track: set it at or
        above the rate frames arrive, or frames queue up and are dropped.
        """
        return cls(
            backend=BACKEND_GSTREAMER,
            gst_source=PUSH_SOURCE,
            camera_name=camera_name,
            fps=fps,
            pinned=True,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        camera_name: str | None = None,
        *,
        fps: int = 30,
        decode: bool = False,
        resolution: tuple[int, int] | None = None,
    ) -> CameraSourceConfig:
        """A camera already streaming on the network.

        By default its H.264 is forwarded without decoding or re-encoding (needs
        GStreamer): ``rtsp(s)://`` (IP camera, NVR), ``rtp://host:port`` (RTP
        over UDP sent to this host; ``0.0.0.0`` listens on every interface, a
        multicast address joins the group) and ``udp://`` / ``tcp://`` (MPEG-TS);
        see :func:`~cyberwave.sensor.gst_sources.h264_source_for_url`.

        With ``decode=True`` OpenCV opens the URL and the SDK re-encodes the
        frames: for streams that are not H.264 (MJPEG over ``http://``, H.265),
        or to draw overlays and run the frame filter. Works without GStreamer.
        *resolution* applies to decoded streams only.

        Raises:
            ValueError: unsupported URL for passthrough.
        """
        if decode:
            return cls(
                backend=BACKEND_CV2,
                camera_id=url.strip(),
                camera_name=camera_name,
                fps=fps,
                resolution=resolution,
                pinned=True,
            )
        return cls(
            backend=BACKEND_GSTREAMER,
            gst_source=h264_source_for_url(url),
            camera_name=camera_name,
            fps=fps,
            pinned=True,
        )

    @classmethod
    def from_env(cls, twin_json: dict[str, Any] | None = None) -> CameraSourceConfig:
        """Resolve the camera from ``CYBERWAVE_METADATA_*`` and the twin JSON sensors."""
        data = load_twin_json() if twin_json is None else twin_json
        sensors = (data.get("capabilities") or {}).get("sensors") or []
        return cls.from_sensors(sensors)

    @classmethod
    def from_sensors(cls, sensors: list[Any]) -> CameraSourceConfig:
        """Resolve the camera for a twin declaring *sensors*, reading metadata from env."""
        sensors = [s for s in sensors if isinstance(s, dict)]
        is_depth = sensors_declare_depth(sensors)
        depth_override = _metadata("IS_DEPTH_CAMERA")
        if depth_override is not None:
            is_depth = depth_override.lower() in _TRUE

        camera_name = None
        if sensors and sensors[0].get("id") is not None:
            camera_name = str(sensors[0]["id"])
        depth_sensor_name = next(
            (str(s["id"]) for s in sensors if s.get("type") == "depth" and s.get("id") is not None),
            "default",
        )

        base = cls(
            resolution=parse_resolution(_metadata("RESOLUTION")),
            fps=env_int("CYBERWAVE_METADATA_FPS", 30),
            depth_fps=env_int("CYBERWAVE_METADATA_DEPTH_FPS", 30),
            camera_name=camera_name,
            depth_sensor_name=depth_sensor_name,
            sensors=sensors,
        )
        raw_device = _metadata("VIDEO_DEVICE")
        serial = _metadata("SERIAL_NUMBER")

        gst_source = _metadata("GST_SOURCE")
        if (
            gst_source is None
            and raw_device
            and "://" in raw_device
            and env_flag("CYBERWAVE_METADATA_H264_PASSTHROUGH")
        ):
            try:
                gst_source = h264_source_for_url(raw_device)
            except ValueError as exc:
                raise CameraConfigurationError(f"h264_passthrough: {exc}") from exc
        if gst_source is not None:
            return replace(base, backend=BACKEND_GSTREAMER, gst_source=gst_source, pinned=True)

        if is_depth and not realsense_available():
            if depth_override is not None:
                raise CameraConfigurationError(
                    "metadata.is_depth_camera is true but librealsense (pyrealsense2) "
                    "is not available here; use an image with RealSense support "
                    "(e.g. the -realsense driver image)"
                )
            # The twin is depth-shaped but this image has no RealSense backend:
            # stream the color node over V4L2 rather than failing to start.
            logger.warning(
                "Twin declares a depth sensor but pyrealsense2 is not available; "
                "streaming color only through OpenCV/V4L2"
            )
            is_depth = False
            serial = None

        if is_depth:
            # librealsense addresses the device by serial; camera_id is inert.
            camera_id = parse_camera_id(raw_device) if raw_device else 0
            return replace(
                base,
                backend=BACKEND_REALSENSE,
                camera_id=camera_id,
                serial_number=serial,
                pinned=serial is not None,
            )

        camera_id, pinned = _resolve_uvc_source(raw_device, serial)
        return replace(base, backend=BACKEND_CV2, camera_id=camera_id, pinned=pinned)


def _resolve_uvc_source(raw_device: str | None, serial: str | None) -> tuple[int | str, bool]:
    """``(camera_id, pinned)`` for an OpenCV/V4L2 camera."""
    # A stream URL outranks the serial: on macOS Edge Core injects an MJPEG
    # bridge URL as video_device while a serial recorded on Linux rides along.
    is_stream_url = raw_device is not None and "://" in raw_device
    if serial and not is_stream_url:
        if not os.path.isdir(V4L_BY_ID_DIR):
            # Older Edge Core runs the container --privileged without the
            # /dev/v4l mount: device nodes exist but udev's symlinks do not.
            logger.warning(
                "%s is not visible in this container; ignoring serial %s and using video_device",
                V4L_BY_ID_DIR,
                serial,
            )
        else:
            resolved = resolve_uvc_serial(serial)
            if resolved is None:
                raise CameraHardwareError(
                    f"No UVC camera with serial '{serial}' under {V4L_BY_ID_DIR}. Check the "
                    "serial, and that the device is attached and visible to this container."
                )
            logger.info("Resolved serial %s -> %s", serial, resolved)
            return resolved, True
    if raw_device is None:
        return 0, False
    camera_id = parse_camera_id(raw_device)
    return camera_id, is_stable_device_identifier(camera_id)


def resolve_usb_serial(config: CameraSourceConfig) -> CameraSourceConfig:
    """*config* with a USB camera's ``serial_number`` resolved to its ``/dev/v4l/by-id`` node.

    Unchanged for any other source. Called each time the camera opens, so a
    camera plugged in later, or into another port, is found.

    Raises:
        CameraHardwareError: no attached camera has that serial.
    """
    if config.backend != BACKEND_CV2 or not config.serial_number:
        return config
    path = resolve_uvc_serial(config.serial_number)
    if path is None:
        raise CameraHardwareError(
            f"No USB camera with serial '{config.serial_number}' under {V4L_BY_ID_DIR}. "
            "Check the serial, and that the camera is attached and /dev/v4l is visible "
            "to this container."
        )
    return replace(config, camera_id=path, pinned=True)
