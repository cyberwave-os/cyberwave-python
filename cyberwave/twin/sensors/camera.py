"""Twin camera façade — unified :meth:`TwinCameraHandle.get_frame` only."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import threading
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from ..compat import time

from ...exceptions import CyberwaveError
from ...manifest.driver_config import (
    TWIN_CAMERA_PHOTO_TOPIC_SLUG,
    supported_mqtt_commands,
)
from ...mqtt.listen import MqttMessage

from .._helpers import (
    _decode_frame,
    _default_control_source_type,
)
from ..runtime_state import RUNTIME_MODE_SIMULATION, active_runtime_mode

if TYPE_CHECKING:
    from ..base import Twin

logger = logging.getLogger(__name__)

FrameSource = Literal["cloud", "local", "zenoh", "remote_edge"]
LocalCaptureMode = Literal["auto", "streamer", "device"]

_FRAME_SOURCE_ALIASES = {
    "edge": "remote_edge",
    "edge_photo": "remote_edge",
    "remote": "remote_edge",
}

# Advertised on handles and in twin.describe() for REPL / agent discovery.
CAMERA_HANDLE_PUBLIC_METHODS: tuple[str, ...] = (
    "get_frame",
    "get_frames",
    "stream",
    "read",
    "snapshot",
)


class TwinCameraHandle:
    """Camera operations on a twin. Use :meth:`get_frame` for all frame reads."""

    def __init__(self, twin: "Twin", *, sensor_id: Optional[str] = None):
        self._twin = twin
        self._sensor_id = sensor_id

    @property
    def sensor_id(self) -> Optional[str]:
        """Bound imaging sensor id (``None`` only on the default twin.camera handle)."""
        return self._sensor_id

    def __repr__(self) -> str:
        sid = self._sensor_id or "default"
        methods = ", ".join(CAMERA_HANDLE_PUBLIC_METHODS)
        return f"{type(self).__name__}(sensor_id={sid!r}; {methods})"

    def __dir__(self) -> List[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(CAMERA_HANDLE_PUBLIC_METHODS)
        return sorted(names)

    def get_frame(
        self,
        format: str = "bytes",
        *,
        path: Optional[str] = None,
        source: FrameSource | str = "cloud",
        sensor_id: Optional[str] = None,
        mock: bool = False,
        idx: int | str = 0,
        max_age_ms: float | None = None,
        zenoh_timeout_s: float = 3.0,
        edge_timeout_s: float = 5.0,
    ) -> Any | None:
        """Return one camera frame from the selected transport.

        Args:
            format: ``"bytes"``, ``"numpy"``, ``"pil"``, or ``"path"`` (JPEG on disk).
            path: Destination file when ``format="path"``; temp file when omitted.
            source: Frame transport:

                - ``"cloud"`` (default) — platform REST ``latest-frame`` (fail-soft).
                  Uses ``sim`` vs ``tele`` from :meth:`~cyberwave.client.Cyberwave.affect`.
                - ``"local"`` — active ``twin.stream()`` frame if running, else USB
                  camera at ``idx`` (fail-soft).
                - ``"zenoh"`` — ``cw.data`` ``frames`` via subscribe (fail-soft).
                - ``"remote_edge"`` — MQTT ``take_photo`` + photo topic
                  (raises :class:`~cyberwave.exceptions.CyberwaveError` on failure).
            sensor_id: Sensor id; defaults to first imaging sensor on the twin.
            mock: Deterministic mock frame (``source="cloud"`` only).
            idx: OpenCV device index when ``source="local"`` and no stream is active.
            max_age_ms: Zenoh staleness threshold (``source="zenoh"``).
            zenoh_timeout_s: Zenoh subscribe wait timeout (``source="zenoh"``).
            edge_timeout_s: MQTT wait for ``remote_edge`` photo response.

        Note:
            In ``simulation`` runtime mode (``config.runtime_mode``), only
            ``source="cloud"`` is available; other transports are rejected.

        Returns:
            Frame in the requested format, or ``None`` when unavailable (except
            ``remote_edge``, which raises on timeout / driver error). For
            ``format="path"``, returns an absolute ``.jpg`` path string.
        """
        if format != "path" and path is not None:
            raise ValueError("path is only valid when format='path'")

        normalized = _FRAME_SOURCE_ALIASES.get(
            str(source).strip().lower(), str(source).strip().lower()
        )
        if active_runtime_mode(self._twin.client) == RUNTIME_MODE_SIMULATION:
            if normalized != "cloud":
                raise ValueError(
                    f"Frame source {source!r} is not available in simulation runtime "
                    "mode; use source='cloud'."
                )

        resolved_sensor = self._twin._resolve_sensor_id(
            sensor_id if sensor_id is not None else self._sensor_id
        )
        if normalized == "cloud":
            frame = self._get_frame_cloud(
                format,
                sensor_id=resolved_sensor,
                mock=mock,
            )
        elif normalized == "local":
            frame = self._get_frame_local(format, idx=idx)
        elif normalized == "zenoh":
            frame = self._get_frame_zenoh(
                format,
                sensor_id=resolved_sensor,
                max_age_ms=max_age_ms,
                timeout_s=zenoh_timeout_s,
            )
        elif normalized == "remote_edge":
            frame = self._get_frame_remote_edge(format, timeout_s=edge_timeout_s)
        else:
            raise ValueError(
                f"Unknown frame source {source!r}; use 'cloud', 'local', 'zenoh', or 'remote_edge'."
            )

        if format != "path":
            return frame
        return self._resolve_frame_path(frame, path)

    def get_frames(
        self,
        count: int,
        *,
        interval_ms: int = 0,
        format: str = "path",
        directory: Optional[str] = None,
        path: Optional[str] = None,
        source: FrameSource | str = "cloud",
        sensor_id: Optional[str] = None,
        mock: bool = False,
        idx: int | str = 0,
        max_age_ms: float | None = None,
        zenoh_timeout_s: float = 3.0,
        edge_timeout_s: float = 5.0,
    ) -> List[Any] | str:
        """Grab multiple frames by calling :meth:`get_frame` in a loop.

        When ``format="path"`` (default), writes numbered JPEGs under ``directory``
        (or a new temp folder) and returns that folder path. Otherwise returns a
        list of frames in the requested format.

        Args:
            count: Number of frames (must be >= 1).
            interval_ms: Delay between grabs.
            format: Passed to each :meth:`get_frame` call.
            directory: Output folder when ``format="path"``; temp dir if omitted.
            path: Ignored unless ``count == 1`` and ``format="path"`` (single file).
            Remaining kwargs: forwarded to each :meth:`get_frame` call.
        """
        if count < 1:
            raise ValueError("count must be >= 1")

        frame_kwargs = {
            "source": source,
            "sensor_id": sensor_id,
            "mock": mock,
            "idx": idx,
            "max_age_ms": max_age_ms,
            "zenoh_timeout_s": zenoh_timeout_s,
            "edge_timeout_s": edge_timeout_s,
        }

        if format == "path":
            if count == 1 and path is not None:
                single = self.get_frame("path", path=path, **frame_kwargs)
                if single is None:
                    return None
                return single

            folder = directory
            if folder is None:
                folder = tempfile.mkdtemp(prefix="cyberwave_frames_")
            else:
                os.makedirs(folder, exist_ok=True)

            for i in range(count):
                dest = os.path.join(folder, f"frame_{i:04d}.jpg")
                saved = self.get_frame("path", path=dest, **frame_kwargs)
                if saved is None:
                    return None
                if i < count - 1 and interval_ms > 0:
                    time.sleep(interval_ms / 1000.0)
            return os.path.abspath(folder)

        frames: List[Any] = []
        for i in range(count):
            frame = self.get_frame(format, **frame_kwargs)
            frames.append(frame)
            if i < count - 1 and interval_ms > 0:
                time.sleep(interval_ms / 1000.0)
        return frames

    @staticmethod
    def _resolve_frame_path(frame: Any | None, path: Optional[str]) -> str | None:
        if frame is None:
            return None
        temp_path = os.path.abspath(str(frame))
        if path is None:
            return temp_path
        dest = os.path.abspath(path)
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(temp_path, dest)
        if temp_path != dest:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return dest

    def _get_frame_cloud(
        self,
        format: str,
        *,
        sensor_id: Optional[str],
        mock: bool,
    ) -> Any | None:
        try:
            resolved_source_type = self._twin.client.config.runtime_mode
            manager_kwargs: Dict[str, Any] = {
                "sensor_id": sensor_id,
                "mock": mock,
            }
            if resolved_source_type in {"simulation", "live"}:
                manager_kwargs["source_type"] = resolved_source_type

            jpeg = self._twin.client.twins.get_latest_frame(
                self._twin.uuid, **manager_kwargs
            )
            if jpeg is None:
                return None
            return _decode_frame(jpeg, format)
        except Exception:
            return None

    def _get_frame_local(
        self,
        format: str,
        *,
        idx: int | str,
    ) -> Any | None:
        frame = self._capture_local_array(idx=idx)
        if frame is None:
            return None
        return self._format_local_array(frame, format)

    def _get_frame_zenoh(
        self,
        format: str,
        *,
        sensor_id: Optional[str],
        max_age_ms: float | None,
        timeout_s: float,
    ) -> Any | None:
        try:
            client = self._twin.client
            sensor_name = self._zenoh_frame_sensor_name(sensor_id)
            fetch = getattr(client, "fetch_zenoh_frame", None)
            if fetch is None:
                return None
            payload = fetch(
                self._twin.uuid,
                sensor_name=sensor_name,
                timeout_s=timeout_s,
                max_age_ms=max_age_ms,
            )
            if payload is None:
                return None
            return self._format_zenoh_sample(payload, format)
        except Exception:
            logger.warning(
                "zenoh get_frame unavailable for twin %s (sensor=%s)",
                self._twin.uuid,
                self._zenoh_frame_sensor_name(sensor_id),
                exc_info=True,
            )
            return None

    @staticmethod
    def _format_zenoh_sample(sample: Any, format: str) -> Any | None:
        if format == "numpy" and isinstance(sample, (bytes, bytearray)):
            return _decode_frame(bytes(sample), "numpy")
        return TwinCameraHandle._format_local_array(sample, format)

    def _get_frame_remote_edge(self, format: str, *, timeout_s: float) -> Any:
        """MQTT take_photo via ``twin.commands`` + ``twin.listen`` on photo topic."""
        twin = self._twin
        schema = twin.driver.get_mqtt_schema()
        if "take_photo" not in supported_mqtt_commands(schema):
            raise CyberwaveError("twin cannot support take photo command")

        result_holder: Dict[str, Any] = {}
        event = threading.Event()

        def _on_photo(message: MqttMessage) -> None:
            payload = message.payload
            if isinstance(payload, dict):
                result_holder["data"] = payload
            else:
                result_holder["error"] = "edge photo payload must be a JSON object"
            event.set()

        with twin.listen(
            filters=["camera"],
            handlers={TWIN_CAMERA_PHOTO_TOPIC_SLUG: _on_photo},
        ):
            twin.commands.take_photo(
                source_type=_default_control_source_type(twin.client)
            )

            if not event.wait(timeout_s):
                raise CyberwaveError(
                    f"Timed out waiting for take_photo response after {timeout_s}s"
                )

            if "error" in result_holder:
                raise CyberwaveError(
                    f"Failed to parse edge photo response: {result_holder['error']}"
                )

            data = result_holder["data"]

            if data.get("status") == "error":
                raise CyberwaveError(data.get("message", "Edge returned an error"))

            if "image" not in data:
                raise CyberwaveError("Edge photo response missing 'image' field")

            jpeg_bytes = base64.b64decode(data["image"])
            return _decode_frame(jpeg_bytes, format)

    def _zenoh_frame_sensor_name(self, sensor_id: Optional[str]) -> str:
        """Zenoh ``frames/<segment>`` sensor — matches edge drivers (sensor id, not label)."""
        resolved = self._twin._resolve_sensor_id(
            sensor_id if sensor_id is not None else self._sensor_id
        )
        if not resolved:
            return "default"
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            aliases = {
                str(value)
                for key in ("id", "name", "role")
                if (value := entry.get(key))
            }
            if resolved in aliases:
                wire_id = entry.get("id")
                if wire_id:
                    return str(wire_id)
                break
        return str(resolved)

    @staticmethod
    def _format_local_array(frame: Any, format: str) -> Any | None:
        if format == "numpy":
            return frame
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return _decode_frame(encoded.tobytes(), format)

    def _capture_local_array(
        self,
        *,
        local_mode: str = "auto",
        idx: int | str = 0,
    ) -> Any | None:
        mode = local_mode.strip().lower()
        if mode in {"auto", "streamer"}:
            streamer = getattr(self._twin, "_camera_streamer", None)
            if streamer is not None:
                track = getattr(streamer, "streamer", None)
                current = getattr(track, "_current_frame", None) if track else None
                if current is not None:
                    return current
            if mode == "streamer":
                return None
        try:
            import cv2

            cap = cv2.VideoCapture(idx)
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None
        except Exception:
            return None

    def stream(self, fps: int = 30, camera_id: int | str = 0) -> None:
        if not hasattr(self._twin, "start_streaming"):
            raise CyberwaveError(
                "Video streaming requires a twin with camera sensors."
            )
        self._twin.start_streaming(fps=fps, camera_id=camera_id)

    def rotate(self, *args: Any, **kwargs: Any) -> None:
        """PR1 stub — real gimbal/camera rotate ships in a later PR."""
        warnings.warn(
            "twin.camera.rotate() is not implemented in PR1",
            UserWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "camera.rotate is not implemented in the mock-twin PR1 slice"
        )

    # --- Deprecated aliases (use twin.get_frame / twin.camera.get_frame) ---

    def latest_frame(self, *args: Any, **kwargs: Any) -> Any | None:
        warnings.warn(
            "camera.latest_frame() is deprecated; use twin.get_frame(source='cloud')",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_frame(*args, source="cloud", **kwargs)

    def capture(self, *args: Any, **kwargs: Any) -> Any | None:
        warnings.warn(
            "camera.capture() is deprecated; use twin.get_frame(source='local')",
            DeprecationWarning,
            stacklevel=2,
        )
        local_mode = kwargs.pop("source", "auto")
        if local_mode not in {"auto", "streamer", "device"}:
            local_mode = "auto"
        kwargs.pop("full_resolution", None)
        kwargs.pop("mock", None)
        fmt = args[0] if args else kwargs.pop("format", "bytes")
        idx = kwargs.pop("idx", 0)
        frame = self._capture_local_array(local_mode=str(local_mode), idx=idx)
        if frame is None:
            return None
        if fmt == "path":
            return self._resolve_frame_path(
                self._format_local_array(frame, "path"),
                kwargs.pop("path", None),
            )
        return self._format_local_array(frame, fmt)

    def read(self, format: str = "numpy", **kwargs: Any) -> Any:
        warnings.warn(
            "camera.read() is deprecated; use twin.get_frame(source='local')",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.get_frame(format, source="local", **kwargs)
        if result is None:
            raise CyberwaveError("No frame available from local capture")
        return result

    def snapshot(self, path: Optional[str] = None, **kwargs: Any) -> str:
        warnings.warn(
            "camera.snapshot() is deprecated; use twin.get_frame('path', path=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("source", "local")
        result = self.get_frame("path", path=path, **kwargs)
        if result is None:
            raise CyberwaveError("No frame available for snapshot")
        return result

    def edge_photo(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "camera.edge_photo() is deprecated; use twin.get_frame(source='remote_edge')",
            DeprecationWarning,
            stacklevel=2,
        )
        timeout = kwargs.pop("timeout", kwargs.pop("edge_timeout_s", 5.0))
        return self.get_frame(*args, source="remote_edge", edge_timeout_s=timeout, **kwargs)

    def edge_photos(
        self,
        count: int,
        interval_ms: int = 100,
        format: str = "bytes",
        *,
        timeout: float = 5.0,
    ) -> List[Any]:
        warnings.warn(
            "camera.edge_photos() is deprecated; loop twin.get_frame(source='remote_edge')",
            DeprecationWarning,
            stacklevel=2,
        )
        frames: List[Any] = []
        for i in range(count):
            frames.append(
                self.get_frame(format, source="remote_edge", edge_timeout_s=timeout)
            )
            if i < count - 1:
                time.sleep(interval_ms / 1000.0)
        return frames
