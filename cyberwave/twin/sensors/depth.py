"""Depth sensor handle — RGB-style frames (REST) + raw depth & point cloud (MQTT)."""

from __future__ import annotations

import base64
import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from ...consumers.callback_hub import StateSubscription
from ...consumers.mqtt_snapshot import FIRST_READ_TIMEOUT_S, MqttSensorStreamHandle
from ...exceptions import CyberwaveError, DepthTransportNotMQTTError
from ...utils.depth import depth_array_to_meters
from ..simulation_support import SimLevel, simulation_level
from .camera import TwinCameraHandle
from .pointcloud import PointCloudCapableMixin

if TYPE_CHECKING:
    from ..base import Twin

DEPTH_STREAM = "depth"

# Fallback depth range (metres) when the sensor capabilities omit min/max_depth.
_DEFAULT_MIN_DEPTH_M = 0.1
_DEFAULT_MAX_DEPTH_M = 5.0


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _decode_depth(payload: dict[str, Any], *, with_encoding: bool = False) -> Any:
    """``/depth`` payload -> numpy ``H×W`` depth (new or legacy format), or None.

    Parsed with the payload's declared ``dtype``; encoding metadata travels with
    the cached array so numeric reads cannot mix calibration between frames.
    """
    import numpy as np

    if payload.get("type") != "depth_data":
        return None
    data = payload.get("data")
    if isinstance(data, dict) and data.get("depth_binary"):
        b64 = data["depth_binary"]
        width = data.get("width")
        height = data.get("height")
        wire_dtype = str(data.get("dtype") or "uint16").lower()
    elif isinstance(data, str):
        b64 = data
        width = payload.get("width")
        height = payload.get("height")
        wire_dtype = str(payload.get("dtype") or "uint16").lower()
    else:
        return None
    try:
        np_dtype = np.dtype(wire_dtype)
    except TypeError:
        return None
    arr = np.frombuffer(base64.b64decode(b64), dtype=np_dtype)
    if width and height:
        arr = arr.reshape(int(height), int(width))
    if with_encoding:
        # Keep encoding and pixels in the same cached snapshot. A separate
        # mutable "last metadata" field can race the next frame's calibration.
        encoding = {**payload, **data} if isinstance(data, dict) else payload
        return arr, {
            key: encoding[key]
            for key in (
                "output_mode",
                "depth_scale",
                "min_depth",
                "max_depth",
                "depth_map_metric",
            )
            if key in encoding
        }
    return arr


_decode_depth_with_encoding = partial(_decode_depth, with_encoding=True)


class DepthSensorHandle(PointCloudCapableMixin, TwinCameraHandle):
    """Depth camera: visual frames (REST) + raw ``uint16`` depth and point cloud (MQTT)."""

    def __init__(self, twin: "Twin", sensor_id: str) -> None:
        TwinCameraHandle.__init__(self, twin, sensor_id=sensor_id)
        MqttSensorStreamHandle.__init__(self, twin, sensor_id)
        self._depth_using_rest: bool | None = None

    @simulation_level(SimLevel.MUJOCO)
    def get_frame(  # type: ignore[override]
        self,
        format: str = "numpy",
        *,
        raw: bool = False,
        source: str = "auto",
        timeout: float = FIRST_READ_TIMEOUT_S,
        sensor_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Latest depth frame.

        ``raw`` selects the value representation of ``format="numpy"`` frames:

        - ``raw=False`` (default): a ``float32 H×W`` array of **depth in metres**.
          MQTT reads honor the frame's declared scale/window; explicitly relative
          or unconfirmed normalized frames require ``raw=True``. Untagged
          uint16 retains the legacy millimetre default. The REST ``uint8`` grayscale
          representation carries no absolute unit and is mapped linearly onto the
          sensor's depth range (``min_depth`` / ``max_depth`` from
          ``twin.capabilities``, defaulting to ``0.1``–``5.0`` m). See
          :meth:`_to_meters`.
        - ``raw=True``: the underlying single-channel depth image — the native
          ``H×W`` array from MQTT (in its declared storage units), or the ``uint8 H×W`` grayscale image from
          REST (the RGB channels, which encode the same value, are collapsed to
          one).

        ``raw`` only affects ``format="numpy"``; ``"bytes"`` / ``"pil"`` / ``"path"``
        always return the underlying encoded frame.

        ``source="auto"`` (default): REST visual representation first
        (``twin.get_latest_frame``); on a miss, fall back to the MQTT ``/depth``
        raw ``uint16`` stream and cache the working transport. ``source="cloud"``
        / ``"mqtt"`` force one transport. From the MQTT raw path only
        ``format="numpy"`` and ``format="bytes"`` are valid.

        Note: the ``auto`` transport choice is cached for the lifetime of this
        handle — once REST misses and MQTT is selected, ``auto`` stays on MQTT
        (it does not re-probe REST). Pass ``source="cloud"`` explicitly to force
        the REST path again.
        """
        # Numpy conversion (grayscale collapse + metric mapping) only applies to
        # ``format="numpy"``; encoded formats (bytes/pil/path) pass through untouched.
        is_numpy = format == "numpy"

        normalized = str(source).strip().lower()
        if normalized == "auto":
            if self._depth_using_rest is not False:
                frame = super().get_frame(
                    format, source="cloud", sensor_id=sensor_id, **kwargs
                )
                if frame is not None:
                    self._depth_using_rest = True
                    return self._rest_numpy(frame, raw=raw) if is_numpy else frame
                self._depth_using_rest = False
            return self._get_depth_mqtt(format, timeout=timeout, raw=raw)
        if normalized == "mqtt":
            return self._get_depth_mqtt(format, timeout=timeout, raw=raw)
        # cloud / local / zenoh / remote_edge -> camera handle behavior
        frame = super().get_frame(format, source=source, sensor_id=sensor_id, **kwargs)
        if is_numpy and frame is not None:
            return self._rest_numpy(frame, raw=raw)
        return frame

    def _mqtt_numpy(self, arr: Any, *, raw: bool, encoding: dict[str, Any]) -> Any:
        """MQTT ``/depth`` numpy frame: raw ``uint16`` (``raw``) or ``float32`` metres."""
        import numpy as np

        a = np.asarray(arr)
        if raw:
            return a
        metric = encoding.get("depth_map_metric")
        if ("depth_map_metric" in encoding and metric is not True) or (
            encoding.get("output_mode") == "normalized_uint16" and metric is not True
        ):
            raise CyberwaveError(
                "Depth frame has no confirmed metric scale. Use raw=True for its "
                "display values, or select a metric depth source."
            )
        try:
            return depth_array_to_meters(
                a,
                **{
                    key: value
                    for key, value in encoding.items()
                    if key != "depth_map_metric"
                },
            )
        except (TypeError, ValueError) as exc:
            raise CyberwaveError(f"Invalid depth encoding: {exc}") from exc

    def _rest_numpy(self, frame: Any, *, raw: bool) -> Any:
        """REST numpy frame collapsed to grayscale: ``uint8`` (``raw``) or metres."""
        import numpy as np

        arr = np.asarray(frame)
        if arr.ndim == 3:
            # RGB channels encode the same grayscale value; average to reduce JPEG noise.
            gray = arr[..., :3].astype(np.float32).mean(axis=2)
            gray = np.clip(gray.round(), 0, 255).astype(np.uint8)
        else:
            gray = arr
        return gray if raw else self._to_meters(gray)

    def _to_meters(self, arr: Any) -> Any:
        """Map a raw depth frame onto **metres**.

        - MQTT float depth (``float16``/``float32``/``float64``): values are
          already absolute **metres**; ``0`` / non-finite = invalid (kept ``0.0``).
        - MQTT ``uint16`` depth: absolute **millimetres**; ``/1000`` → metres
          (``0`` = invalid stays ``0.0``).
        - REST grayscale (``uint8``): a normalized ``0–255`` image with no
          absolute unit — mapped linearly onto the sensor's metric range from
          ``twin.capabilities`` (default 0.1–5.0 m).
        """
        import numpy as np

        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.floating):
            depth = a.astype(np.float32)
            return np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(
                np.float32
            )
        if a.dtype == np.uint16:
            # Absolute millimetres → metres (0 = invalid preserved as 0.0).
            return (a.astype(np.float32) / 1000.0).astype(np.float32)
        # REST grayscale (uint8): normalized by full scale and mapped onto the
        # sensor's [min, max] metric range.
        full_scale = (
            float(np.iinfo(a.dtype).max) if np.issubdtype(a.dtype, np.integer) else 1.0
        )
        norm = a.astype(np.float32) / full_scale
        min_m, max_m = self._depth_range_m()
        return (min_m + norm * (max_m - min_m)).astype(np.float32)

    def _depth_range_m(self) -> tuple[float, float]:
        """``(min_depth, max_depth)`` in metres from capabilities; defaults to 0.1–5.0."""
        entry = self._capability_entry()
        params = entry.get("parameters") if isinstance(entry, dict) else None
        params = params if isinstance(params, dict) else {}
        min_m = _coerce_finite_float(params.get("min_depth"), _DEFAULT_MIN_DEPTH_M)
        max_m = _coerce_finite_float(params.get("max_depth"), _DEFAULT_MAX_DEPTH_M)
        if min_m < 0 or max_m <= min_m:
            return _DEFAULT_MIN_DEPTH_M, _DEFAULT_MAX_DEPTH_M
        return min_m, max_m

    def _capability_entry(self) -> dict[str, Any]:
        """Capability entry for this sensor (matched by id/name), else first depth sensor."""
        target = str(self.sensor_id) if self.sensor_id is not None else None
        first_depth: dict[str, Any] | None = None
        for entry in self._twin.capabilities.get("sensors", []):
            if not isinstance(entry, dict):
                continue
            if first_depth is None and str(entry.get("type")) == "depth":
                first_depth = entry
            ids = {str(entry.get("id") or ""), str(entry.get("name") or "")}
            if target is not None and target in ids:
                return entry
        return first_depth or {}

    def _get_depth_mqtt(self, format: str, *, timeout: float, raw: bool = True) -> Any:
        arr, encoding = self._get_latest(
            DEPTH_STREAM, _decode_depth_with_encoding, timeout=timeout
        )
        if format == "numpy":
            return self._mqtt_numpy(arr, raw=raw, encoding=encoding)
        if format == "bytes":
            return arr.tobytes()
        raise CyberwaveError(
            f"depth format {format!r} is only available from the REST visual path "
            "(source='cloud'); use format='numpy' or 'bytes' for raw MQTT depth"
        )

    def on_update(self, callback: Callable[[Any], None]) -> StateSubscription:
        """Run *callback* on every inbound raw MQTT depth frame (numpy ``uint16``).

        Raises :exc:`DepthTransportNotMQTTError` if the handle has already
        determined that this twin serves depth over REST (i.e. ``get_frame()``
        succeeded via the cloud path).  In that case there is no live MQTT depth
        stream to subscribe to — use a polling loop around ``get_frame()`` instead.
        """
        if self._depth_using_rest is True:
            raise DepthTransportNotMQTTError(
                "Depth frames for this twin are served over REST, not MQTT — "
                "on_update() requires a live MQTT /depth stream.\n"
                "Use a polling loop instead:\n\n"
                "    import time\n"
                "    while True:\n"
                "        frame = twin.camera['depth_camera'].get_frame(source='cloud')\n"
                "        process(frame)\n"
                "        time.sleep(interval)"
            )
        return self._register_callback(
            DEPTH_STREAM, _decode_depth_with_encoding, lambda frame: callback(frame[0])
        )

    def __dir__(self) -> list[str]:
        names = {n for n in object.__dir__(self) if not n.startswith("_")}
        names.update(
            ("get_frame", "get_frames", "on_update", "get_pointcloud", "on_pointcloud")
        )
        return sorted(names)

    def __repr__(self) -> str:
        sid = self.sensor_id or "default"
        return (
            f"{type(self).__name__}(sensor_id={sid!r}; "
            "get_frame, on_update, get_pointcloud, on_pointcloud)"
        )
