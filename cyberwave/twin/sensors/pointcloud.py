"""Point-cloud read path (MQTT /pointcloud) shared by LiDAR and depth handles."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Callable

from ...consumers.callback_hub import StateSubscription
from ...consumers.mqtt_snapshot import FIRST_READ_TIMEOUT_S, MqttSensorStreamHandle
from ...exceptions import CyberwaveError
from ..simulation_support import SimLevel, simulation_level

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

POINTCLOUD_STREAM = "pointcloud"

# Guard so the "no point_stride, guessing" advisory is emitted at most once per
# guessed stride value instead of on every streamed frame (point clouds arrive
# continuously, which would otherwise flood the console/logs).
_warned_missing_stride: set[int] = set()


def _infer_pointcloud_stride(arr: "np.ndarray") -> int:
    """Best-effort stride (3 or 6) guess for a flat float32 buffer without ``point_stride``.

    Heuristic: if the buffer is divisible by 6 *and* every value in the candidate
    RGB columns (indices 3,4,5 of each 6-float group) lies within [0, 1], treat the
    data as XYZRGB (stride 6).  Otherwise fall back to stride 3.

    This is inherently ambiguous and can misclassify in both directions:

    * A stride-3 XYZ cloud with an even point count whose coordinates all lie in
      [0, 1] (normalized / close-range / unit-cube data) looks like stride 6 and
      loses half its points.
    * A stride-6 XYZRGB cloud whose colors are 0-255 (not normalized) looks like
      stride 3 and doubles its point count.

    Senders should always include ``point_stride`` in the payload; this guess is a
    last resort and callers get a warning whenever it is used.
    """
    import numpy as np

    if arr.size % 6 == 0:
        pts = arr.reshape(-1, 6)
        rgb = pts[:, 3:]
        if bool(np.all((rgb >= 0.0) & (rgb <= 1.0))):
            return 6
    return 3


def _decode_pointcloud(payload: dict[str, Any]) -> Any:
    """``{"type":"pointcloud","data":<b64 float32>}`` -> numpy ``N×3`` XYZ (or None).

    Handles both XYZ-only (stride 3) and XYZRGB (stride 6) payloads.

    * If the sender includes ``point_stride`` in the JSON, that value is used directly.
    * Otherwise the stride is *guessed* (see :func:`_infer_pointcloud_stride`) and a
      warning is logged, because the guess can silently corrupt the cloud. Drivers
      should always publish ``point_stride``.

    RGB columns are always stripped so callers receive a consistent ``N×3`` array.
    """
    import numpy as np

    if payload.get("type") != "pointcloud":
        return None
    data = payload.get("data")
    if not isinstance(data, str):
        return None

    arr = np.frombuffer(base64.b64decode(data), dtype=np.float32)

    raw_stride = payload.get("point_stride")
    if raw_stride is not None:
        stride = int(raw_stride)
        if stride not in (3, 6):
            raise CyberwaveError(
                f"unsupported point_stride {stride!r} (expected 3 or 6)"
            )
    else:
        stride = _infer_pointcloud_stride(arr)
        if stride not in _warned_missing_stride:
            _warned_missing_stride.add(stride)
            logger.warning(
                "point cloud payload has no 'point_stride'; guessed stride=%d from "
                "the data. This guess can silently drop or duplicate points — publish "
                "'point_stride' (3 or 6) from the driver to make decoding "
                "deterministic. (Further identical warnings are suppressed.)",
                stride,
            )

    if arr.size % stride != 0:
        raise CyberwaveError(
            f"point cloud payload size {arr.size} not divisible by stride {stride}"
        )

    pts = arr.reshape(-1, stride)
    # ``ascontiguousarray`` both drops the (possibly non-contiguous) RGB view and
    # releases the base64-decoded backing buffer once the caller copies the result.
    return np.ascontiguousarray(pts[:, :3])  # drop RGB; always return XYZ


class PointCloudCapableMixin(MqttSensorStreamHandle):
    """Adds ``get_pointcloud`` / ``on_pointcloud`` over the twin ``/pointcloud`` topic."""

    @simulation_level(SimLevel.MUJOCO)
    def get_pointcloud(self, *, timeout: float = FIRST_READ_TIMEOUT_S) -> Any:
        """A fresh point cloud snapshot as a numpy ``N×3`` float32 array (XYZ).

        Waits up to *timeout* for a message newer than the last one returned,
        so repeated calls don't silently hand back the same stale snapshot.
        Falls back to the last known value if nothing fresher arrives in time.
        """
        return self._get_latest(POINTCLOUD_STREAM, _decode_pointcloud, timeout=timeout)

    def on_pointcloud(self, callback: Callable[[Any], None]) -> StateSubscription:
        """Run *callback* on every inbound point cloud; returns a cancellable sub."""
        return self._register_callback(POINTCLOUD_STREAM, _decode_pointcloud, callback)
