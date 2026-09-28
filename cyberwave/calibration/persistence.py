"""Persist a solved hand-eye transform onto a docked camera twin.

This is where the OpenCV optical frame gets converted into the frame the platform
actually renders and exports from, and where the result is written to the twin's
``attach_offset_*`` fields — the same fields ``Scene.dock`` sets and the URDF
exporter reads as a docked twin's spawn pose.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..exceptions import CyberwaveValidationError
from .frames import (
    OPTICAL_TO_SENSOR,
    invert,
    matrix_to_pose,
    pose_to_matrix,
)
from .handeye import HandEyeResult

#: Twin metadata key for hand-eye provenance.
#:
#: Deliberately *not* ``"calibration"``: the twin-update endpoint already writes
#: joint (servo) calibration there — see ``update_twin`` in the backend's
#: ``src/app/api/twins.py`` — and reusing the key would clobber it.
HAND_EYE_METADATA_KEY = "hand_eye_calibration"

_IMAGING_SENSOR_TYPES = frozenset({"rgb", "depth", "rgbd", "camera"})


def _imaging_sensors(twin: Any) -> list[dict[str, Any]]:
    return [
        sensor
        for sensor in twin.sensors
        if isinstance(sensor, dict)
        and str(sensor.get("type", "")).lower() in _IMAGING_SENSOR_TYPES
    ]


def resolve_sensor_offset(twin: Any, sensor_id: str | None = None) -> np.ndarray:
    """The camera twin's own sensor-to-twin-origin transform, as a 4x4.

    A camera twin's schema places its sensor at ``sensor.offset`` relative to
    ``sensor.parent_link``. ``attach_offset_*`` positions the twin *origin* on the
    arm link, so that sensor offset sits between the two and has to be divided out
    of the solved transform. Identity for most camera assets, which is why getting
    this wrong is easy to miss.
    """
    sensors = _imaging_sensors(twin)
    if not sensors:
        return np.eye(4)
    if sensor_id is None:
        sensor = sensors[0]
    else:
        matches = [
            s
            for s in sensors
            if str(s.get("id")) == str(sensor_id) or str(s.get("name")) == str(sensor_id)
        ]
        if not matches:
            available = [s.get("id") or s.get("name") for s in sensors]
            raise CyberwaveValidationError(
                f"Twin {twin.uuid} has no imaging sensor {sensor_id!r}. "
                f"Available: {available}"
            )
        sensor = matches[0]
    offset = sensor.get("offset") or {}
    return pose_to_matrix(offset.get("position"), offset.get("rotation"))


def optical_to_attach_offset(
    gripper_to_camera_optical: np.ndarray,
    sensor_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a solved optical-frame transform into a twin ``attach_offset``.

    The platform composes a docked camera's sensor pose as::

        link -> attach_offset -> sensor.offset -> (sensor frame)

    and the sensor frame differs from OpenCV's optical frame by a half turn about
    X (see :data:`~cyberwave.calibration.frames.OPTICAL_TO_SENSOR`). Solving that
    chain for ``attach_offset`` gives::

        attach_offset = X_optical @ OPTICAL_TO_SENSOR @ inv(sensor.offset)
    """
    offset = np.eye(4) if sensor_offset is None else np.asarray(sensor_offset, float)
    return np.asarray(gripper_to_camera_optical, float) @ OPTICAL_TO_SENSOR @ invert(offset)


class CameraCalibrationHandle:
    """Read and write a camera twin's hand-eye calibration.

    Reached as ``camera_twin.calibration``. Shaped after
    ``twin.joints.calibration`` (``get`` / ``set`` / ``delete``), and like it,
    REST-only — no MQTT, nothing is commanded.
    """

    def __init__(self, twin: Any) -> None:
        self._twin = twin

    def get(self) -> dict[str, Any] | None:
        """Stored hand-eye provenance, or ``None`` if never calibrated.

        Returns the recorded solve (residuals, sample count, board, intrinsics),
        not the transform itself — the transform lives in the twin's
        ``attach_offset_*`` fields, since that is what the platform renders from.
        """
        metadata = self._twin.metadata or {}
        record = metadata.get(HAND_EYE_METADATA_KEY)
        return record if isinstance(record, dict) else None

    def set(
        self,
        result: HandEyeResult,
        *,
        board: Any = None,
        intrinsics: Mapping[str, Any] | Sequence[Any] | None = None,
        dist_coeffs: Sequence[float] | None = None,
        fk_frame: str | None = None,
        sensor_id: str | None = None,
        solved_at: str | None = None,
        quality_thresholds: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Write *result* to the twin's docking offset, with provenance.

        Args:
            result: The solved transform, in the OpenCV optical frame.
            board: Board used, for provenance. Anything with ``to_metadata()``.
            intrinsics: Intrinsics used, for provenance. **Record these** — nothing
                else in the platform stores camera intrinsics today, so this blob is
                the only trace of what the solve was based on.
            dist_coeffs: Distortion used, for provenance.
            fk_frame: Link the poses were measured about; defaults to the twin's
                current ``attach_to_link``.
            sensor_id: Which imaging sensor was calibrated; defaults to the first.
            solved_at: ISO-8601 timestamp for provenance. Caller-supplied so this
                stays free of implicit clock reads.
            quality_thresholds: The stability/residual bars this run was judged
                against. Stored because they are measured per arm: a reader
                rendering a verdict from this record months later otherwise has
                to assume one robot's numbers.

        Returns:
            The provenance record that was stored.
        """
        if not isinstance(result, HandEyeResult):
            raise CyberwaveValidationError(
                f"Expected a HandEyeResult, got {type(result).__name__}"
            )

        sensor_offset = resolve_sensor_offset(self._twin, sensor_id)
        attach_offset = optical_to_attach_offset(
            result.gripper_to_camera, sensor_offset
        )
        position, rotation = matrix_to_pose(attach_offset)

        record: dict[str, Any] = {
            **result.to_metadata(),
            "fk_frame": fk_frame
            if fk_frame is not None
            else self._twin._data_get("attach_to_link"),
            "sensor_id": sensor_id,
            "sensor_offset_applied": not np.allclose(sensor_offset, np.eye(4)),
            "solved_at": solved_at,
        }
        if board is not None:
            record["board"] = board.to_metadata() if hasattr(board, "to_metadata") else board
        if intrinsics is not None:
            record["intrinsics"] = (
                dict(intrinsics)
                if isinstance(intrinsics, Mapping)
                else np.asarray(intrinsics, float).tolist()
            )
        if dist_coeffs is not None:
            record["dist_coeffs"] = list(np.asarray(dist_coeffs, float).ravel())
        if quality_thresholds is not None:
            record["quality_thresholds"] = {
                k: float(v) for k, v in quality_thresholds.items()
            }

        payload: dict[str, Any] = {
            "attach_offset_x": position["x"],
            "attach_offset_y": position["y"],
            "attach_offset_z": position["z"],
            "attach_offset_rotation_w": rotation["w"],
            "attach_offset_rotation_x": rotation["x"],
            "attach_offset_rotation_y": rotation["y"],
            "attach_offset_rotation_z": rotation["z"],
            # Shallow-merged server-side, so this replaces only our own key.
            "metadata": {HAND_EYE_METADATA_KEY: record},
        }
        # `position_*`/`rotation_*` are deliberately NOT written. For a docked twin
        # they hold the world pose it had before docking, which is what it falls
        # back to on undock (see migration 0225_backfill_twin_attach_offsets) and
        # what the editor draws while the parent link is still resolving. The
        # dashboard reads `attach_offset_*` while docked and does not read these
        # -- useTwinDocking.ts: "stay world-only and are deliberately not read
        # while docked". Writing a link-relative offset here would put the camera
        # at the world origin on undock, unrecoverably.

        self._twin.client.twins.update(self._twin.uuid, **payload)
        self._twin.refresh()
        return record

    def delete(self) -> None:
        """Remove the stored provenance record.

        Leaves ``attach_offset_*`` untouched: forgetting how a mount was measured
        must not silently relocate the camera. Re-dock or call :meth:`set` to change
        the transform itself.
        """
        self._twin.client.twins.update(
            self._twin.uuid, metadata={HAND_EYE_METADATA_KEY: None}
        )
        self._twin.refresh()

    def __repr__(self) -> str:
        return (
            f"CameraCalibrationHandle(twin={self._twin.uuid}, "
            f"calibrated={self.get() is not None})"
        )


__all__ = [
    "HAND_EYE_METADATA_KEY",
    "CameraCalibrationHandle",
    "optical_to_attach_offset",
    "resolve_sensor_offset",
]
