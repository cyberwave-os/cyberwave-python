"""SE(3) helpers for calibration — 4x4 homogeneous transforms and ``(w, x, y, z)`` quaternions.

Quaternion convention matches the platform: Hamilton product, ``(w, x, y, z)``
component order, same rotation matrix as the backend's
``src/lib/transform_utils.py`` (``rotation_matrix`` / ``_quat_multiply_dict``).
Transforms compose with plain ``@``: ``T_a_c = T_a_b @ T_b_c``.

numpy only — no OpenCV, so this module is importable without the ``camera`` extra.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .._geometry import core as _geometry_core


def quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix for a ``(w, x, y, z)`` quaternion (normalized first)."""
    geometry = _geometry_core()
    try:
        rotation = geometry.quat.normalize(
            geometry.Quaternion.from_wxyz([float(v) for v in quaternion])
        )
    except geometry.GeometryError as error:
        # Same contract as before: a non-finite or vanishing quaternion is a
        # caller error here, not something to approximate. The core's floor is
        # 1e-9 rather than the 1e-12 this used to test -- both reject a value
        # that carries no usable rotation.
        raise ValueError(f"Not a usable quaternion: {tuple(quaternion)!r}") from error
    return np.array(geometry.quat.to_matrix(rotation), dtype=float).reshape(3, 3)


def matrix_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """``(w, x, y, z)`` quaternion for a 3x3 rotation matrix.

    Branches on the largest diagonal term rather than always using the trace, so
    the divisor stays well away from zero for 180-degree rotations — which is
    exactly the region a wrist-mounted camera lands in (the optical-frame flip is
    a half turn about X).
    """
    # geometry-core-exempt: the core's from_matrix rejects anything that is not
    # orthonormal within 1e-6, which is right for a matrix it produced but wrong
    # here: this reads matrices straight out of OpenCV hand-eye calibration and
    # out of _mean_transform's averaged rotations, which drift further than that
    # and are still the best estimate available. Swapping it in would turn a
    # working calibration into a hard failure. Migrate only behind an explicit
    # re-orthonormalization step at the boundary.
    m = np.asarray(rotation, dtype=float)[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    # Canonical sign (w >= 0): q and -q are the same rotation, but the persisted
    # value is read back by humans in the editor and diffed between calibrations.
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform from a 3x3 rotation and a 3-vector."""
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def invert(transform: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose the rotation, no matrix solve)."""
    t = np.asarray(transform, dtype=float)
    rotation = t[:3, :3].T
    return make_transform(rotation, -rotation @ t[:3, 3])


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Geodesic angle of a 3x3 rotation, in degrees (0..180)."""
    m = np.asarray(rotation, dtype=float)[:3, :3]
    cos_angle = (m[0, 0] + m[1, 1] + m[2, 2] - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0)))))


def rotation_axis(rotation: np.ndarray) -> np.ndarray | None:
    """Unit rotation axis of a 3x3 rotation, or ``None`` when the angle is ~0."""
    m = np.asarray(rotation, dtype=float)[:3, :3]
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def pose_to_matrix(
    position: Mapping[str, Any] | None,
    rotation: Mapping[str, Any] | None,
) -> np.ndarray:
    """4x4 transform from the platform's ``{x,y,z}`` / ``{w,x,y,z}`` pose dicts.

    Missing keys and ``None`` default to identity, matching the backend's
    ``coerce_position`` / ``coerce_rotation`` leniency — persisted schema poses
    are frequently partial.
    """
    pos = position or {}
    rot = rotation or {}
    translation = [float(pos.get(axis) or 0.0) for axis in ("x", "y", "z")]
    quaternion = [
        float(rot.get("w", 1.0) or 0.0),
        float(rot.get("x", 0.0) or 0.0),
        float(rot.get("y", 0.0) or 0.0),
        float(rot.get("z", 0.0) or 0.0),
    ]
    if not any(quaternion):
        quaternion = [1.0, 0.0, 0.0, 0.0]
    return make_transform(quat_wxyz_to_matrix(quaternion), translation)


def matrix_to_pose(transform: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Split a 4x4 transform into ``({x,y,z}, {w,x,y,z})`` platform pose dicts."""
    t = np.asarray(transform, dtype=float)
    w, x, y, z = matrix_to_quat_wxyz(t[:3, :3])
    return (
        {"x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3])},
        {"w": w, "x": x, "y": y, "z": z},
    )


def quat_wxyz_to_xyzw(quaternion: Sequence[float]) -> list[float]:
    """Reorder ``(w, x, y, z)`` to the ``[x, y, z, w]`` that ``Scene.dock`` expects."""
    w, x, y, z = (float(v) for v in quaternion)
    return [x, y, z, w]


# OpenCV's optical frame is X right, Y down, Z forward along the view axis.
# Cyberwave's sensor frame is X right, Y up, Z backward — the frontend converts
# optical to sensor as ``[x, -y, -z]`` (see
# cyberwave-frontend/lib/utils/sensor-transform.ts::depthCameraOpticalToSensorLocal),
# which is a half turn about X. A solver result expressed in the optical frame is
# therefore right-multiplied by this to land in the sensor frame.
#
# This is its own involution: OPTICAL_TO_SENSOR == invert(OPTICAL_TO_SENSOR).
OPTICAL_TO_SENSOR: np.ndarray = make_transform(
    np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.0]
)


def describe_pose(transform: np.ndarray) -> dict[str, Any]:
    """Human-readable view of a 4x4: position, quaternion and axis-angle.

    Axis-angle rather than roll/pitch/yaw on purpose -- Euler conventions are the
    single easiest thing to misread when two tools disagree about a rotation.

    This is the one shape for a pose written to a JSON artifact, so a pose logged
    by one tool can be diffed against the same pose logged by another. Keep the
    key names stable.
    """
    matrix = np.asarray(transform, dtype=float)
    axis = rotation_axis(matrix[:3, :3])
    return {
        "position_m": [round(float(v), 6) for v in matrix[:3, 3]],
        "quaternion_wxyz": [
            round(float(v), 6) for v in matrix_to_quat_wxyz(matrix[:3, :3])
        ],
        "rotation_angle_deg": round(rotation_angle_deg(matrix[:3, :3]), 4),
        "rotation_axis": None if axis is None else [round(float(v), 6) for v in axis],
    }


__all__ = [
    "OPTICAL_TO_SENSOR",
    "describe_pose",
    "invert",
    "make_transform",
    "matrix_to_pose",
    "matrix_to_quat_wxyz",
    "pose_to_matrix",
    "quat_wxyz_to_matrix",
    "quat_wxyz_to_xyzw",
    "rotation_angle_deg",
    "rotation_axis",
]
