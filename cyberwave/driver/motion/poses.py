"""Pose conversions shared by every planned-motion backend.

Three representations of one TCP pose meet in the planned-motion layer: the
wire's named ``(x, y, z, w)`` :class:`~cyberwave.driver.motion.scene.Pose`, the
operator's extrinsic-XYZ RPY ``ee_pose`` array, and the 4x4 transform the
in-process solver works in. The rotation maths is the shared geometry core's
(:func:`cyberwave._geometry.core`), never a formula written here: a second
quaternion implementation is how two backends end up disagreeing about the
same orientation.
"""

from __future__ import annotations

import numpy as np

from cyberwave._geometry import core as _geometry_core

from .scene import Pose


def pose_from_xyz_rpy(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> Pose:
    """The planning-frame :class:`Pose` for an extrinsic-XYZ RPY target.

    The convention is fixed-axis XYZ, ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` --
    the URDF/ROS one, and the one ``EEPose`` and ``pose_to_matrix`` use.
    Getting it wrong is silent: the arm goes to a real, reachable, wrong
    orientation.
    """
    quaternion = _geometry_core().quat.from_rpy(float(roll), float(pitch), float(yaw))
    x_q, y_q, z_q, w_q = quaternion.to_xyzw()
    return Pose(
        (float(x), float(y), float(z)),
        (float(x_q), float(y_q), float(z_q), float(w_q)),
    )


def pose_to_transform(pose: Pose) -> np.ndarray:
    """The 4x4 homogeneous transform of a named xyzw :class:`Pose`."""
    geometry = _geometry_core()
    rotation = geometry.quat.to_matrix(geometry.Quaternion.from_xyzw(pose.orientation))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = pose.position
    return transform


def transform_to_pose(transform: np.ndarray) -> Pose:
    """The named xyzw :class:`Pose` of a 4x4 (rotation block must be orthonormal)."""
    matrix = np.asarray(transform, dtype=float)
    quaternion = _geometry_core().quat.from_matrix(
        [float(value) for value in matrix[:3, :3].reshape(9)]
    )
    x_q, y_q, z_q, w_q = quaternion.to_xyzw()
    return Pose(
        (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        (float(x_q), float(y_q), float(z_q), float(w_q)),
    )


__all__ = ["pose_from_xyz_rpy", "pose_to_transform", "transform_to_pose"]
