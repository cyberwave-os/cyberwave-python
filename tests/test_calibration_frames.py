"""SE(3) helpers in ``cyberwave.calibration.frames``.

The composition and inversion semantics here must agree with the backend's
``src/lib/transform_utils.py`` (``compose_poses`` / ``invert_pose``), which is
what actually resolves docking offsets server-side. Those functions are
reimplemented in this file rather than imported — the backend is a separate
package and is not installed alongside the SDK — so a divergence in either
direction shows up as a failure here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyberwave.calibration.frames import (
    OPTICAL_TO_SENSOR,
    invert,
    make_transform,
    matrix_to_pose,
    matrix_to_quat_wxyz,
    pose_to_matrix,
    quat_wxyz_to_matrix,
    quat_wxyz_to_xyzw,
    rotation_angle_deg,
    rotation_axis,
)


# --- reference implementations mirrored from the backend -------------------
#
# These are deliberately independent of the shared core. The claim under test is
# that the SDK's calibration helpers agree with the convention the backend
# persists -- Hamilton product, (w, x, y, z) order -- so the expectation has to
# be computed from that convention spelled out, not from the same core the code
# under test now calls. Rewriting these against the core would leave the test
# checking the core against itself.


# geometry-core-exempt: independent reference implementation, see above.
def _quat_multiply_dict(lhs: dict, rhs: dict) -> dict:
    lw, lx, ly, lz = lhs["w"], lhs["x"], lhs["y"], lhs["z"]
    rw, rx, ry, rz = rhs["w"], rhs["x"], rhs["y"], rhs["z"]
    return {
        "w": lw * rw - lx * rx - ly * ry - lz * rz,
        "x": lw * rx + lx * rw + ly * rz - lz * ry,
        "y": lw * ry - lx * rz + ly * rw + lz * rx,
        "z": lw * rz + lx * ry - ly * rx + lz * rw,
    }


# geometry-core-exempt: independent reference implementation, see above.
def _rotate_vector(quaternion: dict, vector: dict) -> dict:
    vector_quat = {"w": 0.0, **vector}
    conjugate = {
        "w": quaternion["w"],
        "x": -quaternion["x"],
        "y": -quaternion["y"],
        "z": -quaternion["z"],
    }
    rotated = _quat_multiply_dict(_quat_multiply_dict(quaternion, vector_quat), conjugate)
    return {"x": rotated["x"], "y": rotated["y"], "z": rotated["z"]}


def _compose_poses(parent_pos, parent_rot, local_pos, local_rot):
    """Backend ``transform_utils.compose_poses``."""
    rotated = _rotate_vector(parent_rot, local_pos)
    position = {axis: parent_pos[axis] + rotated[axis] for axis in ("x", "y", "z")}
    return position, _quat_multiply_dict(parent_rot, local_rot)


def _invert_pose(position, rotation):
    """Backend ``transform_utils.invert_pose``."""
    inverse_rotation = {
        "w": rotation["w"],
        "x": -rotation["x"],
        "y": -rotation["y"],
        "z": -rotation["z"],
    }
    rotated = _rotate_vector(inverse_rotation, position)
    return {axis: -rotated[axis] for axis in ("x", "y", "z")}, inverse_rotation


# --- fixtures --------------------------------------------------------------


def _random_transforms(count: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        out.append(
            make_transform(quat_wxyz_to_matrix(quaternion), rng.uniform(-2.0, 2.0, 3))
        )
    return out


# --- quaternion round trips ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_quaternion_matrix_round_trip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:  # canonical w >= 0, as matrix_to_quat_wxyz returns
        quaternion = -quaternion

    recovered = matrix_to_quat_wxyz(quat_wxyz_to_matrix(quaternion))

    assert np.allclose(recovered, quaternion, atol=1e-12)


def test_quaternion_round_trip_at_half_turn_about_x() -> None:
    """The optical-frame flip is a 180-degree rotation — the trace branch's worst case."""
    quaternion = matrix_to_quat_wxyz(OPTICAL_TO_SENSOR[:3, :3])

    assert np.allclose(quat_wxyz_to_matrix(quaternion), OPTICAL_TO_SENSOR[:3, :3], atol=1e-12)
    assert rotation_angle_deg(OPTICAL_TO_SENSOR[:3, :3]) == pytest.approx(180.0)


@pytest.mark.parametrize(
    "rotation_matrix_fn",
    [
        lambda: np.diag([1.0, -1.0, -1.0]),  # 180 about X
        lambda: np.diag([-1.0, 1.0, -1.0]),  # 180 about Y
        lambda: np.diag([-1.0, -1.0, 1.0]),  # 180 about Z
        lambda: np.eye(3),
    ],
)
def test_quaternion_round_trip_at_degenerate_rotations(rotation_matrix_fn) -> None:
    rotation = rotation_matrix_fn()

    assert np.allclose(quat_wxyz_to_matrix(matrix_to_quat_wxyz(rotation)), rotation, atol=1e-12)


def test_matrix_to_quat_returns_canonical_positive_w() -> None:
    for transform in _random_transforms(20, seed=7):
        assert matrix_to_quat_wxyz(transform[:3, :3])[0] >= 0.0


def test_quat_wxyz_to_xyzw_reorders_for_scene_dock() -> None:
    assert quat_wxyz_to_xyzw([0.941, 0.0, -0.339, 0.0]) == [0.0, -0.339, 0.0, 0.941]


def test_quat_wxyz_to_matrix_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="usable quaternion"):
        quat_wxyz_to_matrix([0.0, 0.0, 0.0, 0.0])


# --- invert ----------------------------------------------------------------


def test_invert_is_a_true_inverse() -> None:
    for transform in _random_transforms(20, seed=11):
        assert np.allclose(transform @ invert(transform), np.eye(4), atol=1e-12)


def test_invert_matches_backend_invert_pose() -> None:
    for transform in _random_transforms(20, seed=13):
        position, rotation = matrix_to_pose(transform)

        expected_position, expected_rotation = _invert_pose(position, rotation)
        actual_position, actual_rotation = matrix_to_pose(invert(transform))

        assert np.allclose(
            [actual_position[a] for a in "xyz"],
            [expected_position[a] for a in "xyz"],
            atol=1e-12,
        )
        # q and -q are the same rotation; compare as rotations, not components.
        assert np.allclose(
            quat_wxyz_to_matrix([actual_rotation[k] for k in "wxyz"]),
            quat_wxyz_to_matrix([expected_rotation[k] for k in "wxyz"]),
            atol=1e-12,
        )


# --- composition agrees with the backend -----------------------------------


def test_matrix_product_matches_backend_compose_poses() -> None:
    """``parent @ local`` must equal the backend's ``compose_poses(parent, local)``.

    This is the semantics the URDF exporter relies on when it uses
    ``attach_offset_*`` as a docked twin's spawn pose.
    """
    parents = _random_transforms(10, seed=17)
    locals_ = _random_transforms(10, seed=19)

    for parent, local in zip(parents, locals_):
        parent_position, parent_rotation = matrix_to_pose(parent)
        local_position, local_rotation = matrix_to_pose(local)

        expected_position, expected_rotation = _compose_poses(
            parent_position, parent_rotation, local_position, local_rotation
        )
        actual_position, actual_rotation = matrix_to_pose(parent @ local)

        assert np.allclose(
            [actual_position[a] for a in "xyz"],
            [expected_position[a] for a in "xyz"],
            atol=1e-12,
        )
        assert np.allclose(
            quat_wxyz_to_matrix([actual_rotation[k] for k in "wxyz"]),
            quat_wxyz_to_matrix([expected_rotation[k] for k in "wxyz"]),
            atol=1e-12,
        )


def test_pose_to_matrix_round_trips_through_matrix_to_pose() -> None:
    for transform in _random_transforms(10, seed=23):
        position, rotation = matrix_to_pose(transform)

        assert np.allclose(pose_to_matrix(position, rotation), transform, atol=1e-12)


# --- pose_to_matrix leniency ----------------------------------------------


@pytest.mark.parametrize(
    ("position", "rotation"),
    [
        (None, None),
        ({}, {}),
        ({"x": 0.0}, {"w": 1.0}),
        # A rotation dict of all zeros is what a partially-written schema looks
        # like; it must fall back to identity rather than raise.
        ({"x": 0.0, "y": 0.0, "z": 0.0}, {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}),
    ],
)
def test_pose_to_matrix_defaults_to_identity(position, rotation) -> None:
    assert np.allclose(pose_to_matrix(position, rotation), np.eye(4), atol=1e-12)


def test_pose_to_matrix_reads_partial_position() -> None:
    transform = pose_to_matrix({"y": 1.5}, None)

    assert np.allclose(transform[:3, 3], [0.0, 1.5, 0.0])


# --- rotation angle / axis -------------------------------------------------


@pytest.mark.parametrize("degrees", [0.0, 1.0, 30.0, 90.0, 179.0, 180.0])
def test_rotation_angle_deg_recovers_the_angle(degrees: float) -> None:
    half = math.radians(degrees) / 2.0
    rotation = quat_wxyz_to_matrix([math.cos(half), math.sin(half), 0.0, 0.0])

    assert rotation_angle_deg(rotation) == pytest.approx(degrees, abs=1e-6)


def test_rotation_axis_recovers_the_axis() -> None:
    half = math.radians(40.0) / 2.0
    expected = np.array([0.0, 0.0, 1.0])
    rotation = quat_wxyz_to_matrix([math.cos(half), 0.0, 0.0, math.sin(half)])

    assert np.allclose(rotation_axis(rotation), expected, atol=1e-9)


def test_rotation_axis_is_none_for_identity() -> None:
    assert rotation_axis(np.eye(3)) is None


# --- OPTICAL_TO_SENSOR ----------------------------------------------------


def test_optical_to_sensor_is_its_own_inverse() -> None:
    assert np.allclose(OPTICAL_TO_SENSOR @ OPTICAL_TO_SENSOR, np.eye(4), atol=1e-12)
    assert np.allclose(invert(OPTICAL_TO_SENSOR), OPTICAL_TO_SENSOR, atol=1e-12)


def test_optical_to_sensor_matches_the_frontend_axis_flip() -> None:
    """Must reproduce ``depthCameraOpticalToSensorLocal``: ``[x, -y, -z]``."""
    point = np.array([0.3, 0.4, 0.5, 1.0])

    assert np.allclose((OPTICAL_TO_SENSOR @ point)[:3], [0.3, -0.4, -0.5])


def test_optical_to_sensor_is_a_proper_rotation() -> None:
    """A reflection here would silently mirror every calibration result."""
    assert np.linalg.det(OPTICAL_TO_SENSOR[:3, :3]) == pytest.approx(1.0)
