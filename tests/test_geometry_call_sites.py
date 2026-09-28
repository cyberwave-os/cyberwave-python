"""Every SDK entry point that does rotation maths, against the real core.

``test_geometry_core_available.py`` proves the core loads. This proves the SDK
*calls it correctly* -- that the angles go in the right order, that the returned
components are read in the right order, and that degrees and radians are not
confused at the boundary.

Those are the mistakes a mock cannot catch, and one of these call sites was
asserted only against a mock until now: ``test_twin_mqtt_state`` patches
``edit_rotation`` out entirely, so it tested that the method was called and
nothing about the quaternion it produces.

The expected values are computed with the core itself rather than hard-coded,
except where a hand-checked constant is the point (a quarter turn about Z).
Hard-coding the core's own output would only restate it; what is worth pinning
is the *convention* -- fixed-axis XYZ, `xyzw` on the way out, degrees at the
public surface -- which is what a swapped argument or a reordered tuple breaks.
"""

from __future__ import annotations

import math

import pytest

QUARTER_TURN = math.sqrt(0.5)


# ---------------------------------------------------------------------------
# schema.Quaternion.from_rpy -- radians in, xyzw fields out
# ---------------------------------------------------------------------------


def test_schema_quaternion_from_rpy_is_fixed_axis_xyzw() -> None:
    from cyberwave.schema import Quaternion

    q = Quaternion.from_rpy(0.0, 0.0, math.pi / 2)

    assert q.x == pytest.approx(0.0, abs=1e-12)
    assert q.y == pytest.approx(0.0, abs=1e-12)
    assert q.z == pytest.approx(QUARTER_TURN, abs=1e-12)
    assert q.w == pytest.approx(QUARTER_TURN, abs=1e-12)
    assert q.to_list() == [q.x, q.y, q.z, q.w]


def test_schema_quaternion_from_rpy_matches_the_core_component_for_component() -> None:
    """Catches a field reordered in the dataclass construction."""
    from cyberwave_geometry import quaternion as quat

    from cyberwave.schema import Quaternion

    roll, pitch, yaw = 0.1, -0.35, 1.2
    expected = quat.from_rpy(roll, pitch, yaw)
    q = Quaternion.from_rpy(roll, pitch, yaw)

    assert (q.x, q.y, q.z, q.w) == pytest.approx(
        (expected.x, expected.y, expected.z, expected.w), abs=1e-12
    )


def test_schema_quaternion_from_rpy_composes_as_rz_ry_rx() -> None:
    """Pins the convention by composition, not by a value.

    Fixed-axis (extrinsic) XYZ is ``Rz * Ry * Rx`` -- x applied first. Three.js
    defaults to intrinsic ``"XYZ"``, which is ``Rx * Ry * Rz`` for the same
    three angles and a genuinely different rotation: the two disagree by ~0.17
    in the components at these angles, so this distinguishes them rather than
    merely restating the core's output.
    """
    from cyberwave_geometry import quaternion as quat

    from cyberwave.schema import Quaternion

    roll = pitch = yaw = 0.6
    qx = quat.from_rpy(roll, 0.0, 0.0)
    qy = quat.from_rpy(0.0, pitch, 0.0)
    qz = quat.from_rpy(0.0, 0.0, yaw)

    extrinsic = quat.multiply(quat.multiply(qz, qy), qx)
    intrinsic = quat.multiply(quat.multiply(qx, qy), qz)

    q = Quaternion.from_rpy(roll, pitch, yaw)
    assert (q.x, q.y, q.z, q.w) == pytest.approx(
        (extrinsic.x, extrinsic.y, extrinsic.z, extrinsic.w), abs=1e-12
    )
    assert (q.x, q.y, q.z, q.w) != pytest.approx(
        (intrinsic.x, intrinsic.y, intrinsic.z, intrinsic.w), abs=1e-3
    )


# ---------------------------------------------------------------------------
# Twin._euler_to_quaternion -- degrees in, the path edit_rotation takes
# ---------------------------------------------------------------------------


def test_twin_euler_to_quaternion_takes_degrees_and_returns_xyzw() -> None:
    from cyberwave.twin import Twin

    x, y, z, w = Twin._euler_to_quaternion(0.0, 0.0, 90.0)

    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(QUARTER_TURN, abs=1e-12)
    assert w == pytest.approx(QUARTER_TURN, abs=1e-12)


def test_twin_euler_to_quaternion_agrees_with_the_schema_helper() -> None:
    """Two entry points, one convention: degrees here, radians there."""
    from cyberwave.schema import Quaternion
    from cyberwave.twin import Twin

    roll_deg, pitch_deg, yaw_deg = 12.0, -30.0, 75.0
    from_twin = Twin._euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)
    from_schema = Quaternion.from_rpy(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )

    assert from_twin == pytest.approx(from_schema.to_list(), abs=1e-12)


def test_twin_euler_to_quaternion_treats_its_arguments_as_roll_pitch_yaw() -> None:
    """A swapped roll/yaw is silent at 45 degrees and wrong everywhere else."""
    from cyberwave.twin import Twin

    roll_only = Twin._euler_to_quaternion(90.0, 0.0, 0.0)
    yaw_only = Twin._euler_to_quaternion(0.0, 0.0, 90.0)

    assert roll_only[0] == pytest.approx(QUARTER_TURN, abs=1e-12)  # x carries roll
    assert yaw_only[2] == pytest.approx(QUARTER_TURN, abs=1e-12)  # z carries yaw


def test_edit_rotation_sends_the_quaternion_the_core_computed() -> None:
    """The end-to-end shape of the call, with only the transport stubbed out.

    ``test_twin_mqtt_state`` patches ``edit_rotation`` itself, so nothing there
    reaches the core. Here the maths runs for real and only ``_update_state``
    is replaced, which is what makes this a test of the conversion rather than
    of the mock.
    """
    from cyberwave.twin import Twin

    sent: dict = {}

    class _StubbedTransport(Twin):
        def __init__(self) -> None:  # noqa: D107 - deliberately no super().__init__
            self._rotation = None

        def _update_state(self, update_data: dict) -> None:
            sent.update(update_data)

    twin = _StubbedTransport()
    twin.edit_rotation(yaw=90.0)

    assert sent["rotation_x"] == pytest.approx(0.0, abs=1e-12)
    assert sent["rotation_y"] == pytest.approx(0.0, abs=1e-12)
    assert sent["rotation_z"] == pytest.approx(QUARTER_TURN, abs=1e-12)
    assert sent["rotation_w"] == pytest.approx(QUARTER_TURN, abs=1e-12)
    # The cache and the wire must not disagree about component order.
    assert twin._rotation["z"] == pytest.approx(sent["rotation_z"], abs=1e-12)


def test_edit_rotation_passes_an_explicit_quaternion_through_unchanged() -> None:
    """The quaternion branch must not re-normalize or reorder."""
    from cyberwave.twin import Twin

    sent: dict = {}

    class _StubbedTransport(Twin):
        def __init__(self) -> None:  # noqa: D107
            self._rotation = None

        def _update_state(self, update_data: dict) -> None:
            sent.update(update_data)

    twin = _StubbedTransport()
    twin.edit_rotation(quaternion=[0.0, 0.0, QUARTER_TURN, QUARTER_TURN])

    assert sent["rotation_x"] == 0.0
    assert sent["rotation_z"] == QUARTER_TURN
    assert sent["rotation_w"] == QUARTER_TURN


# ---------------------------------------------------------------------------
# placement.compute_centered_placement -- the core rotates the centering offset
# ---------------------------------------------------------------------------


def test_centered_placement_puts_an_unrotated_asset_where_asked() -> None:
    from cyberwave.placement import GENERIC_CUBE_BOUNDS, compute_centered_placement

    placement = compute_centered_placement(
        center=(1.0, 2.0, 3.0), asset_bounds=GENERIC_CUBE_BOUNDS
    )

    # The generic cube's local centre is offset from its origin, so the link
    # origin is not the requested centre -- but the centre it produces is.
    (mnx, mny, mnz), (mxx, mxy, mxz) = GENERIC_CUBE_BOUNDS
    local_center = ((mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2)
    recovered = tuple(p + c for p, c in zip(placement.position, local_center))

    assert recovered == pytest.approx((1.0, 2.0, 3.0), abs=1e-12)
    assert placement.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)


def test_centered_placement_rotates_the_centering_offset_through_the_core() -> None:
    """With a rotation, the offset must be rotated before it is subtracted.

    Applying the rotation to the wrong side -- or not at all -- still returns a
    plausible position, which is why this needs an asymmetric asset and a
    quarter turn rather than a round trip.
    """
    from cyberwave_geometry import Vector3
    from cyberwave_geometry import quaternion as quat

    from cyberwave.placement import GENERIC_CUBE_BOUNDS, compute_centered_placement

    yaw_90 = (0.0, 0.0, QUARTER_TURN, QUARTER_TURN)
    placement = compute_centered_placement(
        center=(0.0, 0.0, 0.0), asset_bounds=GENERIC_CUBE_BOUNDS, rotation=yaw_90
    )

    (mnx, mny, mnz), (mxx, mxy, mxz) = GENERIC_CUBE_BOUNDS
    local_center = Vector3((mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2)
    rotated = quat.rotate_unit(quat.Quaternion.from_xyzw(yaw_90), local_center)

    assert placement.position == pytest.approx(
        (-rotated.x, -rotated.y, -rotated.z), abs=1e-12
    )
    assert placement.rotation == pytest.approx(yaw_90, abs=1e-12)


def test_centered_placement_rejects_a_degenerate_rotation() -> None:
    """The core refuses an unusable quaternion; the SDK must not swallow it."""
    from cyberwave.placement import GENERIC_CUBE_BOUNDS, compute_centered_placement

    with pytest.raises(ValueError, match="zero or non-finite norm"):
        compute_centered_placement(
            center=(0.0, 0.0, 0.0),
            asset_bounds=GENERIC_CUBE_BOUNDS,
            rotation=(0.0, 0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# calibration.frames -- already covered by test_calibration_frames.py, pinned
# here only for the component order it hands to the scene, which that file's
# round-trip tests would not catch if both directions flipped together.
# ---------------------------------------------------------------------------


def test_calibration_frames_quaternion_order_reaches_the_scene_as_xyzw() -> None:
    from cyberwave.calibration.frames import quat_wxyz_to_xyzw

    assert quat_wxyz_to_xyzw((QUARTER_TURN, 0.0, 0.0, QUARTER_TURN)) == pytest.approx(
        (0.0, 0.0, QUARTER_TURN, QUARTER_TURN), abs=1e-12
    )
