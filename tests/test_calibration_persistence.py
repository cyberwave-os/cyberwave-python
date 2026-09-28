"""``CameraCalibrationHandle`` — the frame conversion and the exact update payload.

Two things are pinned here because they are silent when wrong:

* the optical-to-sensor conversion and the sensor-offset division
  (``optical_to_attach_offset``), and
* the quaternion component order in the persisted payload — the twin fields are
  ``(w, x, y, z)`` while ``Scene.dock`` takes ``[x, y, z, w]``.

The twin is a double: the handle only needs ``uuid``, ``metadata``, ``sensors``,
``_data_get``, ``client.twins.update`` and ``refresh``.
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
    pose_to_matrix,
    quat_wxyz_to_matrix,
)
from cyberwave.calibration.handeye import HandEyeResult
from cyberwave.calibration.persistence import (
    HAND_EYE_METADATA_KEY,
    CameraCalibrationHandle,
    optical_to_attach_offset,
    resolve_sensor_offset,
)
from cyberwave.exceptions import CyberwaveValidationError

CAM_UUID = "22222222-2222-2222-2222-222222222222"
LINK = "openarm_left_link7"


def _rotation(axis: str, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    quaternion = [math.cos(half), 0.0, 0.0, 0.0]
    quaternion[{"x": 1, "y": 2, "z": 3}[axis]] = math.sin(half)
    return quat_wxyz_to_matrix(quaternion)


SOLVED_OPTICAL = make_transform(_rotation("y", -30.0), [0.045, 0.0, 0.099])


def _result(transform: np.ndarray = SOLVED_OPTICAL) -> HandEyeResult:
    return HandEyeResult(
        gripper_to_camera=transform,
        residual_translation_m=0.0012,
        max_residual_translation_m=0.0031,
        residual_rotation_deg=0.08,
        max_residual_rotation_deg=0.21,
        sample_count=12,
        method="park",
    )


# --- doubles ---------------------------------------------------------------


class FakeTwins:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict]] = []

    def update(self, twin_id: str, **kwargs):
        self.updates.append((twin_id, kwargs))
        return {}


class FakeClient:
    def __init__(self) -> None:
        self.twins = FakeTwins()


class FakeTwin:
    def __init__(self, *, sensors=None, metadata=None, attach_to_link=LINK) -> None:
        self.uuid = CAM_UUID
        self.client = FakeClient()
        self.sensors = sensors if sensors is not None else []
        self.metadata = metadata or {}
        self._data = {"attach_to_link": attach_to_link}
        self.refreshed = 0

    def _data_get(self, field: str, default=None):
        return self._data.get(field, default)

    def refresh(self) -> None:
        self.refreshed += 1


def _sensor(offset=None, sensor_id="color_camera", sensor_type="rgb") -> dict:
    return {
        "id": sensor_id,
        "name": sensor_id,
        "type": sensor_type,
        "offset": offset
        or {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
        },
    }


# --- the frame conversion --------------------------------------------------


def test_identity_sensor_offset_applies_only_the_optical_flip() -> None:
    result = optical_to_attach_offset(SOLVED_OPTICAL, np.eye(4))

    assert np.allclose(result, SOLVED_OPTICAL @ OPTICAL_TO_SENSOR)


def test_conversion_defaults_to_an_identity_sensor_offset() -> None:
    assert np.allclose(
        optical_to_attach_offset(SOLVED_OPTICAL),
        optical_to_attach_offset(SOLVED_OPTICAL, np.eye(4)),
    )


def test_conversion_preserves_the_camera_origin() -> None:
    """The flip is a pure rotation, so the camera's position must not move."""
    converted = optical_to_attach_offset(SOLVED_OPTICAL, np.eye(4))

    assert np.allclose(converted[:3, 3], SOLVED_OPTICAL[:3, 3])


def test_conversion_reconstructs_the_documented_chain() -> None:
    """``attach_offset @ sensor.offset @ OPTICAL_TO_SENSOR`` must return the input."""
    sensor_offset = make_transform(_rotation("z", 15.0), [0.01, -0.02, 0.005])

    attach_offset = optical_to_attach_offset(SOLVED_OPTICAL, sensor_offset)

    assert np.allclose(
        attach_offset @ sensor_offset @ OPTICAL_TO_SENSOR, SOLVED_OPTICAL, atol=1e-12
    )


def test_a_nonidentity_sensor_offset_changes_the_result() -> None:
    sensor_offset = make_transform(np.eye(3), [0.0, 0.0, 0.02])

    with_offset = optical_to_attach_offset(SOLVED_OPTICAL, sensor_offset)

    assert not np.allclose(with_offset, optical_to_attach_offset(SOLVED_OPTICAL))


def test_conversion_output_stays_a_rigid_transform() -> None:
    sensor_offset = make_transform(_rotation("x", 40.0), [0.01, 0.02, 0.03])

    converted = optical_to_attach_offset(SOLVED_OPTICAL, sensor_offset)

    rotation = converted[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


# --- sensor offset resolution ---------------------------------------------


def test_no_sensors_resolves_to_identity() -> None:
    assert np.allclose(resolve_sensor_offset(FakeTwin()), np.eye(4))


def test_first_imaging_sensor_is_used_by_default() -> None:
    offset = {"position": {"x": 0.01, "y": 0, "z": 0}, "rotation": {"w": 1}}
    twin = FakeTwin(sensors=[_sensor(offset), _sensor(sensor_id="second")])

    assert np.allclose(resolve_sensor_offset(twin)[:3, 3], [0.01, 0.0, 0.0])


def test_non_imaging_sensors_are_skipped() -> None:
    """An IMU listed first must not be mistaken for the camera."""
    imu = {
        "id": "imu",
        "type": "imu",
        "offset": {"position": {"x": 9.0, "y": 0, "z": 0}, "rotation": {"w": 1}},
    }
    camera = _sensor({"position": {"x": 0.02, "y": 0, "z": 0}, "rotation": {"w": 1}})
    twin = FakeTwin(sensors=[imu, camera])

    assert np.allclose(resolve_sensor_offset(twin)[:3, 3], [0.02, 0.0, 0.0])


def test_a_sensor_can_be_selected_by_id() -> None:
    first = _sensor({"position": {"x": 0.01, "y": 0, "z": 0}, "rotation": {"w": 1}})
    second = _sensor(
        {"position": {"x": 0.05, "y": 0, "z": 0}, "rotation": {"w": 1}},
        sensor_id="tele",
    )
    twin = FakeTwin(sensors=[first, second])

    assert np.allclose(resolve_sensor_offset(twin, "tele")[:3, 3], [0.05, 0.0, 0.0])


def test_an_unknown_sensor_id_is_rejected() -> None:
    twin = FakeTwin(sensors=[_sensor()])

    with pytest.raises(CyberwaveValidationError, match="no imaging sensor"):
        resolve_sensor_offset(twin, "nope")


def test_a_missing_offset_block_resolves_to_identity() -> None:
    twin = FakeTwin(sensors=[{"id": "c", "type": "rgb"}])

    assert np.allclose(resolve_sensor_offset(twin), np.eye(4))


# --- the persisted payload -------------------------------------------------


def test_set_writes_the_converted_offset_to_the_attach_fields() -> None:
    twin = FakeTwin(sensors=[_sensor()])
    expected_position, expected_rotation = matrix_to_pose(
        optical_to_attach_offset(SOLVED_OPTICAL, np.eye(4))
    )

    CameraCalibrationHandle(twin).set(_result())

    (twin_id, payload) = twin.client.twins.updates[0]
    assert twin_id == CAM_UUID
    for axis in ("x", "y", "z"):
        assert payload[f"attach_offset_{axis}"] == pytest.approx(expected_position[axis])
    for component in ("w", "x", "y", "z"):
        assert payload[f"attach_offset_rotation_{component}"] == pytest.approx(
            expected_rotation[component]
        )


def test_persisted_quaternion_is_w_first_not_xyzw() -> None:
    """Guards the ordering trap: DB fields are (w,x,y,z), Scene.dock takes [x,y,z,w]."""
    # A 90-degree turn about Z: w == x == 0 is false, and w != z, so a w/z swap shows.
    transform = make_transform(_rotation("z", 90.0), [0.0, 0.0, 0.0])
    twin = FakeTwin()

    CameraCalibrationHandle(twin).set(_result(transform))

    (_, payload) = twin.client.twins.updates[0]
    persisted = [payload[f"attach_offset_rotation_{c}"] for c in "wxyz"]
    expected = optical_to_attach_offset(transform, np.eye(4))
    assert np.allclose(quat_wxyz_to_matrix(persisted), expected[:3, :3], atol=1e-12)


def test_the_persisted_offset_round_trips_back_to_the_solved_transform() -> None:
    """End-to-end: read the payload back and reconstruct the optical transform."""
    sensor_offset_pose = {
        "position": {"x": 0.01, "y": -0.005, "z": 0.02},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    twin = FakeTwin(sensors=[_sensor(sensor_offset_pose)])

    CameraCalibrationHandle(twin).set(_result())

    (_, payload) = twin.client.twins.updates[0]
    stored = pose_to_matrix(
        {a: payload[f"attach_offset_{a}"] for a in "xyz"},
        {c: payload[f"attach_offset_rotation_{c}"] for c in "wxyz"},
    )
    sensor_offset = pose_to_matrix(
        sensor_offset_pose["position"], sensor_offset_pose["rotation"]
    )
    assert np.allclose(
        stored @ sensor_offset @ OPTICAL_TO_SENSOR, SOLVED_OPTICAL, atol=1e-9
    )


def test_set_leaves_the_world_pose_alone() -> None:
    """`position_*`/`rotation_*` are the undock fallback, not a mirror of the offset.

    A docked twin's placement is read from `attach_offset_*`; the world pose fields
    still hold where the twin stood before it was docked. Writing the link-relative
    offset over them strands the camera at the world origin on undock.
    """
    twin = FakeTwin()

    CameraCalibrationHandle(twin).set(_result())

    (_, payload) = twin.client.twins.updates[0]
    for axis in ("x", "y", "z"):
        assert f"position_{axis}" not in payload
    for component in ("w", "x", "y", "z"):
        assert f"rotation_{component}" not in payload
    # The offset itself is still written -- this test must not pass by the
    # payload simply being empty.
    for axis in ("x", "y", "z"):
        assert f"attach_offset_{axis}" in payload
    for component in ("w", "x", "y", "z"):
        assert f"attach_offset_rotation_{component}" in payload


def test_set_refreshes_the_twin_so_later_reads_see_the_new_offset() -> None:
    twin = FakeTwin()

    CameraCalibrationHandle(twin).set(_result())

    assert twin.refreshed == 1


def test_set_rejects_anything_that_is_not_a_result() -> None:
    twin = FakeTwin()

    with pytest.raises(CyberwaveValidationError, match="HandEyeResult"):
        CameraCalibrationHandle(twin).set(np.eye(4))


# --- provenance ------------------------------------------------------------


def test_provenance_uses_its_own_metadata_key() -> None:
    """Must not be 'calibration' — the twin endpoint writes joint calibration there."""
    twin = FakeTwin()

    CameraCalibrationHandle(twin).set(_result())

    (_, payload) = twin.client.twins.updates[0]
    assert HAND_EYE_METADATA_KEY == "hand_eye_calibration"
    assert set(payload["metadata"]) == {HAND_EYE_METADATA_KEY}
    assert "calibration" not in payload["metadata"]


def test_provenance_carries_the_residuals_and_solver() -> None:
    twin = FakeTwin()

    record = CameraCalibrationHandle(twin).set(_result())

    assert record["method"] == "park"
    assert record["sample_count"] == 12
    assert record["residual_translation_m"] == pytest.approx(0.0012)


def test_provenance_records_board_intrinsics_and_distortion() -> None:
    from cyberwave.calibration.board import CharucoBoard

    twin = FakeTwin()
    board = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)
    intrinsics = {"fx": 615.0, "fy": 615.0, "cx": 319.5, "cy": 239.5}

    record = CameraCalibrationHandle(twin).set(
        _result(),
        board=board,
        intrinsics=intrinsics,
        dist_coeffs=[0.1, -0.2, 0.0, 0.0, 0.0],
    )

    assert record["board"]["type"] == "charuco"
    assert record["board"]["square_size_m"] == 0.03
    assert record["intrinsics"] == intrinsics
    assert record["dist_coeffs"] == [0.1, -0.2, 0.0, 0.0, 0.0]


def test_provenance_defaults_the_fk_frame_to_the_attach_link() -> None:
    twin = FakeTwin(attach_to_link=LINK)

    record = CameraCalibrationHandle(twin).set(_result())

    assert record["fk_frame"] == LINK


def test_provenance_notes_whether_a_sensor_offset_was_divided_out() -> None:
    plain = FakeTwin(sensors=[_sensor()])
    shifted = FakeTwin(
        sensors=[_sensor({"position": {"x": 0.02, "y": 0, "z": 0}, "rotation": {"w": 1}})]
    )

    assert (
        CameraCalibrationHandle(plain).set(_result())["sensor_offset_applied"] is False
    )
    assert (
        CameraCalibrationHandle(shifted).set(_result())["sensor_offset_applied"] is True
    )


def test_provenance_is_json_safe() -> None:
    import json

    from cyberwave.calibration.board import CheckerBoard

    twin = FakeTwin()

    record = CameraCalibrationHandle(twin).set(
        _result(),
        board=CheckerBoard(inner_corners=(9, 6), square_size_m=0.025),
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0},
        dist_coeffs=np.zeros(5),
        solved_at="2026-08-05T12:00:00Z",
    )

    assert json.loads(json.dumps(record)) == record
    assert record["solved_at"] == "2026-08-05T12:00:00Z"


# --- get / delete ----------------------------------------------------------


def test_get_returns_none_when_never_calibrated() -> None:
    assert CameraCalibrationHandle(FakeTwin()).get() is None


def test_get_returns_the_stored_record() -> None:
    stored = {"method": "park", "sample_count": 12}
    twin = FakeTwin(metadata={HAND_EYE_METADATA_KEY: stored})

    assert CameraCalibrationHandle(twin).get() == stored


def test_get_ignores_a_non_dict_record() -> None:
    twin = FakeTwin(metadata={HAND_EYE_METADATA_KEY: "corrupt"})

    assert CameraCalibrationHandle(twin).get() is None


def test_delete_clears_only_the_record_and_leaves_the_transform() -> None:
    """Forgetting how a mount was measured must not relocate the camera."""
    twin = FakeTwin(metadata={HAND_EYE_METADATA_KEY: {"method": "park"}})

    CameraCalibrationHandle(twin).delete()

    (_, payload) = twin.client.twins.updates[0]
    # Explicit null is the backend's delete sentinel for a metadata key.
    assert payload == {"metadata": {HAND_EYE_METADATA_KEY: None}}
    assert not any(key.startswith("attach_offset") for key in payload)


def test_repr_reports_whether_the_camera_is_calibrated() -> None:
    plain = CameraCalibrationHandle(FakeTwin())
    done = CameraCalibrationHandle(
        FakeTwin(metadata={HAND_EYE_METADATA_KEY: {"method": "park"}})
    )

    assert "calibrated=False" in repr(plain)
    assert "calibrated=True" in repr(done)
