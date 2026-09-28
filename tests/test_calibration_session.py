"""``HandEyeSession`` — attachment validation, sample collection, and solving.

Uses hand-written twin doubles rather than the real ``Twin``: the session only
touches ``uuid``, ``_data_get`` and ``camera.get_frame``, and constructing a real
Twin needs a REST client. Images are bypassed entirely by passing ``image=`` and
stubbing detection, so these tests exercise the session's own logic — the
detection and solving paths have their own files.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from cyberwave.calibration.board import CharucoBoard
from cyberwave.calibration.frames import (
    invert,
    make_transform,
    quat_wxyz_to_matrix,
)
from cyberwave.calibration.handeye import (
    BoardNotDetectedError,
    HandEyeSession,
    HandEyeSyncError,
)
from cyberwave.exceptions import CyberwaveValidationError

ARM_UUID = "11111111-1111-1111-1111-111111111111"
CAM_UUID = "22222222-2222-2222-2222-222222222222"
LINK = "openarm_left_link7"

BOARD = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)
INTRINSICS = {"fx": 615.0, "fy": 615.0, "cx": 319.5, "cy": 239.5}

# Detection and solving call into OpenCV, which arrives with the ``camera`` /
# ``calibration`` extras. The bare-install self-containment run has no cv2, so
# the tests that reach those paths skip there rather than fail. Tests that stop
# at argument validation stay unmarked -- they are exactly what that run checks.
_requires_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason='capture and solving require OpenCV: pip install "cyberwave[camera]"',
)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    quaternion = [math.cos(half), 0.0, 0.0, 0.0]
    quaternion[{"x": 1, "y": 2, "z": 3}[axis]] = math.sin(half)
    return quat_wxyz_to_matrix(quaternion)


TRUE_GRIPPER_TO_CAMERA = make_transform(_rotation("y", -30.0), [0.045, 0.0, 0.099])
BASE_TO_TARGET = make_transform(_rotation("z", 12.0), [0.40, -0.05, 0.02])


# --- doubles ---------------------------------------------------------------


class FakeCameraHandle:
    def __init__(self, frames: list | None = None) -> None:
        self.frames = frames if frames is not None else []
        self.calls: list[dict] = []

    def get_frame(self, format="bytes", **kwargs):  # noqa: A002 - mirrors the real API
        self.calls.append({"format": format, **kwargs})
        return self.frames.pop(0) if self.frames else None


class FakeTwin:
    """Minimal stand-in exposing only what HandEyeSession reads."""

    def __init__(self, uuid: str, data: dict | None = None, frames=None) -> None:
        self.uuid = uuid
        self._data = data or {}
        self.camera = FakeCameraHandle(frames)

    def _data_get(self, field: str, default=None):
        return self._data.get(field, default)


def _docked_camera(frames=None, **overrides) -> FakeTwin:
    data = {"attach_to_twin_uuid": ARM_UUID, "attach_to_link": LINK}
    data.update(overrides)
    return FakeTwin(CAM_UUID, data, frames)


class FakeKinematics:
    """Duck-types ``BaseKinematicsManipulator.fk``."""

    def __init__(self, poses: dict[float, np.ndarray]) -> None:
        self.poses = poses
        self.calls: list[dict] = []

    def fk(self, positions: dict[str, float]) -> np.ndarray:
        self.calls.append(dict(positions))
        return self.poses[positions["j0"]]


def _session(**overrides) -> HandEyeSession:
    kwargs = {
        "camera_twin": _docked_camera(),
        "arm_twin": FakeTwin(ARM_UUID),
        "fk_frame": LINK,
        "board": BOARD,
        "intrinsics": INTRINSICS,
    }
    kwargs.update(overrides)
    return HandEyeSession(**kwargs)


def _gripper_poses() -> list[np.ndarray]:
    return [
        make_transform(np.eye(3), [0.20, 0.00, 0.30]),
        make_transform(_rotation("x", 35.0), [0.22, 0.03, 0.28]),
        make_transform(_rotation("z", 40.0), [0.18, -0.04, 0.31]),
        make_transform(_rotation("y", 30.0), [0.21, 0.02, 0.29]),
    ]


def _observation(base_to_gripper: np.ndarray) -> np.ndarray:
    return invert(base_to_gripper @ TRUE_GRIPPER_TO_CAMERA) @ BASE_TO_TARGET


@pytest.fixture
def detect_ok(monkeypatch):
    """Stub board detection so the session's own logic is what is under test.

    These tests pass a 4x4 transform *as* the image and expect it back as the
    board pose. Detection now happens in ``add_sample`` via the board's ``detect``,
    and posing via ``solve_target_pose``, so both are stubbed: the correspondences
    just have to be numerous enough to pass the capture gate, and the pose comes
    from the "image".
    """
    calls: list = []

    def _detect(self, cv2, image, *, minimum=6):
        calls.append(image)
        usable = isinstance(image, np.ndarray) and image.shape == (4, 4)
        if not usable:
            return None
        # Enough corners and markers to clear both capture gates; the point values
        # are never used because _solve_target_pose below ignores them.
        return (np.zeros((8, 3)), np.zeros((8, 2))), 8

    def _solve_target_pose(object_points, image_points, intrinsics, dist_coeffs=None):
        # The session hands the *stub* points through, so recover the pose from
        # the frame the caller supplied instead.
        return calls[-1] if calls else None

    monkeypatch.setattr(CharucoBoard, "detect", _detect, raising=False)
    monkeypatch.setattr(
        "cyberwave.calibration.handeye.solve_target_pose", _solve_target_pose
    )
    return calls


# --- attachment validation -------------------------------------------------


def test_accepts_a_correctly_docked_camera() -> None:
    session = _session()

    assert session.fk_frame == LINK
    assert session.sample_count == 0


def test_rejects_an_undocked_camera() -> None:
    with pytest.raises(CyberwaveValidationError, match="not docked to anything"):
        _session(camera_twin=FakeTwin(CAM_UUID, {}))


def test_rejects_a_camera_docked_to_a_different_twin() -> None:
    other = _docked_camera(attach_to_twin_uuid="99999999-9999-9999-9999-999999999999")

    with pytest.raises(CyberwaveValidationError, match="not to the given arm twin"):
        _session(camera_twin=other)


def test_rejects_an_fk_frame_that_is_not_the_attach_link() -> None:
    """The mismatch that yields a wrong result with a good residual."""
    with pytest.raises(CyberwaveValidationError, match="does not match"):
        _session(fk_frame="openarm_left_link6")


def test_validation_can_be_opted_out_of() -> None:
    session = _session(
        camera_twin=FakeTwin(CAM_UUID, {}), validate_attachment=False
    )

    assert session.fk_frame == LINK


def test_bad_intrinsics_fail_at_construction_not_at_first_capture() -> None:
    with pytest.raises(CyberwaveValidationError, match="missing"):
        _session(intrinsics={"fx": 1.0, "fy": 1.0})


def test_bad_distortion_fails_at_construction() -> None:
    with pytest.raises(CyberwaveValidationError, match="distortion coefficients"):
        _session(dist_coeffs=[0.0, 0.0, 0.0])


# --- gripper pose sourcing -------------------------------------------------


@pytest.fixture
def detect_any_size(monkeypatch):
    """Detection stub that accepts real image shapes, not the (4, 4) convention.

    ``detect_ok`` passes a 4x4 transform *as* the image, so every capture has the
    same shape and cannot exercise the resolution guard. These tests need frames
    of genuinely different sizes, so the pose is fixed rather than recovered.
    """
    pose = np.eye(4)

    def _detect(self, cv2, image, *, minimum=6):
        return (np.zeros((8, 3)), np.zeros((8, 2))), 8

    monkeypatch.setattr(CharucoBoard, "detect", _detect, raising=False)
    monkeypatch.setattr(
        "cyberwave.calibration.handeye.solve_target_pose",
        lambda *a, **k: pose,
    )


@_requires_cv2
def test_add_sample_refuses_a_mid_session_resolution_change(detect_any_size) -> None:
    """One camera model is fitted and posed across every capture in a session.

    A loaded edge device dropping to a smaller frame mid-run would otherwise be
    accepted silently and measured as if it were still the original size -- a
    confident, meaningless result with nothing pointing at the cause. The public
    docs promise this is rejected, and HandEyeFlow / solve_intrinsics both
    already refuse it; the session is the entry point an SDK caller reaches.
    """
    session = _session()
    poses = _gripper_poses()
    session.add_sample(
        base_to_gripper=poses[1], image=np.zeros((720, 1280, 3), np.uint8)
    )

    with pytest.raises(CyberwaveValidationError, match="per-resolution"):
        session.add_sample(
            base_to_gripper=poses[2], image=np.zeros((360, 640, 3), np.uint8)
        )

    assert session.sample_count == 1, "the rejected capture must not be stored"


@_requires_cv2
def test_add_sample_accepts_repeated_captures_at_one_resolution(
    detect_any_size,
) -> None:
    """The guard must not fire on the normal case."""
    session = _session()

    for pose in _gripper_poses()[1:4]:
        session.add_sample(
            base_to_gripper=pose, image=np.zeros((720, 1280, 3), np.uint8)
        )

    assert session.sample_count == 3


@_requires_cv2
def test_add_sample_stores_the_explicit_pose_and_detection(detect_ok) -> None:
    session = _session()
    pose = _gripper_poses()[1]
    observation = _observation(pose)

    sample = session.add_sample(base_to_gripper=pose, image=observation)

    assert session.sample_count == 1
    assert np.allclose(sample.base_to_gripper, pose)
    assert np.allclose(sample.camera_to_target, observation)


@_requires_cv2
def test_add_sample_runs_forward_kinematics_for_joint_positions(detect_ok) -> None:
    pose = _gripper_poses()[2]
    kinematics = FakeKinematics({0.5: pose})
    session = _session(kinematics=kinematics)

    sample = session.add_sample(
        joint_positions={"j0": 0.5}, image=_observation(pose)
    )

    assert kinematics.calls == [{"j0": 0.5}]
    assert np.allclose(sample.base_to_gripper, pose)


def test_joint_positions_without_kinematics_is_rejected(detect_ok) -> None:
    session = _session()

    with pytest.raises(CyberwaveValidationError, match="no kinematics"):
        session.add_sample(joint_positions={"j0": 0.5}, image=np.eye(4))


def test_no_gripper_pose_at_all_is_rejected(detect_ok) -> None:
    session = _session()

    with pytest.raises(CyberwaveValidationError, match="needs the gripper pose"):
        session.add_sample(image=np.eye(4))


def test_supplying_both_pose_sources_is_rejected(detect_ok) -> None:
    session = _session(kinematics=FakeKinematics({}))

    with pytest.raises(CyberwaveValidationError, match="not both"):
        session.add_sample(
            base_to_gripper=np.eye(4), joint_positions={"j0": 0.0}, image=np.eye(4)
        )


def test_a_malformed_gripper_pose_is_rejected(detect_ok) -> None:
    session = _session()

    with pytest.raises(CyberwaveValidationError, match="base_to_gripper"):
        session.add_sample(base_to_gripper=np.eye(3), image=np.eye(4))


# --- image capture ---------------------------------------------------------


@_requires_cv2
def test_captures_from_the_camera_twin_when_no_image_is_given(detect_ok) -> None:
    pose = _gripper_poses()[0]
    camera = _docked_camera(frames=[_observation(pose)])
    session = _session(camera_twin=camera)

    session.add_sample(base_to_gripper=pose)

    assert camera.camera.calls == [{"format": "numpy", "source": "cloud"}]


@_requires_cv2
def test_frame_source_is_forwarded(detect_ok) -> None:
    pose = _gripper_poses()[0]
    camera = _docked_camera(frames=[_observation(pose)])
    session = _session(camera_twin=camera, frame_source="zenoh")

    session.add_sample(base_to_gripper=pose)

    assert camera.camera.calls[0]["source"] == "zenoh"


def test_an_unavailable_frame_raises_rather_than_passing_none_downstream(
    detect_ok,
) -> None:
    session = _session(camera_twin=_docked_camera(frames=[]))

    with pytest.raises(CyberwaveValidationError, match="No frame available"):
        session.add_sample(base_to_gripper=_gripper_poses()[0])


# --- detection failure is fatal -------------------------------------------


@_requires_cv2
def test_an_undetected_board_raises_and_does_not_store_a_sample(detect_ok) -> None:
    """Silently skipping captures is how a session ends up under-constrained."""
    session = _session()

    with pytest.raises(BoardNotDetectedError, match="capture 1"):
        session.add_sample(base_to_gripper=_gripper_poses()[0], image="not-a-pose")
    assert session.sample_count == 0


@_requires_cv2
def test_the_capture_number_in_the_error_counts_from_one(detect_ok) -> None:
    session = _session()
    pose = _gripper_poses()[0]
    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    with pytest.raises(BoardNotDetectedError, match="capture 2"):
        session.add_sample(base_to_gripper=pose, image="not-a-pose")


# --- solve ----------------------------------------------------------------


def test_solve_recovers_the_mount_transform(detect_ok) -> None:
    pytest.importorskip("cv2", reason="solving requires OpenCV")
    session = _session()
    for pose in _gripper_poses():
        session.add_sample(base_to_gripper=pose, image=_observation(pose))

    result = session.solve()

    assert result.sample_count == 4
    assert np.allclose(
        result.gripper_to_camera[:3, 3], TRUE_GRIPPER_TO_CAMERA[:3, 3], atol=1e-6
    )
    assert result.residual_translation_m < 1e-9


def test_solve_below_the_minimum_sample_count_is_rejected(detect_ok) -> None:
    pytest.importorskip("cv2", reason="solving requires OpenCV")
    session = _session()
    pose = _gripper_poses()[0]
    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    with pytest.raises(CyberwaveValidationError, match="at least 3 samples"):
        session.solve()


@_requires_cv2
def test_clear_drops_collected_samples(detect_ok) -> None:
    session = _session()
    for pose in _gripper_poses():
        session.add_sample(base_to_gripper=pose, image=_observation(pose))

    session.clear()

    assert session.sample_count == 0
    assert session.samples == ()


@_requires_cv2
def test_samples_is_an_immutable_snapshot(detect_ok) -> None:
    session = _session()
    pose = _gripper_poses()[0]
    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    snapshot = session.samples
    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1
    assert session.sample_count == 2


def test_repr_shows_the_frame_and_sample_count(detect_ok) -> None:
    session = _session()

    assert LINK in repr(session)
    assert "samples=0" in repr(session)


# --- capture synchronisation ----------------------------------------------


class FakeJointArm(FakeTwin):
    """Arm twin whose joints report a scripted sequence of readings.

    Each ``get_joints()`` returns the next reading, so a test scripts exactly what
    the arm was doing across the settle poll and the capture.
    """

    def __init__(self, readings: list[dict[str, float]]) -> None:
        super().__init__(ARM_UUID)
        self.readings = list(readings)
        self.reads = 0

    def get_joints(self, **kwargs):
        self.reads += 1
        # Hold the last reading once the script runs out: a settled arm keeps
        # reporting the same position indefinitely.
        index = min(self.reads - 1, len(self.readings) - 1)
        return dict(self.readings[index])


def _still(value: float = 0.5, count: int = 4) -> list[dict[str, float]]:
    return [{"j0": value}] * count


@_requires_cv2
def test_a_settled_arm_captures_normally(detect_ok, monkeypatch) -> None:
    monkeypatch.setattr("cyberwave.calibration.handeye.time.sleep", lambda _s: None)
    arm = FakeJointArm(_still())
    pose = _gripper_poses()[0]
    session = _session(
        arm_twin=arm, camera_twin=_docked_camera(frames=[_observation(pose)])
    )

    session.add_sample(base_to_gripper=pose)

    assert session.sample_count == 1


def test_an_arm_that_never_settles_is_rejected(detect_ok, monkeypatch) -> None:
    """Timing out waiting for the arm to stop must not silently capture anyway."""
    monkeypatch.setattr("cyberwave.calibration.handeye.time.sleep", lambda _s: None)
    arm = FakeJointArm([{"j0": 0.5 + 0.1 * i} for i in range(200)])
    pose = _gripper_poses()[0]
    session = _session(
        arm_twin=arm, camera_twin=_docked_camera(frames=[_observation(pose)])
    )

    with pytest.raises(HandEyeSyncError, match="still moving"):
        session.add_sample(base_to_gripper=pose, settle_timeout_s=0.0)

    assert session.sample_count == 0


def test_an_arm_that_moves_during_capture_is_rejected(detect_ok, monkeypatch) -> None:
    """The post-capture check is the one that catches a mispaired frame.

    Two identical readings settle the arm, then it moves — so the frame that was
    grabbed in between belongs to a different position than the pose.
    """
    monkeypatch.setattr("cyberwave.calibration.handeye.time.sleep", lambda _s: None)
    arm = FakeJointArm([{"j0": 0.5}, {"j0": 0.5}, {"j0": 0.9}])
    pose = _gripper_poses()[0]
    session = _session(
        arm_twin=arm, camera_twin=_docked_camera(frames=[_observation(pose)])
    )

    with pytest.raises(HandEyeSyncError, match="moved"):
        session.add_sample(base_to_gripper=pose)

    # Rejected, not stored: a sample this suspect must not reach the solve.
    assert session.sample_count == 0


@_requires_cv2
def test_settle_can_be_turned_off(detect_ok, monkeypatch) -> None:
    monkeypatch.setattr("cyberwave.calibration.handeye.time.sleep", lambda _s: None)
    arm = FakeJointArm([{"j0": 0.5}, {"j0": 0.5}, {"j0": 0.9}])
    pose = _gripper_poses()[0]
    session = _session(
        arm_twin=arm, camera_twin=_docked_camera(frames=[_observation(pose)])
    )

    session.add_sample(base_to_gripper=pose, settle=False)

    assert session.sample_count == 1
    assert arm.reads == 0


@_requires_cv2
def test_a_supplied_image_skips_the_settle_check(detect_ok, monkeypatch) -> None:
    """We cannot vouch for the timing of an image we did not capture."""
    monkeypatch.setattr("cyberwave.calibration.handeye.time.sleep", lambda _s: None)
    arm = FakeJointArm([{"j0": 0.5}, {"j0": 0.5}, {"j0": 0.9}])
    pose = _gripper_poses()[0]
    session = _session(arm_twin=arm)

    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    assert session.sample_count == 1
    assert arm.reads == 0


@_requires_cv2
def test_an_arm_without_joint_states_still_captures(detect_ok) -> None:
    """The gate is a guard on top of manual capture, not a new requirement."""
    pose = _gripper_poses()[0]
    session = _session(camera_twin=_docked_camera(frames=[_observation(pose)]))

    session.add_sample(base_to_gripper=pose)

    assert session.sample_count == 1


# --- solving intrinsics ----------------------------------------------------


def _board_views(count: int, *, seed: int = 4):
    """Views of a board through a camera with known intrinsics (fx=fy=800)."""
    import cv2

    camera = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1]], float)
    board = CharucoBoard(squares=(8, 6), square_size_m=0.03, marker_size_m=0.022)
    cv_board = board._cv_board(cv2)
    flat = cv_board.generateImage((900, 700))
    height, width = flat.shape[:2]
    board_w, board_h = 8 * 0.03, 6 * 0.03
    corners_m = np.float32(
        [[0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0]]
    )
    corners_px = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    rng = np.random.default_rng(seed)
    views = []
    for _ in range(count):
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array([-board_w / 2, -board_h / 2, 0.55]) + rng.uniform(-0.06, 0.06, 3)
        projected, _ = cv2.projectPoints(corners_m, rvec, tvec, camera, None)
        homography = cv2.getPerspectiveTransform(
            corners_px, projected.reshape(-1, 2).astype(np.float32)
        )
        views.append(cv2.warpPerspective(flat, homography, (1280, 720), borderValue=255))
    return board, views


def _varied_poses(count: int) -> list[np.ndarray]:
    return [
        make_transform(_rotation("x", 15.0 * i), [0.20 + 0.01 * i, 0.01 * i, 0.30])
        for i in range(count)
    ]


def _multi_axis_poses(count: int) -> list[np.ndarray]:
    """Poses rotated about cycling axes, so the hand-eye solve is well posed.

    ``_varied_poses`` turns about X only: every relative motion shares one axis,
    which is exactly the degenerate case ``solve_hand_eye`` refuses. Tests that
    actually reach the solver need genuinely non-parallel axes.
    """
    axes = ("x", "y", "z")
    return [
        make_transform(
            _rotation(axes[i % 3], 20.0 + 12.0 * i),
            [0.20 + 0.01 * i, 0.01 * i, 0.30],
        )
        for i in range(count)
    ]


def test_intrinsics_are_optional() -> None:
    """A session with no intrinsics constructs; it solves them from captures."""
    session = _session(intrinsics=None)

    assert session.intrinsics is None
    assert session.solved_intrinsics is None


@_requires_cv2
def test_solved_intrinsics_are_close_to_the_rendering_camera() -> None:
    """Intrinsics are fitted at solve time, from every capture."""
    board, views = _board_views(10)
    session = _session(intrinsics=None, board=board)

    for pose, view in zip(_multi_axis_poses(10), views):
        session.add_sample(base_to_gripper=pose, image=view)

    # Nothing is fitted during capture: corners are collected, that is all.
    assert session.solved_intrinsics is None
    assert session.intrinsics is None

    session.solve(validate=False)

    solved = session.solved_intrinsics
    assert solved is not None
    assert solved.view_count == 10
    assert solved.fx == pytest.approx(800.0, rel=0.10)
    assert solved.rms_reprojection_px < 1.0


@_requires_cv2
def test_capture_needs_no_intrinsics_and_raises_nothing() -> None:
    """The whole point: capture collects corners, which needs no camera model."""
    board, views = _board_views(8)
    session = _session(intrinsics=None, board=board)

    for index, (pose, view) in enumerate(zip(_multi_axis_poses(8), views), start=1):
        session.add_sample(base_to_gripper=pose, image=view)
        # Every accepted capture counts immediately -- no buffered phase.
        assert session.captured_count == index
        assert session.sample_count == index


@_requires_cv2
def test_a_capture_showing_too_little_of_the_board_is_refused() -> None:
    """Refused on the spot, not banked: the operator can re-aim immediately.

    Storing it instead would advance the progress counter on a view no solve can
    use, which misreports how far along the run actually is.
    """
    board, views = _board_views(6)
    # A threshold nothing can clear, so every capture is below it.
    session = _session(intrinsics=None, board=board, min_correspondences=10_000)

    with pytest.raises(BoardNotDetectedError, match="corner"):
        session.add_sample(base_to_gripper=_multi_axis_poses(1)[0], image=views[0])

    # Nothing was stored, so neither counter moved.
    assert session.captured_count == 0
    assert session.sample_count == 0


@_requires_cv2
def test_captures_below_the_marker_minimum_are_refused() -> None:
    """Markers are a separate floor from corners, so it must gate independently.

    ``min_correspondences`` is left at its default here: the captures clear the
    corner floor comfortably and are refused purely on markers read. If the two
    checks were collapsed into one, this would pass by accident.
    """
    board, views = _board_views(6)
    session = _session(intrinsics=None, board=board, min_markers=10_000)

    with pytest.raises(BoardNotDetectedError, match="marker"):
        session.add_sample(base_to_gripper=_multi_axis_poses(1)[0], image=views[0])

    assert session.captured_count == 0
    assert session.sample_count == 0


@_requires_cv2
def test_the_marker_floor_can_be_turned_off() -> None:
    """``min_markers=0`` disables the check; the corner floor still applies."""
    board, views = _board_views(6)
    session = _session(intrinsics=None, board=board, min_markers=0)

    for pose, view in zip(_multi_axis_poses(6), views):
        session.add_sample(base_to_gripper=pose, image=view)

    assert session.captured_count == 6
    assert session.sample_count == 6


def test_a_markerless_board_is_not_failed_by_the_marker_floor() -> None:
    """A chessboard reports ``-1`` markers, which means "not applicable".

    Comparing that against a threshold as if it were zero would discard every
    capture of a board that was in fact fully found.
    """
    from cyberwave.calibration.handeye import HandEyeSample, HandEyeSession

    class _Gate:
        _min_correspondences = 6
        _min_markers = 6
        _is_solvable = HandEyeSession._is_solvable

    enough_corners = (np.zeros((20, 3)), np.zeros((20, 2)))
    markerless = HandEyeSample(
        base_to_gripper=np.eye(4), correspondences=enough_corners, markers_read=-1
    )
    assert _Gate()._is_solvable(markerless) is True


@_requires_cv2
def test_a_board_that_is_absent_entirely_is_still_refused() -> None:
    """Loosening the gate must not turn "no board" into a stored capture."""
    board, _views = _board_views(1)
    session = _session(intrinsics=None, board=board)

    with pytest.raises(BoardNotDetectedError):
        session.add_sample(
            base_to_gripper=_multi_axis_poses(1)[0],
            image=np.zeros((200, 200, 3), dtype=np.uint8),
        )

    assert session.captured_count == 0


@_requires_cv2
def test_solve_poses_every_capture_with_the_fitted_intrinsics() -> None:
    """No capture is left unposed once the camera model exists."""
    board, views = _board_views(9)
    session = _session(intrinsics=None, board=board)
    for pose, view in zip(_multi_axis_poses(9), views):
        session.add_sample(base_to_gripper=pose, image=view)

    assert session.pending_count == 9  # geometry not computed yet

    session.solve(validate=False)

    assert session.pending_count == 0
    assert all(sample.is_paired for sample in session.samples)


@_requires_cv2
def test_clearing_drops_every_capture() -> None:
    board, views = _board_views(4)
    session = _session(intrinsics=None, board=board)
    for pose, view in zip(_multi_axis_poses(4), views):
        session.add_sample(base_to_gripper=pose, image=view)
    assert session.captured_count == 4

    session.clear()

    assert session.captured_count == 0
    assert session.sample_count == 0
    assert session.pending_count == 0


@_requires_cv2
def test_supplied_intrinsics_are_never_overwritten(detect_ok) -> None:
    """Passing intrinsics keeps the old behaviour: no buffering, no solving."""
    pose = _gripper_poses()[0]
    session = _session()

    session.add_sample(base_to_gripper=pose, image=_observation(pose))

    assert session.sample_count == 1
    assert session.intrinsics == INTRINSICS
    assert session.solved_intrinsics is None


@_requires_cv2
def test_the_result_carries_the_intrinsics_it_was_solved_with() -> None:
    """Provenance must say the intrinsics were solved, not assumed."""
    board, views = _board_views(10)
    session = _session(intrinsics=None, board=board)
    poses = [
        make_transform(
            _rotation("x", 18.0 * i) @ _rotation("y", 9.0 * i),
            [0.20 + 0.012 * i, 0.01 * i, 0.30],
        )
        for i in range(10)
    ]

    for pose, view in zip(poses, views):
        session.add_sample(base_to_gripper=pose, image=view)

    result = session.solve()

    assert result.solved_intrinsics is session.solved_intrinsics
    assert result.to_metadata()["solved_intrinsics"]["source"] == "solved"


@_requires_cv2
def test_supplied_intrinsics_leave_the_result_unmarked(detect_ok) -> None:
    """Nothing was solved, so the result must not claim otherwise."""
    session = _session()
    for pose in _gripper_poses():
        session.add_sample(base_to_gripper=pose, image=_observation(pose))

    result = session.solve(validate=False)

    assert result.solved_intrinsics is None
    assert "solved_intrinsics" not in result.to_metadata()


# --- refitting intrinsics from every view ----------------------------------


@_requires_cv2
def test_supplied_intrinsics_are_not_refitted(detect_ok) -> None:
    """Caller-supplied intrinsics are theirs to own — never silently replaced."""
    session = _session()
    for pose in _gripper_poses():
        session.add_sample(base_to_gripper=pose, image=_observation(pose))

    session.solve(validate=False)

    assert session.solved_intrinsics is None
    assert session.intrinsics == INTRINSICS


@_requires_cv2
def test_frames_are_not_retained_when_intrinsics_are_supplied(detect_ok) -> None:
    """Nothing to refit, so holding every frame for the run would be waste."""
    session = _session()
    for pose in _gripper_poses():
        session.add_sample(base_to_gripper=pose, image=_observation(pose))

    assert all(sample.frame is None for sample in session.samples)


# --- capture progress ------------------------------------------------------


@_requires_cv2
def test_solve_uses_park_and_validates_with_stability() -> None:
    """One solver produces the answer, and stability is what validates it.

    Measured over fourteen real runs the same two of the five solvers were flagged
    as outliers *every* time, including on runs that proved reproducible, so
    cross-method agreement was a constant signal carrying no information about a
    given run. Leave-one-out stability discriminated where it did not.
    """
    board, views = _board_views(8)
    session = _session(intrinsics=None, board=board)
    for pose, view in zip(_multi_axis_poses(8), views):
        session.add_sample(base_to_gripper=pose, image=view)

    result = session.solve(validate=True)

    assert result.method == "park"
    assert result.leave_one_out is not None
