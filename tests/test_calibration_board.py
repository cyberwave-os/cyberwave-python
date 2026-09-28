"""Calibration-board specs and target-pose estimation.

The load-bearing test here is
``test_solve_target_pose_recovers_the_pose_used_to_project``: it pins the direction
convention (``solvePnP`` returns *camera_to_target*, no inversion) by projecting a
known pose and solving it back. A sign or inversion error there would flow straight
into the hand-eye solve and produce a plausible, wrong mount transform.

That test uses only ``cv2.projectPoints`` / ``cv2.solvePnP``, whose API is stable
across OpenCV versions. The ChArUco *detection* tests need
``cv2.aruco.CharucoDetector`` (OpenCV >= 4.7) and skip on older builds.
"""

from __future__ import annotations

import json
import math

from dataclasses import replace

import numpy as np
import pytest

from cyberwave.calibration.board import (
    MIN_INTRINSICS_VIEWS,
    MIN_CORRESPONDENCES,
    CharucoBoard,
    CheckerBoard,
    camera_matrix,
    detect_target_pose,
    distortion_vector,
    resolve_dictionary,
    solve_target_pose,
)
from cyberwave.calibration.frames import (
    invert,
    make_transform,
    quat_wxyz_to_matrix,
    rotation_angle_deg,
)
from cyberwave.calibration.board import solve_intrinsics
from cyberwave.exceptions import CyberwaveValidationError

cv2 = pytest.importorskip("cv2", reason="board detection requires OpenCV")

_HAS_CHARUCO_DETECTOR = hasattr(cv2.aruco, "CharucoDetector")
_requires_charuco = pytest.mark.skipif(
    not _HAS_CHARUCO_DETECTOR,
    reason="ChArUco detection requires OpenCV >= 4.7 (cv2.aruco.CharucoDetector)",
)

INTRINSICS = {"fx": 615.0, "fy": 617.0, "cx": 319.5, "cy": 239.5}


def _rotation(axis: str, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    quaternion = [math.cos(half), 0.0, 0.0, 0.0]
    quaternion[{"x": 1, "y": 2, "z": 3}[axis]] = math.sin(half)
    return quat_wxyz_to_matrix(quaternion)


# --- the direction convention ---------------------------------------------


@pytest.mark.parametrize(
    "camera_to_target",
    [
        make_transform(np.eye(3), [0.0, 0.0, 0.5]),
        make_transform(_rotation("x", 20.0), [0.05, -0.03, 0.42]),
        make_transform(_rotation("y", -25.0), [-0.06, 0.02, 0.60]),
        make_transform(_rotation("z", 35.0) @ _rotation("x", 15.0), [0.01, 0.01, 0.55]),
    ],
)
def test_solve_target_pose_recovers_the_pose_used_to_project(camera_to_target) -> None:
    """``solve_target_pose`` must return camera_to_target, not its inverse."""
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)
    object_points = board.object_points()
    rvec, _ = cv2.Rodrigues(camera_to_target[:3, :3])
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        camera_to_target[:3, 3],
        camera_matrix(INTRINSICS),
        distortion_vector(None),
    )

    solved = solve_target_pose(object_points, image_points, INTRINSICS)

    assert solved is not None
    assert np.allclose(solved[:3, 3], camera_to_target[:3, 3], atol=1e-6)
    assert rotation_angle_deg(camera_to_target[:3, :3].T @ solved[:3, :3]) < 1e-3


def test_solved_pose_is_not_the_inverse() -> None:
    """Explicitly rules out the off-by-an-inversion mistake."""
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)
    object_points = board.object_points()
    camera_to_target = make_transform(_rotation("x", 20.0), [0.05, -0.03, 0.42])
    rvec, _ = cv2.Rodrigues(camera_to_target[:3, :3])
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        camera_to_target[:3, 3],
        camera_matrix(INTRINSICS),
        distortion_vector(None),
    )

    solved = solve_target_pose(object_points, image_points, INTRINSICS)

    assert not np.allclose(solved, invert(camera_to_target), atol=1e-3)


def test_board_scale_error_scales_the_solved_translation() -> None:
    """A mis-measured square size shows up as a proportional range error.

    This is why the docstrings insist on calipers: nothing downstream can detect it,
    because the wrong scale is perfectly self-consistent.
    """
    truth = make_transform(np.eye(3), [0.0, 0.0, 0.500])
    correct = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)
    rvec, _ = cv2.Rodrigues(truth[:3, :3])
    image_points, _ = cv2.projectPoints(
        correct.object_points(),
        rvec,
        truth[:3, 3],
        camera_matrix(INTRINSICS),
        distortion_vector(None),
    )

    overstated = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025 * 1.02)
    solved = solve_target_pose(overstated.object_points(), image_points, INTRINSICS)

    assert solved[2, 3] == pytest.approx(0.500 * 1.02, rel=1e-3)


def test_solve_target_pose_rejects_mismatched_counts() -> None:
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)
    points = board.object_points()

    with pytest.raises(CyberwaveValidationError, match="mismatch"):
        solve_target_pose(points, np.zeros((len(points) - 1, 2)), INTRINSICS)


def test_solve_target_pose_returns_none_below_minimum_correspondences() -> None:
    few = MIN_CORRESPONDENCES - 1

    assert solve_target_pose(np.zeros((few, 3)), np.zeros((few, 2)), INTRINSICS) is None


# --- intrinsics / distortion helpers --------------------------------------


def test_camera_matrix_from_dict() -> None:
    assert np.allclose(
        camera_matrix(INTRINSICS),
        [[615.0, 0.0, 319.5], [0.0, 617.0, 239.5], [0.0, 0.0, 1.0]],
    )


def test_camera_matrix_passes_through_a_3x3() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]])

    assert np.allclose(camera_matrix(matrix), matrix)


@pytest.mark.parametrize(
    ("intrinsics", "match"),
    [
        ({"fx": 1.0, "fy": 1.0}, "missing"),
        ({"fx": 0.0, "fy": 1.0, "cx": 1.0, "cy": 1.0}, "positive"),
        ({"fx": -5.0, "fy": 1.0, "cx": 1.0, "cy": 1.0}, "positive"),
        (np.eye(4), "3x3"),
    ],
)
def test_camera_matrix_rejects_bad_intrinsics(intrinsics, match: str) -> None:
    with pytest.raises(CyberwaveValidationError, match=match):
        camera_matrix(intrinsics)


def test_distortion_defaults_to_zero() -> None:
    assert np.allclose(distortion_vector(None), np.zeros((1, 5)))


@pytest.mark.parametrize("count", [4, 5, 8, 12, 14])
def test_distortion_accepts_opencv_lengths(count: int) -> None:
    assert distortion_vector([0.0] * count).shape == (1, count)


def test_distortion_rejects_odd_lengths() -> None:
    with pytest.raises(CyberwaveValidationError, match="distortion coefficients"):
        distortion_vector([0.0, 0.0, 0.0])


# --- board specs -----------------------------------------------------------


def test_charuco_board_normalizes_squares_to_a_tuple() -> None:
    board = CharucoBoard(squares=[5, 7], square_size_m=0.03, marker_size_m=0.022)

    assert board.squares == (5, 7)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"squares": (5, 7), "square_size_m": 0.0, "marker_size_m": 0.02}, "square_size_m"),
        ({"squares": (5, 7), "square_size_m": 0.03, "marker_size_m": 0.0}, "marker_size_m"),
        # A marker at least as large as its square cannot physically fit.
        ({"squares": (5, 7), "square_size_m": 0.03, "marker_size_m": 0.03}, "smaller"),
        ({"squares": (5, 7), "square_size_m": 0.03, "marker_size_m": 0.05}, "smaller"),
        ({"squares": (1, 7), "square_size_m": 0.03, "marker_size_m": 0.02}, "at least 2"),
        ({"squares": (5,), "square_size_m": 0.03, "marker_size_m": 0.02}, "columns, rows"),
    ],
)
def test_charuco_board_validates_its_geometry(kwargs, match: str) -> None:
    with pytest.raises(CyberwaveValidationError, match=match):
        CharucoBoard(**kwargs)


def test_checkerboard_object_points_form_the_expected_grid() -> None:
    board = CheckerBoard(inner_corners=(4, 3), square_size_m=0.05)

    points = board.object_points()

    assert points.shape == (12, 3)
    assert np.allclose(points[:, 2], 0.0)  # planar, Z = 0
    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    # Spacing equals the square size in both directions.
    assert np.isclose(np.max(points[:, 0]), 3 * 0.05)
    assert np.isclose(np.max(points[:, 1]), 2 * 0.05)


def test_checkerboard_rejects_a_rotationally_ambiguous_square_grid() -> None:
    with pytest.raises(CyberwaveValidationError, match="ambiguous"):
        CheckerBoard(inner_corners=(6, 6), square_size_m=0.025)


def test_checkerboard_rejects_nonpositive_square_size() -> None:
    with pytest.raises(CyberwaveValidationError, match="square_size_m"):
        CheckerBoard(inner_corners=(9, 6), square_size_m=-0.01)


def test_board_metadata_is_json_safe() -> None:
    import json

    charuco = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)
    checker = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)

    for board in (charuco, checker):
        metadata = board.to_metadata()
        assert json.loads(json.dumps(metadata)) == metadata
    assert charuco.to_metadata()["type"] == "charuco"
    assert checker.to_metadata()["type"] == "checkerboard"


# --- aruco dictionary resolution ------------------------------------------


@pytest.mark.parametrize(
    "name", ["DICT_5X5_1000", "dict_5x5_1000", " DICT_4X4_50 ", "tag36h11"]
)
def test_resolve_dictionary_accepts_names_and_aliases(name: str) -> None:
    assert resolve_dictionary(cv2, name) is not None


@pytest.mark.parametrize("name", ["DICT_NOPE", "chessboard", "", "COLOR_BGR2GRAY"])
def test_resolve_dictionary_rejects_unknown_names(name: str) -> None:
    with pytest.raises(CyberwaveValidationError, match="ArUco dictionary"):
        resolve_dictionary(cv2, name)


# --- board image generation ------------------------------------------------

_HAS_GENERATE_IMAGE = hasattr(cv2.aruco.CharucoBoard, "generateImage")


@pytest.mark.skipif(
    not _HAS_GENERATE_IMAGE,
    reason="CharucoBoard.generateImage requires OpenCV >= 4.7",
)
def test_generate_image_returns_a_printable_array() -> None:
    board = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)

    image = board.generate_image(pixels_per_square=50)

    assert image.shape == (7 * 50, 5 * 50)


@pytest.mark.skipif(
    _HAS_GENERATE_IMAGE,
    reason="only meaningful on an OpenCV build without CharucoBoard.generateImage",
)
def test_generate_image_raises_a_helpful_error_on_old_opencv() -> None:
    """Regression: this used to be a raw AttributeError from inside cv2."""
    board = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)

    with pytest.raises(ImportError, match="OpenCV >= 4.7"):
        board.generate_image()


# --- end-to-end detection -------------------------------------------------


@_requires_charuco
def test_detects_a_rendered_charuco_board_head_on() -> None:
    """Render the board, treat the render as a fronto-parallel view, recover its pose."""
    board = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)
    pixels_per_square = 100
    image = board.generate_image(pixels_per_square)
    # A synthetic camera whose focal length makes the rendered image exactly a
    # fronto-parallel view at 1 m: f = pixels_per_metre * distance.
    focal = (pixels_per_square / 0.03) * 1.0
    height, width = image.shape[:2]
    intrinsics = {
        "fx": focal,
        "fy": focal,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }

    pose = detect_target_pose(image, board, intrinsics)

    assert pose is not None
    assert pose[2, 3] == pytest.approx(1.0, rel=0.02)
    assert rotation_angle_deg(pose[:3, :3]) < 2.0


@_requires_charuco
def test_returns_none_when_the_board_is_absent() -> None:
    board = CharucoBoard(squares=(5, 7), square_size_m=0.03, marker_size_m=0.022)
    blank = np.full((480, 640), 255, dtype=np.uint8)

    assert detect_target_pose(blank, board, INTRINSICS) is None


@_requires_charuco
def test_returns_none_on_a_dictionary_mismatch() -> None:
    """Wrong dictionary means zero detections, not a wrong pose."""
    rendered = CharucoBoard(
        squares=(5, 7), square_size_m=0.03, marker_size_m=0.022, dictionary="DICT_4X4_50"
    ).generate_image(100)
    mismatched = CharucoBoard(
        squares=(5, 7), square_size_m=0.03, marker_size_m=0.022, dictionary="DICT_6X6_250"
    )

    assert detect_target_pose(rendered, mismatched, INTRINSICS) is None


def test_returns_none_when_no_checkerboard_is_present() -> None:
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)
    noise = np.zeros((240, 320), dtype=np.uint8)

    assert detect_target_pose(noise, board, INTRINSICS) is None


def test_rejects_an_image_with_an_unusable_shape() -> None:
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)

    with pytest.raises(CyberwaveValidationError, match="grayscale"):
        detect_target_pose(np.zeros((4, 4, 2), dtype=np.uint8), board, INTRINSICS)


@pytest.mark.parametrize("shape", [(240, 320), (240, 320, 3), (240, 320, 4)])
def test_accepts_gray_bgr_and_bgra_images(shape) -> None:
    board = CheckerBoard(inner_corners=(9, 6), square_size_m=0.025)

    # No board present, so None — the point is that the shape is accepted at all.
    assert detect_target_pose(np.zeros(shape, dtype=np.uint8), board, INTRINSICS) is None


# --- intrinsics ------------------------------------------------------------

#: Ground truth for the rendered views below.
_TRUE_FX = _TRUE_FY = 800.0
_TRUE_CX, _TRUE_CY = 640.0, 360.0
_IMAGE_SIZE = (1280, 720)
_INTRINSICS_BOARD = CharucoBoard(
    squares=(8, 6), square_size_m=0.03, marker_size_m=0.022
)


def _render_board_views(count: int, *, seed: int = 4, size=_IMAGE_SIZE):
    """Views of the board seen through a camera with known intrinsics.

    The flat board image is warped by the homography that its four corners project
    to under a real pinhole model, so the result is a geometrically consistent
    view rather than an arbitrary distortion.
    """
    camera = np.array(
        [[_TRUE_FX, 0, _TRUE_CX], [0, _TRUE_FY, _TRUE_CY], [0, 0, 1]], float
    )
    cv_board = _INTRINSICS_BOARD._cv_board(cv2)
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
        views.append(cv2.warpPerspective(flat, homography, size, borderValue=255))
    return views


def test_solve_intrinsics_recovers_the_camera_it_was_rendered_with() -> None:
    result = solve_intrinsics(_render_board_views(14), _INTRINSICS_BOARD)

    assert result.view_count == 14
    assert result.image_size == _IMAGE_SIZE
    # Warping a rendered board resamples pixels, so this is not exact; the point
    # is that the focal length is recovered to within a few percent rather than
    # being the arbitrary starting guess.
    assert result.fx == pytest.approx(_TRUE_FX, rel=0.10)
    assert result.fy == pytest.approx(_TRUE_FY, rel=0.10)
    assert result.cx == pytest.approx(_TRUE_CX, rel=0.10)
    assert result.cy == pytest.approx(_TRUE_CY, rel=0.10)


def test_a_good_solve_reports_a_low_reprojection_error() -> None:
    result = solve_intrinsics(_render_board_views(14), _INTRINSICS_BOARD)

    assert result.rms_reprojection_px < 1.0
    assert result.principal_point_is_centred


def test_as_dict_is_the_shape_the_rest_of_the_package_takes() -> None:
    """The result must feed straight into camera_matrix() and HandEyeSession."""
    result = solve_intrinsics(_render_board_views(8), _INTRINSICS_BOARD)

    matrix = camera_matrix(result.as_dict())

    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(result.fx)
    assert matrix[1, 2] == pytest.approx(result.cy)


def test_too_few_usable_views_is_rejected() -> None:
    views = _render_board_views(MIN_INTRINSICS_VIEWS - 1)

    with pytest.raises(CyberwaveValidationError, match="at least"):
        solve_intrinsics(views, _INTRINSICS_BOARD)


def test_blank_frames_do_not_count_as_views() -> None:
    """Undetectable frames are skipped, and the shortfall is reported."""
    blanks = [np.full((720, 1280), 255, np.uint8) for _ in range(9)]

    with pytest.raises(CyberwaveValidationError, match="detectable board"):
        solve_intrinsics(blanks, _INTRINSICS_BOARD)


def test_mixed_resolutions_are_rejected() -> None:
    """Intrinsics are per-resolution; mixing sizes yields a meaningless result."""
    views = _render_board_views(6)
    views.append(_render_board_views(1, seed=9, size=(640, 480))[0])

    with pytest.raises(CyberwaveValidationError, match="per-resolution"):
        solve_intrinsics(views, _INTRINSICS_BOARD)


def test_an_off_centre_principal_point_is_flagged() -> None:
    """The centring check catches a bad solve the RMS alone can absorb."""
    result = solve_intrinsics(_render_board_views(14), _INTRINSICS_BOARD)
    shifted = replace(result, cx=result.image_size[0] * 0.9)

    assert result.principal_point_is_centred
    assert not shifted.principal_point_is_centred


def test_intrinsics_metadata_is_json_safe() -> None:
    result = solve_intrinsics(_render_board_views(8), _INTRINSICS_BOARD)
    metadata = result.to_metadata()

    json.dumps(metadata)  # must not raise
    assert metadata["source"] == "solved"
    assert metadata["view_count"] == 8
