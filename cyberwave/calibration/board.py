"""Calibration-target description and pose estimation.

A board's job here is to turn one image into ``camera_to_target`` — the board's
pose in the camera's OpenCV optical frame — which is the second half of what
:func:`~cyberwave.calibration.handeye.solve_hand_eye` needs.

Two board types, sharing the same ``solvePnP`` path:

* :class:`CharucoBoard` — chessboard with ArUco markers in the white squares.
  Tolerates partial views and occlusion, so prefer it.
* :class:`CheckerBoard` — plain chessboard. Needs no ArUco dictionary and prints
  from anywhere, but must be *fully* visible in every frame.

**Measure your printed board.** ``square_size_m`` is the single scale input to the
whole calibration: printers rescale, so a nominal 30 mm square that prints at
29.4 mm puts a 2% error into every solved translation. Measure with calipers
across several squares and divide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..exceptions import CyberwaveValidationError
from .frames import make_transform

_CV2_INSTALL_HINT = (
    "OpenCV is required for calibration-board detection. Install it with: "
    'pip install "cyberwave[calibration]"'
)

# ChArUco detection via cv2.aruco.CharucoDetector landed in OpenCV 4.7. The SDK
# pins opencv-python-headless ^4.8, so that is the API written against here.
_ARUCO_API_HINT = (
    "ChArUco detection needs OpenCV >= 4.7 (cv2.aruco.CharucoDetector). Upgrade with: "
    'pip install --upgrade "opencv-python-headless>=4.8" — or use CheckerBoard, '
    "which works on older builds."
)

_DICTIONARY_ALIASES: dict[str, str] = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "apriltag_36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "apriltag_25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
    "apriltag_16h5": "DICT_APRILTAG_16h5",
}

#: solvePnP needs at least four coplanar point correspondences; fewer than six
#: gives a pose too unstable to feed a calibration.
MIN_CORRESPONDENCES = 6


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415  (lazy by design)
    except ImportError as exc:
        raise ImportError(_CV2_INSTALL_HINT) from exc
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "This OpenCV build has no aruco module. Install the wheel: "
            'pip install "cyberwave[calibration]"'
        )
    return cv2


def resolve_dictionary(cv2: Any, name: str) -> Any:
    """Predefined ArUco dictionary by name, accepting ``DICT_*`` or AprilTag aliases."""
    key = name.strip()
    attr = _DICTIONARY_ALIASES.get(key.lower(), key.upper())
    if not attr.startswith("DICT_") or not hasattr(cv2.aruco, attr):
        raise CyberwaveValidationError(
            f"Unknown or unavailable ArUco dictionary {name!r}. Expected a cv2.aruco "
            "DICT_* name such as 'DICT_5X5_1000'."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attr))


def camera_matrix(intrinsics: Mapping[str, Any] | Sequence[Any] | np.ndarray) -> np.ndarray:
    """3x3 pinhole camera matrix from ``{fx, fy, cx, cy}`` or an existing 3x3 array.

    The dict form matches the ``camera_intrinsics`` shape the SDK already uses for
    ML inference (``cyberwave/mlmodels/client.py``).
    """
    if isinstance(intrinsics, Mapping):
        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in intrinsics]
        if missing:
            raise CyberwaveValidationError(
                f"Intrinsics are missing {missing}. Expected keys: fx, fy, cx, cy."
            )
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        if fx <= 0.0 or fy <= 0.0:
            raise CyberwaveValidationError(
                f"Focal lengths must be positive, got fx={fx}, fy={fy}."
            )
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    matrix = np.asarray(intrinsics, dtype=float)
    if matrix.shape != (3, 3):
        raise CyberwaveValidationError(
            f"Intrinsics must be a 3x3 camera matrix or a dict of fx/fy/cx/cy, "
            f"got shape {matrix.shape}"
        )
    return matrix


def distortion_vector(dist_coeffs: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """Distortion coefficients as a float row vector; ``None`` means no distortion."""
    if dist_coeffs is None:
        return np.zeros((1, 5))
    vector = np.asarray(dist_coeffs, dtype=float).reshape(1, -1)
    if vector.shape[1] not in (4, 5, 8, 12, 14):
        raise CyberwaveValidationError(
            f"Expected 4, 5, 8, 12 or 14 distortion coefficients, got {vector.shape[1]}"
        )
    return vector


def _validate_grid(squares: Sequence[int], *, field: str, minimum: int) -> tuple[int, int]:
    if len(squares) != 2:
        raise CyberwaveValidationError(f"{field} must be a (columns, rows) pair")
    cols, rows = (int(v) for v in squares)
    if cols < minimum or rows < minimum:
        raise CyberwaveValidationError(
            f"{field} must be at least {minimum} in each direction, got ({cols}, {rows})"
        )
    return cols, rows


@dataclass(frozen=True)
class CharucoBoard:
    """A ChArUco board: chessboard with an ArUco marker inside each white square.

    Args:
        squares: ``(columns, rows)`` of chessboard squares — the full square count,
            not interior corners.
        square_size_m: Printed chessboard square edge length, in metres. Measure it.
        marker_size_m: Printed ArUco marker edge length, in metres. Must be smaller
            than the square.
        dictionary: ``cv2.aruco`` predefined dictionary name. Must match the
            dictionary the board image was generated with, or nothing is detected.
    """

    squares: tuple[int, int]
    square_size_m: float
    marker_size_m: float
    dictionary: str = "DICT_5X5_1000"

    def __post_init__(self) -> None:
        cols, rows = _validate_grid(self.squares, field="squares", minimum=2)
        object.__setattr__(self, "squares", (cols, rows))
        if self.square_size_m <= 0.0:
            raise CyberwaveValidationError("square_size_m must be positive")
        if self.marker_size_m <= 0.0:
            raise CyberwaveValidationError("marker_size_m must be positive")
        if self.marker_size_m >= self.square_size_m:
            raise CyberwaveValidationError(
                f"marker_size_m ({self.marker_size_m}) must be smaller than "
                f"square_size_m ({self.square_size_m}) — the marker sits inside a square"
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "charuco",
            "squares": list(self.squares),
            "square_size_m": self.square_size_m,
            "marker_size_m": self.marker_size_m,
            "dictionary": self.dictionary,
        }

    def _cv_board(self, cv2: Any) -> Any:
        if not hasattr(cv2.aruco, "CharucoBoard"):
            raise ImportError(_ARUCO_API_HINT)
        return cv2.aruco.CharucoBoard(
            self.squares,
            float(self.square_size_m),
            float(self.marker_size_m),
            resolve_dictionary(cv2, self.dictionary),
        )

    def correspondences(
        self, cv2: Any, image: np.ndarray, *, minimum: int = MIN_CORRESPONDENCES
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(object_points, image_points)`` for the detected corners, or ``None``.

        *minimum* is the fewest corners worth returning. It defaults to
        :data:`MIN_CORRESPONDENCES` — the floor for a usable ``solvePnP`` — but a
        caller that only wants to know whether the board was seen at all can lower
        it to 1 and apply its own threshold later.
        """
        detection = self.detect(cv2, image, minimum=minimum)
        return None if detection is None else detection[0]

    def detect(
        self, cv2: Any, image: np.ndarray, *, minimum: int = MIN_CORRESPONDENCES
    ) -> tuple[tuple[np.ndarray, np.ndarray], int] | None:
        """``((object_points, image_points), markers_read)``, or ``None``.

        Same detection as :meth:`correspondences`, but also reports how many ArUco
        markers were decoded. The two counts are not interchangeable: interpolation
        can yield more chessboard corners than markers (each corner is shared
        between neighbours) or fewer (a marker at the edge places no interior
        corner), so a caller that wants to threshold on markers read cannot infer
        it from the corner count. Both come out of one ``detectBoard`` pass.
        """
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise ImportError(_ARUCO_API_HINT)
        floor = max(int(minimum), 1)
        board = self._cv_board(cv2)
        corners, ids, _marker_corners, marker_ids = cv2.aruco.CharucoDetector(
            board
        ).detectBoard(image)
        markers_read = 0 if marker_ids is None else len(marker_ids)
        if ids is None or len(ids) < floor:
            return None
        object_points, image_points = board.matchImagePoints(corners, ids)
        if object_points is None or len(object_points) < floor:
            return None
        return (
            (
                np.asarray(object_points, dtype=float).reshape(-1, 3),
                np.asarray(image_points, dtype=float).reshape(-1, 2),
            ),
            markers_read,
        )

    def generate_image(self, pixels_per_square: int = 120) -> np.ndarray:
        """Render the board as a printable grayscale image."""
        cv2 = _import_cv2()
        board = self._cv_board(cv2)
        if not hasattr(board, "generateImage"):
            raise ImportError(_ARUCO_API_HINT)
        cols, rows = self.squares
        size = (cols * int(pixels_per_square), rows * int(pixels_per_square))
        return board.generateImage(size)


@dataclass(frozen=True)
class CheckerBoard:
    """A plain chessboard.

    Args:
        inner_corners: ``(columns, rows)`` of *interior* corners — a board with 10x7
            squares has 9x6 interior corners.
        square_size_m: Printed square edge length, in metres. Measure it.

    A plain chessboard has no per-corner identity, so OpenCV can only find it when
    the whole board is visible, and the corner ordering flips under 180-degree
    in-plane rotation. Keep the board's orientation roughly consistent across
    captures, or use :class:`CharucoBoard`.
    """

    inner_corners: tuple[int, int]
    square_size_m: float

    def __post_init__(self) -> None:
        cols, rows = _validate_grid(self.inner_corners, field="inner_corners", minimum=2)
        object.__setattr__(self, "inner_corners", (cols, rows))
        if self.square_size_m <= 0.0:
            raise CyberwaveValidationError("square_size_m must be positive")
        if cols == rows:
            raise CyberwaveValidationError(
                f"A square {cols}x{rows} corner grid is rotationally ambiguous — "
                "OpenCV cannot pin down its orientation. Use a non-square grid "
                "(e.g. 9x6) or a CharucoBoard."
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "checkerboard",
            "inner_corners": list(self.inner_corners),
            "square_size_m": self.square_size_m,
        }

    def object_points(self) -> np.ndarray:
        """Corner coordinates in the board frame: Z=0 plane, X across, Y down."""
        cols, rows = self.inner_corners
        grid = np.zeros((rows * cols, 3), dtype=float)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return grid * float(self.square_size_m)

    def correspondences(
        self, cv2: Any, image: np.ndarray, *, minimum: int = MIN_CORRESPONDENCES
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(object_points, image_points)``, or ``None`` when the board is absent.

        *minimum* is accepted for signature parity with :class:`CharucoBoard` and
        otherwise unused: ``findChessboardCorners`` is all-or-nothing, so a plain
        chessboard is either fully found or not found.
        """
        gray = _as_gray(cv2, image)
        found, corners = cv2.findChessboardCorners(
            gray,
            self.inner_corners,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        return self.object_points(), np.asarray(refined, dtype=float).reshape(-1, 2)

    def detect(
        self, cv2: Any, image: np.ndarray, *, minimum: int = MIN_CORRESPONDENCES
    ) -> tuple[tuple[np.ndarray, np.ndarray], int] | None:
        """``((object_points, image_points), markers_read)``, or ``None``.

        A plain chessboard carries no ArUco markers, so *markers_read* is reported
        as ``None``'s numeric stand-in: ``-1``. A caller thresholding on markers
        must treat that as "not applicable" rather than as zero, or it would
        discard every capture of a board that was in fact fully found.
        """
        correspondences = self.correspondences(cv2, image, minimum=minimum)
        return None if correspondences is None else (correspondences, -1)


Board = CharucoBoard | CheckerBoard


def _as_gray(cv2: Any, image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise CyberwaveValidationError(
            f"Expected a 2D grayscale or 3-/4-channel image, got shape {array.shape}"
        )
    return gray if gray.dtype == np.uint8 else cv2.convertScaleAbs(gray)


#: Views below this leave the focal length and distortion badly constrained.
#: ``calibrateCamera`` will return numbers from fewer; they just will not be right.
MIN_INTRINSICS_VIEWS = 5

#: A principal point further than this fraction of the image from its centre almost
#: always means a wrong board spec or too few distinct viewpoints, not a real sensor
#: offset.
_PRINCIPAL_POINT_TOLERANCE = 0.15


@dataclass(frozen=True)
class IntrinsicsResult:
    """Solved camera intrinsics, with the numbers needed to judge them.

    ``rms_reprojection_px`` is the honest quality signal: under ~1 px is good, over
    ~2 px means the solve should be redone with more varied views. Unlike the
    hand-eye residual this *is* a direct error measure — it is how far the model
    misses the corners it was fitted to.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, ...]
    rms_reprojection_px: float
    image_size: tuple[int, int]
    view_count: int

    def as_dict(self) -> dict[str, float]:
        """``{fx, fy, cx, cy}`` — the shape every other calibration entry point takes."""
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @property
    def principal_point_is_centred(self) -> bool:
        """Whether ``(cx, cy)`` sits plausibly near the image centre."""
        width, height = self.image_size
        return (
            abs(self.cx - width / 2.0) / width <= _PRINCIPAL_POINT_TOLERANCE
            and abs(self.cy - height / 2.0) / height <= _PRINCIPAL_POINT_TOLERANCE
        )

    def to_metadata(self) -> dict[str, Any]:
        """JSON-safe summary for persisting as calibration provenance."""
        return {
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "dist_coeffs": [round(v, 8) for v in self.dist_coeffs],
            "rms_reprojection_px": round(self.rms_reprojection_px, 4),
            "image_size": list(self.image_size),
            "view_count": self.view_count,
            "source": "solved",
        }


def solve_intrinsics(
    images: Sequence[np.ndarray],
    board: Board,
    *,
    min_views: int = MIN_INTRINSICS_VIEWS,
) -> IntrinsicsResult:
    """Estimate ``fx/fy/cx/cy`` and distortion from views of *board*.

    Intrinsics are an **input** to hand-eye that it cannot recover from: every
    ``solvePnP`` board pose inherits the error, and it surfaces only as an inflated
    hand-eye residual with no indication of the cause. Solving them from the same
    board is the only way to get metric ones.

    Intrinsics are **resolution specific**. Every image must be the same size, and
    the result is valid only for that size.

    Args:
        images: Views of the board, held at varied angles *and* distances. Views
            that are all fronto-parallel leave the focal length poorly constrained
            however many there are.
        board: The calibration target. Its measured square size is the only metric
            input to the whole calibration — a 2% error there is a 2% error in
            every distance downstream.
        min_views: Reject fewer usable views than this.

    Returns:
        An :class:`IntrinsicsResult`. **Check ``rms_reprojection_px``** before using
        it; also check :attr:`~IntrinsicsResult.principal_point_is_centred`, which
        catches a wrong board spec that the RMS alone can absorb.

    Raises:
        CyberwaveValidationError: too few usable views, or mixed image sizes.
    """
    cv2 = _import_cv2()

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for index, image in enumerate(images):
        array = np.asarray(image)
        if array.ndim < 2:
            raise CyberwaveValidationError(
                f"images[{index}] is not an image (shape {array.shape})"
            )
        size = (int(array.shape[1]), int(array.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            # Silently mixing resolutions produces a confident, meaningless result.
            raise CyberwaveValidationError(
                f"images[{index}] is {size[0]}x{size[1]} but earlier views are "
                f"{image_size[0]}x{image_size[1]}. Intrinsics are per-resolution; "
                "calibrate each resolution separately."
            )

        correspondences = board.correspondences(cv2, array)
        if correspondences is None:
            continue
        objects, pixels = correspondences
        object_points.append(np.asarray(objects, dtype=np.float32).reshape(-1, 1, 3))
        image_points.append(np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2))

    if len(object_points) < min_views:
        raise CyberwaveValidationError(
            f"Need at least {min_views} views with a detectable board, got "
            f"{len(object_points)} from {len(list(images))} image(s). Hold the whole "
            "board in frame, in focus and evenly lit, and vary its angle and "
            "distance between views."
        )

    assert image_size is not None  # implied by the view count check above
    return solve_intrinsics_from_correspondences(
        list(zip(object_points, image_points)), image_size, min_views=min_views
    )


def solve_intrinsics_from_correspondences(
    correspondences: Sequence[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    *,
    min_views: int = MIN_INTRINSICS_VIEWS,
) -> IntrinsicsResult:
    """Fit intrinsics from already-detected point correspondences.

    The detection half of :func:`solve_intrinsics` does not depend on the
    intrinsics, so a caller that refits repeatedly over a growing set of views
    can detect each frame once and pass the points here instead of re-running
    board detection on every fit.

    Args:
        correspondences: One ``(object_points, image_points)`` pair per view.
        image_size: ``(width, height)`` the views were captured at. Intrinsics
            are per-resolution, so every view must share this size.
        min_views: Reject fewer usable views than this.

    Returns:
        An :class:`IntrinsicsResult`; judge it by ``rms_reprojection_px`` and
        :attr:`~IntrinsicsResult.principal_point_is_centred` as usual.

    Raises:
        CyberwaveValidationError: fewer than *min_views* correspondences.
    """
    cv2 = _import_cv2()
    if len(correspondences) < min_views:
        raise CyberwaveValidationError(
            f"Need at least {min_views} views with a detectable board, got "
            f"{len(correspondences)}. Hold the whole board in frame, in focus and "
            "evenly lit, and vary its angle and distance between views."
        )

    object_points = [
        np.asarray(objects, dtype=np.float32).reshape(-1, 1, 3)
        for objects, _ in correspondences
    ]
    image_points = [
        np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
        for _, pixels in correspondences
    ]
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    return IntrinsicsResult(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        dist_coeffs=tuple(float(v) for v in np.asarray(distortion).ravel()),
        rms_reprojection_px=float(rms),
        image_size=image_size,
        view_count=len(correspondences),
    )


def solve_target_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: Any,
    dist_coeffs: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    """``camera_to_target`` 4x4 from point correspondences, or ``None`` if unsolved.

    ``solvePnP``'s ``rvec``/``tvec`` map board-frame points into the camera frame,
    which is exactly the pose of the target expressed in the camera — the
    ``camera_to_target`` convention used throughout this package. No inversion.
    """
    cv2 = _import_cv2()
    objects = np.asarray(object_points, dtype=float).reshape(-1, 1, 3)
    images = np.asarray(image_points, dtype=float).reshape(-1, 1, 2)
    if len(objects) != len(images):
        raise CyberwaveValidationError(
            f"Correspondence count mismatch: {len(objects)} object points vs "
            f"{len(images)} image points"
        )
    if len(objects) < MIN_CORRESPONDENCES:
        return None

    found, rvec, tvec = cv2.solvePnP(
        objects,
        images,
        camera_matrix(intrinsics),
        distortion_vector(dist_coeffs),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not found:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return make_transform(rotation, np.asarray(tvec, dtype=float).reshape(3))


def detect_target_pose(
    image: np.ndarray,
    board: Board,
    intrinsics: Any,
    dist_coeffs: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Locate *board* in *image* and return its 4x4 pose in the camera optical frame.

    Returns ``None`` when the board is not found or too little of it is visible —
    callers decide whether that is fatal.
    """
    cv2 = _import_cv2()
    correspondences = board.correspondences(cv2, np.asarray(image))
    if correspondences is None:
        return None
    object_points, image_points = correspondences
    return solve_target_pose(object_points, image_points, intrinsics, dist_coeffs)


__all__ = [
    "MIN_CORRESPONDENCES",
    "MIN_INTRINSICS_VIEWS",
    "Board",
    "CharucoBoard",
    "CheckerBoard",
    "IntrinsicsResult",
    "camera_matrix",
    "detect_target_pose",
    "distortion_vector",
    "resolve_dictionary",
    "solve_intrinsics",
    "solve_intrinsics_from_correspondences",
    "solve_target_pose",
]
