"""Eye-in-hand calibration: solve the fixed transform between a robot link and a camera.

Naming convention throughout this module: ``x_to_y`` is the transform that maps
points from frame *y* into frame *x*, i.e. the pose of *y* expressed in *x*. So
``base_to_gripper`` is where the gripper is in base coordinates, and the solved
``gripper_to_camera`` is where the camera sits on the gripper.

The result is expressed in the **OpenCV optical frame** (X right, Y down, Z along
the view axis) because that is what ``cv2`` works in. Converting it to the
platform's sensor frame is the persistence layer's job — see
``frames.OPTICAL_TO_SENSOR`` and ``CameraCalibrationHandle``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..exceptions import CyberwaveValidationError
from .board import (
    MIN_CORRESPONDENCES,
    MIN_INTRINSICS_VIEWS,
    Board,
    IntrinsicsResult,
    camera_matrix,
    distortion_vector,
    solve_intrinsics_from_correspondences,
    solve_target_pose,
)
from .frames import (
    make_transform,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    rotation_angle_deg,
)

_CV2_INSTALL_HINT = (
    "OpenCV is required for hand-eye calibration. Install it with: "
    'pip install "cyberwave[calibration]"'
)

#: Fewer than three absolute poses gives fewer than two relative motions.
MIN_SAMPLES = 3

#: Fewest ArUco markers a capture must decode before it is used in a solve. Corners
#: alone do not say how much of the *grid* was seen: a few markers clustered in one
#: corner still interpolate corners, but pin the board pose from a patch too small
#: to condition it. Not applicable to marker-less boards, which report ``-1``.
MIN_MARKERS = 6

#: Joint movement below this between two reads counts as stationary, in radians.
#: ~0.11 degrees: tighter than the repeatability of the hobby servos this is aimed
#: at, so a settled arm reliably passes.
DEFAULT_SETTLE_TOLERANCE_RAD = 0.002

#: How long :meth:`HandEyeSession.add_sample` waits for the arm to stop, in seconds.
DEFAULT_SETTLE_TIMEOUT_S = 0.5

#: Gap between successive joint reads while waiting for the arm to settle.
_SETTLE_POLL_INTERVAL_S = 0.05

METHODS: dict[str, str] = {
    "park": "CALIB_HAND_EYE_PARK",
    "horaud": "CALIB_HAND_EYE_HORAUD",
    "andreff": "CALIB_HAND_EYE_ANDREFF",
    "tsai": "CALIB_HAND_EYE_TSAI",
    "daniilidis": "CALIB_HAND_EYE_DANIILIDIS",
}

#: Robust across both moderate and large rotations; see :data:`METHODS`.
DEFAULT_METHOD = "park"


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(_CV2_INSTALL_HINT) from exc
    return cv2


class HandEyeDegenerateError(CyberwaveValidationError):
    """Raised when the captured poses cannot constrain the calibration.

    Almost always means the arm was translated but not rotated enough between
    captures. Re-capture with the wrist tilted to genuinely different orientations
    rather than sliding it around at a fixed orientation.
    """


@dataclass(frozen=True)
class HandEyeResult:
    """A solved eye-in-hand transform plus the numbers needed to judge it.

    ``gripper_to_camera`` is in the OpenCV optical frame. The residuals are the
    real output: for a correct solve, ``base_to_gripper @ X @ camera_to_target``
    is the (fixed) board pose in base coordinates and must agree across every
    sample. Disagreement bounds how wrong the transform is.

    **What the residual does not cover.** It measures whether the observations are
    consistent *with each other*, so it catches noise, a board that moved, and bad
    intrinsics. It cannot catch a systematically mislabelled input: swap the two
    argument lists and the resulting system is still exactly consistent (it solves
    for the inverse board pose) and reports a machine-zero residual. Same for
    forward kinematics reported about the wrong link. Those have to be prevented by
    construction, which is why :class:`HandEyeSession` assembles both lists itself
    and validates the link name.
    """

    gripper_to_camera: np.ndarray
    #: Standard deviation of the implied board position across samples, in metres:
    #: the per-axis standard deviation combined into one length. See
    #: :func:`_residuals`.
    residual_translation_m: float
    #: Largest single-sample distance from the average implied board position, in
    #: metres. A value several times :attr:`residual_translation_m` points at one
    #: bad capture rather than a bad fit.
    max_residual_translation_m: float
    #: Standard deviation of the per-sample angle to the average implied board
    #: orientation, in degrees.
    residual_rotation_deg: float
    #: Largest single-sample angle to the average implied board orientation.
    max_residual_rotation_deg: float
    sample_count: int
    method: str
    #: RMS relative rotation error of the AX = XB fit, in degrees — how well the
    #: solved transform reconciles the motions actually observed. See
    #: :func:`relative_errors`.
    relative_rotation_deg: float = 0.0
    #: RMS relative translation error, in metres.
    relative_translation_m: float = 0.0
    #: :attr:`relative_translation_m` over the motion magnitude: dimensionless, so
    #: comparable across arms and scales.
    relative_translation_ratio: float = 0.0
    #: Leave-one-out stability, when the solve was run through
    #: :meth:`HandEyeSession.solve` with validation on. ``None`` means not checked,
    #: and also covers "too few samples to say" — see
    #: :func:`leave_one_out_stability`.
    leave_one_out: LeaveOneOutResult | None = None
    #: Intrinsics solved from this session's own captures, when none were supplied.
    #: ``None`` means they were given, so judging them is the caller's business.
    solved_intrinsics: IntrinsicsResult | None = None

    def with_validation(
        self,
        *,
        leave_one_out: LeaveOneOutResult | None,
        solved_intrinsics: IntrinsicsResult | None = None,
    ) -> HandEyeResult:
        """Copy of this result carrying the given validation output."""
        return replace(
            self,
            leave_one_out=leave_one_out,
            solved_intrinsics=solved_intrinsics,
        )

    def to_metadata(self) -> dict[str, Any]:
        """JSON-safe summary for persisting as calibration provenance."""
        metadata: dict[str, Any] = {
            "method": self.method,
            "sample_count": self.sample_count,
            "residual_translation_m": round(self.residual_translation_m, 6),
            "max_residual_translation_m": round(self.max_residual_translation_m, 6),
            "residual_rotation_deg": round(self.residual_rotation_deg, 4),
            "max_residual_rotation_deg": round(self.max_residual_rotation_deg, 4),
            "relative_rotation_deg": round(self.relative_rotation_deg, 4),
            "relative_translation_m": round(self.relative_translation_m, 6),
            "relative_translation_ratio": round(self.relative_translation_ratio, 4),
        }
        # Omitted rather than null when unchecked: a reader should not have to
        # distinguish "validation absent" from "validation returned nothing".
        if self.leave_one_out is not None:
            metadata["leave_one_out"] = self.leave_one_out.to_metadata()
        if self.solved_intrinsics is not None:
            metadata["solved_intrinsics"] = self.solved_intrinsics.to_metadata()
        return metadata


def _as_transform(value: Any, *, label: str, index: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        raise CyberwaveValidationError(
            f"{label}[{index}] must be a 4x4 homogeneous transform, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise CyberwaveValidationError(f"{label}[{index}] contains non-finite values")
    rotation = array[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise CyberwaveValidationError(
            f"{label}[{index}] rotation block is not orthonormal — "
            "check that this is a pose matrix and not a scaled or transposed one"
        )
    if np.linalg.det(rotation) < 0.0:
        raise CyberwaveValidationError(
            f"{label}[{index}] rotation block is a reflection (negative determinant)"
        )
    return array


def _mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Representative pose of a tight cluster: mean translation, averaged rotation.

    Quaternions are sign-aligned before averaging (``q`` and ``-q`` are the same
    rotation, so a raw mean can cancel to zero). This is the standard small-spread
    approximation and is only ever used as a residual reference, never as output.
    """
    translation = np.mean([t[:3, 3] for t in transforms], axis=0)
    reference = np.array(matrix_to_quat_wxyz(transforms[0][:3, :3]))
    accumulator = np.zeros(4)
    for transform in transforms:
        quaternion = np.array(matrix_to_quat_wxyz(transform[:3, :3]))
        if np.dot(quaternion, reference) < 0.0:
            quaternion = -quaternion
        accumulator += quaternion
    norm = float(np.linalg.norm(accumulator))
    quaternion = reference if norm < 1e-12 else accumulator / norm
    return make_transform(quat_wxyz_to_matrix(quaternion), translation)


def _residuals(
    base_to_gripper: Sequence[np.ndarray],
    camera_to_target: Sequence[np.ndarray],
    gripper_to_camera: np.ndarray,
) -> tuple[float, float, float, float]:
    """Spread of the implied board pose across samples: (std_t, max_t, std_r, max_r).

    For a correct ``gripper_to_camera``, ``base_to_gripper[k] @ X @
    camera_to_target[k]`` is the same rigid transform for every sample ``k`` -- the
    fixed pose of the board in the arm's base frame. The spread across samples
    therefore bounds how wrong ``X`` is, with no ground truth to compare against.

    The two headline figures measure how much the samples *disagree among
    themselves*, which is what a residual is for; see the note on
    :class:`HandEyeResult` for what they consequently cannot see. They are not
    computed the same way, because the two quantities are not the same shape:

    * **translation** -- the per-axis standard deviation of the implied board
      position, combined into one length. The per-axis values are *signed*
      offsets, so their standard deviation is a genuine spread.
    * **rotation** -- the root-mean-square of the per-sample angle to the average
      orientation. Those angles are already unsigned deviations, so taking their
      standard deviation would measure the spread *of the deviations* and report
      zero for samples that sit equally far from the mean while disagreeing
      wildly with each other.

    The ``max_*`` values are deviations from the average, so they stay comparable
    to a physical tolerance and still name a single bad capture. ``max_t >= std_t``
    and ``max_r >= rms_r`` both hold: each headline figure is a root-mean-square of
    deviations and so cannot exceed the largest of them (equality only in the
    degenerate two-cluster case).
    """
    implied = [
        a @ gripper_to_camera @ b
        for a, b in zip(base_to_gripper, camera_to_target)
    ]
    reference = _mean_transform(implied)
    inverse_reference_rotation = reference[:3, :3].T

    positions = np.array([t[:3, 3] for t in implied])
    # Per-axis standard deviation, combined into a single length. Kept per-axis
    # rather than as the deviation of the norms because the three axes of an
    # eye-in-hand rig are constrained very differently -- the optical axis is
    # always the weakest -- and collapsing to a norm first hides which one moved.
    translation_std = float(np.linalg.norm(positions.std(axis=0)))
    deviations = [
        float(np.linalg.norm(position - reference[:3, 3])) for position in positions
    ]

    angles = [
        rotation_angle_deg(inverse_reference_rotation @ t[:3, :3]) for t in implied
    ]
    # RMS, not standard deviation: each angle is already a deviation from the mean
    # orientation, so the std of the magnitudes reports zero whenever every sample
    # sits the same distance from the mean -- however far that is, and however much
    # the samples disagree with each other. The translation branch above is immune
    # because it takes the std of *signed* per-axis values.
    return (
        translation_std,
        float(np.max(deviations)),
        float(np.sqrt(np.mean(np.square(angles)))),
        float(np.max(angles)),
    )


def relative_errors(
    base_to_gripper: Sequence[np.ndarray],
    camera_to_target: Sequence[np.ndarray],
    gripper_to_camera: np.ndarray,
) -> tuple[float, float, float]:
    """Relative rotation and translation error of the AX = XB fit.

    Returns ``(rotation_deg, translation_m, translation_ratio)``, each an RMS over
    every pair of captures.

    This measures *accuracy of fit*: how well the solved transform reconciles the
    relative motions the arm and the camera actually observed. It complements
    :func:`leave_one_out_stability`, which measures whether the answer is stable
    under changes to the evidence — a set can fit tightly and still be unstable,
    or vice versa, so both are reported.

    For each pair of captures ``(i, j)`` the relative motions are
    ``A = inv(A_i) A_j`` (arm) and ``B = B_i inv(B_j)`` (board), which is the pair
    satisfying ``A X = X B``. Expanding its translation part gives the residual
    used here::

        R_A t_X + t_A - R_X t_B - t_X

    Getting those signs wrong (or applying ``R_B`` where ``R_X`` belongs) inflates
    the result by roughly the size of the motions themselves rather than the error,
    so the expression is written out rather than factored.

    ``translation_ratio`` is that residual over the motion magnitude ``|t_A|``,
    which is dimensionless and therefore comparable across arms and scales.
    All pairs are used, not consecutive ones: the pairing is otherwise an artefact
    of capture order.
    """
    RX = gripper_to_camera[:3, :3]
    tX = gripper_to_camera[:3, 3]
    rotations: list[float] = []
    translations: list[float] = []
    ratios: list[float] = []
    count = len(base_to_gripper)
    for i in range(count):
        for j in range(i + 1, count):
            motion_arm = np.linalg.inv(base_to_gripper[i]) @ base_to_gripper[j]
            motion_board = camera_to_target[i] @ np.linalg.inv(camera_to_target[j])
            RA, tA = motion_arm[:3, :3], motion_arm[:3, 3]
            RB, tB = motion_board[:3, :3], motion_board[:3, 3]
            rotations.append(rotation_angle_deg((RX @ RB).T @ (RA @ RX)))
            residual = float(np.linalg.norm(RA @ tX + tA - RX @ tB - tX))
            translations.append(residual)
            motion = float(np.linalg.norm(tA))
            # A pair with no translation between it says nothing about translation
            # error; including it would divide by ~0 and swamp the ratio.
            if motion > 1e-6:
                ratios.append(residual / motion)
    if not rotations:
        return (0.0, 0.0, 0.0)

    def _rms(values: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0

    return (_rms(rotations), _rms(translations), _rms(ratios))


def solve_hand_eye(
    base_to_gripper: Iterable[Any],
    camera_to_target: Iterable[Any],
    *,
    method: str = DEFAULT_METHOD,
) -> HandEyeResult:
    """Solve for the camera's pose on the gripper from paired observations.

    Pure math — no twins, no network, no image decoding.

    Args:
        base_to_gripper: Per-sample 4x4 pose of the gripper link in the arm's base
            frame (forward kinematics). Must be the pose of the *same* link the
            camera twin is docked to.
        camera_to_target: Per-sample 4x4 pose of the calibration board in the
            camera's optical frame, as returned by ``solvePnP``. The board must not
            move between samples.
        method: One of :data:`METHODS`.

    Returns:
        A :class:`HandEyeResult`. **Check its residuals before trusting it** — a
        well-posed solve always returns a number, and only the residual indicates
        whether the observations back it up. Note the documented limits of that
        check on :class:`HandEyeResult`: it cannot detect swapped or mislabelled
        inputs, only mutually inconsistent ones.

    Raises:
        CyberwaveValidationError: mismatched or too-few samples, or a malformed pose.
        HandEyeDegenerateError: the solver returned no valid transform, so the
            poses do not determine one.
    """
    if method not in METHODS:
        raise CyberwaveValidationError(
            f"Unknown hand-eye method {method!r}. Available: {sorted(METHODS)}"
        )

    gripper_poses = [
        _as_transform(t, label="base_to_gripper", index=i)
        for i, t in enumerate(base_to_gripper)
    ]
    target_poses = [
        _as_transform(t, label="camera_to_target", index=i)
        for i, t in enumerate(camera_to_target)
    ]

    if len(gripper_poses) != len(target_poses):
        raise CyberwaveValidationError(
            f"Sample count mismatch: {len(gripper_poses)} gripper poses vs "
            f"{len(target_poses)} target poses"
        )
    if len(gripper_poses) < MIN_SAMPLES:
        raise CyberwaveValidationError(
            f"Hand-eye calibration needs at least {MIN_SAMPLES} samples, "
            f"got {len(gripper_poses)}. Capture 10-15 for a usable result."
        )

    cv2 = _import_cv2()
    solved_rotation, solved_translation = cv2.calibrateHandEye(
        R_gripper2base=[t[:3, :3] for t in gripper_poses],
        t_gripper2base=[t[:3, 3].reshape(3, 1) for t in gripper_poses],
        R_target2cam=[t[:3, :3] for t in target_poses],
        t_target2cam=[t[:3, 3].reshape(3, 1) for t in target_poses],
        method=getattr(cv2, METHODS[method]),
    )
    gripper_to_camera = make_transform(solved_rotation, np.asarray(solved_translation).reshape(3))

    # A rank-deficient system makes cv2 return NaN or a non-rotation rather than
    # failing, so check before anything downstream tries to interpret it. This is
    # the only degeneracy check: it tests the answer itself rather than guessing
    # from the poses whether one could be found.
    if not np.all(np.isfinite(gripper_to_camera)) or not np.allclose(
        gripper_to_camera[:3, :3] @ gripper_to_camera[:3, :3].T, np.eye(3), atol=1e-4
    ):
        raise HandEyeDegenerateError(
            f"The {method!r} solver returned no valid transform — the sample set does "
            f"not determine one over {len(gripper_poses)} samples. Re-capture with "
            "the wrist rotated about several distinct axes."
        )

    mean_t, max_t, mean_r, max_r = _residuals(
        gripper_poses, target_poses, gripper_to_camera
    )
    rel_r, rel_t, rel_ratio = relative_errors(
        gripper_poses, target_poses, gripper_to_camera
    )
    return HandEyeResult(
        gripper_to_camera=gripper_to_camera,
        residual_translation_m=mean_t,
        max_residual_translation_m=max_t,
        residual_rotation_deg=mean_r,
        max_residual_rotation_deg=max_r,
        sample_count=len(gripper_poses),
        method=method,
        relative_rotation_deg=rel_r,
        relative_translation_m=rel_t,
        relative_translation_ratio=rel_ratio,
    )


#: Leave-one-out needs enough samples that dropping one still leaves a solvable
#: system. That floor is :data:`MIN_SAMPLES`: a solve needs three absolute poses,
#: so a subset with one dropped needs four to exist at all.
#:
#: Set to ``MIN_SAMPLES + 1`` rather than higher on purpose. Stability is the
#: statistic the verdict is keyed on, and gating it behind a larger sample count
#: meant that a short run reported *no* stability at all — which the UI renders
#: as no verdict badge, the one state that tells the operator nothing. A spread
#: measured over few subsets is noisy, and the caller should treat it as weaker
#: evidence, but it is strictly more informative than silence.
#:
#: The noise this admits is bounded by the function's own guards rather than by
#: this number: subsets that come out degenerate are skipped, and fewer than two
#: solved subsets still returns ``None``.
MIN_LEAVE_ONE_OUT_SAMPLES = MIN_SAMPLES + 1


@dataclass(frozen=True)
class LeaveOneOutResult:
    """How much the answer moves when any single sample is dropped.

    A tight spread means no individual observation is carrying the solve. A wide
    one means either a bad sample or too little rotation diversity for the sample
    count — ``worst_sample_index`` says which sample to look at first, which is
    what makes this actionable rather than merely alarming.
    """

    translation_stddev_m: tuple[float, float, float]
    max_stddev_m: float
    worst_sample_index: int
    subset_count: int

    def to_metadata(self) -> dict[str, Any]:
        """JSON-safe summary for persisting as calibration provenance."""
        return {
            "translation_stddev_m": [round(v, 6) for v in self.translation_stddev_m],
            "max_stddev_m": round(self.max_stddev_m, 6),
            "worst_sample_index": self.worst_sample_index,
            "subset_count": self.subset_count,
        }


def leave_one_out_stability(
    base_to_gripper: Iterable[Any],
    camera_to_target: Iterable[Any],
    *,
    method: str = DEFAULT_METHOD,
) -> LeaveOneOutResult | None:
    """Re-solve with each sample dropped in turn and report the spread.

    Args:
        base_to_gripper: As :func:`solve_hand_eye`.
        camera_to_target: As :func:`solve_hand_eye`.
        method: Solver to use for every subset.

    Returns:
        A :class:`LeaveOneOutResult`, or ``None`` when there are fewer than
        :data:`MIN_LEAVE_ONE_OUT_SAMPLES` samples (one more than a solve needs, so
        that dropping one still leaves a solvable subset), or when fewer than two
        subsets could be solved at all.

        At low sample counts the spread is dominated by *which* sample was
        removed, so read it as weak evidence rather than a precise figure —
        :attr:`LeaveOneOutResult.subset_count` says how many subsets it came from.
        It is still reported, because the alternative is reporting nothing, and a
        missing stability figure leaves the caller with no verdict to show.
    """
    gripper_poses = list(base_to_gripper)
    target_poses = list(camera_to_target)
    count = len(gripper_poses)
    if count < MIN_LEAVE_ONE_OUT_SAMPLES:
        return None

    cv2 = _import_cv2()
    solved: dict[int, np.ndarray] = {}
    for dropped in range(count):
        keep = [i for i in range(count) if i != dropped]
        try:
            result = solve_hand_eye(
                [gripper_poses[i] for i in keep],
                [target_poses[i] for i in keep],
                method=method,
            )
        except (cv2.error, CyberwaveValidationError, HandEyeDegenerateError):
            # Includes HandEyeDegenerateError: one sample can be the only thing
            # keeping the set well posed, and that subset simply has no answer.
            continue
        solved[dropped] = result.gripper_to_camera[:3, 3]

    if len(solved) < 2:
        return None

    subsets = np.array(list(solved.values()))
    stddev = subsets.std(axis=0)
    # The sample whose removal moves the result furthest from where the other
    # subsets land: the one to re-capture first.
    centre = subsets.mean(axis=0)
    worst = max(solved, key=lambda i: float(np.linalg.norm(solved[i] - centre)))

    return LeaveOneOutResult(
        translation_stddev_m=(
            float(stddev[0]),
            float(stddev[1]),
            float(stddev[2]),
        ),
        max_stddev_m=float(stddev.max()),
        worst_sample_index=worst,
        subset_count=len(solved),
    )


class BoardNotDetectedError(CyberwaveValidationError):
    """Raised when a capture does not contain a usable view of the board.

    Deliberately fatal rather than skipped: silently dropping captures is how a
    session ends up with only a handful of similar views and a confident, wrong
    result. The caller decides whether to reposition and retry.
    """


class HandEyeSyncError(CyberwaveValidationError):
    """Raised when a sample's image and gripper pose describe different instants.

    A frame captured while the arm was still moving pairs one robot position with
    another position's image. The solve absorbs that as a plausible-looking
    transform: nothing downstream can detect it, because each sample is internally
    well formed. It has to be caught at capture time or not at all.
    """


@dataclass
class HandEyeSample:
    """One capture: where the gripper was, and where the board appeared.

    A capture taken before the intrinsics are known has no board pose yet —
    ``camera_to_target`` is ``None`` and :attr:`is_paired` is false. It is still a
    capture the operator made, and it becomes a usable sample as soon as the
    intrinsics solve. Holding unpaired captures in this same list (rather than a
    separate buffer) is what keeps "everything captured" and "everything usable"
    two views of one collection instead of two collections to keep in step.
    """

    base_to_gripper: np.ndarray
    #: Board pose in the camera's optical frame, or ``None`` until intrinsics
    #: exist. Provisional even once set: it was measured against whatever
    #: intrinsics were current, and :meth:`HandEyeSession.solve` re-detects it.
    camera_to_target: np.ndarray | None = None
    #: The image this capture was taken from, kept so intrinsics can be refitted
    #: from every view and the board poses re-detected against the result.
    #: ``None`` when the frame was not retained, i.e. the caller supplied
    #: intrinsics so there is nothing to refit.
    frame: Any = None
    #: Cached ``(object_points, image_points)`` from the last detection of
    #: :attr:`frame`, so refitting intrinsics does not redo the detection work.
    #: ``None`` when the board was not found or the frame was not retained.
    correspondences: tuple[np.ndarray, np.ndarray] | None = None
    #: How many ArUco markers were decoded in :attr:`frame`. ``-1`` for a board
    #: that has no markers (a plain chessboard), which means "not applicable" and
    #: must not be compared against a marker threshold as if it were zero.
    markers_read: int = -1

    @property
    def is_paired(self) -> bool:
        """Whether this capture has a board pose and can be solved from."""
        return self.camera_to_target is not None


class HandEyeSession:
    """Collects paired observations for an eye-in-hand calibration, then solves.

    Manual capture: you position the arm (teleop, hand-guiding, whatever), then call
    :meth:`add_sample`. The session never commands motion.

    Constructing the session validates that the camera twin really is docked to the
    arm twin at *fk_frame*. That check matters because forward kinematics reported
    about the wrong link produces a solve that succeeds, has a small residual, and
    is wrong — the residual cannot detect it (see :class:`HandEyeResult`).

    Args:
        camera_twin: The docked camera twin. Needs an imaging sensor.
        arm_twin: The parent arm twin.
        fk_frame: Name of the link the camera is docked to, and the link
            ``base_to_gripper`` poses are measured about. Must match the twin's
            ``attach_to_link``.
        board: The calibration target. Keep it clamped and still for the session.
        intrinsics: ``{fx, fy, cx, cy}`` or a 3x3 camera matrix.
        dist_coeffs: Distortion coefficients; ``None`` means none.
        kinematics: Optional object with ``fk(joint_positions) -> 4x4``, used when
            :meth:`add_sample` is given ``joint_positions`` instead of a pose.
            ``cyberwave.driver.kinematics.arm.BaseKinematicsManipulator`` satisfies
            this (needs the ``drivers`` extra).
        frame_source: Passed to ``twin.camera.get_frame``.
        validate_attachment: Set ``False`` only to calibrate a camera that is not
            docked yet — you then own the correctness of *fk_frame*.
        min_correspondences: Corners a capture must place to be accepted at all.
            :meth:`add_sample` refuses anything below this rather than banking it,
            so the operator can re-aim while still at the arm. Cannot go below
            :data:`~cyberwave.calibration.board.MIN_CORRESPONDENCES`, the floor
            for a usable ``solvePnP``.
        min_markers: ArUco markers a capture must decode to be accepted,
            defaulting to :data:`MIN_MARKERS`. Enforced alongside
            *min_correspondences* and ignored for boards without markers. Pass ``0``
            to disable it.

    Note:
        **Frame freshness is on you.** No transport the platform currently offers
        guarantees that ``get_frame`` returns an image captured *after* the arm
        stopped moving: ``cloud`` serves the latest published frame whatever its
        age, and only ``zenoh`` honours ``max_age_ms``. A stale image paired with
        the current pose is a silently corrupt sample. Let the arm settle before
        calling :meth:`add_sample`, or pass an ``image`` you captured yourself.
    """

    def __init__(
        self,
        *,
        camera_twin: Any,
        arm_twin: Any,
        fk_frame: str,
        board: Board,
        intrinsics: Any = None,
        dist_coeffs: Sequence[float] | None = None,
        kinematics: Any = None,
        frame_source: str = "cloud",
        validate_attachment: bool = True,
        min_correspondences: int = MIN_CORRESPONDENCES,
        min_markers: int = MIN_MARKERS,
    ) -> None:
        self._camera_twin = camera_twin
        self._arm_twin = arm_twin
        self._fk_frame = str(fk_frame)
        self._board = board
        self._intrinsics = intrinsics
        self._dist_coeffs = dist_coeffs
        self._kinematics = kinematics
        self._frame_source = frame_source
        # Frames are retained only while the session is fitting its own
        # intrinsics: with caller-supplied ones there is nothing to refit, and the
        # images would just be held for the life of the run. This was a constructor
        # argument, but no caller ever set it and the one reachable value was this
        # expression, so it is derived rather than asked for.
        self._keep_frames = intrinsics is None
        #: Corners a capture must place to be accepted by :meth:`add_sample`.
        self._min_correspondences = max(int(min_correspondences), MIN_CORRESPONDENCES)
        #: ArUco markers a capture must decode to be accepted. Enforced alongside
        #: ``_min_correspondences`` and skipped for marker-less boards. Floored at 0
        #: rather than at ``MIN_MARKERS`` so a caller can turn this check off; the
        #: corner floor still applies and keeps a solve arithmetically possible.
        self._min_markers = max(int(min_markers), 0)
        self._samples: list[HandEyeSample] = []
        #: Frame size of the first capture. Every later one must match: see add_sample.
        self._frame_shape: tuple[int, int] | None = None

        #: Set when intrinsics were solved from the session's own captures rather
        #: than supplied. Carried into provenance so a stored calibration says
        #: where its intrinsics came from.
        self._solved_intrinsics: IntrinsicsResult | None = None

        # Fail fast on the intrinsics rather than on the first capture. Skipped
        # when they are to be solved: there is nothing to validate yet.
        if intrinsics is not None:
            camera_matrix(intrinsics)
        distortion_vector(dist_coeffs)
        if validate_attachment:
            self._validate_attachment()

    def _validate_attachment(self) -> None:
        parent_uuid = self._camera_twin._data_get("attach_to_twin_uuid")
        if not parent_uuid:
            raise CyberwaveValidationError(
                f"Camera twin {self._camera_twin.uuid} is not docked to anything. Dock it "
                "to the arm link the camera is physically mounted on first "
                "(scene.dock(...)), or pass validate_attachment=False."
            )
        if str(parent_uuid) != str(self._arm_twin.uuid):
            raise CyberwaveValidationError(
                f"Camera twin {self._camera_twin.uuid} is docked to twin {parent_uuid}, "
                f"not to the given arm twin {self._arm_twin.uuid}."
            )
        attach_to_link = self._camera_twin._data_get("attach_to_link")
        if str(attach_to_link or "") != self._fk_frame:
            raise CyberwaveValidationError(
                f"fk_frame {self._fk_frame!r} does not match the camera twin's "
                f"attach_to_link {attach_to_link!r}. These must be the same link: the "
                "solved transform is expressed relative to whatever link the camera is "
                "docked to, so a mismatch yields a wrong result with a good residual."
            )

    @property
    def samples(self) -> tuple[HandEyeSample, ...]:
        """Every accepted capture, in capture order."""
        return tuple(self._samples)

    @property
    def sample_count(self) -> int:
        """Captures a solve can use.

        Equal to :attr:`captured_count` in normal use: :meth:`add_sample` refuses
        a view that shows too little of the board, so an accepted capture already
        clears both floors. The two can still differ for a session constructed
        with samples added by other means.
        """
        return sum(1 for s in self._samples if self._is_solvable(s))

    @property
    def pending_count(self) -> int:
        """Captures with no board pose yet.

        Before :meth:`solve` this is every capture: corners are found at capture
        time, but turning them into a pose needs the camera model, which is
        fitted at solve. Afterwards it is normally zero, since every accepted
        capture clears the floors and gets posed.
        """
        return sum(1 for s in self._samples if not s.is_paired)

    @property
    def captured_count(self) -> int:
        """Everything the operator has captured, paired or not.

        This — not :attr:`sample_count` — is what an operator-facing progress
        indicator should report, because it never goes backwards and never sits
        at zero while captures are accumulating.
        """
        return len(self._samples)

    @property
    def board(self) -> Board:
        return self._board

    @property
    def fk_frame(self) -> str:
        return self._fk_frame

    def clear(self) -> None:
        """Drop all collected samples (e.g. after the board was bumped)."""
        # One list now, so nothing can be left behind in a side buffer.
        self._samples.clear()

    def _resolve_gripper_pose(
        self,
        base_to_gripper: Any | None,
        joint_positions: Mapping[str, float] | None,
    ) -> np.ndarray:
        if base_to_gripper is not None and joint_positions is not None:
            raise CyberwaveValidationError(
                "Pass either base_to_gripper or joint_positions, not both."
            )
        if base_to_gripper is not None:
            return _as_transform(
                base_to_gripper, label="base_to_gripper", index=self.sample_count
            )
        if joint_positions is None:
            raise CyberwaveValidationError(
                "add_sample needs the gripper pose: pass base_to_gripper=<4x4>, or "
                "joint_positions=<dict> together with a kinematics= object on the "
                "session. The arm twin's pose handle reports the twin's own pose in "
                "the world, not the end-effector, so it cannot be used here."
            )
        if self._kinematics is None:
            raise CyberwaveValidationError(
                "joint_positions was given but the session has no kinematics= object "
                "to run forward kinematics with. Pass one (e.g. "
                "BaseKinematicsManipulator, from the 'drivers' extra) or supply "
                "base_to_gripper directly."
            )
        return _as_transform(
            self._kinematics.fk(dict(joint_positions)),
            label="base_to_gripper",
            index=self.sample_count,
        )

    def _capture_image(self) -> Any:
        frame = self._camera_twin.camera.get_frame(
            format="numpy", source=self._frame_source
        )
        if frame is None:
            raise CyberwaveValidationError(
                f"No frame available from source {self._frame_source!r}. Check the "
                "camera driver is publishing, or pass image= explicitly."
            )
        return frame

    @property
    def intrinsics(self) -> Any:
        """The intrinsics in use, or ``None`` while they are still being solved."""
        return self._intrinsics

    @property
    def solved_intrinsics(self) -> IntrinsicsResult | None:
        """The solve, when intrinsics were derived from this session's own captures."""
        return self._solved_intrinsics

    def _is_solvable(self, sample: HandEyeSample) -> bool:
        """Whether *sample* shows enough of the board to take part in a solve.

        Two independent floors, because they catch different failures. Corners are
        what ``solvePnP`` consumes, so too few is arithmetically unsolvable. Markers
        are what carry board *identity*: a handful of decoded markers can still
        interpolate corners, but from so small a patch of the grid that the pose is
        badly conditioned. A board with no markers reports ``-1`` and skips the
        second floor rather than failing it.
        """
        if sample.correspondences is None:
            return False
        if len(sample.correspondences[0]) < self._min_correspondences:
            return False
        if sample.markers_read < 0:
            return True
        return sample.markers_read >= self._min_markers

    def _sample_correspondences(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The board correspondences of every capture, in capture order.

        Detection already happened in :meth:`add_sample`, so this is a collection
        rather than a recomputation. Captures below ``min_correspondences`` are
        skipped: too few points make a view that drags the camera model rather
        than constraining it.
        """
        return [s.correspondences for s in self._samples if self._is_solvable(s)]

    def _fit_intrinsics_from_all_views(self) -> bool:
        """Fit intrinsics from every captured view. True when they were updated."""
        correspondences = self._sample_correspondences()
        image_size = self._image_size()
        if image_size is None:
            return False
        try:
            self._solved_intrinsics = solve_intrinsics_from_correspondences(
                correspondences, image_size
            )
        except CyberwaveValidationError:
            # Not enough usable views yet. Keep collecting.
            return False

        self._adopt_solved_intrinsics()
        return True

    def _image_size(self) -> tuple[int, int] | None:
        """``(width, height)`` of the retained frames, or ``None`` if there are none."""
        for sample in self._samples:
            if sample.frame is None:
                continue
            shape = np.asarray(sample.frame).shape
            return (int(shape[1]), int(shape[0]))
        return None

    def _adopt_solved_intrinsics(self) -> None:
        """Take the fitted intrinsics (and distortion) as the ones in use."""
        assert self._solved_intrinsics is not None
        self._intrinsics = self._solved_intrinsics.as_dict()
        # Distortion is solved alongside and matters as much as the focal length
        # for a webcam, so adopt it too — unless the caller pinned their own.
        if self._dist_coeffs is None:
            self._dist_coeffs = list(self._solved_intrinsics.dist_coeffs)

    def _solve_intrinsics_and_pose_samples(self) -> None:
        """Fit intrinsics from every capture, then pose every capture with them.

        This is the whole intrinsics story: capture collects corners, and the
        camera model is solved once, here, from all of them. ``calibrateCamera``
        returns the focal length, principal point *and* distortion together, so
        nothing about the camera is guessed or carried over from an earlier fit.

        Both halves matter. Fitting alone would leave the board poses derived
        from whatever intrinsics happened to exist before, so each pose is
        recomputed from its cached correspondences: afterwards every pose in the
        solve shares one camera model.

        No-op when the caller supplied intrinsics -- those are theirs to own.
        """
        if self._intrinsics is not None and self._solved_intrinsics is None:
            # Caller pinned them before any fit of ours; leave well alone.
            return
        if not self._fit_intrinsics_from_all_views():
            return

        # Cached correspondences make this cheap: no board detection is repeated.
        # A capture whose points are missing keeps its previous pose rather than
        # being dropped -- silently changing the sample set at solve time would
        # be worse than one stale pose.
        for sample in self._samples:
            if not self._is_solvable(sample):
                # Left unposed on purpose, so it stays visible as a capture that
                # was made and excluded rather than silently vanishing.
                continue
            object_points, image_points = sample.correspondences
            pose = solve_target_pose(
                object_points, image_points, self._intrinsics, self._dist_coeffs
            )
            if pose is not None:
                sample.camera_to_target = pose

    def _read_joint_positions(self) -> dict[str, float] | None:
        """Current joint positions of the arm twin, or ``None`` if unreadable.

        Returns ``None`` rather than raising: a twin with no live joint feed is a
        perfectly valid manual-capture setup, and the settle check is an optional
        guard on top of it, not a new requirement.
        """
        try:
            view = self._arm_twin.get_joints()
        except Exception:
            return None
        try:
            positions = dict(view)
        except (TypeError, ValueError):
            return None
        numeric = {
            str(name): float(value)
            for name, value in positions.items()
            if isinstance(value, (int, float))
        }
        return numeric or None

    @staticmethod
    def _max_joint_drift(
        first: Mapping[str, float], second: Mapping[str, float]
    ) -> float | None:
        """Largest per-joint change between two reads, over the joints in both."""
        shared = set(first) & set(second)
        if not shared:
            return None
        return max(abs(first[name] - second[name]) for name in shared)

    def _wait_until_settled(
        self, *, tolerance_rad: float, timeout_s: float
    ) -> dict[str, float] | None:
        """Poll joint positions until two consecutive reads agree, or time out.

        Returns the settled positions, or ``None`` when joint states are not
        readable at all (in which case the caller skips the check entirely).
        """
        previous = self._read_joint_positions()
        if previous is None:
            return None

        deadline = time.monotonic() + timeout_s
        while True:
            time.sleep(_SETTLE_POLL_INTERVAL_S)
            current = self._read_joint_positions()
            if current is None:
                return None
            drift = self._max_joint_drift(previous, current)
            if drift is None or drift <= tolerance_rad:
                return current
            if time.monotonic() >= deadline:
                raise HandEyeSyncError(
                    f"Arm still moving after {timeout_s:.1f}s: largest joint drift "
                    f"{drift:.4f} rad exceeds the {tolerance_rad:.4f} rad settle "
                    "tolerance. Wait for the arm to stop before capturing, or raise "
                    "settle_timeout_s if the move is genuinely this slow."
                )
            previous = current

    def add_sample(
        self,
        *,
        base_to_gripper: Any | None = None,
        joint_positions: Mapping[str, float] | None = None,
        image: Any | None = None,
        settle: bool = True,
        settle_tolerance_rad: float = DEFAULT_SETTLE_TOLERANCE_RAD,
        settle_timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
    ) -> HandEyeSample:
        """Pair the current gripper pose with a detection of the board.

        When *settle* is set and the arm twin reports joint states, the arm is
        confirmed stationary before the frame is captured **and again afterwards**.
        The second check is the one that matters: an arm that was stationary before
        the grab and is in the same place after it cannot have been moving during
        it. That turns frame/pose pairing from an assumption into something
        checked, without needing a capture timestamp — which no transport currently
        exposes (``get_frame`` returns the frame alone; ``zenoh``'s ``max_age_ms``
        is applied inside the transport and never surfaces the stamp).

        Args:
            base_to_gripper: 4x4 pose of *fk_frame* in the arm's base frame.
            joint_positions: Joint angles to run through ``kinematics.fk`` instead.
            image: Image to use; captured from the camera twin when omitted. When
                given, the settle check is skipped — you captured it, so its
                pairing is yours to vouch for.
            settle: Wait for the arm to stop, and verify it stayed put across the
                capture. No-op when the arm twin has no readable joint states.
            settle_tolerance_rad: Per-joint movement counted as stationary.
            settle_timeout_s: How long to wait for the arm to stop.

        Returns:
            The stored :class:`HandEyeSample`.

        Raises:
            CyberwaveValidationError: no gripper pose given, or a malformed one.
            BoardNotDetectedError: the board was not found in the image.
            HandEyeSyncError: the arm never settled, or it moved during the
                capture — the sample is rejected rather than stored.
        """
        gripper_pose = self._resolve_gripper_pose(base_to_gripper, joint_positions)

        # Only meaningful when we capture the frame ourselves: a caller-supplied
        # image was taken at a time we know nothing about.
        check_settle = settle and image is None
        before = (
            self._wait_until_settled(
                tolerance_rad=settle_tolerance_rad, timeout_s=settle_timeout_s
            )
            if check_settle
            else None
        )

        frame = self._capture_image() if image is None else image

        # Intrinsics are per-resolution, and a session fits and poses every
        # capture against ONE camera model -- so a mid-session mode change (a
        # loaded edge device dropping to a smaller frame, say) silently mixes
        # scales and yields a confident, meaningless result. `_image_size`
        # reports whichever size the first retained frame had, so nothing
        # downstream can notice. Guarded here rather than at the fit because the
        # supplied-intrinsics path keeps no frames at all, yet still applies one
        # resolution's intrinsics to another's corners. `HandEyeFlow` and
        # `solve_intrinsics` already refuse this; the session is the one entry
        # point an SDK caller reaches directly, and the public docs promise it
        # ("mixing frame sizes in one session is rejected").
        # Only 2-D+ frames carry a size. Anything else is not an image, and the
        # detector below already reports that as BoardNotDetectedError -- the
        # message a caller is meant to see, so do not pre-empt it here.
        raw_shape = np.asarray(frame).shape
        frame_shape = tuple(int(v) for v in raw_shape[:2]) if len(raw_shape) >= 2 else None
        if frame_shape is None:
            pass
        elif self._frame_shape is None:
            self._frame_shape = frame_shape
        elif frame_shape != self._frame_shape:
            raise CyberwaveValidationError(
                f"Capture {self.captured_count + 1} is "
                f"{frame_shape[1]}x{frame_shape[0]} but earlier captures are "
                f"{self._frame_shape[1]}x{self._frame_shape[0]}. Intrinsics are "
                "per-resolution, so captures at different sizes cannot be mixed. "
                "Restart the session with a stable stream."
            )

        if before is not None:
            after = self._read_joint_positions()
            drift = None if after is None else self._max_joint_drift(before, after)
            if drift is not None and drift > settle_tolerance_rad:
                raise HandEyeSyncError(
                    f"Arm moved {drift:.4f} rad during capture "
                    f"{self.captured_count + 1} (tolerance {settle_tolerance_rad:.4f} "
                    "rad). The frame and the gripper pose describe different "
                    "instants, so this sample was discarded rather than stored."
                )

        # Detect the board and keep the correspondences. This needs no
        # intrinsics -- finding the corners is independent of the camera model;
        # only turning them into a pose is not. So a capture can be accepted or
        # refused on the spot, and the intrinsics become purely a solve-time
        # concern.
        cv2 = _import_cv2()
        # Refuse on the spot when too little of the board was seen. The operator is
        # standing there and can re-aim immediately, which is worth more than
        # banking a view that would only be dropped at solve time -- and a capture
        # counter that advances on unusable views misreports progress.
        detection = self._board.detect(cv2, np.asarray(frame), minimum=1)
        if detection is None:
            raise BoardNotDetectedError(
                f"Could not locate the calibration board in capture "
                f"{self.captured_count + 1}. Check the board is in frame, in focus "
                "and evenly lit, and that the board spec (dictionary, square count) "
                "matches what was printed."
            )
        correspondences, markers_read = detection
        # ``-1`` means the board has no markers at all (a plain chessboard), which
        # is "not applicable" rather than a failing count.
        if 0 <= markers_read < self._min_markers:
            raise BoardNotDetectedError(
                f"Only {markers_read} of the board's markers were read in capture "
                f"{self.captured_count + 1}; {self._min_markers} are needed. Bring "
                "more of the board into view, square-on, and check it is in focus "
                "and evenly lit."
            )
        if len(correspondences[0]) < self._min_correspondences:
            raise BoardNotDetectedError(
                f"Only {len(correspondences[0])} board corner(s) were placed in "
                f"capture {self.captured_count + 1}; {self._min_correspondences} are "
                "needed to resolve a pose. Bring more of the board into view."
            )

        sample = HandEyeSample(
            base_to_gripper=gripper_pose,
            frame=frame if self._keep_frames else None,
            correspondences=correspondences,
            markers_read=markers_read,
        )
        # Pose it against whatever intrinsics exist now, if any. Provisional
        # either way: solve() fits intrinsics from every capture and re-poses
        # them all, which is the only point at which the geometry is final.
        if self._intrinsics is not None and self._is_solvable(sample):
            object_points, image_points = correspondences
            sample.camera_to_target = solve_target_pose(
                object_points, image_points, self._intrinsics, self._dist_coeffs
            )
        self._samples.append(sample)
        return sample

    def solve(
        self,
        *,
        method: str = DEFAULT_METHOD,
        validate: bool = True,
    ) -> HandEyeResult:
        """Solve for the camera's pose on the gripper from the collected samples.

        The two pose lists are assembled here rather than by the caller, which is
        what rules out the swapped-argument mistake described on
        :class:`HandEyeResult`.

        Intrinsics are solved here, from every capture, before the hand-eye solve
        runs — see :meth:`_solve_intrinsics_and_pose_samples`. Capture only
        collects board corners, which needs no camera model, so the focal length,
        principal point and distortion are all fitted once from the full set and
        every board pose is measured against that one model.

        Args:
            method: Solver for the returned transform, defaulting to
                :data:`DEFAULT_METHOD`.
            validate: Also run :func:`leave_one_out_stability`, attaching it to the
                result. Costs one solve per sample — negligible against the capture
                it took to get here. Turn it off for a tight loop over synthetic
                data.
        """
        # Intrinsics first, then the geometry they imply.
        self._solve_intrinsics_and_pose_samples()
        # Every capture should be posed by now. Filter anyway rather than hand a
        # None to the solver: a capture whose points would not resolve to a pose
        # even under the final intrinsics is better skipped than fatal.
        paired = [s for s in self._samples if s.is_paired]
        if not paired:
            raise CyberwaveValidationError(
                f"None of the {len(self._samples)} capture(s) could be posed: none "
                f"showed both the {self._min_correspondences} board corners and the "
                f"{self._min_markers} markers a solve needs, or fewer than "
                f"{MIN_INTRINSICS_VIEWS} usable views were available to fit the "
                "camera model. Capture more with more of the board in frame, "
                "varying its angle and distance."
            )
        gripper_poses = [s.base_to_gripper for s in paired]
        target_poses = [s.camera_to_target for s in paired]
        result = solve_hand_eye(
            gripper_poses,
            target_poses,
            method=method,
        )
        if not validate:
            return replace(result, solved_intrinsics=self._solved_intrinsics)
        return result.with_validation(
            solved_intrinsics=self._solved_intrinsics,
            leave_one_out=leave_one_out_stability(
                gripper_poses, target_poses, method=method
            ),
        )

    def __repr__(self) -> str:
        return (
            f"HandEyeSession(fk_frame={self._fk_frame!r}, "
            f"samples={self.sample_count})"
        )


__all__ = [
    "DEFAULT_METHOD",
    "DEFAULT_SETTLE_TIMEOUT_S",
    "DEFAULT_SETTLE_TOLERANCE_RAD",
    "METHODS",
    "MIN_LEAVE_ONE_OUT_SAMPLES",
    "MIN_SAMPLES",
    "BoardNotDetectedError",
    "HandEyeDegenerateError",
    "HandEyeResult",
    "HandEyeSample",
    "HandEyeSession",
    "HandEyeSyncError",
    "LeaveOneOutResult",
    "leave_one_out_stability",
    "relative_errors",
    "solve_hand_eye",
]
