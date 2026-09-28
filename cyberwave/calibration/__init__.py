"""Eye-in-hand camera calibration.

Measures where a camera actually sits on a robot link, instead of that transform
being entered by eye in the editor. Solves for the camera's pose on the link and
writes it back to the camera twin's docking offset::

    from cyberwave.calibration import CharucoBoard, HandEyeSession

    session = HandEyeSession(
        camera_twin=camera,            # already docked to the arm's wrist link
        arm_twin=arm,
        fk_frame="wrist_link",         # must equal camera's attach_to_link
        board=CharucoBoard(squares=(5, 7), square_size_m=0.030, marker_size_m=0.022),
        intrinsics={"fx": 615.0, "fy": 615.0, "cx": 319.5, "cy": 239.5},
    )

    for _ in range(12):
        input("Reorient the arm, let it settle, then press Enter...")
        session.add_sample(base_to_gripper=my_fk_4x4)

    result = session.solve()
    stability = result.leave_one_out
    if stability is not None and stability.max_stddev_m < 0.004:
        camera.calibration.set(result, board=session.board)

Three things decide whether the result is any good:

1. **Rotation, not translation.** The wrist must be *reoriented* about several
   distinct axes between captures. Sliding it around at a fixed orientation leaves
   the problem unsolvable; the session refuses rather than returning a guess.
2. **A measured board.** ``square_size_m`` is the only scale input — measure the
   print with calipers.
3. **Stability first, residual second.** Leave-one-out stability re-solves with
   each sample dropped and reports how far the *answer* moves, so it is the number
   the verdict is keyed on. The residual is a secondary check for a grossly bad
   fit. Note the blind spot both share, documented on
   :class:`~cyberwave.calibration.handeye.HandEyeResult`: they detect observations
   that disagree with each other, not inputs that were mislabelled.

Intrinsics are an input to the hand-eye solve, but you do not have to supply them:
omit ``intrinsics`` and :class:`~cyberwave.calibration.handeye.HandEyeSession`
solves real ones from the same board views it captures (see
:func:`~cyberwave.calibration.board.solve_intrinsics`). ``solve`` then refits them
from *every* captured view and re-detects each board pose against the result, so a
long run is solved with long-run intrinsics rather than the handful that happened to
be available when the fit first succeeded. The platform can also store intrinsics
per stream profile — :func:`~cyberwave.calibration.flow.intrinsics_from_twin` reads
them back via ``twin.camera.intrinsics_for(width, height)`` — and whichever set was
used is recorded in the calibration provenance either way.

``frames`` is numpy-only. Everything else needs OpenCV
(``pip install "cyberwave[calibration]"``), imported lazily so that importing this
package never requires it.
"""

from __future__ import annotations

from .board import (
    MIN_INTRINSICS_VIEWS,
    Board,
    CharucoBoard,
    CheckerBoard,
    IntrinsicsResult,
    camera_matrix,
    detect_target_pose,
    solve_intrinsics,
    solve_intrinsics_from_correspondences,
    solve_target_pose,
)
from .frames import (
    OPTICAL_TO_SENSOR,
    describe_pose,
    invert,
    make_transform,
    matrix_to_pose,
    matrix_to_quat_wxyz,
    pose_to_matrix,
    quat_wxyz_to_matrix,
    quat_wxyz_to_xyzw,
)
from .handeye import (
    DEFAULT_METHOD,
    DEFAULT_SETTLE_TIMEOUT_S,
    DEFAULT_SETTLE_TOLERANCE_RAD,
    METHODS,
    MIN_LEAVE_ONE_OUT_SAMPLES,
    MIN_MARKERS,
    MIN_SAMPLES,
    BoardNotDetectedError,
    HandEyeDegenerateError,
    HandEyeResult,
    HandEyeSample,
    HandEyeSession,
    HandEyeSyncError,
    LeaveOneOutResult,
    leave_one_out_stability,
    relative_errors,
    solve_hand_eye,
)
from .persistence import (
    HAND_EYE_METADATA_KEY,
    CameraCalibrationHandle,
    optical_to_attach_offset,
    resolve_sensor_offset,
)

# The edge-side guided flow. Imported lazily via __getattr__ below rather than
# here: `flow` pulls in `presentation` and `config`, and a caller that only wants
# the maths (or only wants to render an alert) should not pay for the rest. The
# public names stay in __all__ so `from cyberwave.calibration import HandEyeFlow`
# works exactly as if they were imported eagerly.
_LAZY_FLOW = {
    "HandEyeError": "flow",
    "HandEyeFlow": "flow",
    "HandEyeRunner": "flow",
    "intrinsics_from_twin": "flow",
    "resolve_docked_fk_frame": "flow",
    "DEFAULT_TARGET_SAMPLES": "config",
    "HandEyeFlowConfig": "config",
    "ALERT_TYPE": "presentation",
    "STATE_APPLIED": "presentation",
    "STATE_CAPTURING": "presentation",
    "STATE_ERROR": "presentation",
    "STATE_SOLVED": "presentation",
    "build_button": "presentation",
    "buttons_for_state": "presentation",
    "capture_mode_warning": "presentation",
    "describe_state": "presentation",
    "preflight_findings": "presentation",
    "preflight_warning": "presentation",
    "stability_from_result": "presentation",
    "verdict": "presentation",
    "FrameSource": "sources",
    "JointSource": "sources",
}


def __getattr__(name: str):
    """Resolve the flow/presentation/config names on first use."""
    module = _LAZY_FLOW.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)

__all__ = [
    "ALERT_TYPE",
    "DEFAULT_METHOD",
    "DEFAULT_SETTLE_TIMEOUT_S",
    "DEFAULT_SETTLE_TOLERANCE_RAD",
    "DEFAULT_TARGET_SAMPLES",
    "HAND_EYE_METADATA_KEY",
    "METHODS",
    "MIN_INTRINSICS_VIEWS",
    "MIN_LEAVE_ONE_OUT_SAMPLES",
    "MIN_MARKERS",
    "MIN_SAMPLES",
    "OPTICAL_TO_SENSOR",
    "Board",
    "BoardNotDetectedError",
    "CameraCalibrationHandle",
    "CharucoBoard",
    "CheckerBoard",
    "FrameSource",
    "HandEyeDegenerateError",
    "HandEyeError",
    "HandEyeFlow",
    "HandEyeFlowConfig",
    "HandEyeRunner",
    "HandEyeResult",
    "HandEyeSample",
    "HandEyeSession",
    "HandEyeSyncError",
    "IntrinsicsResult",
    "JointSource",
    "LeaveOneOutResult",
    "STATE_APPLIED",
    "STATE_CAPTURING",
    "STATE_ERROR",
    "STATE_SOLVED",
    "build_button",
    "buttons_for_state",
    "camera_matrix",
    "capture_mode_warning",
    "describe_state",
    "describe_pose",
    "detect_target_pose",
    "preflight_findings",
    "preflight_warning",
    "intrinsics_from_twin",
    "invert",
    "leave_one_out_stability",
    "relative_errors",
    "resolve_docked_fk_frame",
    "make_transform",
    "matrix_to_pose",
    "matrix_to_quat_wxyz",
    "optical_to_attach_offset",
    "pose_to_matrix",
    "quat_wxyz_to_matrix",
    "quat_wxyz_to_xyzw",
    "resolve_sensor_offset",
    "solve_hand_eye",
    "solve_intrinsics",
    "solve_intrinsics_from_correspondences",
    "stability_from_result",
    "verdict",
    "solve_target_pose",
]
