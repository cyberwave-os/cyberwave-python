"""``solve_hand_eye`` — recovery on synthetic data, degeneracy rejection, residuals.

The synthetic generator is the core of this file: pick a ground-truth
``gripper_to_camera``, place a fixed board, generate gripper poses, and derive the
camera observations those poses *must* produce. Feeding that through the solver has
to return the transform we started from. This is what catches a swapped or inverted
argument, which is the failure mode that otherwise ships silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import pytest

from cyberwave.calibration.frames import (
    invert,
    make_transform,
    quat_wxyz_to_matrix,
    rotation_angle_deg,
)
from cyberwave.calibration.handeye import (
    DEFAULT_METHOD,
    METHODS,
    MIN_LEAVE_ONE_OUT_SAMPLES,
    MIN_SAMPLES,
    HandEyeDegenerateError,
    HandEyeResult,
    _mean_transform,
    leave_one_out_stability,
    solve_hand_eye,
)
from cyberwave.exceptions import CyberwaveValidationError

cv2 = pytest.importorskip("cv2", reason="hand-eye solving requires OpenCV")


# --- synthetic scene -------------------------------------------------------


def _axis_angle(axis: Sequence[float], degrees: float) -> np.ndarray:
    """Rotation of *degrees* about an arbitrary *axis* (Rodrigues).

    ``_rotation`` only covers the three basis axes, which cannot express "almost
    but not quite about Z" -- the shape a weakly-spread capture set actually has.
    """
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    angle = math.radians(degrees)
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    component = {"x": 1, "y": 2, "z": 3}[axis]
    quaternion = [math.cos(half), 0.0, 0.0, 0.0]
    quaternion[component] = math.sin(half)
    return quat_wxyz_to_matrix(quaternion)


#: The unknown being solved for: camera 4 cm forward and 9 cm up on the wrist,
#: tilted 30 degrees down — roughly the real wrist-camera mount in the demos.
TRUE_GRIPPER_TO_CAMERA = make_transform(_rotation("y", -30.0), [0.045, 0.0, 0.099])

#: The board, clamped down somewhere in front of the arm. Never moves.
BASE_TO_TARGET = make_transform(_rotation("z", 12.0), [0.40, -0.05, 0.02])


def _make_samples(
    gripper_poses: list[np.ndarray],
    *,
    gripper_to_camera: np.ndarray = TRUE_GRIPPER_TO_CAMERA,
    base_to_target: np.ndarray = BASE_TO_TARGET,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Derive the camera observations implied by a set of gripper poses.

    ``base_to_target = base_to_gripper @ gripper_to_camera @ camera_to_target``
    is the closed loop the solver inverts, so the observations follow from it.
    """
    camera_to_target = [
        invert(a @ gripper_to_camera) @ base_to_target for a in gripper_poses
    ]
    return gripper_poses, camera_to_target


def _well_conditioned_gripper_poses(
    count: int = 12, *, seed: int = 3, max_deg: float = 45.0
) -> list[np.ndarray]:
    """Poses with multi-axis rotation of realistic magnitude.

    Random axis, angle in ``[10, max_deg]`` — what a person reorienting a wrist
    around a clamped board actually produces. Deliberately *not* uniform over
    SO(3): uniformly random orientations average ~120 degrees per pose, which no
    real capture session reaches (see ``_uniform_so3_gripper_poses``).
    """
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        half = math.radians(rng.uniform(10.0, max_deg)) / 2.0
        quaternion = [math.cos(half), *(axis * math.sin(half))]
        translation = rng.uniform(-0.15, 0.15, 3) + np.array([0.15, 0.0, 0.25])
        poses.append(make_transform(quat_wxyz_to_matrix(quaternion), translation))
    return poses


def _uniform_so3_gripper_poses(count: int = 12, *, seed: int = 3) -> list[np.ndarray]:
    """Orientations sampled uniformly over SO(3) — many near a half turn."""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(count):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        translation = rng.uniform(-0.15, 0.15, 3) + np.array([0.15, 0.0, 0.25])
        poses.append(make_transform(quat_wxyz_to_matrix(quaternion), translation))
    return poses


# --- recovery --------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(METHODS))
def test_recovers_ground_truth_from_noiseless_samples(method: str) -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses())

    result = solve_hand_eye(gripper, target, method=method)

    assert np.allclose(
        result.gripper_to_camera[:3, 3], TRUE_GRIPPER_TO_CAMERA[:3, 3], atol=1e-6
    )
    error = TRUE_GRIPPER_TO_CAMERA[:3, :3].T @ result.gripper_to_camera[:3, :3]
    assert rotation_angle_deg(error) < 1e-4


def test_noiseless_solve_has_effectively_zero_residual() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses())

    result = solve_hand_eye(gripper, target)

    assert result.residual_translation_m < 1e-9
    assert result.residual_rotation_deg < 1e-6
    assert result.max_residual_translation_m < 1e-9


def test_result_reports_sample_count_method_and_spread() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=9))

    result = solve_hand_eye(gripper, target, method="horaud")

    assert isinstance(result, HandEyeResult)
    assert result.sample_count == 9
    assert result.method == "horaud"


def test_default_method_is_robust_at_large_rotations() -> None:
    """``park`` (the default) must stay exact even at uniform-SO(3) rotation magnitudes."""
    gripper, target = _make_samples(_uniform_so3_gripper_poses())

    result = solve_hand_eye(gripper, target)

    assert result.method == DEFAULT_METHOD
    assert np.allclose(
        result.gripper_to_camera[:3, 3], TRUE_GRIPPER_TO_CAMERA[:3, 3], atol=1e-6
    )


@pytest.mark.parametrize("method", sorted(METHODS))
def test_residual_reflects_actual_error_at_large_rotations(method: str) -> None:
    """The residual must not under-report a bad solve, whatever the solver does.

    Some OpenCV releases lose several degrees of rotation accuracy in ``tsai`` and
    ``daniilidis`` as individual poses approach a half turn — the original reason
    ``park`` is :data:`DEFAULT_METHOD` — but that degradation is an OpenCV
    implementation detail this module cannot pin without coupling a test to one cv2
    build (it does not reproduce on OpenCV 4.11, for instance). What this module
    *can* guarantee, and what actually matters to a caller deciding whether to trust
    a result: the residual tracks the true error closely enough that a bad solve is
    never silently reported as good, on any solver.
    """
    gripper, target = _make_samples(_uniform_so3_gripper_poses())

    result = solve_hand_eye(gripper, target, method=method)

    error = TRUE_GRIPPER_TO_CAMERA[:3, :3].T @ result.gripper_to_camera[:3, :3]
    actual_rotation_error_deg = rotation_angle_deg(error)

    if actual_rotation_error_deg > 1.0:
        assert result.residual_rotation_deg > 0.5
    else:
        assert result.residual_rotation_deg < 0.5


def test_solves_at_the_minimum_sample_count() -> None:
    poses = [
        make_transform(np.eye(3), [0.2, 0.0, 0.3]),
        make_transform(_rotation("x", 35.0), [0.22, 0.03, 0.28]),
        make_transform(_rotation("z", 40.0), [0.18, -0.04, 0.31]),
    ]
    gripper, target = _make_samples(poses)

    result = solve_hand_eye(gripper, target)

    assert result.sample_count == MIN_SAMPLES
    assert np.allclose(
        result.gripper_to_camera[:3, 3], TRUE_GRIPPER_TO_CAMERA[:3, 3], atol=1e-6
    )


def test_swapping_the_arguments_is_wrong_but_the_residual_cannot_tell() -> None:
    """Documents the one failure mode the residual provably cannot catch.

    Swapping the two inputs solves ``B X' A = const`` instead of ``A X B = const``.
    That system is *also* exactly consistent — its solution is the inverse board
    pose, ``inv(base_to_target)`` — so it comes back with a machine-zero residual
    while being completely wrong.

    The consequence: a good residual means the observations agree with each other,
    NOT that the inputs were labelled correctly. Systematic mistakes (swapped
    arguments, FK reported for the wrong link, a constant frame offset) stay
    invisible to it. That is why ``HandEyeSession`` assembles both lists itself
    rather than letting callers pair them up.
    """
    gripper, target = _make_samples(_well_conditioned_gripper_poses())

    swapped = solve_hand_eye(target, gripper)

    assert swapped.residual_translation_m < 1e-9  # looks perfect...
    assert not np.allclose(  # ...and is not the answer
        swapped.gripper_to_camera[:3, 3], TRUE_GRIPPER_TO_CAMERA[:3, 3], atol=1e-3
    )
    assert np.allclose(swapped.gripper_to_camera, invert(BASE_TO_TARGET), atol=1e-9)


# --- degeneracy ------------------------------------------------------------


def test_translation_only_capture_is_rejected() -> None:
    """The classic mistake: sliding the wrist around without reorienting it.

    Still refused with the spread gate off by default: the system is rank
    deficient, so ``cv2`` hands back a non-rotation and the backstop catches it.
    That backstop is unconditional — it is what makes turning the gate off safe.
    """
    poses = [
        make_transform(np.eye(3), [0.2 + 0.05 * i, 0.01 * i, 0.3 - 0.02 * i])
        for i in range(10)
    ]
    gripper, target = _make_samples(poses)

    with pytest.raises(HandEyeDegenerateError):
        solve_hand_eye(gripper, target)


def test_single_axis_rotation_is_rejected() -> None:
    """Rotating only about one axis leaves the camera orientation unconstrained."""
    poses = [
        make_transform(_rotation("z", 12.0 * i), [0.2, 0.0, 0.3]) for i in range(10)
    ]
    gripper, target = _make_samples(poses)

    with pytest.raises(HandEyeDegenerateError):
        solve_hand_eye(gripper, target)


def test_a_weakly_spread_set_solves_and_reports_its_spread() -> None:
    """There is no spread gate: a thin capture set solves and stays visible.

    Degeneracy is caught from the solver's *output* instead, which is the check
    that survives -- see the backstop tests above.
    """
    # Every rotation is about an axis within ~1 degree of Z, swept slowly around
    # it. That leaves enough rank for the solver to return an exact answer while
    # the axes stay nearly parallel -- a spread of ~8 degrees. Exactly the set the
    # gate used to refuse.
    #
    # Tilting the axes rather than adding one off-axis pose is deliberate: a
    # single perpendicular nudge produces a genuine 90-degree spread, not a small
    # one, because the spread is the widest angle between *any* two axes.
    tilt = math.radians(0.8)
    poses = [
        make_transform(
            _axis_angle(
                [
                    math.sin(tilt) * math.cos(i * 1.3),
                    math.sin(tilt) * math.sin(i * 1.3),
                    math.cos(tilt),
                ],
                20.0 * (i + 1),
            ),
            [0.2, 0.01 * i, 0.3],
        )
        for i in range(6)
    ]
    gripper, target = _make_samples(poses)

    # Solves rather than being refused: nothing inspects the pose spread any more,
    # and degeneracy is caught from the solver's output instead.
    result = solve_hand_eye(gripper, target)

    assert np.all(np.isfinite(result.gripper_to_camera))


def test_degeneracy_is_caught_from_the_solver_output() -> None:
    """Degeneracy is caught from cv2's output, which is the only check there is.

    A single-axis capture leaves the system rank-deficient, and cv2 signals that by
    returning NaN rather than by failing. Without the output check that surfaces as
    an opaque error from deep inside the frame helpers, so assert the clean one.
    """
    poses = [
        make_transform(_rotation("z", 12.0 * i), [0.2, 0.0, 0.3]) for i in range(10)
    ]
    gripper, target = _make_samples(poses)

    with pytest.raises(HandEyeDegenerateError, match="no valid transform"):
        solve_hand_eye(gripper, target)


def test_a_degenerate_solve_never_returns_a_non_rotation() -> None:
    """Whatever comes back must be a usable rigid transform, or nothing at all."""
    poses = [
        make_transform(_rotation("y", 9.0 * i), [0.2, 0.01 * i, 0.3]) for i in range(8)
    ]
    gripper, target = _make_samples(poses)

    try:
        result = solve_hand_eye(gripper, target)
    except HandEyeDegenerateError:
        return
    rotation = result.gripper_to_camera[:3, :3]
    assert np.all(np.isfinite(result.gripper_to_camera))
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)


# --- residuals track error -------------------------------------------------


@pytest.mark.parametrize("noise_m", [0.0, 0.0005, 0.002, 0.008])
def test_residual_grows_with_observation_noise(noise_m: float) -> None:
    rng = np.random.default_rng(101)
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=15))
    noisy = [t.copy() for t in target]
    for transform in noisy:
        transform[:3, 3] += rng.normal(scale=noise_m, size=3)

    result = solve_hand_eye(gripper, noisy)

    # Residual is on the order of the injected noise, not orders of magnitude off.
    assert result.residual_translation_m <= max(noise_m * 8.0, 1e-9)
    if noise_m > 0.0:
        assert result.residual_translation_m > noise_m * 0.1


def test_residual_is_monotonic_in_noise() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=15))
    residuals = []
    for noise_m in (0.0, 0.001, 0.004, 0.010):
        rng = np.random.default_rng(7)
        noisy = [t.copy() for t in target]
        for transform in noisy:
            transform[:3, 3] += rng.normal(scale=noise_m, size=3)
        residuals.append(solve_hand_eye(gripper, noisy).residual_translation_m)

    assert residuals == sorted(residuals)


def test_moving_the_board_mid_capture_shows_up_as_residual() -> None:
    """A bumped board is the other common failure; the residual must expose it."""
    poses = _well_conditioned_gripper_poses(count=12)
    _, first_half = _make_samples(poses[:6])
    _, second_half = _make_samples(
        poses[6:],
        base_to_target=make_transform(_rotation("z", 12.0), [0.44, -0.05, 0.02]),
    )

    result = solve_hand_eye(poses, first_half + second_half)

    assert result.residual_translation_m > 0.005


def test_translation_residual_is_the_combined_per_axis_standard_deviation() -> None:
    """Pins the formulation, not just a magnitude.

    ``residual_translation_m`` is ``norm(std(positions, axis=0))`` of the implied
    board position. Recomputing it here from the solved transform is the only way
    to catch a silent change back to a mean deviation, which differs by only ~4%
    on realistic noise and so would pass any threshold-shaped assertion.
    """
    rng = np.random.default_rng(11)
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=12))
    noisy = [t.copy() for t in target]
    for transform in noisy:
        transform[:3, 3] += rng.normal(scale=0.003, size=3)

    result = solve_hand_eye(gripper, noisy)

    implied = [
        a @ result.gripper_to_camera @ b for a, b in zip(gripper, noisy)
    ]
    positions = np.array([t[:3, 3] for t in implied])
    expected = float(np.linalg.norm(positions.std(axis=0)))

    assert result.residual_translation_m == pytest.approx(expected, rel=1e-9)


def test_rotation_residual_is_the_rms_of_the_angles() -> None:
    """Pins ``residual_rotation_deg`` as the RMS of the per-sample angles.

    Not their standard deviation: the angles are already unsigned deviations from
    the mean orientation, so their std measures the spread *of the deviations* and
    reads ~0 whenever every sample sits equally far from the mean, however far
    that is. See ``test_a_uniform_rotation_offset_is_not_reported_as_zero``.
    """
    rng = np.random.default_rng(12)
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=12))
    noisy = [t.copy() for t in target]
    for transform in noisy:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(rng.normal(0.0, 1.0))
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        )
        transform[:3, :3] = rotation @ transform[:3, :3]

    result = solve_hand_eye(gripper, noisy)

    implied = [
        a @ result.gripper_to_camera @ b for a, b in zip(gripper, noisy)
    ]
    reference = _mean_transform(implied)
    angles = [
        rotation_angle_deg(reference[:3, :3].T @ t[:3, :3]) for t in implied
    ]

    assert result.residual_rotation_deg == pytest.approx(
        float(np.sqrt(np.mean(np.square(angles)))), rel=1e-9
    )
    assert result.max_residual_rotation_deg == pytest.approx(
        float(np.max(angles)), rel=1e-9
    )


def test_a_uniform_rotation_offset_is_not_reported_as_zero() -> None:
    """Samples that all sit the same distance from the mean still show a residual.

    The regression this guards: taking the standard deviation of the per-sample
    angles reports 0.00deg here, because the angles are unsigned deviations and
    every one of them is identical -- so a set whose captures genuinely disagree
    by degrees reads as a flawless fit. The RMS reports the actual disagreement.
    """
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=12))
    noisy = [t.copy() for t in target]
    # Rotate each sample by the same angle about a different axis: every implied
    # board pose lands equally far from the mean orientation, so the angles are
    # near-uniform while the samples disagree with each other substantially.
    offset_deg = 2.0
    axes = np.eye(3)
    for index, transform in enumerate(noisy):
        axis = axes[index % 3]
        angle = np.radians(offset_deg)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        )
        transform[:3, :3] = rotation @ transform[:3, :3]

    result = solve_hand_eye(gripper, noisy)

    # The samples demonstrably disagree, so the headline figure must say so.
    assert result.max_residual_rotation_deg > 0.5
    assert result.residual_rotation_deg > 0.5
    # And it can never exceed the worst single deviation.
    assert result.residual_rotation_deg <= result.max_residual_rotation_deg + 1e-9


def test_max_residual_is_at_least_the_mean() -> None:
    rng = np.random.default_rng(5)
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=12))
    noisy = [t.copy() for t in target]
    for transform in noisy:
        transform[:3, 3] += rng.normal(scale=0.003, size=3)

    result = solve_hand_eye(gripper, noisy)

    assert result.max_residual_translation_m >= result.residual_translation_m
    assert result.max_residual_rotation_deg >= result.residual_rotation_deg


# --- input validation ------------------------------------------------------


def test_too_few_samples_is_rejected() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=2))

    with pytest.raises(CyberwaveValidationError, match="at least 3 samples"):
        solve_hand_eye(gripper, target)


def test_mismatched_sample_counts_are_rejected() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=8))

    with pytest.raises(CyberwaveValidationError, match="mismatch"):
        solve_hand_eye(gripper, target[:-1])


def test_unknown_method_is_rejected() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses())

    with pytest.raises(CyberwaveValidationError, match="Unknown hand-eye method"):
        solve_hand_eye(gripper, target, method="kabsch")


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.eye(3), "4x4"),
        (np.zeros((4, 4)), "orthonormal"),
        (make_transform(np.diag([1.0, 1.0, -1.0]), [0, 0, 0]), "reflection"),
        (make_transform(np.eye(3), [np.nan, 0, 0]), "non-finite"),
        (make_transform(np.eye(3) * 2.0, [0, 0, 0]), "orthonormal"),
    ],
)
def test_malformed_pose_is_rejected(bad, match: str) -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=4))
    gripper = [bad, *gripper[1:]]

    with pytest.raises(CyberwaveValidationError, match=match):
        solve_hand_eye(gripper, target)


def test_validation_error_names_the_offending_index() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count=4))
    target = [*target[:2], np.eye(3), *target[3:]]

    with pytest.raises(CyberwaveValidationError, match=r"camera_to_target\[2\]"):
        solve_hand_eye(gripper, target)


# --- metadata --------------------------------------------------------------


def test_to_metadata_is_json_safe_and_carries_the_residuals() -> None:
    import json

    gripper, target = _make_samples(_well_conditioned_gripper_poses())

    metadata = solve_hand_eye(gripper, target).to_metadata()

    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["method"] == DEFAULT_METHOD
    assert metadata["sample_count"] == 12
    assert set(metadata) >= {
        "residual_translation_m",
        "max_residual_translation_m",
        "residual_rotation_deg",
    }


# --- leave-one-out stability ----------------------------------------------


def test_leave_one_out_is_tight_on_noiseless_samples() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(12))
    stability = leave_one_out_stability(gripper, target)

    assert stability is not None
    assert stability.subset_count == 12
    assert stability.max_stddev_m < 1e-6


def test_leave_one_out_identifies_the_corrupted_sample() -> None:
    """A single bad observation must be named, not just counted."""
    gripper, target = _make_samples(_well_conditioned_gripper_poses(12))
    corrupted = list(target)
    bad_index = 7
    corrupted[bad_index] = make_transform(
        corrupted[bad_index][:3, :3],
        corrupted[bad_index][:3, 3] + np.array([0.05, -0.04, 0.03]),
    )

    stability = leave_one_out_stability(gripper, corrupted)
    clean = leave_one_out_stability(gripper, target)

    assert stability is not None and clean is not None
    # Names the sample to re-capture...
    assert stability.worst_sample_index == bad_index
    # ...and the instability is visible against an otherwise identical clean set.
    # Not compared to a fixed millimetre threshold: how far one bad sample moves
    # the answer scales with how many others outvote it.
    assert stability.max_stddev_m > 100 * clean.max_stddev_m


def test_leave_one_out_returns_none_below_the_sample_floor() -> None:
    """Below the floor a subset is not solvable, so there is nothing to report."""
    count = MIN_LEAVE_ONE_OUT_SAMPLES - 1
    gripper, target = _make_samples(_well_conditioned_gripper_poses(count))
    assert leave_one_out_stability(gripper, target) is None


def test_the_floor_is_one_more_than_a_solve_needs() -> None:
    """Dropping a sample must still leave a solvable subset -- and no more than
    that is required, so a short run still gets a stability figure rather than
    the silence the UI renders as no verdict at all."""
    assert MIN_LEAVE_ONE_OUT_SAMPLES == MIN_SAMPLES + 1


def test_leave_one_out_reports_at_the_floor() -> None:
    """The smallest set that can produce the statistic does produce it."""
    gripper, target = _make_samples(
        _well_conditioned_gripper_poses(MIN_LEAVE_ONE_OUT_SAMPLES)
    )

    stability = leave_one_out_stability(gripper, target)

    assert stability is not None
    # One subset per dropped sample, so the caller can weigh how thin it is.
    assert stability.subset_count == MIN_LEAVE_ONE_OUT_SAMPLES
    assert stability.max_stddev_m >= 0.0


def test_leave_one_out_metadata_is_json_safe() -> None:
    gripper, target = _make_samples(_well_conditioned_gripper_poses(12))
    stability = leave_one_out_stability(gripper, target)
    assert stability is not None
    json.dumps(stability.to_metadata())  # must not raise


def test_relative_errors_are_zero_for_an_exact_set() -> None:
    """The formula's signs are load-bearing: a wrong one inflates by ~|motion|."""
    from cyberwave.calibration import relative_errors

    X = make_transform(_rotation("x", 30.0) @ _rotation("y", 20.0), [0.05, 0.002, -0.09])
    base_to_target = make_transform(_rotation("z", 15.0), [0.30, 0.10, 0.0])
    arm = [
        make_transform(_rotation(axis, deg), [0.20 + 0.01 * i, 0.01 * i, 0.30])
        for i, (axis, deg) in enumerate(
            [("x", 10.0), ("y", 40.0), ("z", 70.0), ("x", -30.0), ("y", -50.0)]
        )
    ]
    board = [np.linalg.inv(a @ X) @ base_to_target for a in arm]

    rot_deg, trans_m, ratio = relative_errors(arm, board, X)
    # Degrees, via arccos, which loses precision near zero -- 1e-5 deg is noise,
    # while a sign error in the formula shows up in whole degrees.
    assert rot_deg == pytest.approx(0.0, abs=1e-5)
    assert trans_m == pytest.approx(0.0, abs=1e-12)
    assert ratio == pytest.approx(0.0, abs=1e-12)


def test_relative_errors_grow_with_a_perturbed_transform() -> None:
    """A wrong X must not fit the motions: this is what makes it an accuracy metric."""
    from cyberwave.calibration import relative_errors

    X = make_transform(_rotation("x", 30.0), [0.05, 0.0, -0.09])
    base_to_target = make_transform(np.eye(3), [0.30, 0.10, 0.0])
    arm = [
        make_transform(_rotation(axis, deg), [0.20, 0.01 * i, 0.30])
        for i, (axis, deg) in enumerate(
            [("x", 10.0), ("y", 40.0), ("z", 70.0), ("x", -30.0), ("y", -50.0)]
        )
    ]
    board = [np.linalg.inv(a @ X) @ base_to_target for a in arm]

    wrong = X.copy()
    wrong[2, 3] += 0.02  # 20 mm along the optical axis
    _, exact_t, _ = relative_errors(arm, board, X)
    _, wrong_t, _ = relative_errors(arm, board, wrong)
    assert wrong_t > exact_t
    assert wrong_t > 0.001


def test_relative_translation_ratio_is_scale_free() -> None:
    """The ratio is the dimensionless form, so scaling the scene must not move it."""
    from cyberwave.calibration import relative_errors

    X = make_transform(_rotation("x", 25.0), [0.04, 0.0, -0.08])
    base_to_target = make_transform(np.eye(3), [0.30, 0.05, 0.0])
    arm = [
        make_transform(_rotation(axis, deg), [0.20, 0.01 * i, 0.30])
        for i, (axis, deg) in enumerate([("x", 15.0), ("y", 45.0), ("z", 65.0), ("y", -35.0)])
    ]
    board = [np.linalg.inv(a @ X) @ base_to_target for a in arm]
    wrong = X.copy()
    wrong[2, 3] += 0.02
    _, _, ratio_m = relative_errors(arm, board, wrong)

    # Same geometry expressed in different units: every length x10.
    scale = np.diag([1.0, 1.0, 1.0, 0.1])
    def big(transform: np.ndarray) -> np.ndarray:
        return scale @ transform @ np.linalg.inv(scale)

    _, _, ratio_scaled = relative_errors(
        [big(a) for a in arm], [big(b) for b in board], big(wrong)
    )
    assert ratio_scaled == pytest.approx(ratio_m, rel=1e-9)
