"""Timing and smoothing of local-IK joint paths: pure numerics, no solver."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cyberwave.driver.kinematics.arm.trajectory import (
    LocalTrajectory,
    PosePathPreflightError,
    WaypointSampling,
    interpolate_joint_transitions,
    smooth_trajectory,
    time_parameterise,
    waypoint_count,
)

JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_START = dict.fromkeys(JOINTS, 0.0)
_ACCELERATION = 2.0
#: A hardware speed cap of 10 % against a 3 rad/s controller: the planner times
#: against 0.3 rad/s per joint while the acceleration bound keeps its value.
_CAPPED_VELOCITY = dict.fromkeys(JOINTS, 0.3)
_UNRESTRICTED_VELOCITY = dict.fromkeys(JOINTS, 3.0)


def _timed(path, start=None, *, velocity_limits=None, acceleration=None):
    return time_parameterise(
        path,
        _START if start is None else start,
        velocity_limits=velocity_limits or _CAPPED_VELOCITY,
        max_joint_acceleration=_ACCELERATION if acceleration is None else acceleration,
    )


def _long_path(step: float = 0.2, count: int = 60) -> list[dict[str, float]]:
    """12 rad of joint1 travel in 0.2 rad samples: long enough to cruise."""
    return [{**_START, "joint1": step * (index + 1)} for index in range(count)]


def _cruise_velocity(path, times) -> float:
    """Joint-space speed across the middle of the profile."""
    middle = len(path) // 2
    travel = path[middle + 1]["joint1"] - path[middle - 1]["joint1"]
    return travel / (times[middle + 1] - times[middle - 1])


def _peak_velocities(path, times, start=_START):
    """Highest |dq_i| / dt reached on any segment, per joint."""
    peaks = dict.fromkeys(JOINTS, 0.0)
    previous, elapsed = start, 0.0
    for point, time_from_start in zip(path, times):
        span = time_from_start - elapsed
        for joint in JOINTS:
            delta = abs(point[joint] - previous.get(joint, point[joint]))
            peaks[joint] = max(peaks[joint], delta / span)
        previous, elapsed = point, time_from_start
    return peaks


# ── Sampling ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "distance,expected", [(0.0, 4), (0.01, 4), (0.06, 6), (0.5, 50), (5.0, 60)]
)
def test_waypoint_count_scales_with_distance_and_is_bounded(distance, expected) -> None:
    assert waypoint_count(distance) == expected


def test_a_longer_move_is_sampled_more_finely() -> None:
    """A straight line only stays straight if it is sampled."""
    assert waypoint_count(0.4) > waypoint_count(0.05)


def test_rotation_only_paths_are_sampled_too() -> None:
    """Four waypoints for a 0.12 rad wrist jog let one joint jump past its cap."""
    assert waypoint_count(0.0, 0.12) == 6
    assert waypoint_count(0.0, -0.12) == 6


def test_sampling_density_and_bounds_are_parameters() -> None:
    coarse = WaypointSampling(per_metre=10.0, per_radian=5.0, minimum=2, maximum=8)
    assert waypoint_count(0.5, sampling=coarse) == 5
    assert waypoint_count(0.0, sampling=coarse) == 2
    assert waypoint_count(5.0, sampling=coarse) == 8


def test_joint_transitions_are_expanded_at_the_sample_interval() -> None:
    path = [{**_START, "joint1": -4.561}]
    expanded = interpolate_joint_transitions(path, _START)

    assert len(expanded) == 23
    assert expanded[-1]["joint1"] == pytest.approx(-4.561)
    assert all(
        abs(point["joint1"] - previous["joint1"]) <= 0.2
        for point, previous in zip(expanded, [_START, *expanded[:-1]])
    )
    assert len(interpolate_joint_transitions(path, _START, sample_rad=1.0)) == 5
    with pytest.raises(ValueError):
        interpolate_joint_transitions(path, _START, sample_rad=0.0)


def test_large_joint_transition_gets_proportionally_more_time() -> None:
    expanded = interpolate_joint_transitions([{**_START, "joint1": -4.561}], _START)
    times = _timed(expanded)
    # 4.561 rad at 0.3 rad/s cruise, with a physical 2 rad/s² ramp.
    assert times[-1] == pytest.approx(4.561 / 0.3 + 0.3 / 2.0)


# ── Timing must match the speed the arm is actually told to move at ─────────


def test_timing_is_scaled_to_the_configured_ceiling() -> None:
    """A trajectory timed for an unrestricted arm while the hardware is capped
    lags by the cap ratio, and the trajectory controller aborts it."""
    path = _long_path()
    fast = _timed(path, velocity_limits=_UNRESTRICTED_VELOCITY)
    slow = _timed(path)
    # Only velocity scales with the cap; the acceleration bound stays physical,
    # so the slow ramp takes 0.15 s vs 1.5 s.
    assert _cruise_velocity(path, slow) == pytest.approx(
        _cruise_velocity(path, fast) / 10
    )
    assert slow[-1] == pytest.approx(fast[-1] * 10 - 9.9 * 1.5, rel=0.01)


def test_cruise_velocity_is_exactly_the_configured_per_joint_limit() -> None:
    path = _long_path()
    times = _timed(path, velocity_limits=_UNRESTRICTED_VELOCITY)
    assert _cruise_velocity(path, times) == pytest.approx(3.0)
    # 12 rad: 4 s at cruise plus one ramp's worth lost to starting and stopping.
    assert times[-1] == pytest.approx(12.0 / 3.0 + 3.0 / _ACCELERATION)


def test_the_profile_starts_and_ends_at_rest() -> None:
    """A constant-velocity law commands cruise speed from the first sample, which
    no real actuator can follow."""
    path = _long_path()
    times = _timed(path, velocity_limits=_UNRESTRICTED_VELOCITY)
    cruise = _cruise_velocity(path, times)
    assert 0.2 / times[0] < cruise / 2
    assert 0.2 / (times[-1] - times[-2]) < cruise / 2
    assert all(later > earlier for earlier, later in pairwise(times))


def test_a_short_multi_joint_jog_uses_a_slow_triangular_profile() -> None:
    largest_joint_delta = 0.04
    target = {
        **_START,
        "joint1": 0.01,
        "joint2": -0.02,
        "joint3": largest_joint_delta,
        "joint4": -0.03,
    }
    times = _timed([target])
    expected_duration = 2.0 * (largest_joint_delta / _ACCELERATION) ** 0.5
    assert times == pytest.approx([expected_duration])
    assert largest_joint_delta / times[0] < _CAPPED_VELOCITY["joint3"]


def test_times_are_strictly_increasing() -> None:
    """Two points at the same time_from_start is an infinite velocity."""
    times = _timed([dict(_START) for _ in range(5)])
    assert all(later > earlier for earlier, later in pairwise(times))
    assert times[0] == pytest.approx(0.02)


def test_no_segment_exceeds_its_own_joint_velocity_limit() -> None:
    path = _long_path()
    for joint, peak in _peak_velocities(path, _timed(path)).items():
        assert peak <= _CAPPED_VELOCITY[joint] * (1.0 + 1e-9)


def test_asymmetric_limits_bind_per_joint_rather_than_collapsing_to_a_scalar() -> None:
    limits = {**_UNRESTRICTED_VELOCITY, "joint2": 0.3}
    path = [
        {**_START, "joint1": 0.3 * (n + 1), "joint2": 0.3 * (n + 1)} for n in range(20)
    ]
    peaks = _peak_velocities(path, _timed(path, velocity_limits=limits))
    assert peaks["joint2"] <= 0.3 * (1.0 + 1e-9)
    assert peaks["joint1"] <= 0.3 * (1.0 + 1e-9)  # in lockstep with joint2
    faster = _timed(path, velocity_limits=_UNRESTRICTED_VELOCITY)
    assert faster[-1] < _timed(path, velocity_limits=limits)[-1]


def test_only_the_named_joints_are_timed() -> None:
    """A gripper in ``velocity_limits`` but absent from the path is not timed."""
    limits = {**_CAPPED_VELOCITY, "gripper": 0.01}
    times = time_parameterise(
        [{**_START, "joint1": 0.04}],
        _START,
        velocity_limits=limits,
        max_joint_acceleration=_ACCELERATION,
        joints=JOINTS,
    )
    assert times == pytest.approx([2.0 * (0.04 / _ACCELERATION) ** 0.5])


# ── Smoothing: limits hold BETWEEN the transmitted points ───────────────────


def _jtc_spline(trajectory):
    """Reconstruct a quintic-interpolating controller's segments from the
    actual transmitted fields."""
    interpolate = pytest.importorskip("scipy.interpolate")
    derivatives = [
        [
            [p[j] for j in JOINTS],
            [v[j] for j in JOINTS],
            [a[j] for j in JOINTS],
        ]
        for p, v, a in zip(
            trajectory.positions, trajectory.velocities, trajectory.accelerations
        )
    ]
    return interpolate.BPoly.from_derivatives(trajectory.time_from_start_s, derivatives)


def assert_smooth_limits(trajectory, velocity_limits, acceleration):
    curve = _jtc_spline(trajectory)
    times = np.linspace(0, trajectory.duration_s, 10001)
    assert np.all(
        np.abs(curve(times, 1)) <= np.array([velocity_limits[j] for j in JOINTS]) + 1e-8
    )
    assert np.max(np.abs(curve(times, 2))) <= acceleration + 1e-8
    for when in trajectory.time_from_start_s[1:-1]:
        # Both sides of a waypoint must join at the same acceleration.
        assert np.max(np.abs(curve(when - 1e-8, 2) - curve(when + 1e-8, 2))) < 1e-5
    assert np.max(np.abs(curve([0, trajectory.duration_s], 1))) < 1e-10
    assert np.max(np.abs(curve([0, trajectory.duration_s], 2))) < 1e-10
    np.testing.assert_allclose(
        curve(trajectory.time_from_start_s),
        [[p[j] for j in JOINTS] for p in trajectory.positions],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "travel", [[0.04], [0.2, 0.4, 0.6, 0.8], [0.04, 0.04, -0.01, 0.02], [0, 0, 0]]
)
def test_transmitted_quintic_respects_limits_between_points(travel):
    pytest.importorskip("scipy")
    limits = {**dict.fromkeys(JOINTS, 0.27), "joint2": 0.05}
    path = [
        {**_START, "joint1": q, "joint2": 0.1 * index} for index, q in enumerate(travel)
    ]
    trajectory = smooth_trajectory(
        path,
        _START,
        velocity_limits=limits,
        max_joint_acceleration=0.2,
        requested_pose_count=len(path),
    )
    assert isinstance(trajectory, LocalTrajectory)
    assert trajectory.positions[0] == _START
    assert trajectory.time_from_start_s[0] == 0
    assert list(trajectory.positions[0]) == list(JOINTS)
    assert trajectory.skipped_pose_count == 0
    assert_smooth_limits(trajectory, limits, 0.2)


def test_smoothing_refuses_joint_limit_overshoot_between_valid_waypoints():
    """The unconstrained quintic bows past joint1's limit on the plateau even
    though no waypoint requests it. Slowing the curve cannot make that legal."""
    pytest.importorskip("scipy")
    path = [
        {**_START, "joint1": q, "joint2": 0.04 * (i + 1)}
        for i, q in enumerate([0.1, 0.1, 0.1, 0.0])
    ]
    with pytest.raises(PosePathPreflightError, match="joint1 limits"):
        smooth_trajectory(
            path,
            _START,
            velocity_limits=dict.fromkeys(JOINTS, 0.27),
            max_joint_acceleration=0.2,
            requested_pose_count=len(path),
            joint_limits={**dict.fromkeys(JOINTS, (-5, 5)), "joint1": (0, 0.1)},
        )


def test_smoothing_checks_collisions_between_ik_samples():
    pytest.importorskip("scipy")
    with pytest.raises(PosePathPreflightError, match="smoothed trajectory: obstacle"):
        smooth_trajectory(
            [{**_START, "joint1": 0.1}],
            _START,
            velocity_limits=_CAPPED_VELOCITY,
            max_joint_acceleration=_ACCELERATION,
            requested_pose_count=1,
            collision_reason=lambda point: (
                "obstacle" if 0.04 < point["joint1"] < 0.06 else None
            ),
        )


@pytest.mark.parametrize(
    "path, limits, acceleration",
    [
        ([], _CAPPED_VELOCITY, 1.0),
        ([{**_START, "joint1": float("nan")}], _CAPPED_VELOCITY, 1.0),
        ([{**_START, "joint1": 0.1}], {**_CAPPED_VELOCITY, "joint1": 0.0}, 1.0),
        ([{**_START, "joint1": 0.1}], _CAPPED_VELOCITY, 0.0),
    ],
)
def test_smoothing_refuses_degenerate_input(path, limits, acceleration):
    with pytest.raises(PosePathPreflightError):
        smooth_trajectory(
            path,
            _START,
            velocity_limits=limits,
            max_joint_acceleration=acceleration,
            requested_pose_count=1,
        )
