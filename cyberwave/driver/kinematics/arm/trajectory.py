"""Timed, rest-to-rest joint trajectories from solved IK paths. Pure numerics.

The geometry half of local Cartesian planning lives in
:class:`~cyberwave.driver.kinematics.arm.controller.ArmController` (IK one
waypoint at a time). This module is the timing half: how many waypoints a
Cartesian path needs, how long each segment may take under per-joint velocity
and a physical acceleration bound, and a quintic spline fitted through them
whose derivatives -- including BETWEEN the IK samples -- respect those limits.

The output is shaped for a ``joint_trajectory_controller``-style executor that
interpolates quintically from positions, velocities and accelerations: every
waypoint carries all three, so the executor's own interpolation reproduces the
audited curve rather than inventing one.

No robot specifics: joint names come from the caller (or the keys of
``velocity_limits``), and every rate is an argument. scipy is imported lazily,
like :mod:`cyberwave.driver.control.motion`, so importing this module costs
nothing for a driver that never times a path.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

#: Keep distinct waypoint timestamps even for repeated positions: two points at
#: the same ``time_from_start`` is an infinite velocity to a trajectory
#: controller. This is the initial minimum between spline knots, independent of
#: publication rate; retiming may increase it to satisfy the spline's limits.
MIN_SEGMENT_S = 0.02

#: Joint travel (rad) between collision samples on the smoothed spline. The
#: check is sampled, not continuous; this bounds how much motion a sample can
#: hide between two checks.
COLLISION_SAMPLE_RAD = 0.02

#: Interval for expanding a joint-space PTP transition into audited samples.
#: This is not a rejection threshold: every legal delta is expanded and timed.
JOINT_SAMPLE_RAD = 0.2

_SCIPY_INSTALL_HINT = (
    "scipy is required for local trajectory smoothing. Install it with: "
    'pip install "cyberwave[drivers]"'
)


def _import_spline() -> tuple[Any, Any]:
    try:
        from scipy.interpolate import (
            PPoly,
            make_interp_spline,
        )
    except ImportError as exc:
        raise ImportError(_SCIPY_INSTALL_HINT) from exc
    return PPoly, make_interp_spline


class PosePathPreflightError(RuntimeError):
    """A complete pose path could not become one safe, timed trajectory."""


@dataclass(frozen=True, slots=True)
class PosePathPreflight:
    """Motion-free result of checking an entire Cartesian pose path.

    ``positions`` includes interpolated joint-space PTP samples across any IK
    branch transition, and the start snapshot as its first row. Every position
    has passed joint limits in IK and whatever collision model the solver
    exposes; ``time_from_start_s`` is ready for a trajectory controller.
    """

    positions: tuple[dict[str, float], ...]
    time_from_start_s: tuple[float, ...]
    velocities: tuple[dict[str, float], ...]
    accelerations: tuple[dict[str, float], ...]
    requested_pose_count: int
    solved_pose_count: int

    @property
    def skipped_pose_count(self) -> int:
        return self.requested_pose_count - self.solved_pose_count

    @property
    def duration_s(self) -> float:
        return self.time_from_start_s[-1] if self.time_from_start_s else 0.0


#: The same thing, named for what it carries: a timed trajectory.
LocalTrajectory = PosePathPreflight


@dataclass(frozen=True, slots=True)
class WaypointSampling:
    """How densely a straight Cartesian path is sampled for IK.

    A straight line only stays straight if it is sampled: the executor
    interpolates in JOINT space between samples, so the tool point bows off the
    line within one sample interval. Rotation-only motions need sampling too --
    too few waypoints for a small wrist jog lets one joint jump past a per-step
    cap at a single waypoint.

    The defaults (1 cm, ~1.1 deg, 4..60 samples) are generic: they describe
    planning resolution, not any arm's hardware.
    """

    per_metre: float = 100.0
    per_radian: float = 50.0
    minimum: int = 4
    maximum: int = 60

    def count(self, distance_m: float, angle_rad: float = 0.0) -> int:
        """Samples for a path of ``distance_m`` travel and ``angle_rad`` rotation."""
        raw = max(
            math.ceil(abs(distance_m) * self.per_metre),
            math.ceil(abs(angle_rad) * self.per_radian - 1e-9),
        )
        return max(self.minimum, min(self.maximum, raw))


DEFAULT_WAYPOINT_SAMPLING = WaypointSampling()


def waypoint_count(
    distance_m: float,
    angle_rad: float = 0.0,
    *,
    sampling: WaypointSampling = DEFAULT_WAYPOINT_SAMPLING,
) -> int:
    """Sample both TCP translation and rotation along a Cartesian path."""
    return sampling.count(distance_m, angle_rad)


def _joint_names(
    joints: Sequence[str] | None, velocity_limits: Mapping[str, float]
) -> tuple[str, ...]:
    return tuple(joints) if joints is not None else tuple(velocity_limits)


def time_parameterise(
    path: list[dict[str, float]],
    start: dict[str, float],
    *,
    velocity_limits: Mapping[str, float],
    max_joint_acceleration: float,
    joints: Sequence[str] | None = None,
    min_segment_s: float = MIN_SEGMENT_S,
) -> list[float]:
    """Initial timing estimate for each waypoint, in seconds.

    A trapezoidal profile over a NORMALIZED path coordinate: each segment costs
    ``max_i(|dq_i| / v_i)`` seconds at full rate, where ``v_i`` is that joint's
    own physical velocity limit. A path rate of at most 1 therefore guarantees
    every joint stays under its own bound -- which is the whole point of
    normalizing per joint rather than against one scalar. It starts and ends at
    rest; a short path becomes a triangle.

    This bounds segment averages, not the interpolated joint derivatives.
    :func:`smooth_trajectory` enforces the limits on the final spline. Timing
    and the arm's commanded cap must agree: hardware that executes one global
    speed setting lags a trajectory timed faster than it will move, and the
    trajectory controller aborts on path tolerance. A constant-velocity law
    fails the same way at t=0 -- real actuators cannot step to cruise speed.

    ``joints`` defaults to the keys of ``velocity_limits``.
    """
    names = _joint_names(joints, velocity_limits)
    limits = {joint: max(float(velocity_limits[joint]), 1e-6) for joint in names}
    # Full rate in the normalized coordinate is 1.0 by construction.
    limit = 1.0
    accel = min(
        max(float(max_joint_acceleration), 1e-6) / velocity
        for velocity in limits.values()
    )
    distances: list[float] = []
    previous, cumulative = start, 0.0
    for point in path:
        travel = max(
            (
                abs(point[joint] - previous.get(joint, point[joint])) / limits[joint]
                for joint in names
            ),
            default=0.0,
        )
        cumulative += travel
        distances.append(cumulative)
        previous = point
    length = distances[-1] if distances else 0.0
    ramp_distance = limit * limit / (2.0 * accel)
    if length >= 2.0 * ramp_distance:
        peak, ramp_time = limit, limit / accel
    else:
        peak = math.sqrt(max(length * accel, 0.0))
        ramp_time = peak / accel
        ramp_distance = length / 2.0

    def time_at(distance: float) -> float:
        if distance <= ramp_distance:
            return math.sqrt(2.0 * distance / accel) if accel > 0.0 else 0.0
        if distance >= length - ramp_distance:
            remaining = max(length - distance, 0.0)
            return (
                2.0 * ramp_time
                + (length - 2.0 * ramp_distance) / peak
                - math.sqrt(2.0 * remaining / accel)
            )
        return ramp_time + (distance - ramp_distance) / peak

    times: list[float] = []
    elapsed = 0.0
    for distance in distances:
        elapsed = max(time_at(distance), elapsed + min_segment_s)
        times.append(elapsed)
    return times


def smooth_trajectory(
    path: list[dict[str, float]],
    start: dict[str, float],
    *,
    velocity_limits: Mapping[str, float],
    max_joint_acceleration: float,
    requested_pose_count: int,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
    collision_reason: Callable[[dict[str, float]], str | None] | None = None,
    joints: Sequence[str] | None = None,
    min_segment_s: float = MIN_SEGMENT_S,
    collision_sample_rad: float = COLLISION_SAMPLE_RAD,
) -> LocalTrajectory:
    """Fit and retime a rest-to-rest joint spline for quintic interpolation.

    The start snapshot is part of the trajectory. Exact polynomial extrema bound
    velocity, acceleration and joint positions, including between IK samples:
    an unconstrained quintic can bow past a joint limit on a plateau even when
    every requested waypoint is legal, and slowing the curve cannot fix that,
    so it is refused. Velocity and acceleration overshoot is fixed by uniform
    time dilation, which preserves the checked joint curve.

    Collision checking samples the spline at <= ``collision_sample_rad`` of
    joint travel through ``collision_reason`` (``None`` = collision free); it is
    not a continuous collision guarantee.

    ``joints`` defaults to the keys of ``velocity_limits`` and fixes the key
    order of every returned row.
    """
    if not path:
        raise PosePathPreflightError("cannot time an empty joint path")
    names = _joint_names(joints, velocity_limits)
    limits = np.array([velocity_limits[joint] for joint in names], dtype=float)
    acceleration = float(max_joint_acceleration)
    positions = np.array(
        [[point[joint] for joint in names] for point in [start, *path]],
        dtype=float,
    )
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(limits).all()
        or np.any(limits <= 0)
        or not math.isfinite(acceleration)
        or acceleration <= 0
    ):
        raise PosePathPreflightError(
            "joint positions and positive motion limits must be finite"
        )
    PPoly, make_interp_spline = _import_spline()
    times = np.array(
        [
            0.0,
            *time_parameterise(
                path,
                start,
                velocity_limits=velocity_limits,
                max_joint_acceleration=acceleration,
                joints=names,
                min_segment_s=min_segment_s,
            ),
        ]
    )
    zero = np.zeros(len(names))
    spline = make_interp_spline(
        times,
        positions,
        k=5,
        bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]),
    )
    scale = 1.0
    peak_velocity = 0.0
    audit_times = set(times)
    for index, joint in enumerate(names):
        polynomial = PPoly.from_spline((spline.t, spline.c[:, index], spline.k))
        for order in (0, 1, 2):
            derivative = polynomial.derivative(order)
            roots = derivative.derivative().roots(extrapolate=False)
            roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= times[-1])]
            values = derivative(np.concatenate((times, roots)))
            if order == 0:
                audit_times.update(roots)
                if joint_limits is not None:
                    lower, upper = joint_limits[joint]
                    if np.min(values) < lower - 1e-9 or np.max(values) > upper + 1e-9:
                        raise PosePathPreflightError(
                            f"smoothed trajectory exceeds {joint} limits"
                        )
            elif order == 1:
                peak = float(np.max(np.abs(values)))
                peak_velocity = max(peak_velocity, peak)
                scale = max(scale, peak / limits[index])
            else:
                scale = max(
                    scale, math.sqrt(float(np.max(np.abs(values))) / acceleration)
                )

    if collision_reason is not None:
        for left, right in pairwise(times):
            samples = max(
                2, math.ceil(peak_velocity * (right - left) / collision_sample_rad)
            )
            audit_times.update(np.linspace(left, right, samples + 1))
        for when in sorted(audit_times):
            point = dict(zip(names, map(float, spline(when))))
            reason = collision_reason(point)
            if reason:
                raise PosePathPreflightError(f"smoothed trajectory: {reason}")

    # Uniform time dilation preserves the checked joint curve. Its derivatives
    # scale by 1/s and 1/s²; a tiny margin absorbs duration serialization rounding.
    scale *= 1.0 + 1e-9
    velocities = spline(times, 1) / scale
    accelerations = spline(times, 2) / scale**2
    velocities[[0, -1]] = 0.0
    accelerations[[0, -1]] = 0.0

    def rows(values: np.ndarray) -> tuple[dict[str, float], ...]:
        return tuple(dict(zip(names, map(float, row))) for row in values)

    return LocalTrajectory(
        positions=rows(positions),
        time_from_start_s=tuple(map(float, times * scale)),
        velocities=rows(velocities),
        accelerations=rows(accelerations),
        requested_pose_count=requested_pose_count,
        solved_pose_count=requested_pose_count,
    )


def interpolate_joint_transitions(
    path: list[dict[str, float]],
    start: dict[str, float],
    *,
    sample_rad: float = JOINT_SAMPLE_RAD,
) -> list[dict[str, float]]:
    """Expand joint-space PTP segments; geometry only, angles in radians.

    Each segment is split so no joint moves more than ``sample_rad`` between
    consecutive samples, giving a collision checker something to check along a
    transition that IK alone would describe only by its endpoints.
    """
    if not math.isfinite(sample_rad) or sample_rad <= 0.0:
        raise ValueError("sample_rad must be finite and greater than 0")
    expanded: list[dict[str, float]] = []
    previous = dict(start)
    for target in path:
        largest_delta = max(
            (
                abs(float(value) - float(previous.get(name, value)))
                for name, value in target.items()
            ),
            default=0.0,
        )
        samples = max(1, math.ceil(largest_delta / sample_rad))
        for sample_index in range(1, samples + 1):
            alpha = sample_index / samples
            expanded.append(
                {
                    name: float(previous.get(name, value))
                    + alpha * (float(value) - float(previous.get(name, value)))
                    for name, value in target.items()
                }
            )
        previous = dict(target)
    return expanded
