"""Core dataclasses for joint control (pure data, numpy-free).

All joint channels are ``dict[str, float]`` keyed by joint name — a target may
cover any subset of a robot's joints. Ordered arrays exist only at the ROS
message boundary (see ``driver/ros2/joint_names.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _immutable(mapping: Mapping) -> Mapping:
    return MappingProxyType(dict(mapping)) if mapping else MappingProxyType({})


@dataclass(frozen=True)
class JointCommand:
    """One-shot actuation command. Channels keyed by joint name."""

    positions: dict[str, float]
    velocities: dict[str, float] | None = None
    efforts: dict[str, float] | None = None


@dataclass(frozen=True)
class MotionRequest:
    """Input to a :data:`MotionGenerator`.

    ``velocities`` holds the *resolved* per-joint constraint (rad/s) for every
    joint in ``target`` — resolution happens in ``JointController.plan``.
    """

    current: dict[str, float]
    target: dict[str, float]
    velocities: dict[str, float]
    rate_hz: float


@dataclass(frozen=True)
class TrajectoryWaypoint:
    """One point on a generated motion. ``time_from_start`` in seconds."""

    positions: dict[str, float]
    velocities: dict[str, float] | None
    time_from_start: float


@dataclass(frozen=True)
class TrajectoryPlan:
    """Result of planning a move. Empty ``waypoints`` means nothing to do.

    ``joints`` is the plan's joint universe, used for overlap-based stream
    superseding (disjoint plans may run concurrently).
    """

    waypoints: tuple[TrajectoryWaypoint, ...]
    duration: float
    joints: frozenset[str]


MotionGenerator = Callable[[MotionRequest], tuple[TrajectoryWaypoint, ...]]
"""Shapes a move. Returns () when there is nothing to do."""


@dataclass(frozen=True)
class JointControllerConfig:
    """Pure-data controller parametrization. Never loads a URDF.

    Safe to build before ``configure()`` (manifest export). Limits resolution
    precedence lives in ``JointController`` — config values here override the
    opportunistic kinematics fallbacks.
    """

    # Joint universe: filters incoming targets (parse_joint_mqtt_payload
    # controllable_names) and scopes overlap superseding. None => accept all.
    joints: tuple[str, ...] | None = None
    # Excluded from shaping — applied at full target from the first waypoint
    # (e.g. binarized grippers, where intermediates fight gripper_hold).
    passthrough_joints: tuple[str, ...] = ()
    # Velocity-constraint fallbacks (rad/s).
    default_velocity: float = 1.0
    default_velocities: Mapping[str, float] = field(default_factory=dict)
    # Optional explicit position limits {joint: (lo, hi)}.
    position_limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # Sampling rate for generated motion.
    rate_hz: float = 50.0
    # Extra channels populated on outgoing commands ("position" implied).
    command_interfaces: tuple[str, ...] = ("position",)

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive (got {self.rate_hz!r})")
        if self.default_velocity <= 0:
            raise ValueError(
                f"default_velocity must be positive (got {self.default_velocity!r})"
            )
        object.__setattr__(self, "default_velocities", _immutable(self.default_velocities))
        object.__setattr__(self, "position_limits", _immutable(self.position_limits))
