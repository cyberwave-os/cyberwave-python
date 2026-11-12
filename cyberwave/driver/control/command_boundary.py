"""Source-scoped joint-command validation shared by arm command arbiters.

An arm that accepts motion from more than one producer -- live teleop, a
trajectory controller streaming setpoints, a separate gripper channel -- needs
one place that decides, per command, whether it is *complete* for its source,
*finite* and *inside the joint limits*. This module is that place, and it is
deliberately pure: no ROS, no clock, no driver, no MQTT. Ownership, faults and
cancellation belong to :mod:`cyberwave.driver.control.arbiter`; converting a
validated command into a vendor's wire format belongs to an injected
:class:`VendorAdapter`, because that format (speed slots, effort slots, joint
renames) is the one thing that really differs between robots.

## Why each source has an exact scope

* **Trajectory** setpoints cover the arm joints and nothing else. A trajectory
  message carrying a gripper value means something upstream disagrees about who
  owns the gripper; it is refused, not trimmed.
* **Manual** commands cover the arm plus the gripper (the teleop contract):
  a partial manual command would have to be completed with *some* value, and
  there is no safe default -- 0.0 is a perfectly plausible-looking joint angle
  and, on most grippers, "fully closed".
* **Gripper** commands cover only the gripper joints; the arbiter supplies the
  arm setpoint it already owns.

All-or-nothing: no missing value is ever defaulted, and validation never raises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

#: How far outside a joint limit a value may sit and still be treated as being
#: AT the limit. 1e-3 rad is 0.057 deg -- far above encoder quantisation, far
#: below anything mechanically meaningful.
#:
#: A hard bound plus a real sensor is a trap: a joint whose lower limit is
#: exactly 0.0 reports 0 plus or minus noise at rest, and a command carries
#: measured positions for every joint the operator is not moving. A reading of
#: -1e-6 would refuse the WHOLE command with the arm sitting still and healthy.
#: Values inside this band are clamped into range rather than refused; what
#: reaches the arm is still never outside the limit.
LIMIT_BOUNDARY_TOLERANCE_RAD = 1e-3

#: Commands stamped this far in the future are still accepted (clock skew
#: between the producer and this process); anything later is refused.
_FUTURE_STAMP_TOLERANCE_NS = 50_000_000


class MotionSource(Enum):
    """Who produced a command. Scope and ownership policy follow from this."""

    MANUAL = "manual"
    TRAJECTORY = "trajectory"
    GRIPPER = "gripper"


class CommandRejection(Enum):
    """Closed set, so a refusal is greppable and a caller can branch on it."""

    EMPTY = "empty"
    LENGTH_MISMATCH = "length_mismatch"
    INCOMPLETE_SCOPE = "incomplete_scope"
    DUPLICATE_JOINT = "duplicate_joint"
    UNKNOWN_JOINT = "unknown_joint"
    NON_FINITE = "non_finite"
    OUT_OF_LIMIT = "out_of_limit"
    MISSING_LIMIT = "missing_limit"
    INVALID_SPEED = "invalid_speed"
    NO_MEASURED_GRIPPER = "no_measured_gripper"
    NO_MEASURED_ARM = "no_measured_arm"
    #: A command carried its own velocity array while the vendor speed has
    #: exactly one writer; a second one is a stale publisher, not an override,
    #: and is refused rather than quietly obeyed or quietly dropped.
    UNSUPPORTED_VELOCITY_OVERRIDE = "unsupported_velocity_override"
    #: No honest vendor speed could be resolved at startup, so no vendor
    #: command exists. Inhibits ALL actuation; observation is unaffected.
    VELOCITY_CONTRACT_UNRESOLVED = "velocity_contract_unresolved"
    TEACHING_MODE = "teaching_mode"


@dataclass(frozen=True, slots=True)
class CommandScopes:
    """The exact joint sets each :class:`MotionSource` must cover.

    ``manual`` defaults to ``arm + gripper``. It may be given explicitly (for
    example arm-only teleop with a separate gripper channel), but it must still
    contain every arm joint and nothing outside ``arm + gripper``: the arbiter
    keeps the last manual *arm* setpoint and the standing gripper target, and a
    manual scope that cannot supply both has no coherent meaning.
    """

    arm: tuple[str, ...]
    gripper: tuple[str, ...] = ()
    manual: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        arm, gripper = tuple(self.arm), tuple(self.gripper)
        manual = tuple(self.manual) if self.manual else (*arm, *gripper)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "gripper", gripper)
        object.__setattr__(self, "manual", manual)
        if not arm:
            raise ValueError("CommandScopes.arm must name at least one joint")
        for label, names in (("arm", arm), ("gripper", gripper), ("manual", manual)):
            if len(set(names)) != len(names):
                raise ValueError(f"CommandScopes.{label} has duplicate joints: {names}")
        if set(arm) & set(gripper):
            raise ValueError("CommandScopes.arm and .gripper must not overlap")
        if not set(arm) <= set(manual) <= set(arm) | set(gripper):
            raise ValueError(
                "CommandScopes.manual must contain every arm joint and only "
                "arm/gripper joints"
            )

    def for_source(self, source: MotionSource) -> tuple[str, ...]:
        if source is MotionSource.TRAJECTORY:
            return self.arm
        if source is MotionSource.GRIPPER:
            return self.gripper
        return self.manual

    def matches(self, source: MotionSource, names: Sequence[str]) -> bool:
        """Whether *names* is exactly *source*'s scope (any order, no extras)."""
        scope = self.for_source(source)
        return bool(scope) and len(names) == len(scope) and set(names) == set(scope)

    def classify(self, names: Sequence[str]) -> MotionSource | None:
        """The source whose exact scope *names* is, or ``None``.

        Trajectory is tried first, so with an empty gripper scope an arm-only
        set classifies as trajectory; input topics should check
        :meth:`matches` for their own source instead of classifying.
        """
        for source in (
            MotionSource.TRAJECTORY,
            MotionSource.MANUAL,
            MotionSource.GRIPPER,
        ):
            if self.matches(source, names):
                return source
        return None

    def gripper_of(self, positions: Mapping[str, float]) -> dict[str, float] | None:
        """The complete gripper target in *positions*, or ``None`` if absent/partial."""
        if not self.gripper or not all(name in positions for name in self.gripper):
            return None
        return {name: float(positions[name]) for name in self.gripper}


def source_for_names(
    names: Sequence[str], scopes: CommandScopes
) -> MotionSource | None:
    """Classify the exact joint scope of *names* (see :meth:`CommandScopes.classify`)."""
    return scopes.classify(names)


def command_stamp_is_fresh(
    stamp_sec: int, stamp_nsec: int, *, now_ns: int, max_age_s: float
) -> bool:
    """Reject queued or unstamped commands before arbitration or vendor output.

    A command that sat in a queue describes a world that has since moved; an
    unstamped one cannot prove it did not. Up to 50 ms of future skew is
    tolerated for producers on another clock.
    """
    if stamp_sec <= 0 or not 0 <= stamp_nsec < 1_000_000_000:
        return False
    age_ns = now_ns - (stamp_sec * 1_000_000_000 + stamp_nsec)
    return -_FUTURE_STAMP_TOLERANCE_NS <= age_ns <= int(max_age_s * 1_000_000_000)


@dataclass(frozen=True, slots=True)
class ValidatedCommand:
    """A complete, finite, in-limit command covering exactly its source's scope."""

    source: MotionSource
    positions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class VendorJointCommand:
    """A joint-state-shaped outbound command (names + parallel arrays).

    The default shape an adapter produces and the ROS arbiter node publishes as
    ``sensor_msgs/JointState``. Vendors that need speed or effort slots fill
    ``velocities``/``efforts`` in their own adapter.
    """

    names: list[str]
    positions: list[float]
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    command: ValidatedCommand | None = None
    reason: CommandRejection | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """A vendor command (any type the publish callable accepts) or a refusal."""

    command: Any = None
    reason: CommandRejection | None = None
    detail: str = ""


class VendorAdapter(Protocol):
    """Convert a validated command into the vendor's outbound command.

    ``held_gripper`` is the standing gripper target to carry on an arm-only
    command (the last commanded one, else measured); ``None`` when there is
    none. ``gripper_effort`` is the standing gripper effort/torque limit, or
    ``None`` for the vendor default. Return a refusal instead of guessing.
    """

    def __call__(
        self,
        command: ValidatedCommand,
        *,
        held_gripper: Mapping[str, float] | None,
        gripper_effort: float | None,
    ) -> AdaptationResult: ...


def validate_command(
    source: MotionSource,
    names: Sequence[str],
    positions: Sequence[float],
    *,
    limits: Mapping[str, tuple[float, float]],
    scopes: CommandScopes,
    boundary_tolerance: float = LIMIT_BOUNDARY_TOLERANCE_RAD,
) -> ValidationResult:
    """Total and pure: never raises, never partially succeeds.

    ``limits`` is injected so this stays independent of which joint-limit table
    (URDF, controller profile, configured envelope) a deployment resolves. A
    joint absent from it is a configuration failure and fails closed.
    """
    scope = scopes.for_source(source)
    names = list(names)
    positions = list(positions)

    if not names or not positions:
        return ValidationResult(reason=CommandRejection.EMPTY, detail="no joints")
    if len(names) != len(positions):
        return ValidationResult(
            reason=CommandRejection.LENGTH_MISMATCH,
            detail=f"{len(names)} names vs {len(positions)} positions",
        )
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        return ValidationResult(
            reason=CommandRejection.DUPLICATE_JOINT, detail=",".join(duplicates)
        )

    unknown = [name for name in names if name not in scope]
    if unknown:
        # Refused rather than dropped: an out-of-scope joint means a producer
        # disagrees with us about who owns it, and ignoring it hides that.
        return ValidationResult(
            reason=CommandRejection.UNKNOWN_JOINT,
            detail=f"{','.join(unknown)} not in {source.value} scope",
        )

    observed = dict(zip(names, positions))
    missing = [joint for joint in scope if joint not in observed]
    if missing:
        return ValidationResult(
            reason=CommandRejection.INCOMPLETE_SCOPE, detail=",".join(missing)
        )

    clamped: dict[str, float] = {}
    for joint in scope:
        try:
            value = float(observed[joint])
        except (TypeError, ValueError):
            return ValidationResult(
                reason=CommandRejection.NON_FINITE,
                detail=f"{joint}={observed[joint]!r}",
            )
        if not math.isfinite(value):
            return ValidationResult(
                reason=CommandRejection.NON_FINITE, detail=f"{joint}={value}"
            )
        bounds = limits.get(joint)
        if bounds is None:
            # Fail closed: an unlimited joint means a misconfigured arbiter
            # validates nothing while reporting every command as valid.
            return ValidationResult(
                reason=CommandRejection.MISSING_LIMIT,
                detail=f"no configured limit for {joint}",
            )
        lower, upper = float(bounds[0]), float(bounds[1])
        # Snap a value that is outside the range only by sensor noise. Beyond
        # the tolerance it is a real out-of-limit target and is still refused.
        if lower - boundary_tolerance <= value < lower:
            value = lower
        elif upper < value <= upper + boundary_tolerance:
            value = upper
        elif not (lower <= value <= upper):
            return ValidationResult(
                reason=CommandRejection.OUT_OF_LIMIT,
                detail=f"{joint}={value:.4f} outside [{lower:.4f},{upper:.4f}]",
            )
        clamped[joint] = value

    return ValidationResult(
        command=ValidatedCommand(source=source, positions=MappingProxyType(clamped))
    )


def joint_state_adapter(scopes: CommandScopes) -> VendorAdapter:
    """The default adapter: a plain joint-state command, arm then gripper order.

    No speed slot and no per-joint effort beyond the standing gripper effort.
    A gripper-only command is refused (the arbiter supplies the arm setpoint
    before adapting), and an arm-only command on an arm with a gripper needs a
    held gripper target -- never a default one.
    """

    def adapt(
        command: ValidatedCommand,
        *,
        held_gripper: Mapping[str, float] | None,
        gripper_effort: float | None,
    ) -> AdaptationResult:
        if command.source is MotionSource.GRIPPER:
            return AdaptationResult(
                reason=CommandRejection.NO_MEASURED_ARM,
                detail="gripper-only input needs the arbiter's current arm setpoint",
            )
        gripper = scopes.gripper_of(command.positions)
        if gripper is None and scopes.gripper:
            gripper = (
                scopes.gripper_of(held_gripper) if held_gripper is not None else None
            )
            if gripper is None or not all(math.isfinite(v) for v in gripper.values()):
                return AdaptationResult(
                    reason=CommandRejection.NO_MEASURED_GRIPPER,
                    detail="arm-only command needs a held gripper target",
                )
        names = [*scopes.arm, *scopes.gripper]
        values = {
            **{j: float(command.positions[j]) for j in scopes.arm},
            **(gripper or {}),
        }
        efforts: list[float] = []
        if gripper_effort is not None and scopes.gripper:
            efforts = [0.0] * len(scopes.arm) + [float(gripper_effort)] * len(
                scopes.gripper
            )
        return AdaptationResult(
            command=VendorJointCommand(
                names=names,
                positions=[values[name] for name in names],
                efforts=efforts,
            )
        )

    return adapt


__all__ = [
    "LIMIT_BOUNDARY_TOLERANCE_RAD",
    "AdaptationResult",
    "CommandRejection",
    "CommandScopes",
    "MotionSource",
    "ValidatedCommand",
    "ValidationResult",
    "VendorAdapter",
    "VendorJointCommand",
    "command_stamp_is_fresh",
    "joint_state_adapter",
    "source_for_names",
    "validate_command",
]
