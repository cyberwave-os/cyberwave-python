"""The one conversion from a physical joint-velocity ceiling to what an arm stack needs.

An operator configures a single number: a commanded upper bound, in rad/s, on the
absolute angular velocity of every arm joint. It is a ceiling, not a promise --
synchronized motion, short travel, acceleration limits and a vendor controller's
own re-profiling all make joints slower.

Three different numbers feed this and must never be substituted for each other:

* the **requested ceiling** ``R``, which is operator policy;
* the **URDF velocity limit** ``U_i``, which is the model's own bound and what
  every planner is allowed to time against;
* the **controller maximum** ``H_i``, commissioned hardware evidence linked to a
  firmware version, and the only honest denominator for a controller-side speed
  scale.

The URDF cannot be that denominator. Descriptions routinely declare velocities
that differ from what the controller actually applies (Piper's URDF says 5 rad/s
on most joints while its commissioned controller reports 3 rad/s), so a speed
scale computed against the URDF silently delivers a fraction of what was asked.

Most arm controllers apply **one global speed scale** to every joint (a vendor
"speed percentage", a UR speed slider). This module models exactly that: a
fraction ``f`` of each joint's controller maximum. Controllers that encode ``f``
as an integer (1..100 %) pass a :class:`SpeedScaleQuantization`; the resolver
then floors to that resolution so the hardware can never exceed the request.

Pure: no ROS, no config loading, no logging. Everything it needs is an argument,
so a driver, its launch files and ``move_group`` all resolve the same numbers
from the same code instead of each carrying a copy of the formula. The driver
supplies its joint names, its commissioned :class:`ControllerVelocityProfile`
table and (optionally) its quantization; nothing robot-specific lives here.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

#: Planning headroom for a controller's own re-profiling. Internal tracking
#: margin, NOT a second operator setting: trajectories are timed against 90 % of
#: the ceiling the controller is capped at, so a trajectory controller does not
#: abort on ``goal_time`` because the hardware ran a little slower than planned.
PLANNING_VELOCITY_SAFETY = 0.9

#: Floating-point slack when comparing a request against a derived bound. Well
#: below any realistic controller resolution, so it can never admit a real
#: overshoot.
_EPSILON_RAD_S = 1e-9


@dataclass(frozen=True, slots=True)
class ControllerVelocityProfile:
    """Per-joint maximum angular velocity the physical controller will apply.

    Calibration and provenance, not a speed setting. A driver stores these in
    code as reviewed hardware evidence and selects one by the configured firmware
    version. A diagnostic override that waives a geometry gate must never invent
    one of these: a made-up denominator turns the operator's rad/s into a number
    nobody has measured.
    """

    firmware_version: str
    max_velocity_rad_s: Mapping[str, float]
    evidence: str


@dataclass(frozen=True, slots=True)
class SpeedScaleQuantization:
    """How a controller encodes its global speed scale as an integer.

    ``steps`` is the integer meaning "100 % of the controller maximum" (100 for
    a percentage); ``min_steps`` is the smallest integer the controller accepts.
    Many vendor interfaces treat 0 as out of range rather than "stopped", which
    is why a request that floors below ``min_steps`` is refused instead of being
    rounded up to it.

    ``label`` names the encoded value in the startup record (e.g.
    ``"vendor_speed_percent"``) so logs keep the vendor's own vocabulary.
    """

    steps: int = 100
    min_steps: int = 1
    label: str = "speed_scale_steps"

    def __post_init__(self) -> None:
        if int(self.steps) < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps!r}")
        if not 1 <= int(self.min_steps) <= int(self.steps):
            raise ValueError(
                f"min_steps must be in 1..{self.steps}, got {self.min_steps!r}"
            )


#: A percentage with a 1 % floor: the common vendor encoding.
PERCENT_QUANTIZATION = SpeedScaleQuantization(steps=100, min_steps=1)


class MotionLimitRefusal(Enum):
    """Closed set, so a startup status can report the reason without a traceback."""

    INVALID_REQUEST = "invalid_request"
    NO_FIRMWARE = "no_firmware"
    UNKNOWN_FIRMWARE = "unknown_firmware"
    INVALID_CONTROLLER_PROFILE = "invalid_controller_profile"
    URDF_UNREADABLE = "urdf_unreadable"
    URDF_MISSING_VELOCITY_LIMIT = "urdf_missing_velocity_limit"
    BELOW_MINIMUM_RESOLUTION = "below_minimum_resolution"
    ABOVE_HARDWARE_CEILING = "above_hardware_ceiling"


class MotionLimitError(ValueError):
    """Resolution failed. Carries the closed-set reason, never a bare KeyError."""

    def __init__(self, reason: MotionLimitRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ResolvedMotionLimits:
    """Everything derived from one requested ceiling, with its provenance.

    ``speed_scale`` is the global fraction of the controller maximum the arm is
    capped at. ``speed_scale_steps`` is the same value as the integer the
    controller is commanded with when a :class:`SpeedScaleQuantization` applied
    (e.g. a vendor percentage), else ``None``. Both are observations about how
    the request was encoded, never settings.
    """

    joints: tuple[str, ...]
    requested_rad_s: float
    urdf_velocity_rad_s: Mapping[str, float]
    controller_velocity_rad_s: Mapping[str, float]
    speed_scale: float
    speed_scale_steps: int | None
    effective_velocity_rad_s: Mapping[str, float]
    planner_velocity_rad_s: Mapping[str, float]
    firmware_version: str
    evidence: str
    speed_scale_label: str = "speed_scale"

    def startup_record(self) -> str:
        """One line naming the request, the evidence and every derived bound."""

        def _joints(values: Mapping[str, float]) -> str:
            return "[" + ", ".join(f"{values[j]:.4g}" for j in self.joints) + "]"

        scale = (
            f"{self.speed_scale_label}={self.speed_scale_steps}"
            if self.speed_scale_steps is not None
            else f"{self.speed_scale_label}={self.speed_scale:.4g}"
        )
        return (
            f"max_joint_velocity_rad_s={self.requested_rad_s:g} "
            f"firmware={self.firmware_version} "
            f"controller_max={_joints(self.controller_velocity_rad_s)} "
            f"urdf_max={_joints(self.urdf_velocity_rad_s)} "
            f"{scale} "
            f"effective={_joints(self.effective_velocity_rad_s)} "
            f"planner={_joints(self.planner_velocity_rad_s)} "
            f"evidence={self.evidence}"
        )


def _positive_float(value: object) -> float | None:
    """*value* as a finite positive float, or None. ``True`` is not a velocity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def urdf_joint_velocity_limits(
    urdf_path: str, joints: Sequence[str]
) -> dict[str, float]:
    """``joint -> <limit velocity>`` for exactly *joints*, in rad/s.

    Plain ElementTree, like :func:`urdf_joint_limits`: this runs at launch time,
    and making a limit lookup depend on a full kinematics stack would couple
    plain teleop to solver availability.

    Fails closed. A joint with no ``velocity`` attribute is not an unlimited
    joint, it is a description nobody can plan against.
    """
    wanted = tuple(joints)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        raise MotionLimitError(
            MotionLimitRefusal.URDF_UNREADABLE,
            f"{urdf_path}: {type(exc).__name__}: {exc}",
        ) from exc

    described: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        bound = joint.find("limit")
        if name not in wanted or bound is None:
            continue
        velocity = _positive_float(bound.get("velocity"))
        if velocity is None:
            raise MotionLimitError(
                MotionLimitRefusal.URDF_MISSING_VELOCITY_LIMIT,
                f"{name} has velocity={bound.get('velocity')!r} in {urdf_path}",
            )
        described[str(name)] = velocity

    missing = [joint for joint in wanted if joint not in described]
    if missing:
        raise MotionLimitError(
            MotionLimitRefusal.URDF_MISSING_VELOCITY_LIMIT,
            f"{','.join(missing)} carry no <limit velocity> in {urdf_path}",
        )
    return described


def controller_velocity_profile(
    firmware_version: str | None,
    profiles: Mapping[str, ControllerVelocityProfile],
    *,
    firmware_hint: str = "",
) -> ControllerVelocityProfile:
    """The commissioned profile for *firmware_version*, or a closed-set refusal.

    *firmware_hint* is appended to the ``NO_FIRMWARE`` detail so the operator is
    told how this particular arm reports its firmware (e.g. which probe to run).
    """
    if not firmware_version:
        detail = "no firmware_version configured"
        if firmware_hint:
            detail = f"{detail}; {firmware_hint}"
        raise MotionLimitError(MotionLimitRefusal.NO_FIRMWARE, detail)
    profile = profiles.get(firmware_version)
    if profile is None:
        raise MotionLimitError(
            MotionLimitRefusal.UNKNOWN_FIRMWARE,
            f"firmware {firmware_version!r} has no commissioned controller velocity "
            f"profile; known: {sorted(profiles)}",
        )
    return profile


def resolve_motion_limits(
    requested_rad_s: object,
    *,
    joints: Sequence[str],
    firmware_version: str | None,
    profiles: Mapping[str, ControllerVelocityProfile],
    urdf_path: str,
    quantization: SpeedScaleQuantization | None = None,
    planning_safety: float = PLANNING_VELOCITY_SAFETY,
    firmware_hint: str = "",
) -> ResolvedMotionLimits:
    """Turn one physical ceiling into a controller speed scale and planner limits.

    For every joint ``i`` in *joints*, with ``R`` requested, ``U_i`` from the
    URDF and ``H_i`` from the controller profile::

        f_raw = min(1, min_i(min(R, U_i) / H_i))
        f     = floor(steps * f_raw) / steps     # only with a quantization
        E_i   = H_i * f
        planner_i = min(R, U_i, E_i) * planning_safety

    Flooring is mandatory. Rounding or ceiling would let the hardware exceed the
    operator's requested maximum, which is the one thing the setting promises.
    With a quantization the smallest ceiling the arm can represent is
    ``H_min * min_steps / steps``; a request below that is refused rather than
    rounded up, and a request the controller cannot reach is refused rather than
    silently delivered as something slower.
    """
    arm_joints = tuple(joints)
    if not arm_joints:
        raise ValueError("joints must name at least one arm joint")

    requested = _positive_float(requested_rad_s)
    if requested is None:
        raise MotionLimitError(
            MotionLimitRefusal.INVALID_REQUEST,
            f"max_joint_velocity_rad_s must be a finite positive number, got "
            f"{requested_rad_s!r}",
        )

    profile = controller_velocity_profile(
        firmware_version, profiles, firmware_hint=firmware_hint
    )
    controller: dict[str, float] = {}
    for joint in arm_joints:
        hardware = _positive_float(profile.max_velocity_rad_s.get(joint))
        if hardware is None:
            raise MotionLimitError(
                MotionLimitRefusal.INVALID_CONTROLLER_PROFILE,
                f"profile {profile.firmware_version!r} has "
                f"{joint}={profile.max_velocity_rad_s.get(joint)!r}",
            )
        controller[joint] = hardware

    described = urdf_joint_velocity_limits(urdf_path, arm_joints)

    # The fastest ceiling this controller can apply at all: 100 % of its
    # slowest joint. A request above it is non-operative and is refused rather
    # than accepted as a value the arm silently delivers as something slower.
    #
    # The URDF deliberately does NOT enter this bound. A description tighter
    # than the request is an expected, documented reason for a joint to end up
    # slower than the ceiling (the setting is an upper bound, not a promise), so
    # it constrains the derived scale below instead of refusing the config.
    ceiling = min(controller[joint] for joint in arm_joints)
    if requested > ceiling + _EPSILON_RAD_S:
        raise MotionLimitError(
            MotionLimitRefusal.ABOVE_HARDWARE_CEILING,
            f"{requested} rad/s is above the largest ceiling the commissioned "
            f"controller can apply ({ceiling} rad/s at 100 %); it would be "
            "delivered as something slower rather than refused",
        )

    raw = min(
        min(requested, described[joint]) / controller[joint] for joint in arm_joints
    )
    steps: int | None = None
    if quantization is None:
        fraction = min(1.0, raw)
    else:
        steps = min(int(quantization.steps), math.floor(quantization.steps * raw))
        if steps < int(quantization.min_steps):
            smallest = ceiling * quantization.min_steps / quantization.steps
            raise MotionLimitError(
                MotionLimitRefusal.BELOW_MINIMUM_RESOLUTION,
                f"{requested} rad/s quantizes to {steps}/{quantization.steps} on "
                f"this controller; its resolution is 1/{quantization.steps} of the "
                f"controller maximum, so the smallest representable ceiling is "
                f"{smallest} rad/s. It is never rounded up",
            )
        fraction = steps / quantization.steps

    effective = {joint: controller[joint] * fraction for joint in arm_joints}
    for joint, value in effective.items():
        # The formula guarantees these; asserting them is what makes a future
        # edit to the quantization fail here rather than on the arm.
        assert 0.0 < value <= requested + _EPSILON_RAD_S, (joint, value, requested)
        assert value <= described[joint] + _EPSILON_RAD_S, (joint, value)

    safety = float(planning_safety)
    planner = {
        joint: min(requested, described[joint], effective[joint]) * safety
        for joint in arm_joints
    }

    return ResolvedMotionLimits(
        joints=arm_joints,
        requested_rad_s=requested,
        urdf_velocity_rad_s=MappingProxyType(described),
        controller_velocity_rad_s=MappingProxyType(controller),
        speed_scale=float(fraction),
        speed_scale_steps=steps,
        effective_velocity_rad_s=MappingProxyType(effective),
        planner_velocity_rad_s=MappingProxyType(planner),
        firmware_version=profile.firmware_version,
        evidence=profile.evidence,
        speed_scale_label=(
            "speed_scale" if quantization is None else quantization.label
        ),
    )


# ── Position limits ─────────────────────────────────────────────────────────


def urdf_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """``joint -> (lower, upper)`` straight from the description XML.

    Plain ElementTree rather than pinocchio: this runs at launch time, and making
    a driver's managed launch depend on a full kinematics stack to look up two
    XML attributes would couple plain teleop to solver availability. Joints with
    no ``<limit>`` or no ``lower``/``upper`` (continuous, fixed) are omitted.
    """
    limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        bound = joint.find("limit")
        name = joint.get("name")
        if bound is None or name is None:
            continue
        lower, upper = bound.get("lower"), bound.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = (float(lower), float(upper))
    return limits


def intersect_joint_limits(
    described: Mapping[str, tuple[float, float]],
    overrides: Mapping[str, tuple[float, float]] | None,
    joints: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """The description's limits for *joints*, narrowed by *overrides*.

    Intersected rather than assigned: an override table must never authorise
    motion the description says is impossible, and in practice every curated
    table is wider than the description somewhere. ``None`` (or a joint absent
    from *overrides*) means "the description's own limits".

    No permissive fallback. A joint the description does not bound raises,
    because a wide default accepts targets the arm cannot reach and reports them
    as validated; an override that leaves an empty range raises for the same
    reason.
    """
    limits: dict[str, tuple[float, float]] = {}
    for joint in joints:
        model = described.get(joint)
        if model is None:
            raise ValueError(
                f"{joint} has no <limit> in the robot description; refusing to "
                "hand a validator an unbounded joint"
            )
        bound = None if overrides is None else overrides.get(joint)
        if bound is None:
            limits[joint] = (float(model[0]), float(model[1]))
            continue
        lower = max(float(bound[0]), float(model[0]))
        upper = min(float(bound[1]), float(model[1]))
        if lower >= upper:
            raise ValueError(
                f"joint limit override leaves {joint} with an empty range: model "
                f"({model[0]:.4f}, {model[1]:.4f}) vs override "
                f"({bound[0]:.4f}, {bound[1]:.4f})"
            )
        limits[joint] = (lower, upper)
    return limits


# ── Planner configuration bodies ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CartesianLimitBases:
    """Cartesian (Pilz LIN/CIRC-style) bounds at a speed scale of 1.0.

    Scaled by the resolved ``speed_scale``, exactly like the joint velocities,
    so a slow arm is not handed a fast straight-line move.
    """

    max_trans_vel_m_s: float
    max_trans_acc_m_s2: float
    max_rot_vel_rad_s: float


@dataclass(frozen=True)
class PlanningLimits:
    """Ready-to-dump ``joint_limits.yaml`` and Cartesian-limits bodies."""

    joint_limits: dict[str, Any]
    cartesian_limits: dict[str, Any]


def planning_limits_from(
    limits: ResolvedMotionLimits,
    *,
    joint_acceleration_rad_s2: float,
    cartesian_bases: CartesianLimitBases | None = None,
) -> PlanningLimits:
    """The ONE resolver from the physical ceiling to planner limit files.

    Takes the resolved limits rather than a speed scale, so it cannot be called
    with a number nobody validated: the scale is a transport detail, and the
    per-joint velocities here are the physical policy.

    Velocities are written PER JOINT, straight from the resolver. The joint
    acceleration bound is physical rad/s², independent of the speed scale:
    measured manual motion at a low vendor percentage accelerates far faster
    than a fraction-scaled bound. Cartesian bounds (when given) remain scaled by
    the resolved fraction. The body shapes follow MoveIt's ``joint_limits.yaml``
    and ``pilz_cartesian_limits.yaml``; nothing here imports MoveIt.
    """
    fraction = limits.speed_scale
    max_acceleration = float(joint_acceleration_rad_s2)
    joint_limits = {
        "joint_limits": {
            name: {
                "has_velocity_limits": True,
                "max_velocity": limits.planner_velocity_rad_s[name],
                "has_acceleration_limits": True,
                "max_acceleration": max_acceleration,
                # Pilz reads its own deceleration bound and warns ("Failed
                # loading deceleration limits") for every joint without one.
                # Negative, as Pilz reads it.
                "has_deceleration_limits": True,
                "max_deceleration": -max_acceleration,
            }
            for name in limits.joints
        }
    }
    cartesian_limits: dict[str, Any] = {}
    if cartesian_bases is not None:
        cartesian_limits = {
            "cartesian_limits": {
                "max_trans_vel": cartesian_bases.max_trans_vel_m_s * fraction,
                "max_trans_acc": cartesian_bases.max_trans_acc_m_s2 * fraction,
                "max_trans_dec": cartesian_bases.max_trans_acc_m_s2 * fraction,
                "max_rot_vel": cartesian_bases.max_rot_vel_rad_s * fraction,
            }
        }
    return PlanningLimits(joint_limits=joint_limits, cartesian_limits=cartesian_limits)
