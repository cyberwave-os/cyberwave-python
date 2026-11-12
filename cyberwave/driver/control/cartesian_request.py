"""One entry point for every Cartesian intent, and one place they are refused.

Absolute (``twin.pose.set``, MQTT ``ee_pose``) and relative (``ee_move``,
``ee_rotate_*``) requests are the same thing once a relative delta has been
resolved against the measured TCP: an absolute TCP pose the arm should reach.
This module validates that request and hands it to a planner. It does not
execute anything and it owns no ROS, fieldbus or solver code.

**Local IK is pre-flight validation, not an executor.** Whatever planner a
driver composes (an external collision-aware planner, or the in-process
:class:`~cyberwave.driver.control.local_planner.LocalCartesianPlanner`) returns
a joint trajectory to the arm's trajectory controller. The solver answers "is
this target reachable, in-limit, non-singular and collision-free" before
anything is planned, which is exactly what it is good at.

When no planner is available the request is REFUSED with a reason. It is never
quietly downgraded to streaming locally-solved waypoints: that path bypasses
collision-aware planning, trajectory tolerances, preemption and cancellation
while looking, from the outside, like the same command succeeding.

Pure Python: no numpy, no pinocchio, no ROS.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class CartesianRejection(Enum):
    """Closed set, so a refusal is greppable and reportable to the operator.

    The first six values are the pre-planning gates, and their strings are
    identical to the matching :class:`~cyberwave.driver.motion.guards.MotionRefusal`
    members on purpose: ``set_pose`` and ``ee_pose`` are the same physical act,
    so a caller must not have to learn two vocabularies for the same refusal.
    ``tests/test_motion_vocabulary.py`` asserts the two enums agree.
    """

    ARM_FAULTED = "arm_faulted"
    CONTROL_MODE_UNAVAILABLE = "control_mode_unavailable"
    HARDWARE_UNVERIFIED = "hardware_unverified"
    GEOMETRY_UNVALIDATED = "geometry_unvalidated"
    COLLISION_MODEL_INVALID = "collision_model_invalid"
    NO_MEASURED_STATE = "no_measured_state"
    UNKNOWN_FRAME = "unknown_frame"
    NON_FINITE_TARGET = "non_finite_target"
    UNREACHABLE = "unreachable"
    OUT_OF_LIMIT = "out_of_limit"
    SELF_COLLISION = "self_collision"
    SINGULAR = "singular"
    PLANNER_UNAVAILABLE = "planner_unavailable"
    PLANNING_FAILED = "planning_failed"
    SUPERSEDED = "superseded"


#: Frames a Cartesian request may be expressed in. Anything else is refused
#: rather than silently treated as one of these -- a request in an unrecognised
#: frame is a request nobody can check.
SUPPORTED_FRAMES: frozenset[str] = frozenset({"base", "tool"})


@dataclass(frozen=True, slots=True)
class CartesianRequest:
    """An absolute TCP target, whatever intent produced it."""

    #: x, y, z, roll, pitch, yaw in the robot base frame.
    target: tuple[float, float, float, float, float, float]
    #: Where it came from, for status and logs (``ee_pose``, ``ee_move``, ...).
    source: str
    #: Joint solution found during pre-flight validation. A planning SEED and a
    #: reachability proof -- never something to stream at the arm directly.
    seed_joints: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class CartesianDecision:
    """Accepted (``request`` set) or refused (``rejection`` + ``detail``)."""

    request: CartesianRequest | None = None
    rejection: CartesianRejection | None = None
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.request is not None

    def status(self) -> str:
        """One line an operator can act on."""
        if self.accepted:
            return "accepted"
        code = self.rejection.value if self.rejection else "unknown"
        return f"{code}: {self.detail}" if self.detail else code


@runtime_checkable
class CartesianPlanner(Protocol):
    """What the driver needs from a Cartesian planner, and all it needs.

    Kept as a protocol so the driver depends on the capability rather than on a
    particular planner process, and so the "planner absent" path is testable
    without ROS.
    """

    @property
    def available(self) -> bool:
        """True when a planner is up and able to accept goals right now."""
        ...

    def plan_and_execute(self, request: CartesianRequest) -> CartesianDecision:
        """Plan to ``request.target`` and submit the trajectory for execution."""
        ...


@dataclass(frozen=True, slots=True)
class Preconditions:
    """Everything checked before a target is worth planning to.

    Passed in rather than read, so this module has no opinion about where the
    driver keeps its state and the whole matrix is testable directly. The same
    object feeds :func:`~cyberwave.driver.motion.guards.check_motion_preconditions`
    for the planned-motion primitives, so both paths refuse identically.
    """

    arm_faulted: bool = False
    control_mode_ready: bool = True
    control_mode_detail: str = ""
    hardware_verified: bool = True
    hardware_detail: str = ""
    geometry_validated: bool = True
    geometry_detail: str = ""
    collision_model_valid: bool = True
    collision_detail: str = ""
    #: Has the arm ever reported where it is? Cartesian motion is relative to the
    #: current configuration, and ``ArmController._start`` completes any joint the
    #: feedback did not supply with 0.0 -- so with no feedback at all the solver
    #: plans from a FABRICATED all-zeros posture rather than refusing. On many
    #: arms that posture sits on a joint limit or a wrist singularity, so the
    #: visible symptom is "unreachable" rather than "I do not know where the arm
    #: is". Defaults True so pure unit tests and manifest export are unaffected;
    #: the driver supplies the real answer from its joint-feedback bookkeeping.
    measured_state_seen: bool = True
    measured_state_detail: str = ""


def validate_cartesian_request(
    target: Any,
    *,
    source: str,
    preconditions: Preconditions,
    frame: str = "base",
    solve: Callable[[Any], dict[str, float] | None] | None = None,
    solve_refusal: Callable[[], str | None] | None = None,
    supported_frames: frozenset[str] = SUPPORTED_FRAMES,
) -> CartesianDecision:
    """Pre-flight a Cartesian intent. Pure, total, and fails closed.

    ``target`` is anything with ``x, y, z, roll, pitch, yaw`` attributes (e.g.
    :class:`~cyberwave.driver.control.cartesian.EEPose`).

    ``solve`` is the reachability check -- the arm's IK in production --
    returning joint positions or ``None``. It proves the target is reachable,
    in-limit and collision-free before a planner is asked, and its result is
    carried as a planning seed. It is NOT an execution path.

    The pre-planning gates are :func:`~cyberwave.driver.motion.guards.check_motion_preconditions`,
    the same matrix ``set_pose`` runs, so the two can never disagree: a faulted
    arm is reported as faulted whatever else is also wrong, and nothing is
    solved before the arm has reported a measured state.
    """
    from ..motion.guards import check_motion_preconditions

    gate = check_motion_preconditions(preconditions, plan_only=False)
    if gate is not None:
        return CartesianDecision(
            rejection=CartesianRejection(gate.code.value), detail=gate.detail
        )
    if frame not in supported_frames:
        return CartesianDecision(
            rejection=CartesianRejection.UNKNOWN_FRAME,
            detail=f"{frame!r} is not one of {sorted(supported_frames)}",
        )

    values = tuple(
        float(getattr(target, name)) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    )
    if not all(math.isfinite(value) for value in values):
        return CartesianDecision(
            rejection=CartesianRejection.NON_FINITE_TARGET, detail=repr(values)
        )

    seed = None
    if solve is not None:
        try:
            seed = solve(target)
        except Exception as exc:  # noqa: BLE001 - a solver crash is a refusal
            return CartesianDecision(
                rejection=CartesianRejection.UNREACHABLE,
                detail=f"{type(exc).__name__}: {exc}",
            )
        if seed is None:
            # A collision-aware solver folds reachability, joint limits and
            # self-collision into one answer, so this covers all three; a planner
            # would only rediscover it more slowly. ``solve_refusal`` asks the
            # caller which of the three it actually was -- "no in-limit,
            # collision-free solution" names three causes and identifies none,
            # and "the gripper is under the table" and "that pose is outside the
            # travel" have nothing in common but that message.
            detail = "no in-limit, collision-free solution for this target"
            if solve_refusal is not None:
                # Diagnosis must never turn a refusal into a crash.
                with contextlib.suppress(Exception):
                    detail = solve_refusal() or detail
            return CartesianDecision(
                rejection=CartesianRejection.UNREACHABLE, detail=detail
            )

    return CartesianDecision(
        request=CartesianRequest(
            target=values,  # type: ignore[arg-type]
            source=source,
            seed_joints=seed,
        )
    )


#: Default refusal when no planner is composed. A driver with a named external
#: planner passes its own detail so the operator knows WHICH process is missing.
DEFAULT_UNAVAILABLE_DETAIL = (
    "no Cartesian planner is available. Cartesian motion is planned; it is not "
    "executed by streaming locally-solved IK waypoints."
)


class UnavailablePlanner:
    """The planner when none is composed or its backing process is not running.

    Refuses, with a reason. It exists so "no planner" is a first-class,
    reportable state rather than the absence of one -- the alternative is a
    driver that quietly executes Cartesian motion some other way.
    """

    def __init__(self, detail: str = "") -> None:
        self._detail = detail or DEFAULT_UNAVAILABLE_DETAIL

    @property
    def available(self) -> bool:
        return False

    def plan_and_execute(self, request: CartesianRequest) -> CartesianDecision:
        return CartesianDecision(
            rejection=CartesianRejection.PLANNER_UNAVAILABLE, detail=self._detail
        )
