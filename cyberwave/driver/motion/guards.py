"""Closed-set motion refusals, and the guards run before any planner is asked.

Every guard is a pure function returning ``Refusal | None`` rather than raising,
because the caller runs several in a fixed order and the ORDER is part of the
contract: a stale robot state must not be reported when the scene revision was
already wrong, or an operator chases the wrong problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .status import MotionState


class MotionRefusal(str, Enum):
    """Why a motion request was refused. Greppable, and reported verbatim.

    ONE closed set on the wire. The first values are the pre-planning gates,
    and their strings are identical to the matching ``CartesianRejection``
    members on purpose: ``set_pose`` and ``ee_pose`` are the same physical act,
    so a caller must not have to learn two vocabularies for the same refusal.
    ``tests/test_motion_vocabulary.py`` asserts the two enums agree.
    """

    # ── pre-planning gates, shared verbatim with CartesianRejection ──────────
    ARM_FAULTED = "arm_faulted"
    CONTROL_MODE_UNAVAILABLE = "control_mode_unavailable"
    HARDWARE_UNVERIFIED = "hardware_unverified"
    GEOMETRY_UNVALIDATED = "geometry_unvalidated"
    COLLISION_MODEL_INVALID = "collision_model_invalid"
    NO_MEASURED_STATE = "no_measured_state"

    # ── planner availability ────────────────────────────────────────────────
    # The backend answering the request cannot accept goals now; the detail
    # names the backend (see backend_unavailable), whichever backend it is.
    BACKEND_UNAVAILABLE = "backend_unavailable"

    # ── shared task-level refusals ──────────────────────────────────────────
    SCENE_REVISION_UNKNOWN = "scene_revision_unknown"
    STALE_ROBOT_STATE = "stale_robot_state"
    BUSY = "busy"
    INVALID_SCENE = "invalid_scene"
    INVALID_REQUEST = "invalid_request"
    PAYLOAD_ALREADY_ATTACHED = "payload_already_attached"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Refusal:
    """A refusal code plus the one sentence an operator can act on."""

    code: MotionRefusal
    detail: str = ""

    def status_line(self) -> str:
        """``"<code>: <detail>"`` — the shape the driver's failure hook logs."""
        return f"{self.code.value}: {self.detail}" if self.detail else self.code.value

    def to_payload(self, command: str | None) -> dict[str, Any]:
        """``last_refusal`` on ``state/motion``: which command, the code, why."""
        return {
            "command": command,
            "code": self.code.value,
            "detail": self.detail,
        }


def backend_unavailable(backend: str, detail: str) -> Refusal:
    """``backend_unavailable`` for the backend named *backend*, with the reason.

    What a :class:`~cyberwave.driver.motion.backend.MotionBackend` returns from
    ``unavailable_refusal`` (and for a scene, attach or detach its planner did
    not acknowledge): one code for every backend, built in or registered, and
    a detail that says which one and why.
    """
    reason = detail.strip() or "not available"
    return Refusal(MotionRefusal.BACKEND_UNAVAILABLE, f"{backend}: {reason}")


def check_revision(requested: str | None, active: str | None) -> Refusal | None:
    """Bind a request to the observation that produced it.

    ``None`` (the default) passes deliberately: the request plans against
    whatever scene the planner holds. Operator commands (``ee_pose``, jogs, a
    plain ``set_pose``) are not derived from a perception snapshot, so
    demanding one would make manual motion impossible until an executor happened
    to be running. An executor sends the revision its targets were computed
    against, and a mismatch means the scene the planner holds is not that one.
    """
    if requested is None:
        return None
    if active is None:
        return Refusal(
            MotionRefusal.SCENE_REVISION_UNKNOWN,
            "no scene has been set since this driver started; send scene_set first",
        )
    if requested != active:
        return Refusal(
            MotionRefusal.SCENE_REVISION_UNKNOWN,
            f"requested {requested!r} but the active scene is {active!r}",
        )
    return None


def check_start_state(
    expected: Mapping[str, float] | None,
    measured: Mapping[str, float],
    tolerance_rad: float,
) -> Refusal | None:
    """Refuse when the arm is not where the requester last observed it.

    The executor reads joint state, computes a grasp from a camera frame taken at
    that posture, and sends a target. Anything that moved the arm in between --
    a teleop nudge, a home return, a previous goal still settling -- invalidates
    the hand-eye chain behind that target. Comparing the requester's snapshot
    against live feedback is what turns that into a refusal instead of a
    confidently wrong motion.

    A missing joint is a refusal, never a pass: an absent measurement is exactly
    the case where "close enough" cannot be established.
    """
    if expected is None:
        return None
    for name, value in expected.items():
        current = measured.get(name)
        if current is None:
            return Refusal(
                MotionRefusal.STALE_ROBOT_STATE,
                f"{name} has no measured value; the arm has not reported it",
            )
        if abs(float(current) - float(value)) > tolerance_rad:
            return Refusal(
                MotionRefusal.STALE_ROBOT_STATE,
                f"{name} measured {float(current):.4f} rad, request expected "
                f"{float(value):.4f} (tolerance {tolerance_rad})",
            )
    return None


def check_motion_preconditions(
    preconditions: Any, *, plan_only: bool
) -> Refusal | None:
    """The gates a request must clear before any planner is asked anything.

    The ONE matrix: ``set_pose``, ``ee_pose`` (via ``validate_cartesian_request``)
    and every planned primitive run exactly this, in this order. Without it the
    primitives reached the planner while the arbiter silently dropped every
    resulting command, and the operator saw ``execution_failed`` for a request
    that was refused on policy before it ever left the driver.

    Order is part of the contract: a faulted arm is reported as faulted whatever
    else is also wrong (a faulted arm often stops reporting too, and "no joint
    feedback" would send the operator to the wrong place). The measured-state
    gate comes last and always runs, because a solve seeded from an assumed
    posture is a wrong answer, not a slow one.

    ``plan_only`` relaxes the gates that exist to prevent MOTION, because a
    plan-only probe moves nothing: a faulted arm, unverified firmware and
    unvalidated wrist geometry all still permit asking "could this be reached".
    What it does NOT relax is the measured-state requirement -- planning from a
    posture nobody reported is not a probe, it is a guess.
    """
    if not plan_only:
        if getattr(preconditions, "arm_faulted", False):
            return Refusal(
                MotionRefusal.ARM_FAULTED, "arm is in a fault state; target refused"
            )
        if not getattr(preconditions, "control_mode_ready", True):
            return Refusal(
                MotionRefusal.CONTROL_MODE_UNAVAILABLE,
                getattr(preconditions, "control_mode_detail", "")
                or "the arm is not in its command mode",
            )
        if not getattr(preconditions, "hardware_verified", True):
            return Refusal(
                MotionRefusal.HARDWARE_UNVERIFIED,
                getattr(preconditions, "hardware_detail", "")
                or "firmware/URDF not verified",
            )
        if not getattr(preconditions, "geometry_validated", True):
            return Refusal(
                MotionRefusal.GEOMETRY_UNVALIDATED,
                getattr(preconditions, "geometry_detail", "")
                or "wrist geometry unvalidated",
            )
        if not getattr(preconditions, "collision_model_valid", True):
            return Refusal(
                MotionRefusal.COLLISION_MODEL_INVALID,
                getattr(preconditions, "collision_detail", "")
                or "no reviewed collision model",
            )
    if not getattr(preconditions, "measured_state_seen", True):
        return Refusal(
            MotionRefusal.NO_MEASURED_STATE,
            getattr(preconditions, "measured_state_detail", "")
            or "no joint feedback received yet; refusing to plan from an "
            "assumed configuration",
        )
    return None


def check_not_busy(state: MotionState) -> Refusal | None:
    """One motion at a time, and no scene edits underneath a running one.

    Not merely a concurrency guard: changing the planning scene or the attached
    payload while a trajectory executes would mean the collision decision that
    authorised the running motion no longer describes the world it is moving
    through, with no way to notice.
    """
    if state in (MotionState.PLANNING, MotionState.EXECUTING):
        return Refusal(MotionRefusal.BUSY, f"a motion is {state.value}")
    return None
