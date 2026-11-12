"""The contract every planned-motion backend satisfies, whichever planner it wraps.

A driver names its default kinematics backend -- the built-in in-process
``"pinocchio"`` one, or any backend the driver registers itself -- and a
request may name another registered one. The orchestrator
(:class:`~cyberwave.driver.motion.planner.PlannedMotionController`) routes
``set_pose`` / ``grasp`` / ``scene_set`` / ``attach_payload`` /
``detach_payload`` / ``ee_pose`` to it. Everything a caller can observe -- the
guards, their order, the refusal codes, the ``state/motion`` payload, the scene
revision and the attached payload -- lives in the orchestrator, so choosing a
backend changes WHICH planner answers, never what the answer looks like.

A backend therefore owns only the planner-specific part:

* whether it can accept goals now, and the operator-facing reason when not;
* planning (and executing, unless ``plan_only``) one pose goal of a declared
  motion type -- ``PTP`` (any collision-free path the backend can find) or
  ``LIN`` (a straight tool line, refused rather than bent) -- or one ranked
  grasp search (:class:`GraspGoal`);
* applying a validated scene, and carrying / dropping one payload;
* cancelling the active goal.

Targets reach a backend in its planning frame: the orchestrator has already
resolved relative targets and reference frames.

Pure Python: no ROS, no pinocchio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .guards import Refusal
from .scene import AttachRequest, Pose, SceneDiff, SceneRequest

#: The name of the built-in in-process backend
#: (:class:`~cyberwave.driver.motion.pinocchio_backend.PinocchioMotionBackend`):
#: IK and collision checks on the URDF, executed through the arm's trajectory
#: controller. Every other name is one a driver registers itself.
PINOCCHIO_BACKEND = "pinocchio"


class MotionType(str, Enum):
    """What kind of path a goal asks for. Never changed as a fallback."""

    #: Any collision-free path the backend can find to the pose.
    PTP = "ptp"
    #: A straight line of the tool point, refused rather than bent.
    LIN = "lin"


class MotionOutcome(str, Enum):
    """The terminal outcome of one goal, as reported in ``last_result``."""

    #: Planned and executed to the end.
    SUCCEEDED = "succeeded"
    #: A ``plan_only`` probe found a plan; nothing moved.
    PLANNED = "planned"
    #: Planning or execution failed; ``refusal`` says which and why.
    FAILED = "failed"
    #: Stopped or superseded; ``refusal.code`` is ``cancelled``.
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class MotionGoal:
    """One pose goal in the planning frame, validated and guarded by the orchestrator."""

    target: Pose
    motion_type: MotionType
    #: Ask whether a plan exists under the current scene and budget; move nothing.
    plan_only: bool = False
    #: The command that produced it (``set_pose``, ``ee_pose``), for logs only.
    command: str = ""


@dataclass(frozen=True)
class GraspCandidate:
    """One grasp TCP pose and the direction the tool travels onto it.

    Both in the planning frame; ``approach`` is a unit vector.
    """

    pose: Pose
    approach: tuple[float, float, float]


@dataclass(frozen=True)
class GraspGoal:
    """Ranked grasp candidates: plan the first one that passes, execute it.

    For each candidate in order: a PTP to the pregrasp pose (the grasp pose
    backed off ``approach_distance_m`` against ``approach``), a LIN approach
    onto the grasp pose, and a plan-only LIN lift of ``lift_distance_m`` from
    there. The first candidate whose three phases all plan is executed as
    transit + approach, stopping at the grasp pose; the lift is checked, never
    executed (the gripper has to close first). The result names the
    candidate in ``selected_candidate``.
    """

    candidates: tuple[GraspCandidate, ...]
    approach_distance_m: float
    lift_distance_m: float
    #: Unit vector, planning frame: the direction of the post-grasp lift.
    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    plan_only: bool = False
    command: str = "grasp"


@dataclass(frozen=True)
class MotionUpdate:
    """Progress of the active goal. Every field is optional; ``None`` = unchanged.

    ``planner`` is announced when a goal starts and again whenever a backend
    falls back to another planner, so the status always names the planner that
    is actually working. It is the backend's own name for that planner, a
    plain string reported verbatim (the in-process backend reports
    :attr:`~cyberwave.driver.motion.status.PlannerKind.PINOCCHIO`).
    ``executing`` is set once the trajectory controller has the goal. The plan
    statistics are reported as soon as a backend knows them.
    """

    planner: str | None = None
    executing: bool = False
    plan_time_ms: int | None = None
    trajectory_points: int | None = None
    trajectory_duration_s: float | None = None


@dataclass(frozen=True)
class MotionResult:
    """The terminal result of one goal: the same fields ``state/motion`` reports."""

    outcome: MotionOutcome
    planner: str | None = None
    plan_time_ms: int | None = None
    trajectory_points: int | None = None
    trajectory_duration_s: float | None = None
    #: Set for ``FAILED`` and ``CANCELLED``: the closed-set code and the one
    #: sentence an operator can act on.
    refusal: Refusal | None = None
    #: A :class:`GraspGoal`'s chosen candidate (index in the ranked list).
    selected_candidate: int | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in (MotionOutcome.SUCCEEDED, MotionOutcome.PLANNED)


@runtime_checkable
class MotionBackend(Protocol):
    """What the orchestrator needs from a planner, and all it needs.

    Implement this to plug another planner (an external motion-planning
    framework, a vendor planner, a test fake) into
    :class:`~cyberwave.driver.control.planned_motion.PlannedMotionMixin`, then
    register it from the driver's ``_motion_backend_factories``. It answers
    every request when it is the driver's ``kinematics_backend``, and any
    request that names it in ``backend``. The SDK never imports it: a backend
    that talks to ROS lives in the driver package that owns that dependency.

    **Identity.** ``name`` is the name the backend is registered under.
    ``planning_frame`` is the one frame scene geometry and pose targets are
    expressed in. Every backend a driver registers plans in the same frame;
    when it is not the arm's base frame, the driver also overrides
    ``_motion_planning_frame`` so the command catalog describes the same frame.

    **Availability.** ``available()`` is cheap and side-effect free: it is
    polled for every command and every status payload. When it is ``False``,
    ``unavailable_refusal()`` says why -- normally
    :func:`~cyberwave.driver.motion.guards.backend_unavailable` with the
    operator-facing detail (``backend_unavailable``, the detail naming the
    backend). A backend that needs a connection (an action client, a socket)
    builds it lazily and reports a failure to build it here, so a driver
    started without that dependency degrades to refusals instead of crashing.

    **Plan / execute.** ``validate_goal`` refuses, before anything starts, a
    goal this backend can never honour (a :class:`GraspGoal` on a backend
    without a grasp search, say) -- never answer it with a different planner.
    ``plan_and_execute`` is asynchronous: it returns at once and reports
    through two callbacks. ``on_update`` announces the planner name as soon as
    it is known (again on every fallback), ``executing=True`` once the
    trajectory controller has the goal, and the plan statistics when known.
    ``on_done`` is called exactly once per goal, possibly before
    ``plan_and_execute`` returns, with a :class:`MotionResult`: ``SUCCEEDED``,
    ``PLANNED`` (a ``plan_only`` probe), or ``FAILED`` / ``CANCELLED`` with a
    :class:`~cyberwave.driver.motion.guards.Refusal` (``planning_failed`` when
    no plan was found, ``execution_failed`` when the arm moved and the
    controller did not follow). A backend must drop callbacks of a goal it has
    already reported terminal.

    **Scene, attach, detach.** Synchronous; each returns ``None`` once the
    planner has acknowledged the change, or the refusal to report. The
    acknowledgement is what makes a scene revision active, so the orchestrator
    does not advance before it has one. A backend built after a scene was set
    receives that scene (and any carried payload) before its first goal. ``apply_scene`` receives the full
    validated request and the :class:`~cyberwave.driver.motion.scene.SceneDiff`
    against the previous revision (ids to remove, objects to add, touch
    permissions to grant and revoke). ``attach`` carries a box on the tool
    frame in every later plan, collision-checked against everything but the
    gripper holding it; ``detach`` drops it from the robot AND the world.

    **Cancel.** Fire and forget: abandon the active goal; its ``on_done``
    reports ``CANCELLED`` (or whatever terminal answer arrives first).
    """

    @property
    def name(self) -> str:
        """The registered backend name, reported as ``kinematics_backend``."""
        ...

    @property
    def planning_frame(self) -> str:
        """The one frame scene geometry and pose targets are expressed in."""
        ...

    def available(self) -> bool:
        """True when the backend can accept a goal right now."""
        ...

    def unavailable_refusal(self) -> Refusal:
        """The refusal (code + operator detail naming the backend) when not available."""
        ...

    def validate_goal(self, goal: MotionGoal | GraspGoal) -> Refusal | None:
        """Refuse a goal this backend cannot honour BEFORE anything starts."""
        ...

    def plan_and_execute(
        self,
        goal: MotionGoal | GraspGoal,
        on_update: Callable[[MotionUpdate], None],
        on_done: Callable[[MotionResult], None],
    ) -> None: ...

    def apply_scene(self, request: SceneRequest, diff: SceneDiff) -> Refusal | None:
        """Make *request* the world the planner checks against."""
        ...

    def attach(self, request: AttachRequest) -> Refusal | None:
        """Carry an inflated box on the tool frame in every later plan."""
        ...

    def detach(self, object_id: str) -> Refusal | None:
        """Drop the carried payload from the robot AND from the world."""
        ...

    def cancel(self) -> None:
        """Abandon the active goal. Fire and forget; the result reports it."""
        ...


__all__ = [
    "PINOCCHIO_BACKEND",
    "GraspCandidate",
    "GraspGoal",
    "MotionBackend",
    "MotionGoal",
    "MotionOutcome",
    "MotionResult",
    "MotionType",
    "MotionUpdate",
]
