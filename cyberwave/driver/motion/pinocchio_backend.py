"""The in-process (pinocchio) :class:`~cyberwave.driver.motion.backend.MotionBackend`.

Plans with :class:`~cyberwave.driver.control.local_planner.LocalCartesianPlanner`
on the arm's own kinematics -- normally
:class:`~cyberwave.driver.kinematics.arm.collision.CollisionAwareKinematics` --
and executes through an injected trajectory executor (the SDK's
:class:`~cyberwave.driver.ros2.trajectory_executor.TrajectoryExecutor` fits).
No external planner process.

What each request means here:

* **PTP** -- multi-seed IK to the target, then the joint-space interpolation to
  it sampled and collision-checked. A blocked interpolation is REFUSED; there is
  no sampling planner to route around the obstacle.
* **LIN** -- the tool line sampled at the planner's waypoint resolution, each
  sample solved on one IK branch and checked; refused rather than bent.
* **grasp** -- the ranked candidates in order, each with a fresh planning
  window: PTP to the pregrasp pose, LIN approach, LIN lift checked from the
  grasp pose. The first candidate that passes is executed as ONE timed
  trajectory (transit + approach); every refused candidate is reported
  through ``on_candidate_refused`` before the next is tried.
* **scene_set** -- the collision model is rebuilt with the scene's objects
  through an injected ``rebuild_scene`` hook (a driver composing
  :class:`~cyberwave.driver.control.planned_motion.PlannedMotionMixin` gets one
  that keeps its ``WorldObstaclesMixin`` world obstacles too). ``allow_touch``
  objects are deliberately NOT added: the in-process model has no
  per-link allowance, and leaving the target out is the only rule that makes a
  grasp pose plannable at all. That is coarser than a planner with a per-link
  allowed-collision matrix (MoveIt, for example), which still checks the
  target against every non-gripper link.
* **attach / detach** -- the payload becomes a box carried on the tool frame
  (``CollisionAwareKinematics(attached_payload=...)``): checked against the
  world and every arm link except the gripper holding it. Attaching an object
  that is in the scene removes it from the world; detaching
  removes it from the robot and does not put it back.

Pure Python over numpy: no ROS. Execution and the model rebuild are injected.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..kinematics.arm.trajectory import LocalTrajectory, PosePathPreflightError
from .backend import (
    PINOCCHIO_BACKEND,
    GraspGoal,
    MotionGoal,
    MotionOutcome,
    MotionResult,
    MotionUpdate,
)
from .guards import MotionRefusal, Refusal, backend_unavailable
from .poses import pose_to_transform
from .scene import (
    PLANNING_FRAME,
    AttachRequest,
    Primitive,
    SceneDiff,
    SceneRequest,
)
from .status import PlannerKind

#: ``(objects, payload) -> None``: install a collision model holding exactly
#: these task-scene objects (the driver adds its own standing world, e.g. a
#: floor) and this carried payload (pose relative to the TCP), or raise.
SceneRebuild = Callable[[tuple[Primitive, ...], Primitive | None], None]

DEFAULT_UNAVAILABLE_DETAIL = (
    "the in-process planner is unavailable (no arm controller, or no "
    "velocity/acceleration limits to time a trajectory with)"
)


class TrajectoryExecution(Protocol):
    """Submit one fully planned trajectory and report its terminal result.

    Returns ``False`` when the submission was not accepted. Otherwise it calls
    ``on_active()`` once the controller accepts the goal and
    ``on_done(outcome, detail)`` exactly once at the end, with ``outcome`` one
    of ``"succeeded"``, ``"cancelled"`` or anything else for a failure.
    """

    def __call__(
        self,
        trajectory: LocalTrajectory,
        on_active: Callable[[], None],
        on_done: Callable[[str, str], None],
    ) -> bool: ...


@dataclass
class _Run:
    goal: MotionGoal | GraspGoal
    cancelled: bool = False
    finished: bool = False


class PinocchioMotionBackend:
    """Plans in-process and executes through the arm's trajectory controller.

    ``planner`` is the :class:`LocalCartesianPlanner` (anything with
    ``preflight_move`` / ``preflight_grasp`` and ``available``), or a
    zero-argument callable that returns it (or ``None`` while it cannot be
    built) -- so a driver can build it lazily, after its kinematics and
    limits are configured.
    """

    def __init__(
        self,
        *,
        planner: Any,
        current_joints: Callable[[], dict[str, float]],
        execute: TrajectoryExecution,
        cancel_execution: Callable[[], None],
        rebuild_scene: SceneRebuild | None = None,
        planning_frame: str = PLANNING_FRAME,
        executor_name: str = "trajectory executor",
        unavailable_detail: Callable[[], str] | str = DEFAULT_UNAVAILABLE_DETAIL,
        on_candidate_refused: Callable[[str], None] = lambda _detail: None,
    ) -> None:
        self._planner = planner
        self._current_joints = current_joints
        self._execute = execute
        self._cancel_execution = cancel_execution
        self._rebuild_scene = rebuild_scene
        self._planning_frame = planning_frame
        self._executor_name = executor_name
        self._unavailable_detail = unavailable_detail
        self._on_candidate_refused = on_candidate_refused
        #: The task scene in the collision model: the acknowledged revision's
        #: objects minus allow_touch, minus anything since attached.
        self._objects: tuple[Primitive, ...] = ()
        self._payload: Primitive | None = None
        self._run: _Run | None = None

    # ── identity and availability ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return PINOCCHIO_BACKEND

    @property
    def planning_frame(self) -> str:
        return self._planning_frame

    @property
    def scene_objects(self) -> tuple[Primitive, ...]:
        """Task-scene objects currently in the collision model."""
        return self._objects

    @property
    def payload(self) -> Primitive | None:
        """The carried payload (pose relative to the TCP), or ``None``."""
        return self._payload

    def _resolved_planner(self) -> Any:
        planner = self._planner
        if planner is not None and not hasattr(planner, "preflight_move"):
            planner = planner()
        return planner

    def available(self) -> bool:
        try:
            planner = self._resolved_planner()
        except Exception:  # noqa: BLE001 - a builder failure is "unavailable"
            return False
        return planner is not None and bool(getattr(planner, "available", True))

    def unavailable_refusal(self) -> Refusal:
        detail = self._unavailable_detail
        text = detail() if callable(detail) else detail
        return backend_unavailable(
            PINOCCHIO_BACKEND, text or DEFAULT_UNAVAILABLE_DETAIL
        )

    # ── goals ────────────────────────────────────────────────────────────────

    def validate_goal(self, goal: MotionGoal | GraspGoal) -> Refusal | None:
        """PTP, LIN and the grasp search all exist here."""
        del goal
        return None

    def plan_and_execute(
        self,
        goal: MotionGoal | GraspGoal,
        on_update: Callable[[MotionUpdate], None],
        on_done: Callable[[MotionResult], None],
    ) -> None:
        run = _Run(goal)
        self._run = run

        def finish(result: MotionResult) -> None:
            if run.finished:
                return
            run.finished = True
            if self._run is run:
                self._run = None
            on_done(result)

        planner_kind = PlannerKind.PINOCCHIO
        on_update(MotionUpdate(planner=planner_kind))
        started = time.perf_counter()
        selected: int | None = None
        try:
            planner = self._resolved_planner()
            if planner is None:
                finish(
                    MotionResult(
                        MotionOutcome.FAILED,
                        planner=planner_kind,
                        refusal=self.unavailable_refusal(),
                    )
                )
                return
            if isinstance(goal, GraspGoal):
                selected, trajectory = self._plan_grasp(planner, goal)
            else:
                trajectory = planner.preflight_move(
                    self._current_joints(),
                    pose_to_transform(goal.target),
                    motion_type=goal.motion_type.value,
                )
        except PosePathPreflightError as exc:
            finish(self._planning_failed(goal, str(exc), started))
            return
        except Exception as exc:  # noqa: BLE001 - a solver crash is a refusal
            finish(self._planning_failed(goal, f"{type(exc).__name__}: {exc}", started))
            return
        stats: dict[str, Any] = {
            "planner": planner_kind,
            "plan_time_ms": max(0, round((time.perf_counter() - started) * 1000)),
            "trajectory_points": len(trajectory.positions),
            "trajectory_duration_s": float(trajectory.duration_s),
            "selected_candidate": selected,
        }
        if run.cancelled or self._run is not run:
            # Planning finished after a stop or a newer goal: never execute a
            # displaced plan.
            finish(
                MotionResult(
                    MotionOutcome.CANCELLED,
                    refusal=Refusal(
                        MotionRefusal.CANCELLED, "superseded during planning"
                    ),
                    **stats,
                )
            )
            return
        if goal.plan_only:
            finish(MotionResult(MotionOutcome.PLANNED, **stats))
            return
        on_update(
            MotionUpdate(
                **{
                    key: value
                    for key, value in stats.items()
                    if key != "selected_candidate"
                }
            )
        )

        def on_active() -> None:
            if run.finished:
                return
            if run.cancelled:
                # The stop raced goal acceptance: re-assert it now the goal exists.
                self._cancel_execution()
                return
            on_update(MotionUpdate(executing=True))

        def on_finished(outcome: str, detail: str) -> None:
            if outcome == "succeeded":
                finish(MotionResult(MotionOutcome.SUCCEEDED, **stats))
                return
            cancelled = outcome == "cancelled"
            code = (
                MotionRefusal.CANCELLED if cancelled else MotionRefusal.EXECUTION_FAILED
            )
            finish(
                MotionResult(
                    MotionOutcome.CANCELLED if cancelled else MotionOutcome.FAILED,
                    refusal=Refusal(code, detail or f"trajectory {outcome}"),
                    **stats,
                )
            )

        if not self._execute(trajectory, on_active, on_finished):
            finish(
                MotionResult(
                    MotionOutcome.FAILED,
                    refusal=Refusal(
                        MotionRefusal.EXECUTION_FAILED,
                        f"{self._executor_name} did not accept submission",
                    ),
                    **stats,
                )
            )

    def _plan_grasp(self, planner: Any, goal: GraspGoal) -> tuple[int, LocalTrajectory]:
        """The first ranked candidate whose transit, approach and lift all plan.

        The synchronous solver cannot be interrupted mid-solve, so there is no
        per-candidate deadline to promise: its finite seed and iteration caps
        bound the work, and each refusal reports how long it took.
        """
        start = self._current_joints()
        reasons: list[str] = []
        total = len(goal.candidates)
        for index, candidate in enumerate(goal.candidates):
            began = time.perf_counter()
            try:
                trajectory = planner.preflight_grasp(
                    start,
                    pose_to_transform(candidate.pose),
                    approach=candidate.approach,
                    approach_distance_m=goal.approach_distance_m,
                    lift_distance_m=goal.lift_distance_m,
                    lift_direction=goal.lift_direction,
                    # Every refused candidate is reported, so each needs the
                    # diagnostic re-solve that names its actual guard.
                    diagnose=True,
                )
            except PosePathPreflightError as exc:
                elapsed = time.perf_counter() - began
                reasons.append(str(exc))
                self._on_candidate_refused(
                    f"candidate {index + 1}/{total} refused after "
                    f"{elapsed:.2f}s: {exc}"
                )
                continue
            return index, trajectory
        counts: dict[str, int] = {}
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
        summary = "; ".join(f"{count} x {reason}" for reason, count in counts.items())
        raise PosePathPreflightError(
            f"all {total} candidate(s) refused" + (f": {summary}" if summary else "")
        )

    @staticmethod
    def _planning_failed(
        goal: MotionGoal | GraspGoal, detail: str, started: float
    ) -> MotionResult:
        what = (
            "grasp" if isinstance(goal, GraspGoal) else goal.motion_type.value.upper()
        )
        return MotionResult(
            MotionOutcome.FAILED,
            planner=PlannerKind.PINOCCHIO,
            plan_time_ms=max(0, round((time.perf_counter() - started) * 1000)),
            refusal=Refusal(MotionRefusal.PLANNING_FAILED, f"{what} refused: {detail}"),
        )

    def cancel(self) -> None:
        run = self._run
        if run is None or run.cancelled or run.finished:
            return
        run.cancelled = True
        self._cancel_execution()

    # ── scene and payload ────────────────────────────────────────────────────

    def _rebuild(
        self, objects: tuple[Primitive, ...], payload: Primitive | None, what: str
    ) -> Refusal | None:
        if self._rebuild_scene is None:
            return backend_unavailable(
                PINOCCHIO_BACKEND,
                f"no collision-model rebuild hook is configured, so {what} cannot "
                "reach the in-process planner",
            )
        try:
            self._rebuild_scene(objects, payload)
        except Exception as exc:  # noqa: BLE001 - the old model stays installed
            return backend_unavailable(
                PINOCCHIO_BACKEND,
                f"{what}: the in-process collision model could not be rebuilt "
                f"({type(exc).__name__}: {exc}); the previous model is unchanged",
            )
        return None

    def apply_scene(self, request: SceneRequest, diff: SceneDiff) -> Refusal | None:
        """Rebuild the model with the scene, leaving ``allow_touch`` objects out."""
        del diff  # a full rebuild needs the whole scene, not the delta
        objects = tuple(
            entry for entry in request.objects if entry.id not in request.allow_touch
        )
        refusal = self._rebuild(objects, self._payload, "scene_set")
        if refusal is None:
            self._objects = objects
        return refusal

    def attach(self, request: AttachRequest) -> Refusal | None:
        payload = Primitive(request.object_id, "box", request.dimensions, request.pose)
        # The object leaves the world and becomes part of the robot.
        objects = tuple(entry for entry in self._objects if entry.id != payload.id)
        refusal = self._rebuild(objects, payload, f"attaching {payload.id!r}")
        if refusal is None:
            self._objects, self._payload = objects, payload
        return refusal

    def detach(self, object_id: str) -> Refusal | None:
        refusal = self._rebuild(self._objects, None, f"detaching {object_id!r}")
        if refusal is None:
            self._payload = None
        return refusal


__all__ = ["PinocchioMotionBackend", "SceneRebuild", "TrajectoryExecution"]
