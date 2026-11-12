"""Fakes and fixtures shared by the planned-motion suites.

* :class:`FakeExternalBackend` -- a scriptable external ``MotionBackend``, as a
  driver registers one: records goals and lets a test finish them when and how
  it wants, or auto-finish.
* :class:`FakeExecutor` -- a scriptable ``FollowJointTrajectory`` executor with
  the SDK ``TrajectoryExecutor`` interface (``send`` / ``cancel``).
* the ``collision_arm`` fixture (six revolute joints, primitive collision
  geometry, reviewed adjacent-only SRDF) and poses on it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from cyberwave.driver.motion import (
    GraspGoal,
    MotionOutcome,
    MotionRefusal,
    MotionResult,
    MotionType,
    MotionUpdate,
    Refusal,
    backend_unavailable,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
URDF = FIXTURES / "collision_arm.urdf"
SRDF = FIXTURES / "collision_arm.srdf"
ARM = tuple(f"joint{i}" for i in range(1, 7))
HOME = dict.fromkeys(ARM, 0.0)
#: A well-conditioned working posture, clear of everything.
READY = {**HOME, "joint2": 0.6, "joint3": 0.9, "joint5": 0.6}


def arm_config():
    from cyberwave.driver.kinematics.arm.config import ArmKinematicsConfig

    return ArmKinematicsConfig(
        urdf_path=str(URDF),
        ee_frame="tool_link",
        arm_joints=ARM,
        locked_joints=("finger_joint",),
        ik_max_iter=200,
        tool_axes={"forward": np.array([0.0, 0.0, 1.0])},
        rot_axes={"yaw": np.array([0.0, 0.0, 1.0])},
    )


def pose_dict(position, orientation=(0.0, 0.0, 0.0, 1.0)) -> dict[str, Any]:
    return {
        "position": dict(zip("xyz", (float(v) for v in position))),
        "orientation": dict(zip("xyzw", (float(v) for v in orientation))),
    }


def matrix_pose_dict(matrix) -> dict[str, Any]:
    from cyberwave.driver.motion.poses import transform_to_pose

    pose = transform_to_pose(matrix)
    return pose_dict(pose.position, pose.orientation)


class FakeExternalBackend:
    """A registered external :class:`MotionBackend`, scripted like a remote planner.

    What a driver plugs in through ``_motion_backend_factories``: it names its
    own planners (``fake_ptp`` / ``fake_lin`` / ``fake_grasp``), acknowledges
    scene and payload changes, and reports ``backend_unavailable`` when
    offline.

    Records goals; finishes them on demand, or at once when ``auto``. An
    auto-finished goal succeeds unless its target is beyond ``reach_m`` from
    the base, in which case it fails ``planning_failed`` -- the same physical
    answer the in-process backend gives the same target.
    """

    PLANNERS: ClassVar[dict[str, str]] = {
        "ptp": "fake_ptp",
        "lin": "fake_lin",
        "grasp": "fake_grasp",
    }

    def __init__(
        self,
        name: str = "external",
        *,
        available: bool = True,
        auto: bool = False,
        planning_frame: str = "base_link",
    ) -> None:
        self._name = name
        self._planning_frame = planning_frame
        self.is_available = available
        self.auto = auto
        self.reach_m = 1.0
        self.scene_ok = True
        self.goals: list[Any] = []
        self.scene_calls: list[Any] = []
        self.attach_calls: list[Any] = []
        self.detach_calls: list[Any] = []
        self.cancelled = 0
        self._pending: tuple[Any, str, Any, Any] | None = None

    # ── identity and availability ───────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._name

    @property
    def planning_frame(self) -> str:
        return self._planning_frame

    def available(self) -> bool:
        return self.is_available

    def unavailable_refusal(self):
        return backend_unavailable(self._name, "the fake planner is offline")

    # ── goals ────────────────────────────────────────────────────────────────
    def validate_goal(self, goal) -> None:
        return None  # PTP, LIN and a grasp search all exist here

    def plan_and_execute(self, goal, on_update, on_done) -> None:
        self.goals.append(goal)
        if isinstance(goal, GraspGoal):
            planner = self.PLANNERS["grasp"]
        elif goal.motion_type is MotionType.LIN:
            planner = self.PLANNERS["lin"]
        else:
            planner = self.PLANNERS["ptp"]
        self._pending = (goal, planner, on_update, on_done)
        on_update(MotionUpdate(planner=planner))
        if self.auto:
            if self._reachable(goal):
                self.activate()
            self.finish()

    def _reachable(self, goal) -> bool:
        target = goal.candidates[0].pose if isinstance(goal, GraspGoal) else goal.target
        return float(np.linalg.norm(target.position)) <= self.reach_m

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def activate(self) -> None:
        """The trajectory controller has the goal."""
        assert self._pending is not None, "no goal in flight"
        self._pending[2](MotionUpdate(executing=True))

    def finish(self, outcome: MotionOutcome | None = None) -> None:
        """Report the terminal result: by reachability unless *outcome* is given."""
        assert self._pending is not None, "no goal in flight"
        goal, planner, _update, on_done = self._pending
        self._pending = None
        if outcome is None:
            if not self._reachable(goal):
                outcome = MotionOutcome.FAILED
            elif goal.plan_only:
                outcome = MotionOutcome.PLANNED
            else:
                outcome = MotionOutcome.SUCCEEDED
        refusal = {
            MotionOutcome.FAILED: Refusal(
                MotionRefusal.PLANNING_FAILED, "fake planner: no IK solution"
            ),
            MotionOutcome.CANCELLED: Refusal(
                MotionRefusal.CANCELLED, "superseded or stopped"
            ),
        }.get(outcome)
        on_done(
            MotionResult(
                outcome,
                planner=planner,
                plan_time_ms=40,
                trajectory_points=10,
                trajectory_duration_s=2.0,
                refusal=refusal,
                selected_candidate=(
                    0 if isinstance(goal, GraspGoal) and refusal is None else None
                ),
            )
        )

    def cancel(self) -> None:
        self.cancelled += 1

    # ── scene and payload ────────────────────────────────────────────────────
    def apply_scene(self, request, diff):
        self.scene_calls.append(diff)
        if not self.scene_ok:
            return backend_unavailable(self._name, "the planner rejected the scene")
        return None

    def attach(self, request) -> None:
        self.attach_calls.append(request)

    def detach(self, object_id) -> None:
        self.detach_calls.append(object_id)


class FakeExecutor:
    """``TrajectoryExecutor``-shaped: ``send`` records, ``finish`` reports.

    ``auto`` accepts, activates and succeeds every goal inside ``send``.
    ``cancel`` only counts: a test reports the controller's answer with
    ``finish("cancelled")`` when it chooses to.
    """

    def __init__(self, *, accept: bool = True, auto: bool = False) -> None:
        self.accept = accept
        self.auto = auto
        self.calls: list[SimpleNamespace] = []
        self.cancelled = 0

    def send(self, trajectory, on_active, on_done) -> bool:
        if not self.accept:
            return False
        call = SimpleNamespace(
            trajectory=trajectory, on_active=on_active, on_done=on_done, done=False
        )
        self.calls.append(call)
        if self.auto:
            on_active()
            self._finish(call, "succeeded", "")
        return True

    __call__ = send

    @property
    def pending(self) -> bool:
        return bool(self.calls) and not self.calls[-1].done

    def activate(self) -> None:
        self.calls[-1].on_active()

    def finish(self, outcome: str = "succeeded", detail: str = "") -> None:
        self._finish(self.calls[-1], outcome, detail)

    @staticmethod
    def _finish(call, outcome, detail) -> None:
        if not call.done:
            call.done = True
            call.on_done(outcome, detail)

    def cancel(self) -> None:
        self.cancelled += 1
