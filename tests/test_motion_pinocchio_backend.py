"""PinocchioMotionBackend on the real collision_arm fixture, with a fake executor.

Planning, the collision model and the payload are real (pinocchio + coal);
only the trajectory controller is faked, so every assertion is about what the
in-process planner actually decided.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

try:
    import pinocchio  # noqa: F401
    import scipy  # noqa: F401
except ImportError:
    pytest.skip("pinocchio/scipy unavailable", allow_module_level=True)

from _planned_motion_fakes import ARM, READY, SRDF, FakeExecutor, arm_config

from cyberwave.driver.control.local_planner import LocalCartesianPlanner
from cyberwave.driver.kinematics.arm.collision import CollisionAwareKinematics
from cyberwave.driver.kinematics.arm.trajectory import (
    LocalTrajectory,
    PosePathPreflightError,
)
from cyberwave.driver.kinematics.arm.controller import ArmController
from cyberwave.driver.motion.backend import (
    PINOCCHIO_BACKEND,
    GraspCandidate,
    GraspGoal,
    MotionBackend,
    MotionGoal,
    MotionOutcome,
    MotionType,
)
from cyberwave.driver.motion.guards import MotionRefusal
from cyberwave.driver.motion.pinocchio_backend import PinocchioMotionBackend
from cyberwave.driver.motion.poses import transform_to_pose
from cyberwave.driver.motion.scene import (
    AttachRequest,
    Pose,
    Primitive,
    SceneRequest,
    SceneTracker,
)
from cyberwave.driver.motion.status import PlannerKind


def _build(config, **kwargs):
    return CollisionAwareKinematics(config, srdf_path=str(SRDF), **kwargs)


class _Rig:
    """An arm, its planner, a fake controller and the backend over them."""

    def __init__(self, *, rebuild: bool = True, executor: FakeExecutor | None = None):
        self.arm = ArmController(arm_config(), build_kinematics=_build)
        self.joints = dict(READY)
        self.executor = executor or FakeExecutor()
        self.rebuilds: list[tuple] = []
        self.planner = LocalCartesianPlanner(
            controller=self.arm,
            current_joints=lambda: dict(self.joints),
            send_trajectory=lambda _t: True,
            velocity_limits=dict.fromkeys(ARM, 1.0),
            max_joint_acceleration=2.0,
            path_ik_kwargs={"restart": False},
        )
        self.backend = PinocchioMotionBackend(
            planner=self.planner,
            current_joints=lambda: dict(self.joints),
            execute=self.executor.send,
            cancel_execution=self.executor.cancel,
            rebuild_scene=self._rebuild if rebuild else None,
            executor_name="arm_controller/follow_joint_trajectory",
        )
        self.updates: list = []
        self.results: list = []

    def _rebuild(self, objects, payload):
        self.rebuilds.append((objects, payload))
        self.arm._kin = _build(
            self.arm.config, obstacles=tuple(objects), attached_payload=payload
        )

    @property
    def tcp(self) -> np.ndarray:
        return self.arm.ee_pose(self.joints)

    def goal(self, matrix, motion_type=MotionType.PTP, **kwargs) -> MotionGoal:
        return MotionGoal(transform_to_pose(matrix), motion_type, **kwargs)

    def run(self, goal: MotionGoal):
        self.backend.plan_and_execute(goal, self.updates.append, self.results.append)
        return self.results[-1] if self.results else None


def _lowered(rig, dz=-0.03) -> np.ndarray:
    target = rig.tcp.copy()
    target[2, 3] += dz
    return target


def _swung(rig, joint1=0.3) -> np.ndarray:
    return rig.arm.ee_pose({**rig.joints, "joint1": joint1})


def _scene(objects=(), allow_touch=()) -> SceneRequest:
    return SceneRequest("r1", tuple(objects), frozenset(allow_touch))


def _box_at(position, size=0.05, identifier="box") -> Primitive:
    return Primitive(
        identifier,
        "box",
        (size, size, size),
        Pose(tuple(float(v) for v in position), (0.0, 0.0, 0.0, 1.0)),
    )


def test_identity_and_planning_frame() -> None:
    rig = _Rig()
    assert rig.backend.name == PINOCCHIO_BACKEND == "pinocchio"
    assert isinstance(rig.backend, MotionBackend)
    assert rig.backend.planning_frame == "base_link"
    assert rig.backend.available()


def test_ptp_plans_executes_and_reports_terminal_only_after_the_controller() -> None:
    rig = _Rig()
    assert rig.run(rig.goal(_swung(rig))) is None
    assert rig.updates[0].planner is PlannerKind.PINOCCHIO
    stats = rig.updates[1]
    assert stats.trajectory_points >= 2 and stats.trajectory_duration_s > 0.0
    rig.executor.activate()
    assert rig.updates[-1].executing is True
    rig.executor.finish("succeeded")
    result = rig.results[-1]
    assert result.outcome is MotionOutcome.SUCCEEDED and result.ok
    assert result.planner is PlannerKind.PINOCCHIO
    assert result.trajectory_points == stats.trajectory_points


def test_lin_samples_the_straight_line() -> None:
    rig = _Rig()
    rig.run(rig.goal(_lowered(rig), MotionType.LIN))
    trajectory = rig.executor.calls[-1].trajectory
    assert trajectory.requested_pose_count > 1
    end = rig.arm.ee_pose(trajectory.positions[-1])
    np.testing.assert_allclose(end[:3, 3], _lowered(rig)[:3, 3], atol=2e-3)


def test_a_probe_moves_nothing_and_reports_planned() -> None:
    rig = _Rig()
    result = rig.run(rig.goal(_swung(rig), plan_only=True))
    assert result.outcome is MotionOutcome.PLANNED
    assert result.trajectory_points >= 2
    assert rig.executor.calls == []


def test_every_goal_kind_is_accepted_up_front() -> None:
    rig = _Rig()
    assert rig.backend.validate_goal(rig.goal(_swung(rig))) is None
    assert rig.backend.validate_goal(rig.goal(_lowered(rig), MotionType.LIN)) is None
    grasp = GraspGoal(
        (GraspCandidate(transform_to_pose(rig.tcp), (0, 0, -1)),), 0.05, 0.05
    )
    assert rig.backend.validate_goal(grasp) is None


def test_an_unreachable_target_is_planning_failed_with_the_guard_named() -> None:
    rig = _Rig()
    far = rig.tcp.copy()
    far[0, 3] += 5.0
    result = rig.run(rig.goal(far))
    assert result.outcome is MotionOutcome.FAILED
    assert result.refusal.code is MotionRefusal.PLANNING_FAILED
    assert "PTP refused" in result.refusal.detail
    assert "no IK solution" in result.refusal.detail
    assert rig.executor.calls == []


def test_a_scene_object_blocks_the_line_and_allow_touch_unblocks_it() -> None:
    rig = _Rig()
    target = _lowered(rig, -0.06)
    # A box right under the tool, on the straight line down.
    blocker = _box_at(rig.tcp[:3, 3] + np.array([0.0, 0.0, -0.075]), 0.03, "target")

    assert rig.backend.apply_scene(_scene([blocker]), None) is None
    assert rig.backend.scene_objects == (blocker,)
    result = rig.run(rig.goal(target, MotionType.LIN, plan_only=True))
    assert result.refusal.code is MotionRefusal.PLANNING_FAILED
    assert "contact with obstacle 'target'" in result.refusal.detail

    # allow_touch: the target is left out of the in-process model entirely.
    assert rig.backend.apply_scene(_scene([blocker], ["target"]), None) is None
    assert rig.backend.scene_objects == ()
    assert rig.run(rig.goal(target, MotionType.LIN, plan_only=True)).ok


def test_a_failed_rebuild_keeps_the_previous_scene() -> None:
    rig = _Rig()
    installed = rig.arm.kinematics

    def broken(objects, payload):
        raise ValueError("geometry exploded")

    rig.backend._rebuild_scene = broken
    refusal = rig.backend.apply_scene(_scene([_box_at((1.0, 1.0, 1.0))]), None)
    assert refusal.code is MotionRefusal.BACKEND_UNAVAILABLE
    assert "geometry exploded" in refusal.detail
    assert rig.backend.scene_objects == ()
    assert rig.arm.kinematics is installed


def test_no_rebuild_hook_refuses_scene_and_payload_changes() -> None:
    rig = _Rig(rebuild=False)
    assert (
        rig.backend.apply_scene(_scene(), None).code
        is MotionRefusal.BACKEND_UNAVAILABLE
    )
    attach = AttachRequest("obj", (0.03, 0.03, 0.03), Pose((0, 0, 0.1), (0, 0, 0, 1)))
    assert rig.backend.attach(attach).code is MotionRefusal.BACKEND_UNAVAILABLE


def test_an_attached_payload_is_carried_and_collision_checked() -> None:
    rig = _Rig()
    # Clear of the tool and fingers (which end 0.08 m past the TCP).
    attach = AttachRequest("obj", (0.03, 0.03, 0.03), Pose((0, 0, 0.12), (0, 0, 0, 1)))
    carried_at = rig.tcp @ np.array([0.0, 0.0, 0.12, 1.0])
    obstacle = _box_at(carried_at[:3], 0.03, "shelf")
    assert rig.backend.apply_scene(_scene([obstacle]), None) is None
    # Nothing of the bare arm reaches the shelf...
    assert rig.arm.kinematics.collision_reason(rig.joints) is None

    assert rig.backend.attach(attach) is None
    kin = rig.arm.kinematics
    assert kin.attached_payload_id == "obj"
    assert rig.backend.payload.id == "obj"
    # ...but the payload it now carries does.
    assert kin.collision_reason(rig.joints) == "contact with obstacle 'shelf'"
    result = rig.run(rig.goal(_lowered(rig), MotionType.LIN, plan_only=True))
    assert result.refusal.code is MotionRefusal.PLANNING_FAILED
    assert "start configuration" in result.refusal.detail

    assert rig.backend.detach("obj") is None
    assert rig.backend.payload is None
    assert rig.arm.kinematics.attached_payload_id is None
    assert rig.arm.kinematics.collision_reason(rig.joints) is None


def test_attaching_a_scene_object_moves_it_from_the_world_to_the_robot() -> None:
    rig = _Rig()
    target = _box_at((0.3, 0.3, 0.05), 0.03, "obj")
    rig.backend.apply_scene(_scene([target, _box_at((0.3, -0.3, 0.05))]), None)
    rig.backend.attach(
        AttachRequest("obj", (0.03, 0.03, 0.03), Pose((0, 0, 0.12), (0, 0, 0, 1)))
    )
    assert [o.id for o in rig.backend.scene_objects] == ["box"]
    objects, payload = rig.rebuilds[-1]
    assert [o.id for o in objects] == ["box"] and payload.id == "obj"
    # Detaching does not put it back: nobody observed where it landed.
    rig.backend.detach("obj")
    assert [o.id for o in rig.rebuilds[-1][0]] == ["box"]


def test_the_payload_is_not_paired_with_the_gripper_holding_it() -> None:
    payload = Primitive(
        "obj", "box", (0.02, 0.02, 0.02), Pose((0, 0, 0.0), (0, 0, 0, 1))
    )
    kin = _build(arm_config(), attached_payload=payload)
    # Sitting at the TCP, the payload overlaps the tool and finger geometry --
    # which hold it by definition -- and nothing else.
    assert kin.collision_reason(READY) is None
    assert not kin.in_self_collision(READY)
    huge = Primitive("huge", "box", (0.4, 0.4, 0.4), Pose((0, 0, 0.0), (0, 0, 0, 1)))
    reason = _build(arm_config(), attached_payload=huge).collision_reason(READY)
    assert reason == "attached payload 'huge' contact with the arm"


def test_cancel_during_execution_reports_cancelled() -> None:
    rig = _Rig()
    rig.run(rig.goal(_swung(rig)))
    rig.executor.activate()
    rig.backend.cancel()
    assert rig.executor.cancelled == 1
    rig.backend.cancel()  # idempotent
    assert rig.executor.cancelled == 1
    rig.executor.finish("cancelled", "")
    result = rig.results[-1]
    assert result.outcome is MotionOutcome.CANCELLED
    assert result.refusal.code is MotionRefusal.CANCELLED


def test_a_cancel_racing_goal_acceptance_is_reasserted() -> None:
    rig = _Rig()
    rig.run(rig.goal(_swung(rig)))
    rig.backend.cancel()
    rig.executor.activate()
    assert rig.executor.cancelled == 2
    assert not any(update.executing for update in rig.updates)


def test_controller_failure_and_refused_submission_are_execution_failed() -> None:
    rig = _Rig()
    rig.run(rig.goal(_swung(rig)))
    rig.executor.finish("aborted", "path tolerance violated")
    assert rig.results[-1].refusal.code is MotionRefusal.EXECUTION_FAILED
    assert rig.results[-1].refusal.detail == "path tolerance violated"

    refused = _Rig(executor=FakeExecutor(accept=False))
    result = refused.run(refused.goal(_swung(refused)))
    assert result.refusal.code is MotionRefusal.EXECUTION_FAILED
    assert (
        "arm_controller/follow_joint_trajectory did not accept" in result.refusal.detail
    )


def test_a_stale_callback_never_reports_twice() -> None:
    rig = _Rig()
    rig.run(rig.goal(_swung(rig)))
    rig.executor.finish("succeeded")
    rig.executor.calls[-1].on_done("failed", "late duplicate")
    assert len(rig.results) == 1


def test_a_lazy_planner_factory_and_its_unavailable_detail() -> None:
    backend = PinocchioMotionBackend(
        planner=lambda: None,
        current_joints=dict,
        execute=FakeExecutor().send,
        cancel_execution=lambda: None,
        unavailable_detail=lambda: "no limits configured",
    )
    assert not backend.available()
    refusal = backend.unavailable_refusal()
    assert refusal.code is MotionRefusal.BACKEND_UNAVAILABLE
    assert refusal.detail == "pinocchio: no limits configured"


def test_the_scene_tracker_is_backend_neutral() -> None:
    tracker = SceneTracker("world")
    request = _scene([_box_at((1, 0, 0))], ["box"])
    diff = tracker.plan_update(request)
    assert diff.frame == "world" and diff.remove_ids == ()
    tracker.commit_update(request)
    assert tracker.active_revision == "r1" and tracker.owns("box")
    assert tracker.plan_update(_scene()).revoke_touch == ("box",)


# ── grasp: ranked candidates, against a scripted planner ────────────────────


class _GraspPlanner:
    """``preflight_grasp`` refuses the candidates whose x is in ``reject_x``."""

    available = True

    def __init__(self, reject_x=()) -> None:
        self.reject_x = set(reject_x)
        self.calls: list[dict] = []

    def preflight_move(self, start, target, *, motion_type):
        raise AssertionError("a grasp never plans a plain move")

    def preflight_grasp(self, start, target, **kwargs):
        self.calls.append({"x": float(target[0, 3]), **kwargs})
        if float(target[0, 3]) in self.reject_x:
            raise PosePathPreflightError(
                "no IK solution from the current configuration"
            )
        return LocalTrajectory(
            positions=(dict.fromkeys(ARM, 0.1),),
            time_from_start_s=(0.5,),
            velocities=(dict.fromkeys(ARM, 0.0),),
            accelerations=(dict.fromkeys(ARM, 0.0),),
            requested_pose_count=1,
            solved_pose_count=1,
        )


def _grasp_backend(planner, executor=None):
    refused: list[str] = []
    executor = executor or FakeExecutor()
    backend = PinocchioMotionBackend(
        planner=planner,
        current_joints=lambda: dict(READY),
        execute=executor.send,
        cancel_execution=executor.cancel,
        on_candidate_refused=refused.append,
    )
    return backend, executor, refused


def _grasp(xs, **kwargs) -> GraspGoal:
    return GraspGoal(
        tuple(
            GraspCandidate(Pose((x, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)), (0.0, 0.0, -1.0))
            for x in xs
        ),
        approach_distance_m=0.06,
        lift_distance_m=0.04,
        lift_direction=(0.0, 0.0, 1.0),
        **kwargs,
    )


def test_a_grasp_executes_the_first_candidate_that_plans() -> None:
    planner = _GraspPlanner(reject_x={0.3})
    backend, executor, refused = _grasp_backend(planner)
    results: list = []
    backend.plan_and_execute(_grasp((0.3, 0.35, 0.4)), lambda _u: None, results.append)
    assert [call["x"] for call in planner.calls] == [0.3, 0.35]
    assert planner.calls[-1]["approach_distance_m"] == 0.06
    assert planner.calls[-1]["lift_distance_m"] == 0.04
    assert planner.calls[-1]["lift_direction"] == (0.0, 0.0, 1.0)
    assert len(refused) == 1 and refused[0].startswith("candidate 1/3 refused after")
    assert "no IK solution" in refused[0]
    executor.activate()
    executor.finish("succeeded")
    assert results[-1].outcome is MotionOutcome.SUCCEEDED
    assert results[-1].selected_candidate == 1


def test_a_grasp_probe_selects_without_moving() -> None:
    backend, executor, _ = _grasp_backend(_GraspPlanner())
    results: list = []
    backend.plan_and_execute(
        _grasp((0.3,), plan_only=True), lambda _u: None, results.append
    )
    assert results[-1].outcome is MotionOutcome.PLANNED
    assert results[-1].selected_candidate == 0
    assert executor.calls == []


def test_a_grasp_with_no_plannable_candidate_counts_the_reasons() -> None:
    backend, executor, refused = _grasp_backend(_GraspPlanner(reject_x={0.3, 0.4}))
    results: list = []
    backend.plan_and_execute(_grasp((0.3, 0.4)), lambda _u: None, results.append)
    result = results[-1]
    assert result.outcome is MotionOutcome.FAILED
    assert result.refusal.code is MotionRefusal.PLANNING_FAILED
    assert result.refusal.detail == (
        "grasp refused: all 2 candidate(s) refused: "
        "2 x no IK solution from the current configuration"
    )
    assert len(refused) == 2 and executor.calls == []


def test_a_refused_candidate_reports_how_long_it_took() -> None:
    backend, _executor, refused = _grasp_backend(_GraspPlanner(reject_x={0.3}))
    backend.plan_and_execute(_grasp((0.3, 0.4)), lambda _u: None, lambda _r: None)
    assert re.match(r"candidate 1/2 refused after \d+\.\d\ds: ", refused[0])
