"""PlannedMotionController: guard order, defaults, targets, backends and the scene.

Against scriptable stub backends, so what is checked is the orchestrator's own
contract -- identical for every backend -- not any planner's behaviour.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cyberwave.driver.control.cartesian_request import Preconditions
from cyberwave.driver.motion import (
    PINOCCHIO_BACKEND,
    FrameSource,
    GraspGoal,
    MotionGoal,
    MotionOutcome,
    MotionRefusal,
    MotionResult,
    MotionUpdate,
    PlannedMotionConfig,
    PlannedMotionController,
    PlannerKind,
    Refusal,
    backend_unavailable,
)
from cyberwave.driver.motion.status import MotionState

POSE = {
    "position": {"x": 0.3, "y": 0.0, "z": 0.2},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}
JOINTS = {"joint1": 0.0, "joint2": 0.5}
HALF = math.sqrt(0.5)


class StubBackend:
    """Records goals and scene calls; finishes goals when the test says so."""

    planning_frame = "base_link"

    def __init__(self, name: str = PINOCCHIO_BACKEND) -> None:
        self.name = name
        self.is_available = True
        self.goals: list[tuple[object, object, object]] = []
        self.scenes: list = []
        self.attached: list = []
        self.detached: list = []
        self.cancelled = 0
        self.scene_ok = True
        self.attach_ok = True
        self.grasp = True

    def available(self) -> bool:
        return self.is_available

    def unavailable_refusal(self) -> Refusal:
        return backend_unavailable(self.name, "stub offline")

    def validate_goal(self, goal) -> Refusal | None:
        if isinstance(goal, GraspGoal) and not self.grasp:
            return Refusal(MotionRefusal.INVALID_REQUEST, "no grasp here")
        return None

    def plan_and_execute(self, goal, on_update, on_done) -> None:
        self.goals.append((goal, on_update, on_done))
        on_update(MotionUpdate(planner=PlannerKind.PINOCCHIO))

    @property
    def goal(self):
        return self.goals[-1][0]

    def finish(self, outcome=MotionOutcome.SUCCEEDED, refusal=None, **extra) -> None:
        _goal, _update, done = self.goals[-1]
        done(
            MotionResult(
                outcome, planner=PlannerKind.PINOCCHIO, refusal=refusal, **extra
            )
        )

    def apply_scene(self, request, diff):
        if not self.scene_ok:
            return backend_unavailable(self.name, "scene rejected")
        self.scenes.append((request, diff))
        return None

    def attach(self, request):
        if not self.attach_ok:
            return backend_unavailable(self.name, "attach rejected")
        self.attached.append(request.object_id)
        return None

    def detach(self, object_id):
        self.detached.append(object_id)

    def cancel(self) -> None:
        self.cancelled += 1


def _frames(tcp=None, lookup=None) -> FrameSource:
    if tcp is None:
        tcp = np.eye(4)
        tcp[:3, 3] = (0.3, 0.0, 0.2)

    def missing(name):
        raise KeyError(name)

    return FrameSource("base_link", tcp, lookup or missing)


def _subject(
    *,
    preconditions=None,
    frames=None,
    config=None,
    other: StubBackend | None = None,
):
    failures: list[tuple[str, str]] = []
    backend = StubBackend()
    registry = {PINOCCHIO_BACKEND: backend}
    if other is not None:
        registry[other.name] = lambda: other
    subject = PlannedMotionController(
        backends=registry,
        default_backend=PINOCCHIO_BACKEND,
        current_joints=lambda: dict(JOINTS),
        preconditions=preconditions,
        config=config or PlannedMotionConfig(start_state_tolerance_rad=0.05),
        frames=(lambda: frames) if frames is not None else None,
        on_failed=lambda command, line: failures.append((command, line)),
        clock=lambda: 1000.0,
    )
    return subject, backend, failures


def _scene(subject, revision="r1", **extra):
    return subject.handle_command(
        "scene_set", {"scene_revision": revision, "objects": [], **extra}
    )


def _set_pose(subject, **extra):
    return subject.handle_command("set_pose", {"pose": POSE, **extra})


def _target(subject, backend, **extra):
    """Send one set_pose and return the planning-frame target it produced."""
    assert _set_pose(subject, **extra) is None
    target = backend.goal.target
    backend.finish()
    return target


# ── guards and defaults ──────────────────────────────────────────────────────


def test_the_guards_run_in_the_documented_order() -> None:
    """Fix one problem at a time; each run must name the NEXT one, never a later one."""
    pre = {"value": Preconditions(arm_faulted=True)}
    subject, backend, _ = _subject(preconditions=lambda: pre["value"])
    backend.is_available = False
    everything_wrong = {
        "scene_revision": "other",
        "expected_joints": {"joint1": 1.0},
        "pose": {"position": {}},
    }

    def code():
        return subject.handle_command("set_pose", dict(everything_wrong)).code

    assert code() is MotionRefusal.BACKEND_UNAVAILABLE
    backend.is_available = True
    assert code() is MotionRefusal.ARM_FAULTED
    pre["value"] = Preconditions()
    assert code() is MotionRefusal.SCENE_REVISION_UNKNOWN
    _scene(subject, "other")
    assert code() is MotionRefusal.STALE_ROBOT_STATE
    everything_wrong["expected_joints"] = JOINTS
    assert code() is MotionRefusal.INVALID_REQUEST  # the pose
    everything_wrong["pose"] = POSE
    assert subject.handle_command("set_pose", everything_wrong) is None


def test_set_pose_needs_only_the_pose() -> None:
    """Every option has a default: absolute, PTP, the default backend, no checks."""
    subject, backend, _ = _subject()
    assert _set_pose(subject) is None
    goal = backend.goal
    assert isinstance(goal, MotionGoal)
    assert goal.motion_type.value == "ptp" and goal.plan_only is False
    assert goal.target.position == (0.3, 0.0, 0.2)
    assert subject.status().backend == PINOCCHIO_BACKEND
    refusal = subject.handle_command("set_pose", {})
    assert (refusal.code, refusal.detail) == (
        MotionRefusal.INVALID_REQUEST,
        "pose is required",
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("motion", "spline", "motion must be one of ['lin', 'ptp'], got 'spline'"),
        (
            "mode",
            "sideways",
            "mode must be one of ['absolute', 'relative'], got 'sideways'",
        ),
        ("plan_only", "yes", "plan_only must be true or false"),
        ("backend", 3, "backend must be a registered name"),
    ],
)
def test_an_option_outside_its_closed_set_is_refused(field, value, expected) -> None:
    subject, _backend, _ = _subject()
    refusal = _set_pose(subject, **{field: value})
    assert (refusal.code, refusal.detail) == (MotionRefusal.INVALID_REQUEST, expected)


def test_scene_edits_check_availability_then_busy_then_the_payload() -> None:
    subject, backend, _ = _subject()
    _scene(subject)
    _set_pose(subject)
    backend.is_available = False
    assert _scene(subject).code is MotionRefusal.BACKEND_UNAVAILABLE
    backend.is_available = True
    bad = {"scene_revision": "", "objects": []}
    assert subject.handle_command("scene_set", bad).code is MotionRefusal.BUSY
    backend.finish()
    assert subject.handle_command("scene_set", bad).code is MotionRefusal.INVALID_SCENE


def test_a_refusal_is_returned_published_and_alerted_without_touching_state() -> None:
    subject, backend, failures = _subject()
    _scene(subject)
    _set_pose(subject)
    backend.goals[-1][1](MotionUpdate(executing=True))
    assert subject.status().motion_state is MotionState.EXECUTING
    refusal = _scene(subject, "r2")
    status = subject.status()
    assert refusal.code is MotionRefusal.BUSY
    assert status.last_refusal == {
        "command": "scene_set",
        "code": "busy",
        "detail": "a motion is executing",
        # The clock stands still here; stamps still never repeat.
        "timestamp_ms": 1_000_001,
    }
    assert status.motion_state is MotionState.EXECUTING  # the running motion
    assert status.command == "set_pose"
    assert failures[-1] == ("scene_set", "busy: a motion is executing")


def test_a_probe_skips_the_motion_only_gates_but_not_measured_state() -> None:
    subject, _backend, _ = _subject(
        preconditions=lambda: Preconditions(arm_faulted=True, geometry_validated=False)
    )
    assert _set_pose(subject, plan_only=True) is None
    blind, _, _ = _subject(
        preconditions=lambda: Preconditions(measured_state_seen=False)
    )
    assert _set_pose(blind, plan_only=True).code is MotionRefusal.NO_MEASURED_STATE


def test_moves_supersede_instead_of_refusing_busy() -> None:
    subject, backend, _ = _subject()
    _set_pose(subject)
    assert _set_pose(subject, motion="lin") is None
    assert backend.cancelled == 1 and len(backend.goals) == 1
    backend.finish()  # the old goal's terminal result
    assert len(backend.goals) == 2 and backend.goal.motion_type.value == "lin"
    status = subject.status()
    assert status.motion_state is MotionState.PLANNING
    # The cancellation names the replaced motion, not the one now planning.
    assert (status.motion_id, status.last_result["motion_id"]) == (2, 1)
    assert status.last_result["outcome"] == "cancelled"


def test_terminal_results_name_the_command_and_backend() -> None:
    subject, backend, failures = _subject()
    _set_pose(subject, motion="lin")
    backend.finish()
    first = subject.status().last_result
    assert first == {
        "command": "set_pose",
        "outcome": "succeeded",
        "backend": "pinocchio",
        "motion": "lin",
        "planner": "pinocchio",
        "motion_id": 1,
        "timestamp_ms": first["timestamp_ms"],
    }
    # Two identical answers in a row: a new motion number, a new stamp.
    _set_pose(subject, motion="lin")
    assert subject.status().motion_id == 2
    backend.finish()
    second = subject.status().last_result
    assert (second["motion_id"], second["timestamp_ms"] > first["timestamp_ms"]) == (
        2,
        True,
    )
    _set_pose(subject)
    backend.finish(
        MotionOutcome.FAILED, Refusal(MotionRefusal.PLANNING_FAILED, "no IK")
    )
    assert subject.status().motion_state is MotionState.FAILED
    assert subject.status().last_refusal["command"] == "set_pose"
    assert failures[-1] == ("set_pose", "planning_failed: no IK")


def test_status_names_the_default_backend_and_its_availability() -> None:
    subject, backend, _ = _subject()
    payload = subject.status().to_payload()
    assert payload["kinematics_backend"] == "pinocchio"
    assert payload["backend_available"] is True
    assert not [key for key in payload if "moveit" in key or key == "command_id"]
    backend.is_available = False
    assert subject.status().backend_available is False


@pytest.mark.parametrize("bad", ["x", 3, [1]])
def test_non_object_data_is_an_invalid_request(bad) -> None:
    subject, _backend, _ = _subject()
    assert subject.handle_command("set_pose", bad).code is MotionRefusal.INVALID_REQUEST


def test_an_unknown_command_is_refused() -> None:
    subject, _backend, _ = _subject()
    refusal = subject.handle_command("plan_move", {"pose": POSE})
    assert refusal.code is MotionRefusal.INVALID_REQUEST
    assert "unknown motion command" in refusal.detail


# ── targets ──────────────────────────────────────────────────────────────────


def test_every_pose_form_reaches_the_backend_in_the_planning_frame() -> None:
    subject, backend, _ = _subject(frames=_frames())
    target = _target(subject, backend, pose=[0.3, 0.1, 0.2, 0.0, 0.0, math.pi / 2])
    assert target.position == pytest.approx((0.3, 0.1, 0.2))
    assert target.orientation == pytest.approx((0, 0, HALF, HALF))
    target = _target(subject, backend, pose=[0.3, 0.1, 0.2, 0.0, 0.0, HALF, HALF])
    assert target.orientation == pytest.approx((0, 0, HALF, HALF))
    waypoint = {
        "position": {"x": 0.25, "y": 0.0, "z": 0.3},
        "rotation": {"w": HALF, "x": 0.0, "y": 0.0, "z": HALF},
        "frame_id": "base_link",
    }
    target = _target(subject, backend, pose=waypoint)
    assert target.position == pytest.approx((0.25, 0.0, 0.3))
    assert target.orientation == pytest.approx((0, 0, HALF, HALF))


def test_a_position_and_rotation_pair_is_a_pose() -> None:
    subject, backend, _ = _subject(frames=_frames())
    target = _target(subject, backend, pose=[[0.3, 0.1, 0.2], [0, 0, math.pi / 2]])
    assert target.position == pytest.approx((0.3, 0.1, 0.2))
    assert target.orientation == pytest.approx((0, 0, HALF, HALF))
    target = _target(subject, backend, pose=[[0.3, 0.1, 0.2], [0, 0, HALF, HALF]])
    assert target.orientation == pytest.approx((0, 0, HALF, HALF))


@pytest.mark.parametrize(
    "pose",
    [
        [0.3, 0.0, 0.1],
        {"position": {"x": 0.3, "y": 0.0, "z": 0.1}},
        {"x": 0.3, "y": 0.0, "z": 0.1},
        [[0.3, 0.0, 0.1], [0.0, 0.0]],
    ],
)
def test_a_pose_without_a_rotation_is_refused(pose) -> None:
    """Cartesian means a position AND a rotation: no point keeps the tool's."""
    subject, backend, _ = _subject(frames=_frames())
    refusal = _set_pose(subject, pose=pose)
    assert refusal.code is MotionRefusal.INVALID_REQUEST
    assert backend.goals == []


def test_a_relative_target_moves_along_the_tool_axes_by_default() -> None:
    tcp = np.eye(4)
    tcp[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # tool x is base y
    tcp[:3, 3] = (0.3, 0.0, 0.2)
    subject, backend, _ = _subject(frames=_frames(tcp=tcp))
    target = _target(subject, backend, pose=[0.05, 0, 0, 0, 0, 0], mode="relative")
    assert target.position == pytest.approx((0.3, 0.05, 0.2))
    target = _target(
        subject,
        backend,
        pose=[0.05, 0, 0, 0, 0, 0],
        mode="relative",
        reference_frame="base",
    )
    assert target.position == pytest.approx((0.35, 0.0, 0.2))
    # A relative rotation turns the tool about its own axis: +90 deg about
    # tool z from a tool already yawed +90 deg is a 180 deg yaw.
    target = _target(
        subject, backend, pose=[0, 0, 0, 0, 0, math.pi / 2], mode="relative"
    )
    assert target.position == pytest.approx((0.3, 0.0, 0.2))
    assert np.abs(target.orientation) == pytest.approx((0, 0, 1, 0))


def test_a_reference_frame_of_the_description_is_read_at_the_current_posture() -> None:
    camera = np.eye(4)
    camera[:3, 3] = (0.1, 0.0, 0.5)
    lookups: list[str] = []

    def lookup(name):
        lookups.append(name)
        if name != "camera_link":
            raise KeyError(name)
        return camera

    subject, backend, _ = _subject(frames=_frames(lookup=lookup))
    target = _target(
        subject, backend, pose=[0.0, 0.0, 0.2, 0, 0, 0], reference_frame="camera_link"
    )
    assert target.position == pytest.approx((0.1, 0.0, 0.7))
    refusal = _set_pose(subject, pose=[0, 0, 0.2, 0, 0, 0], reference_frame="nowhere")
    assert refusal.code is MotionRefusal.INVALID_REQUEST
    assert "'nowhere' is not a frame of the robot description" in refusal.detail
    refusal = _set_pose(
        subject,
        pose={**POSE, "reference_frame": "camera_link"},
        reference_frame="base_link",
    )
    assert "the target is in 'camera_link' but reference_frame is" in refusal.detail


def test_without_forward_kinematics_only_planning_frame_poses_are_accepted() -> None:
    subject, backend, _ = _subject()  # no frame source
    assert _set_pose(subject) is None
    for extra in ({"mode": "relative"}, {"reference_frame": "tool"}):
        refusal = _set_pose(subject, **extra)
        assert refusal.code is MotionRefusal.INVALID_REQUEST
        assert "no forward kinematics" in refusal.detail
    assert len(backend.goals) == 1


def test_ee_pose_is_set_pose_with_a_ptp_from_extrinsic_rpy() -> None:
    from cyberwave.driver.control.cartesian_request import CartesianRequest

    subject, backend, _ = _subject()
    assert subject.plan_and_execute(
        CartesianRequest(
            target=(0.3, 0.0, 0.2, 0.0, 0.0, math.pi / 2), source="ee_pose"
        )
    ).accepted
    goal = backend.goal
    assert goal.motion_type.value == "ptp"
    assert goal.target.position == (0.3, 0.0, 0.2)
    assert goal.target.orientation == pytest.approx((0, 0, HALF, HALF))
    assert subject.status().command == "ee_pose"


def test_an_ee_pose_refusal_reaches_the_motion_stream_like_set_pose() -> None:
    """Same path as set_pose: the refusal is on state/motion, one code, one line."""
    from cyberwave.driver.control.cartesian_request import (
        CartesianRejection,
        CartesianRequest,
    )

    subject, backend, failures = _subject()
    backend.is_available = False
    decision = subject.plan_and_execute(
        CartesianRequest(target=(0.3, 0.0, 0.2, 0.0, 0.0, 0.0), source="ee_pose")
    )
    assert decision.rejection is CartesianRejection.PLANNER_UNAVAILABLE
    refused = subject.status().last_refusal
    assert (refused["command"], refused["code"]) == ("ee_pose", "backend_unavailable")
    assert failures[-1][0] == "ee_pose"
    assert failures[-1][1] == f"backend_unavailable: {decision.detail}"


# ── backends ─────────────────────────────────────────────────────────────────


def test_a_request_may_name_another_registered_backend() -> None:
    other = StubBackend("planner")
    subject, backend, _ = _subject(other=other)
    assert _set_pose(subject, backend="planner") is None
    assert other.goals and not backend.goals
    assert subject.status().backend == "planner"
    assert subject.status().kinematics_backend == "pinocchio"  # still the default
    other.finish()
    assert subject.status().last_result["backend"] == "planner"
    refusal = _set_pose(subject, backend="rrt")
    assert refusal.code is MotionRefusal.INVALID_REQUEST
    assert refusal.detail == (
        "backend 'rrt' is not registered on this driver; "
        "registered: ['pinocchio', 'planner']"
    )


def test_a_named_backend_that_is_down_refuses_with_its_own_reason() -> None:
    other = StubBackend("planner")
    other.is_available = False
    subject, _backend, _ = _subject(other=other)
    refusal = _set_pose(subject, backend="planner")
    assert refusal.status_line() == "backend_unavailable: planner: stub offline"


def test_a_backend_used_later_is_given_the_scene_and_payload_first() -> None:
    other = StubBackend("planner")
    subject, _backend, _ = _subject(other=other)
    box = {
        "id": "box",
        "primitive": "box",
        "dimensions": [0.1, 0.1, 0.1],
        "pose": POSE,
    }
    _scene(subject, objects=[box, {**box, "id": "cup"}], allow_touch=["cup"])
    subject.handle_command(
        "attach_payload", {"object_id": "cup", "dimensions": [0.05] * 3}
    )
    assert other.scenes == []  # never used yet
    _set_pose(subject, backend="planner")
    ((request, diff),) = other.scenes
    assert request.revision == "r1"
    assert [entry.id for entry in request.objects] == ["box"]  # cup is carried
    assert diff.remove_ids == ()
    assert other.attached == ["cup"]
    assert other.goals


def test_a_scene_reaches_every_used_backend_or_none() -> None:
    other = StubBackend("planner")
    subject, backend, _ = _subject(other=other)
    assert _scene(subject, "r1") is None  # the default backend takes r1
    _set_pose(subject, backend="planner")  # and the other is given it
    other.finish()
    assert len(backend.scenes) == 1 and len(other.scenes) == 1
    other.scene_ok = False
    refusal = _scene(subject, "r2")
    assert refusal.status_line() == "backend_unavailable: planner: scene rejected"
    # The backend that took r2 was put back on r1, and r1 is still active.
    assert [request.revision for request, _ in backend.scenes] == ["r1", "r2", "r1"]
    assert subject.status().active_scene_revision == "r1"


def test_attach_reaches_every_used_backend_or_none() -> None:
    other = StubBackend("planner")
    subject, backend, _ = _subject(other=other)
    _scene(subject)  # the default backend first
    _set_pose(subject, backend="planner")
    other.finish()
    other.attach_ok = False
    refusal = subject.handle_command("attach_payload", {"dimensions": [0.05] * 3})
    assert refusal.code is MotionRefusal.BACKEND_UNAVAILABLE
    assert backend.attached == ["payload"] and backend.detached == ["payload"]
    assert subject.status().attached_payload is None


def test_scene_and_payload_options_have_defaults() -> None:
    subject, backend, _ = _subject()
    assert subject.handle_command("scene_set", {}) is None
    assert subject.status().active_scene_revision == "scene-1"
    assert subject.handle_command("scene_set", None) is None
    assert subject.status().active_scene_revision == "scene-2"
    assert subject.handle_command("attach_payload", {"dimensions": [0.05] * 3}) is None
    assert subject.status().attached_payload == "payload"
    assert subject.handle_command("detach_payload", {}) is None
    assert backend.detached == ["payload"]
    refusal = subject.handle_command("detach_payload", {})
    assert refusal.code is MotionRefusal.INVALID_REQUEST


# ── grasp ────────────────────────────────────────────────────────────────────


def _grasp_pose(z=0.1):
    return {
        "position": {"x": 0.3, "y": 0.0, "z": z},
        "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    }


def test_grasp_needs_only_the_candidates() -> None:
    subject, backend, _ = _subject(
        config=PlannedMotionConfig(tool_approach_axis=(0.0, 0.0, 1.0))
    )
    assert (
        subject.handle_command("grasp", {"candidates": [{"pose": _grasp_pose()}]})
        is None
    )
    goal = backend.goal
    assert isinstance(goal, GraspGoal)
    assert (goal.approach_distance_m, goal.lift_distance_m) == (0.06, 0.05)
    # The tool's z axis, flipped by the grasp orientation: approach downwards.
    assert goal.candidates[0].approach == pytest.approx((0.0, 0.0, -1.0))
    backend.finish(selected_candidate=0)
    assert subject.status().last_result["selected_candidate"] == 0
    assert "motion" not in subject.status().last_result


def test_grasp_without_a_tool_axis_needs_an_approach() -> None:
    subject, backend, _ = _subject()
    refusal = subject.handle_command("grasp", {"candidates": [{"pose": _grasp_pose()}]})
    assert refusal.code is MotionRefusal.INVALID_REQUEST
    assert "candidates[0].approach is required" in refusal.detail
    assert (
        subject.handle_command(
            "grasp",
            {"candidates": [{"pose": _grasp_pose(), "approach": [0, 0, -2]}]},
        )
        is None
    )
    assert backend.goal.candidates[0].approach == (0.0, 0.0, -1.0)


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        ({}, "candidates must be a non-empty list"),
        ({"candidates": []}, "candidates must be a non-empty list"),
        ({"candidates": [{}]}, "candidates[0] must be an object with a pose"),
        (
            {"candidates": [{"pose": POSE, "approach": [0, 0, 0]}]},
            "candidates[0].approach must be non-zero",
        ),
        (
            {
                "candidates": [{"pose": POSE, "approach": [0, 0, 1]}],
                "lift_distance_m": 0,
            },
            "lift_distance_m must be positive",
        ),
    ],
)
def test_a_malformed_grasp_is_refused(data, detail) -> None:
    subject, _backend, _ = _subject()
    refusal = subject.handle_command("grasp", data)
    assert (refusal.code, refusal.detail) == (MotionRefusal.INVALID_REQUEST, detail)


def test_a_backend_without_a_grasp_search_refuses_it_up_front() -> None:
    subject, backend, _ = _subject()
    backend.grasp = False
    refusal = subject.handle_command(
        "grasp", {"candidates": [{"pose": POSE, "approach": [0, 0, -1]}]}
    )
    assert refusal.detail == "no grasp here" and backend.goals == []


# ── one motion at a time ─────────────────────────────────────────────────────


def test_driver_owned_motion_waits_for_the_planned_motion_to_end() -> None:
    """A jog sent while a planned goal still owns the controller would race it."""
    subject, backend, _ = _subject()
    ran: list[str] = []
    assert not subject.replace_after_cancel(lambda: ran.append("idle"))
    _set_pose(subject)
    assert subject.replace_after_cancel(lambda: ran.append("jog"))
    assert backend.cancelled == 1 and ran == []
    backend.finish()
    assert ran == ["jog"]
    assert subject.status().motion_state is MotionState.CANCELLED
    assert not subject.active


def test_only_the_newest_replacement_survives_and_stop_drops_it() -> None:
    subject, backend, _ = _subject()
    _set_pose(subject)
    ran: list[str] = []
    subject.replace_after_cancel(lambda: ran.append("first"))
    subject.replace_after_cancel(lambda: ran.append("second"))
    assert backend.cancelled == 1  # one cancel request, not one per call
    backend.finish()
    assert ran == ["second"]

    _set_pose(subject)
    subject.replace_after_cancel(lambda: ran.append("dropped"))
    subject.cancel()
    backend.finish()
    assert ran == ["second"]


def test_cancel_reaches_the_backend_running_the_motion() -> None:
    other = StubBackend("planner")
    subject, backend, _ = _subject(other=other)
    _set_pose(subject, backend="planner")
    subject.cancel()
    assert (other.cancelled, backend.cancelled) == (1, 0)


def test_busy_detail_for_world_obstacle_updates() -> None:
    subject, backend, _ = _subject()
    assert subject.busy_detail() is None
    _set_pose(subject)
    assert subject.busy_detail() == "a planned motion is planning"
    backend.finish()
    assert subject.busy_detail() is None


def test_a_stub_satisfies_the_backend_protocol() -> None:
    from cyberwave.driver.motion import MotionBackend

    assert isinstance(StubBackend(), MotionBackend)
    assert isinstance(SimpleNamespace(), MotionBackend) is False


def test_a_result_cannot_land_between_storing_a_replacement_and_its_cancel() -> None:
    """A pose sent as the running motion finishes is never accepted-and-dropped.

    The backend's result arrives on another thread while handle_command is
    inside the backend's cancel(): it must wait for the command to finish, then
    end the old motion as cancelled and start the new one.
    """
    subject, backend, _ = _subject()
    assert _set_pose(subject) is None
    first_done = backend.goals[-1][2]
    in_cancel, release = threading.Event(), threading.Event()

    def slow_cancel() -> None:
        backend.cancelled += 1
        in_cancel.set()
        release.wait(5)

    backend.cancel = slow_cancel
    answers: list = []
    sender = threading.Thread(target=lambda: answers.append(_set_pose(subject)))
    sender.start()
    assert in_cancel.wait(5)
    result_thread = threading.Thread(
        target=lambda: first_done(
            MotionResult(MotionOutcome.SUCCEEDED, planner=PlannerKind.PINOCCHIO)
        )
    )
    result_thread.start()
    result_thread.join(0.2)
    assert result_thread.is_alive()  # held off until the command is done
    release.set()
    sender.join(5)
    result_thread.join(5)
    assert answers == [None]
    assert len(backend.goals) == 2  # the new pose reached the backend
    status = subject.status()
    assert status.last_result["outcome"] == "cancelled"
    assert status.motion_state is MotionState.PLANNING
    assert status.motion_id == 2
