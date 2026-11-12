"""Robot-agnostic motion vocabulary: status, guards and scene IR."""

from __future__ import annotations

import math

import pytest

from cyberwave.driver.motion.guards import (
    MotionRefusal,
    Refusal,
    backend_unavailable,
    check_not_busy,
    check_revision,
    check_start_state,
)
from cyberwave.driver.motion.scene import (
    Pose,
    SceneError,
    parse_attach,
    parse_pose,
    parse_scene_set,
)
from cyberwave.driver.motion.status import MotionState, MotionStatus, PlannerKind


def test_status_payload_has_the_wire_fields():
    s = MotionStatus(
        kinematics_backend="pinocchio",
        backend_available=True,
        active_scene_revision="r1",
    )
    p = s.to_payload()
    for key in (
        "kinematics_backend",
        "backend_available",
        "wrist_payload_present",
        "active_scene_revision",
        "attached_payload",
        "motion_state",
        "command",
        "backend",
        "planner",
        "plan_time_ms",
        "trajectory_points",
        "trajectory_duration_s",
        "progress",
        "last_result",
        "last_refusal",
    ):
        assert key in p
    assert p["motion_state"] == "idle"
    assert p["planner"] is None
    # No planner-framework names in the wire schema: no per-framework fields...
    assert not [key for key in p if "moveit" in key]
    # ...and the planner is whatever string the answering backend reports.
    s.planner = PlannerKind.PINOCCHIO
    assert s.to_payload()["planner"] == "pinocchio"
    s.planner = "a_backend_planner"
    assert s.to_payload()["planner"] == "a_backend_planner"
    assert {kind.value for kind in PlannerKind} == {"pinocchio"}
    # Correlation is by command name and MQTT v5 replies, not a caller id.
    assert "command_id" not in p


def test_backend_unavailable_names_the_backend():
    refusal = backend_unavailable("sampler", "action server not ready")
    assert refusal.code is MotionRefusal.BACKEND_UNAVAILABLE
    assert refusal.code.value == "backend_unavailable"
    assert refusal.status_line() == (
        "backend_unavailable: sampler: action server not ready"
    )
    assert backend_unavailable("sampler", "").detail == "sampler: not available"
    assert not [m for m in MotionRefusal if "moveit" in m.value]


def test_revision_guard():
    assert check_revision("a", "a") is None
    r = check_revision("a", None)
    assert r is not None and r.code is MotionRefusal.SCENE_REVISION_UNKNOWN
    assert check_revision("a", "b").code is MotionRefusal.SCENE_REVISION_UNKNOWN
    # Operator commands carry no revision and are deliberately not gated.
    assert check_revision(None, "b") is None
    assert check_revision(None, None) is None


def test_start_state_guard():
    measured = {"joint1": 0.0, "joint2": 0.5}
    assert check_start_state(None, measured, 0.02) is None
    assert check_start_state({"joint1": 0.01}, measured, 0.02) is None
    r = check_start_state({"joint1": 0.1}, measured, 0.02)
    assert r.code is MotionRefusal.STALE_ROBOT_STATE and "joint1" in r.detail
    missing = check_start_state({"joint9": 0.0}, measured, 0.02)
    assert (
        missing.code is MotionRefusal.STALE_ROBOT_STATE and "joint9" in missing.detail
    )


def test_busy_guard():
    assert check_not_busy(MotionState.IDLE) is None
    assert check_not_busy(MotionState.FAILED) is None
    assert check_not_busy(MotionState.CANCELLED) is None
    assert check_not_busy(MotionState.EXECUTING).code is MotionRefusal.BUSY
    assert check_not_busy(MotionState.PLANNING).code is MotionRefusal.BUSY


def test_refusal_rendering():
    r = Refusal(MotionRefusal.BUSY, "motion c-1 is executing")
    assert r.status_line() == "busy: motion c-1 is executing"
    assert r.to_payload("set_pose") == {
        "command": "set_pose",
        "code": "busy",
        "detail": "motion c-1 is executing",
    }
    assert Refusal(MotionRefusal.CANCELLED).status_line() == "cancelled"


# ── the pre-planning gates ───────────────────────────────────────────────────


def test_the_two_refusal_enums_agree_on_the_shared_codes():
    """set_pose and ee_pose/ee_move are the same physical act, so a caller must not have
    to learn two vocabularies for the same refusal."""
    from cyberwave.driver.control.cartesian_request import CartesianRejection

    for name in (
        "ARM_FAULTED",
        "CONTROL_MODE_UNAVAILABLE",
        "HARDWARE_UNVERIFIED",
        "GEOMETRY_UNVALIDATED",
        "COLLISION_MODEL_INVALID",
        "NO_MEASURED_STATE",
    ):
        assert MotionRefusal[name].value == CartesianRejection[name].value, name


class _Pre:
    def __init__(self, **overrides):
        self.arm_faulted = False
        self.control_mode_ready = True
        self.control_mode_detail = ""
        self.hardware_verified = True
        self.hardware_detail = ""
        self.geometry_validated = True
        self.geometry_detail = ""
        self.collision_model_valid = True
        self.collision_detail = ""
        self.measured_state_seen = True
        self.measured_state_detail = ""
        self.__dict__.update(overrides)


def test_a_clean_precondition_matrix_passes():
    from cyberwave.driver.motion.guards import check_motion_preconditions

    assert check_motion_preconditions(_Pre(), plan_only=False) is None
    assert check_motion_preconditions(_Pre(), plan_only=True) is None


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"arm_faulted": True}, MotionRefusal.ARM_FAULTED),
        ({"control_mode_ready": False}, MotionRefusal.CONTROL_MODE_UNAVAILABLE),
        ({"hardware_verified": False}, MotionRefusal.HARDWARE_UNVERIFIED),
        ({"geometry_validated": False}, MotionRefusal.GEOMETRY_UNVALIDATED),
        ({"collision_model_valid": False}, MotionRefusal.COLLISION_MODEL_INVALID),
        ({"measured_state_seen": False}, MotionRefusal.NO_MEASURED_STATE),
    ],
)
def test_each_gate_refuses_a_motion(overrides, code):
    from cyberwave.driver.motion.guards import check_motion_preconditions

    assert check_motion_preconditions(_Pre(**overrides), plan_only=False).code is code


@pytest.mark.parametrize(
    "overrides",
    [
        {"arm_faulted": True},
        {"control_mode_ready": False},
        {"hardware_verified": False},
        {"geometry_validated": False},
        {"collision_model_valid": False},
    ],
)
def test_a_probe_is_not_blocked_by_gates_that_only_prevent_motion(overrides):
    """A plan-only probe moves nothing, so a faulted arm or unvalidated wrist
    geometry still permits asking 'could this be reached'."""
    from cyberwave.driver.motion.guards import check_motion_preconditions

    assert check_motion_preconditions(_Pre(**overrides), plan_only=True) is None


def test_a_probe_still_needs_a_measured_start_state():
    """Planning from a posture nobody reported is a guess, not a probe."""
    from cyberwave.driver.motion.guards import check_motion_preconditions

    refusal = check_motion_preconditions(
        _Pre(measured_state_seen=False), plan_only=True
    )
    assert refusal.code is MotionRefusal.NO_MEASURED_STATE


def test_the_gates_run_in_the_documented_order():
    """A stale-state refusal must not mask a faulted arm, and vice versa."""
    from cyberwave.driver.motion.guards import check_motion_preconditions

    every = _Pre(
        arm_faulted=True,
        hardware_verified=False,
        geometry_validated=False,
        collision_model_valid=False,
        measured_state_seen=False,
    )
    assert (
        check_motion_preconditions(every, plan_only=False).code
        is MotionRefusal.ARM_FAULTED
    )
    assert (
        check_motion_preconditions(every, plan_only=True).code
        is MotionRefusal.NO_MEASURED_STATE
    )


def test_ee_pose_and_set_pose_refuse_with_the_same_gate():
    """One matrix: whatever is wrong, both entry points name the same cause."""
    from cyberwave.driver import EEPose
    from cyberwave.driver.control.cartesian_request import (
        Preconditions,
        validate_cartesian_request,
    )
    from cyberwave.driver.motion.guards import check_motion_preconditions

    cases = [
        {"arm_faulted": True, "measured_state_seen": False},
        {"control_mode_ready": False, "collision_model_valid": False},
        {"geometry_validated": False, "measured_state_seen": False},
        {"measured_state_seen": False},
    ]
    for overrides in cases:
        pre = Preconditions(**overrides)
        gate = check_motion_preconditions(pre, plan_only=False)
        decision = validate_cartesian_request(
            EEPose(0.3, 0.0, 0.2, 0.0, 0.0, 0.0), source="ee_pose", preconditions=pre
        )
        assert decision.rejection.value == gate.code.value, overrides
        assert decision.detail == gate.detail, overrides


# ── scene IR ──

IDENTITY = {
    "position": {"x": 0.3, "y": 0.0, "z": 0.1},
    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
}


def _box(id_, dims=(0.1, 0.1, 0.1)):
    return {"id": id_, "primitive": "box", "dimensions": list(dims), "pose": IDENTITY}


def test_parse_pose_requires_unit_quaternion_and_finite_values():
    assert parse_pose(IDENTITY) == Pose((0.3, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(SceneError) as excinfo:
        parse_pose({**IDENTITY, "orientation": {"x": 0, "y": 0, "z": 0, "w": 0.5}})
    assert excinfo.value.refusal.code is MotionRefusal.INVALID_REQUEST
    with pytest.raises(SceneError):
        parse_pose({**IDENTITY, "position": {"x": math.nan, "y": 0, "z": 0}})
    with pytest.raises(SceneError):
        parse_pose({**IDENTITY, "position": {"x": math.inf, "y": 0, "z": 0}})
    with pytest.raises(SceneError):
        parse_pose("not a pose")


def test_scene_poses_accept_every_set_pose_form_but_no_frame():
    """One parser: a scene object takes what set_pose takes, in its fixed frame."""
    identity = Pose((0.3, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))
    for form in (
        [0.3, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        [[0.3, 0.0, 0.1], [0.0, 0.0, 0.0]],
        {"position": [0.3, 0.0, 0.1], "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}},
    ):
        assert parse_pose(form) == identity, form
    with pytest.raises(SceneError) as excinfo:
        parse_pose(
            {**IDENTITY, "reference_frame": "tool"}, code=MotionRefusal.INVALID_SCENE
        )
    assert excinfo.value.refusal.code is MotionRefusal.INVALID_SCENE
    with pytest.raises(SceneError) as excinfo:
        parse_pose([0.3, 0.0, 0.1], code=MotionRefusal.INVALID_SCENE)
    assert excinfo.value.refusal.code is MotionRefusal.INVALID_SCENE


def test_parse_scene_set_accepts_the_three_primitives_and_empty_lists():
    request = parse_scene_set(
        {
            "command_id": "c1",
            "scene_revision": "r1",
            "frame": "base_link",
            "objects": [
                _box("desk", (1.2, 0.8, 0.04)),
                {
                    "id": "can",
                    "primitive": "cylinder",
                    "dimensions": [0.1, 0.03],
                    "pose": IDENTITY,
                },
                {
                    "id": "ball",
                    "primitive": "sphere",
                    "dimensions": [0.04],
                    "pose": IDENTITY,
                },
            ],
            "allow_touch": ["can"],
        }
    )
    assert request.revision == "r1"
    assert [entry.kind for entry in request.objects] == ["box", "cylinder", "sphere"]
    assert request.allow_touch == frozenset({"can"})
    assert parse_scene_set({"scene_revision": "r2", "objects": []}).objects == ()


@pytest.mark.parametrize(
    "bad, code",
    [
        (
            {"scene_revision": "r", "objects": [_box("a"), _box("a")]},
            MotionRefusal.INVALID_SCENE,
        ),
        (
            {
                "scene_revision": "r",
                "objects": [
                    {"id": "m", "primitive": "mesh", "dimensions": [], "pose": IDENTITY}
                ],
            },
            MotionRefusal.INVALID_SCENE,
        ),
        (
            {"scene_revision": "r", "objects": [_box("a", (0.1, 0.1))]},
            MotionRefusal.INVALID_SCENE,
        ),
        (
            {"scene_revision": "r", "objects": [_box("a", (0.1, -0.1, 0.1))]},
            MotionRefusal.INVALID_SCENE,
        ),
        (
            {"scene_revision": "r", "objects": [_box("a", (0.1, 0.0, 0.1))]},
            MotionRefusal.INVALID_SCENE,
        ),
        (
            {"scene_revision": "r", "frame": "tool", "objects": []},
            MotionRefusal.INVALID_SCENE,
        ),
        ({"objects": []}, MotionRefusal.INVALID_SCENE),
        ({"scene_revision": "", "objects": []}, MotionRefusal.INVALID_SCENE),
        ({"scene_revision": "r", "objects": {}}, MotionRefusal.INVALID_SCENE),
        (
            {"scene_revision": "r", "objects": [], "allow_touch": ["ghost"]},
            MotionRefusal.INVALID_SCENE,
        ),
    ],
)
def test_parse_scene_set_refuses_malformed_input(bad, code):
    with pytest.raises(SceneError) as excinfo:
        parse_scene_set(bad)
    assert excinfo.value.refusal.code is code


@pytest.mark.parametrize(
    "bad",
    [
        {"object_id": "", "dimensions": [0.1, 0.1, 0.1], "pose": IDENTITY},
        {"object_id": "o", "dimensions": [0.1, 0.1], "pose": IDENTITY},
        {"object_id": "o", "dimensions": [0.1, 0.1, -1.0], "pose": IDENTITY},
        {"object_id": "o"},
    ],
)
def test_parse_attach_refuses_malformed_input(bad):
    with pytest.raises(SceneError) as excinfo:
        parse_attach(bad)
    assert excinfo.value.refusal.code is MotionRefusal.INVALID_REQUEST


def test_attach_and_scene_set_default_every_option():
    attach = parse_attach({"dimensions": [0.1, 0.2, 0.3]})
    assert attach.object_id == "payload"
    assert attach.pose.position == (0.0, 0.0, 0.0)
    assert attach.pose.orientation == (0.0, 0.0, 0.0, 1.0)
    scene = parse_scene_set({}, default_revision="scene-1")
    assert scene.revision == "scene-1"
    assert scene.objects == () and scene.allow_touch == frozenset()
