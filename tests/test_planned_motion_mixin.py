"""PlannedMotionMixin: one attribute selects the backend; nothing else changes.

The parity test drives the SAME command sequence through two driver classes
that differ only in ``kinematics_backend`` and asserts identical
``state/motion`` payloads, refusal codes and outcomes. The pinocchio side is
the built-in backend and plans for real on the collision_arm fixture; the other
side is a FAKE external backend the driver registers through
``_motion_backend_factories`` -- exactly how a driver plugs in a planner the
SDK does not ship -- and answers the same physical question the same way
(succeed when reachable, planning_failed when not).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

try:
    import pinocchio  # noqa: F401
    import scipy  # noqa: F401
except ImportError:
    pytest.skip("pinocchio/scipy unavailable", allow_module_level=True)

from _planned_motion_fakes import (
    READY,
    SRDF,
    FakeExecutor,
    FakeExternalBackend,
    arm_config,
    matrix_pose_dict,
    pose_dict,
)

from cyberwave.driver import (
    ArmCapabilityMixin,
    EECartesianPoseMixin,
    EEPose,
    MotionBackend,
    PlannedMotionMixin,
    WorldObstaclesMixin,
)
from cyberwave.driver.control.cartesian_request import Preconditions
from cyberwave.driver.interface import DriverOperationMode
from cyberwave.driver.interface.registry import DriverInterfaceRegistry
from cyberwave.driver.motion import MotionOutcome, PinocchioMotionBackend
from cyberwave.driver.motion.scene import Pose, Primitive


class _TerminalBase:
    def define_interface(self, iface):
        pass


class _ArmDriver(
    PlannedMotionMixin,
    WorldObstaclesMixin,
    EECartesianPoseMixin,
    ArmCapabilityMixin,
    _TerminalBase,
):
    """One driver body. Its two concrete classes below differ ONLY in the backend."""

    use_base_commands = True
    use_cartesian_pose = True
    PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2 = 2.0

    def __init__(self, *, auto: bool = True) -> None:
        super().__init__()
        self.joints = dict(READY)
        self.executor = FakeExecutor(auto=auto)
        self.external = FakeExternalBackend("external", auto=auto)
        self.failures: list[tuple[str | None, str]] = []
        self.applied: list[dict[str, float]] = []
        self.preconditions = Preconditions()

    # ── the seams a real driver implements once, for every backend ──────────
    def arm_config(self):
        return arm_config()

    def _current_joint_positions(self):
        return dict(self.joints)

    def _apply_joint_targets(self, positions):
        self.applied.append(dict(positions))

    def _on_arm_command_failed(self, command, reason):
        self.failures.append((command, reason))

    def _motion_preconditions(self):
        return self.preconditions

    def _planned_motion_kinematics_kwargs(self):
        return {"srdf_path": str(SRDF)}

    # ── registering a backend the SDK does not ship ─────────────────────────
    def _motion_backend_factories(self):
        return {**super()._motion_backend_factories(), "external": self._external}

    def _external(self) -> MotionBackend:
        return self.external

    # ── test transports (a real driver uses the SDK defaults) ───────────────
    def _motion_trajectory_executor(self):
        return self.executor

    # ── test helpers ─────────────────────────────────────────────────────────
    def command(self, name, **data):
        self._on_planned_motion_command({"command": name, "data": data})

    def payload(self):
        return self.planned_motion.status().to_payload()


class PinocchioArm(_ArmDriver):
    kinematics_backend = "pinocchio"


class ExternalArm(_ArmDriver):
    kinematics_backend = "external"


def _tcp(driver) -> np.ndarray:
    return driver._arm.ee_pose(READY)


def _swung(driver):
    return matrix_pose_dict(driver._arm.ee_pose({**READY, "joint1": 0.3}))


def _lowered(driver, dz=-0.03):
    target = _tcp(driver).copy()
    target[2, 3] += dz
    return matrix_pose_dict(target)


def _scene_objects(driver):
    tcp = _tcp(driver)[:3, 3]
    return [
        {  # well away from every path in the sequence
            "id": "wall",
            "primitive": "box",
            "dimensions": [0.05, 1.0, 1.0],
            "pose": pose_dict((-0.5, 0.0, 0.5)),
        },
        {  # right under the tool: only plannable because it may be touched
            "id": "target",
            "primitive": "box",
            "dimensions": [0.03, 0.03, 0.03],
            "pose": pose_dict(tcp + np.array([0.0, 0.0, -0.075])),
        },
    ]


#: Fields that NAME the backend or its planner and so must differ, and the
#: timing a real planner and a fake cannot share. Everything else must match.
_BACKEND_IDENTITY = {
    "kinematics_backend",
    "backend",
    "planner",
    "plan_time_ms",
    "trajectory_points",
    "trajectory_duration_s",
}


def _normalized(payload: dict) -> dict:
    out = {k: v for k, v in payload.items() if k not in _BACKEND_IDENTITY}
    if out["last_result"] is not None:
        out["last_result"] = {
            k: v
            for k, v in out["last_result"].items()
            if k not in ("planner", "backend", "timestamp_ms", "motion_id")
        }
    if out["last_refusal"] is not None:
        # The detail is the planner's own sentence; the code is the contract.
        out["last_refusal"] = {
            k: v
            for k, v in out["last_refusal"].items()
            if k not in ("detail", "timestamp_ms", "motion_id")
        }
    return out


def _run_sequence(cls) -> list[tuple[str, dict, list[str]]]:
    driver = cls()
    trace: list[tuple[str, dict, list[str]]] = []

    def step(label, name, **data):
        before = len(driver.failures)
        driver.command(name, **data)
        codes = [line.split(":")[0] for _c, line in driver.failures[before:]]
        trace.append((label, driver.payload(), codes))

    step(
        "scene",
        "scene_set",
        scene_revision="r1",
        objects=_scene_objects(driver),
        allow_touch=["target"],
    )
    step(
        "plan",
        "set_pose",
        scene_revision="r1",
        pose=_swung(driver),
        expected_joints=READY,
    )
    step(
        "line",
        "set_pose",
        scene_revision="r1",
        pose=_lowered(driver, -0.06),
        motion="lin",
    )
    step(
        "attach",
        "attach_payload",
        object_id="target",
        dimensions=[0.03, 0.03, 0.03],
        pose=pose_dict((0.0, 0.0, 0.12)),
    )
    step(
        "probe",
        "set_pose",
        scene_revision="r1",
        pose=_swung(driver),
        plan_only=True,
    )
    step(
        "attach-again",
        "attach_payload",
        object_id="other",
        dimensions=[0.03, 0.03, 0.03],
        pose=pose_dict((0.0, 0.0, 0.12)),
    )
    step("detach", "detach_payload", object_id="target")
    step(
        "stale",
        "set_pose",
        scene_revision="r1",
        pose=_swung(driver),
        expected_joints={**READY, "joint2": READY["joint2"] + 0.3},
    )
    step(
        "wrong-revision",
        "set_pose",
        scene_revision="r0",
        pose=_lowered(driver),
        motion="lin",
    )
    step(
        "unreachable",
        "set_pose",
        scene_revision="r1",
        pose=pose_dict((5.0, 0.0, 0.3)),
    )
    step(
        "bad-scene",
        "scene_set",
        scene_revision="r2",
        objects=[{"id": "x", "primitive": "mesh"}],
    )
    driver.preconditions = Preconditions(arm_faulted=True)
    step("faulted", "set_pose", pose=_swung(driver))
    driver.preconditions = Preconditions()

    # busy: hold the next motion in flight on both backends
    driver.executor.auto = driver.external.auto = False
    step("hold", "set_pose", pose=_swung(driver))
    if driver.executor.pending:
        driver.executor.activate()
    if driver.external.pending:
        driver.external.activate()
    step("busy", "scene_set", scene_revision="r3", objects=[])
    driver._stop_motion()
    if driver.executor.pending:
        driver.executor.finish("cancelled")
    if driver.external.pending:
        driver.external.finish(MotionOutcome.CANCELLED)
    trace.append(("stopped", driver.payload(), []))

    # ee_pose arrives through the EECartesianPoseMixin registration
    driver.executor.auto = driver.external.auto = True
    tcp = _tcp(driver)
    driver._on_ee_pose(
        {
            "command": "ee_pose",
            "data": {
                "pose": [
                    float(tcp[0, 3]),
                    float(tcp[1, 3]),
                    float(tcp[2, 3]) - 0.02,
                    *_rpy(tcp),
                ]
            },
        }
    )
    trace.append(("ee_pose", driver.payload(), []))
    return trace


def _rpy(matrix):
    rotation = matrix[:3, :3]
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(np.arcsin(-rotation[2, 0]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return [roll, pitch, yaw]


def test_both_backends_publish_identical_state_motion_for_the_same_commands() -> None:
    pinocchio_trace = _run_sequence(PinocchioArm)
    external_trace = _run_sequence(ExternalArm)
    assert [label for label, *_ in pinocchio_trace] == [
        label for label, *_ in external_trace
    ]
    for (label, ours, our_codes), (_l, theirs, their_codes) in zip(
        pinocchio_trace, external_trace
    ):
        # Same wire shape, key for key.
        assert set(ours) == set(theirs), label
        assert _normalized(ours) == _normalized(theirs), label
        assert our_codes == their_codes, label


def test_the_sequence_reports_what_actually_happened() -> None:
    """Pin the shared trace itself, so parity cannot pass by both being wrong."""
    trace = {
        label: (payload, codes) for label, payload, codes in _run_sequence(PinocchioArm)
    }

    def result(label):
        return trace[label][0]["last_result"]

    def refusal(label):
        return trace[label][0]["last_refusal"]

    assert result("scene") == {
        "command": "scene_set",
        "outcome": "accepted",
        "scene_revision": "r1",
        "object_count": 2,
        "timestamp_ms": result("scene")["timestamp_ms"],
    }
    assert trace["scene"][0]["active_scene_revision"] == "r1"
    assert result("plan")["outcome"] == "succeeded"
    assert result("plan")["motion"] == "ptp"
    assert result("plan")["backend"] == "pinocchio"
    assert trace["plan"][0]["progress"] == 1.0
    assert result("line")["motion"] == "lin"
    assert trace["attach"][0]["attached_payload"] == "target"
    assert result("probe")["outcome"] == "planned"
    assert trace["probe"][0]["progress"] is None
    assert refusal("attach-again")["code"] == "payload_already_attached"
    assert trace["detach"][0]["attached_payload"] is None
    assert refusal("stale")["code"] == "stale_robot_state"
    assert refusal("wrong-revision")["code"] == "scene_revision_unknown"
    assert refusal("unreachable") == {
        "command": "set_pose",
        "code": "planning_failed",
        "detail": refusal("unreachable")["detail"],
        "motion_id": trace["unreachable"][0]["motion_id"],
        "timestamp_ms": refusal("unreachable")["timestamp_ms"],
    }
    assert trace["unreachable"][0]["motion_state"] == "failed"
    assert refusal("bad-scene")["code"] == "invalid_scene"
    assert refusal("faulted")["code"] == "arm_faulted"
    assert (
        trace["hold"][0]["motion_state"] == "planning"
    )  # before the controller took it
    assert refusal("busy")["code"] == "busy"
    assert trace["busy"][0]["motion_state"] == "executing"
    assert trace["stopped"][0]["motion_state"] == "cancelled"
    assert refusal("stopped")["code"] == "cancelled"
    assert trace["ee_pose"][0]["command"] == "ee_pose"
    assert trace["ee_pose"][0]["motion_state"] == "idle"
    assert result("ee_pose")["outcome"] == "succeeded"
    assert trace["unreachable"][1] == ["planning_failed"]


def test_a_request_may_name_the_other_backend() -> None:
    """One command, any registered backend; the reply names who answered."""
    pin, external = PinocchioArm(), ExternalArm()
    pin.command("set_pose", pose=_swung(pin), backend="external", plan_only=True)
    assert pin.external.goals and pin.executor.calls == []
    result = pin.payload()["last_result"]
    assert (result["backend"], result["planner"]) == ("external", "fake_ptp")
    assert pin.payload()["kinematics_backend"] == "pinocchio"  # the default
    external.command(
        "set_pose", pose=_lowered(external), motion="lin", backend="pinocchio"
    )
    result = external.payload()["last_result"]
    assert (result["outcome"], result["backend"]) == ("succeeded", "pinocchio")
    assert result["planner"] == "pinocchio"
    assert len(external.executor.calls) == 1 and external.external.goals == []


def test_a_backend_used_second_plans_against_the_same_scene() -> None:
    """The in-process backend, first used after a scene_set, still sees it."""
    driver = ExternalArm()
    tcp = _tcp(driver)
    wall = {
        "id": "wall",
        "primitive": "box",
        "dimensions": [0.2, 0.2, 0.01],
        "pose": pose_dict(tcp[:3, 3] + np.array([0.0, 0.0, -0.04])),
    }
    driver.command("scene_set", scene_revision="r1", objects=[wall])
    driver.command(
        "set_pose", pose=_lowered(driver, -0.06), motion="lin", backend="pinocchio"
    )
    refusal = driver.payload()["last_refusal"]
    assert refusal["code"] == "planning_failed"
    assert "wall" in refusal["detail"]


def test_a_grasp_needs_only_its_candidates() -> None:
    driver = PinocchioArm()
    driver.command(
        "grasp", candidates=[{"pose": _lowered(driver, -0.04)}], plan_only=True
    )
    result = driver.payload()["last_result"]
    assert (result["outcome"], result["selected_candidate"]) == ("planned", 0)
    assert driver.executor.calls == []


def test_the_catalog_is_identical_for_both_backends() -> None:
    def supported(cls):
        iface = DriverInterfaceRegistry()
        cls().define_interface(iface)
        commands = iface.to_cw_driver_dict(registry_id="t")["mqtt"]["commands"]
        return [c for c in commands["supported"] if isinstance(c, dict)]

    pin, external = supported(PinocchioArm), supported(ExternalArm)
    assert pin == external
    names = [entry["name"] for entry in pin]
    for name in ("set_pose", "grasp", "scene_set", "attach_payload", "detach_payload"):
        assert name in names
    for gone in ("plan_move", "linear_move", "local_ik_move", "local_grasp_move"):
        assert gone not in names
    assert names.count("ee_pose") == 1  # EECartesianPoseMixin's registration
    set_pose = next(entry for entry in pin if entry["name"] == "set_pose")
    defaults = {arg["name"]: arg["default"] for arg in set_pose["args"]}
    assert defaults == {
        "pose": None,
        "mode": "absolute",
        "motion": "ptp",
        "reference_frame": None,
        "backend": None,
        "expected_joints": None,
        "scene_revision": None,
        "plan_only": False,
    }


def test_ee_pose_and_the_publisher_are_registered_without_the_pose_mixin() -> None:
    class Bare(PlannedMotionMixin, _TerminalBase):
        kinematics_backend = "external"  # resolved on the instance, not here

    iface = DriverInterfaceRegistry()
    Bare().define_interface(iface)
    table = iface.command_dispatch_table(DriverOperationMode.TELEOP_LOCAL)
    assert "ee_pose" in table and "set_pose" in table and "grasp" in table
    assert "set_pose" not in iface.command_dispatch_table(DriverOperationMode.NO_OP)
    publishers = iface.tick_publishers_for_mode(DriverOperationMode.NO_OP)
    assert any(entry.topic.leaf == "motion" for entry in publishers)


def test_the_mixin_must_precede_the_mixins_it_overrides() -> None:
    with pytest.raises(TypeError, match="must come before EECartesianPoseMixin"):

        class Wrong(EECartesianPoseMixin, PlannedMotionMixin, _TerminalBase):
            pass

    for bad in ("", "   ", 3, None):
        with pytest.raises(TypeError, match="kinematics_backend must be a non-empty"):

            class Typo(PlannedMotionMixin, _TerminalBase):
                kinematics_backend = bad


# ── the registry seam ───────────────────────────────────────────────────────


def test_only_pinocchio_is_built_in() -> None:
    class Bare(PlannedMotionMixin, _TerminalBase):
        pass

    assert Bare.kinematics_backend == "pinocchio"
    assert set(Bare()._motion_backend_factories()) == {"pinocchio"}
    assert set(PinocchioArm()._motion_backend_factories()) == {"pinocchio", "external"}


def test_an_unknown_backend_name_fails_loudly_with_the_registered_names() -> None:
    class Typo(_ArmDriver):
        kinematics_backend = "externl"  # defining the class is fine

    driver = Typo()
    with pytest.raises(ValueError, match=r"'externl'.*\['external', 'pinocchio'\]"):
        driver.planned_motion  # noqa: B018
    with pytest.raises(ValueError, match="not a registered motion backend"):
        driver.selected_kinematics_backend  # noqa: B018
    # A command reports the failure instead of crashing the dispatch.
    driver.command("set_pose", pose=pose_dict((0.3, 0.0, 0.3)))
    assert "externl" in driver.failures[-1][1]


def test_switching_the_default_is_one_attribute() -> None:
    driver = PinocchioArm()
    controller = driver.planned_motion
    assert isinstance(controller.backend, PinocchioMotionBackend)
    driver.kinematics_backend = "External"  # e.g. from config or an env var
    assert driver.selected_kinematics_backend == "external"
    assert driver.planned_motion is controller  # one controller, one status
    assert controller.backend is driver.external
    assert driver.payload()["kinematics_backend"] == "external"
    driver.kinematics_backend = "pinocchio"
    assert isinstance(controller.backend, PinocchioMotionBackend)
    assert driver.payload()["kinematics_backend"] == "pinocchio"


def test_a_switch_never_strands_the_motion_in_flight() -> None:
    driver = ExternalArm(auto=False)
    driver.command("set_pose", pose=_swung(driver))
    driver.kinematics_backend = "pinocchio"
    assert driver.payload()["backend"] == "external"  # still the one moving
    driver.external.finish()
    assert driver.payload()["last_result"]["outcome"] == "succeeded"
    assert driver.payload()["last_result"]["backend"] == "external"


def test_an_unavailable_registered_backend_refuses_backend_unavailable() -> None:
    driver = ExternalArm()
    driver.external.is_available = False
    driver.command("scene_set", scene_revision="r", objects=[])
    refusal = driver.payload()["last_refusal"]
    assert refusal["code"] == "backend_unavailable"
    assert refusal["detail"].startswith("external: ")  # names the backend
    assert driver.payload()["backend_available"] is False
    assert driver.payload()["kinematics_backend"] == "external"
    driver.command("set_pose", pose=_swung(driver))
    assert driver.payload()["last_refusal"]["command"] == "set_pose"
    assert driver.payload()["last_refusal"]["code"] == "backend_unavailable"
    assert driver.external.goals == []
    # ee_pose is the same physical act and is refused the same way.
    pose = EEPose.from_payload({"pose": [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]})
    before = len(driver.failures)
    decision = driver.submit_planned_ee_pose(pose)
    assert decision.rejection.value == "planner_unavailable"
    assert decision.detail.startswith("external: ")
    # On the stream under the name the caller sent, and reported exactly once,
    # with one code -- not "planning_failed: backend_unavailable: ...".
    assert driver.payload()["last_refusal"]["command"] == "ee_pose"
    assert driver.payload()["last_refusal"]["code"] == "backend_unavailable"
    assert len(driver.failures) == before + 1
    assert driver.failures[-1][1].startswith("backend_unavailable: ")


def test_a_scene_the_planner_did_not_acknowledge_is_not_active() -> None:
    driver = ExternalArm()
    driver.external.scene_ok = False
    driver.command("scene_set", scene_revision="r", objects=[])
    assert driver.payload()["last_refusal"]["code"] == "backend_unavailable"
    assert driver.payload()["active_scene_revision"] is None


def test_stop_cancels_the_active_planned_motion() -> None:
    for cls in (PinocchioArm, ExternalArm):
        driver = cls(auto=False)
        driver.command("set_pose", pose=_swung(driver))
        assert driver.planned_motion.active
        asyncio.run(driver._on_stop_cmd({"command": "stop"}))
        assert driver.executor.cancelled + driver.external.cancelled == 1, cls.__name__


@pytest.mark.parametrize("cls", [PinocchioArm, ExternalArm])
def test_a_driver_jog_takes_the_arm_only_after_the_planned_motion_ends(cls) -> None:
    driver = cls(auto=False)
    jogs: list[str] = []
    assert not driver.replace_planned_motion(lambda: jogs.append("idle"))
    driver.command("set_pose", pose=_swung(driver))

    assert driver.replace_planned_motion(lambda: jogs.append("jog"))
    assert jogs == []
    assert driver.executor.cancelled + driver.external.cancelled == 1
    if cls is ExternalArm:
        driver.external.finish(MotionOutcome.CANCELLED)
    else:
        driver.executor.finish("cancelled")
    assert jogs == ["jog"]
    assert driver.payload()["motion_state"] == "cancelled"


def test_an_ee_jog_cancels_the_planned_motion_before_it_moves_the_arm() -> None:
    """The SDK owns the handoff: no arm has to copy it to be safe."""
    driver = PinocchioArm(auto=False)
    driver.command("set_pose", pose=_swung(driver))
    assert driver.planned_motion.active
    driver._on_arm_command({"command": "ee_move", "data": {"forward": 0.01}})
    # Cancelled, and nothing sent while the trajectory may still be running.
    assert driver.executor.cancelled == 1
    assert driver.applied == []
    driver.executor.finish("cancelled")
    assert len(driver.applied) == 1


def test_an_ee_jog_with_no_planned_motion_runs_at_once() -> None:
    driver = PinocchioArm(auto=False)
    driver._on_arm_command({"command": "ee_move", "data": {"forward": 0.01}})
    assert driver.executor.cancelled == 0
    assert len(driver.applied) == 1


def test_the_state_motion_tick_is_edge_triggered() -> None:
    driver = ExternalArm()
    assert driver._planned_motion_state_payload() is not None
    assert driver._planned_motion_state_payload() is None
    driver.command("scene_set", scene_revision="r", objects=[])
    assert driver._planned_motion_state_payload()["active_scene_revision"] == "r"


def test_without_an_acceleration_bound_pinocchio_says_why() -> None:
    class Untimed(PinocchioArm):
        PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2 = None

    driver = Untimed()
    driver.command("set_pose", pose=_swung(driver))
    refusal = driver.payload()["last_refusal"]
    assert refusal["code"] == "backend_unavailable"
    assert refusal["detail"].startswith("pinocchio: ")
    assert "PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2" in refusal["detail"]


def test_add_obstacles_keeps_the_task_scene_and_the_payload() -> None:
    driver = PinocchioArm()
    driver.command(
        "scene_set",
        scene_revision="r",
        objects=_scene_objects(driver)[:1],
    )
    driver.command(
        "attach_payload",
        object_id="obj",
        dimensions=[0.03, 0.03, 0.03],
        pose=pose_dict((0.0, 0.0, 0.12)),
    )
    floor = Primitive(
        "floor", "box", (2.0, 2.0, 0.02), Pose((0.0, 0.0, -0.05), (0.0, 0.0, 0.0, 1.0))
    )
    driver.replace_world_obstacles((floor,), persist=False)
    kin = driver._arm.kinematics
    assert kin.attached_payload_id == "obj"
    names = {obj.name for obj in kin._geom_model.geometryObjects}
    assert {"world_floor", "world_wall", "payload_obj"} <= names
    # ...and a later scene_set keeps the standing world obstacles.
    driver.command("scene_set", scene_revision="r2", objects=[])
    names = {obj.name for obj in driver._arm.kinematics._geom_model.geometryObjects}
    assert "world_floor" in names and "world_wall" not in names
    assert driver.current_world_obstacles() == (floor,)


def test_world_obstacle_updates_wait_for_a_planned_motion() -> None:
    driver = PinocchioArm(auto=False)
    driver.command("set_pose", pose=_swung(driver))
    with pytest.raises(ValueError, match="a planned motion is"):
        driver.replace_world_obstacles((), persist=False)
