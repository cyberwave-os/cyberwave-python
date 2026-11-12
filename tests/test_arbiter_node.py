"""ROS transport for the motion arbiter, driven against fake rclpy/message modules.

No ROS installation is needed: the node factory imports rclpy and the message
packages lazily, so the fakes below are installed into ``sys.modules`` for each
test only. They record subscriptions, clients, services and timers so the
transport contract (topics, QoS, goal-status handling, freshness gate, reset
service) is exercised as behaviour rather than read from source text.
"""

from __future__ import annotations

import enum
import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.driver.control.arbiter import MotionOwner
from cyberwave.driver.control.command_boundary import (
    AdaptationResult,
    CommandRejection,
    CommandScopes,
    MotionSource,
)
from cyberwave.driver.ros2.arbiter_node import (
    ArbiterNodeSpec,
    build_motion_arbiter_node_class,
    command_arrays_valid,
    gripper_effort_from_message,
)

ARM = tuple(f"joint{i}" for i in range(1, 7))
SCOPES = CommandScopes(arm=ARM, gripper=("joint7",))
LOWER = [-2.0] * 6 + [0.0]
UPPER = [2.0] * 6 + [0.05]
NOW_NS = 1_000 * 10**9

ACTION_STATUS_QOS = object()


class _ParamType(enum.Enum):
    DOUBLE_ARRAY = "double_array"


class _Uninitialized(Exception):
    pass


class _Param:
    Type = _ParamType

    def __init__(self, name: str, value=None) -> None:
        self.name, self.value = name, value


class _Header:
    def __init__(self) -> None:
        self.stamp = SimpleNamespace(sec=0, nanosec=0)
        self.frame_id = ""


class _JointState:
    def __init__(self) -> None:
        self.header = _Header()
        self.name: list[str] = []
        self.position: list[float] = []
        self.velocity: list[float] = []
        self.effort: list[float] = []


class _GoalStatus:
    STATUS_ACCEPTED = 1
    STATUS_EXECUTING = 2
    STATUS_CANCELING = 3
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


class _CancelGoal:
    Request = MagicMock(name="CancelGoal.Request")


class _FakeClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ready = True
        self.calls: list = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request) -> None:
        self.calls.append(request)


class _FakeNode:
    def __init__(self, name, *, cli_args=None, namespace=None, parameter_overrides=()):
        self.node_name, self.cli_args, self.namespace = name, cli_args, namespace
        self._overrides = {p.name: p.value for p in parameter_overrides}
        self._params: dict[str, object] = {}
        self.subscriptions: list[tuple] = []
        self.clients: list[_FakeClient] = []
        self.services: list[tuple] = []
        self.timers: list[tuple] = []
        self.logger = MagicMock()
        self.now_ns = NOW_NS

    def declare_parameter(self, name, default):
        if name in self._overrides:
            self._params[name] = self._overrides[name]
        elif isinstance(default, _ParamType):
            self._params[name] = _Uninitialized
        else:
            self._params[name] = default

    def get_parameter(self, name):
        value = self._params[name]
        if value is _Uninitialized:
            raise _Uninitialized(name)
        return SimpleNamespace(value=value)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))

    def create_client(self, srv_type, name, callback_group=None):
        client = _FakeClient(name)
        self.clients.append(client)
        return client

    def create_service(self, srv_type, name, callback, callback_group=None):
        self.services.append((srv_type, name, callback))

    def create_timer(self, period, callback):
        self.timers.append((period, callback))

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(
                nanoseconds=self.now_ns,
                to_msg=lambda: SimpleNamespace(
                    sec=self.now_ns // 10**9, nanosec=self.now_ns % 10**9
                ),
            )
        )


@pytest.fixture
def fake_ros(monkeypatch):
    def install(name: str, **attrs) -> None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    install("action_msgs")
    install(
        "action_msgs.msg", GoalStatus=_GoalStatus, GoalStatusArray=type("GSA", (), {})
    )
    install("action_msgs.srv", CancelGoal=_CancelGoal)
    if "rclpy" not in sys.modules:
        install("rclpy")
    install("rclpy.callback_groups", ReentrantCallbackGroup=MagicMock)
    install("rclpy.node", Node=_FakeNode)
    install("rclpy.parameter", Parameter=_Param)
    install(
        "rclpy.qos",
        QoSProfile=lambda **kw: SimpleNamespace(**kw),
        DurabilityPolicy=SimpleNamespace(VOLATILE="volatile"),
        HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        ReliabilityPolicy=SimpleNamespace(
            RELIABLE="reliable", BEST_EFFORT="best_effort"
        ),
        qos_profile_action_status_default=ACTION_STATUS_QOS,
    )
    install("sensor_msgs")
    install("sensor_msgs.msg", JointState=_JointState)
    install("std_srvs")
    install("std_srvs.srv", Trigger=object)


_STATUS_TYPE = type("ArmStatus", (), {})


def _spec(**overrides) -> ArbiterNodeSpec:
    fields: dict = {
        "node_name": "test_motion_arbiter",
        "scopes": SCOPES,
        "command_frame_id": "vendor_frame",
        "arm_status_type": lambda: _STATUS_TYPE,
        "is_fault": lambda msg: bool(msg.fault),
        "is_teaching": lambda msg: bool(msg.teaching),
    }
    fields.update(overrides)
    return ArbiterNodeSpec(**fields)


def _node(*, spec=None, node_class=None, limits=True, planned=True, extra=()):
    cls = node_class or build_motion_arbiter_node_class(spec or _spec())
    overrides = [_Param("planned_motion_allowed", planned), *extra]
    if limits:
        overrides += [
            _Param("joint_limits_lower", LOWER),
            _Param("joint_limits_upper", UPPER),
        ]
    sent: list = []
    node = cls(vendor_send=sent.append, parameter_overrides=overrides)
    node.sent = sent
    node.clock = [10.0]
    node._now = lambda: node.clock[0]
    return node


def _sub(node, topic):
    return next(s for s in node.subscriptions if s[1] == topic)


def _msg(names, positions, *, age_s=0.01, effort=()):
    message = _JointState()
    stamp_ns = NOW_NS - int(age_s * 1e9)
    message.header.stamp = SimpleNamespace(
        sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9
    )
    message.name, message.position, message.effort = (
        list(names),
        list(positions),
        list(effort),
    )
    return message


def _status(*entries):
    return SimpleNamespace(
        status_list=[
            SimpleNamespace(
                status=status,
                goal_info=SimpleNamespace(
                    goal_id=SimpleNamespace(uuid=[index] * 16),
                    stamp=SimpleNamespace(sec=sec, nanosec=0),
                ),
            )
            for index, (status, sec) in enumerate(entries)
        ]
    )


_MANUAL = ([*ARM, "joint7"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02])
_ARM = (list(ARM), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
_MEASURED = ([*ARM, "joint7"], [0.0] * 7)


# ── wiring contract ──────────────────────────────────────────────────────────


def test_action_status_uses_the_ros_action_status_qos(fake_ros) -> None:
    """A late arbiter must still see an already accepted goal (TRANSIENT_LOCAL).

    A generic VOLATILE profile makes goal ownership depend on startup timing and
    can suppress every planned setpoint while the action reports acceptance.
    """
    node = _node()
    _, _, _, qos = _sub(node, "arm_controller/follow_joint_trajectory/_action/status")
    assert qos is ACTION_STATUS_QOS


def test_the_ownership_action_is_configurable_for_a_trajectory_bridge(fake_ros) -> None:
    node = _node(
        extra=[_Param("trajectory_action", "/cw_bridge/follow_joint_trajectory")]
    )
    topics = [s[1] for s in node.subscriptions]
    assert "cw_bridge/follow_joint_trajectory/_action/status" in topics
    assert (
        node.clients[0].name == "cw_bridge/follow_joint_trajectory/_action/cancel_goal"
    )


def test_command_inputs_are_separate_reliable_and_depth_one(fake_ros) -> None:
    node = _node()
    for topic in ("joint_commands", "trajectory_joint_commands", "gripper_commands"):
        qos = _sub(node, topic)[3]
        assert (qos.depth, qos.reliability) == (1, "reliable")
    assert _sub(node, "joint_states")[3].reliability == "best_effort"
    assert node.cli_args == ["--ros-args", "-r", "__node:=test_motion_arbiter"]
    assert [name for _, name, _ in node.services] == ["reset_motion_fault"]


def test_topics_are_relative_and_remappable(fake_ros) -> None:
    node = _node(extra=[_Param("command_topic", "/teleop/joint_commands")])
    assert any(s[1] == "teleop/joint_commands" for s in node.subscriptions)


def test_no_gripper_scope_means_no_gripper_input_and_no_status_without_decoder(
    fake_ros,
) -> None:
    spec = _spec(scopes=CommandScopes(arm=ARM), is_fault=None)
    node = build_motion_arbiter_node_class(spec)(
        vendor_send=lambda _m: None,
        parameter_overrides=[
            _Param("joint_limits_lower", LOWER[:6]),
            _Param("joint_limits_upper", UPPER[:6]),
        ],
    )
    topics = [s[1] for s in node.subscriptions]
    assert "gripper_commands" not in topics
    assert "arm_status" not in topics


def test_missing_limits_are_logged_and_refuse_everything(fake_ros) -> None:
    node = _node(limits=False)
    node.logger.error.assert_called()
    _sub(node, "joint_commands")[2](_msg(*_MANUAL))
    assert node.sent == []
    assert node.core.last_rejection is CommandRejection.MISSING_LIMIT


def test_planned_motion_defaults_closed_at_the_ros_boundary(fake_ros) -> None:
    cls = build_motion_arbiter_node_class(_spec())
    sent: list = []
    node = cls(
        vendor_send=sent.append,
        parameter_overrides=[
            _Param("joint_limits_lower", LOWER),
            _Param("joint_limits_upper", UPPER),
        ],
    )
    node.logger.warning.assert_called()
    node._now = lambda: 10.0
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    _sub(node, "arm_controller/follow_joint_trajectory/_action/status")[2](
        _status((_GoalStatus.STATUS_EXECUTING, 1))
    )
    _sub(node, "trajectory_joint_commands")[2](_msg(*_ARM))
    assert node.core.owner is MotionOwner.TRAJECTORY
    assert node.core.last_rejection is None  # dropped by the gate, not invalid
    assert sent == []


# ── command flow ─────────────────────────────────────────────────────────────


def test_a_goal_grants_trajectory_ownership_and_setpoints_reach_the_vendor(
    fake_ros,
) -> None:
    node = _node()
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    _sub(node, "arm_controller/follow_joint_trajectory/_action/status")[2](
        _status((_GoalStatus.STATUS_EXECUTING, 1))
    )
    assert node.core.owner is MotionOwner.TRAJECTORY
    _sub(node, "trajectory_joint_commands")[2](_msg(*_ARM))
    assert len(node.sent) == 1
    message = node.sent[0]
    assert message.header.frame_id == "vendor_frame"
    assert message.name == [*ARM, "joint7"]
    assert message.position == _ARM[1] + [0.0]


def test_stale_wrong_scope_and_velocity_inputs_are_dropped_before_the_core(
    fake_ros,
) -> None:
    node = _node()
    manual = _sub(node, "joint_commands")[2]
    manual(_msg(*_MANUAL, age_s=1.0))  # stale
    manual(_msg(*_ARM))  # trajectory-shaped on the manual input
    with_velocity = _msg(*_MANUAL)
    with_velocity.velocity = [0.0] * 7
    manual(with_velocity)
    assert node.sent == []
    assert node.core.owner is MotionOwner.IDLE
    manual(_msg(*_MANUAL))
    assert len(node.sent) == 1


def test_the_newest_goal_status_entry_decides_ownership(fake_ros) -> None:
    node = _node()
    status = _sub(node, "arm_controller/follow_joint_trajectory/_action/status")[2]
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    status(_status((_GoalStatus.STATUS_EXECUTING, 5)))
    assert node.core.goal_active is True
    # The same goal (uuid index 0) is now terminal, listed next to an older
    # goal still CANCELING: the newest entry must win.
    status(
        _status((_GoalStatus.STATUS_SUCCEEDED, 5), (_GoalStatus.STATUS_CANCELING, 1))
    )
    assert node.core.goal_active is False


def test_manual_preemption_cancels_through_the_action_cancel_service(fake_ros) -> None:
    node = _node()
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    _sub(node, "arm_controller/follow_joint_trajectory/_action/status")[2](
        _status((_GoalStatus.STATUS_EXECUTING, 1))
    )
    _sub(node, "joint_commands")[2](_msg(*_MANUAL))
    assert node.core.owner is MotionOwner.MANUAL
    period, tick = node.timers[0]
    assert period == pytest.approx(0.1)
    node.clock[0] = 10.2
    tick()
    assert len(node.clients[0].calls) == 1


def test_an_unavailable_cancel_service_is_logged_not_waited_on(fake_ros) -> None:
    node = _node()
    node.clients[0].ready = False
    _sub(node, "arm_controller/follow_joint_trajectory/_action/status")[2](
        _status((_GoalStatus.STATUS_EXECUTING, 1))
    )
    _sub(node, "arm_status")[2](SimpleNamespace(fault=True, teaching=False))
    assert node.core.owner is MotionOwner.FAULT
    assert node.clients[0].calls == []
    assert any(
        "cancel service unavailable" in call.args[0]
        for call in node.logger.warning.call_args_list
    )


def test_arm_status_predicates_drive_teaching_refusal_and_fault_latch(fake_ros) -> None:
    node = _node()
    status = _sub(node, "arm_status")[2]
    manual = _sub(node, "joint_commands")[2]
    status(SimpleNamespace(fault=False, teaching=True))
    manual(_msg(*_MANUAL))
    assert node.core.last_rejection is CommandRejection.TEACHING_MODE
    status(SimpleNamespace(fault=False, teaching=False))
    manual(_msg(*_MANUAL))
    assert len(node.sent) == 1
    status(SimpleNamespace(fault=True, teaching=False))
    assert node.core.owner is MotionOwner.FAULT


def test_the_reset_service_reports_refusal_and_success(fake_ros) -> None:
    node = _node()
    reset = node.services[0][2]
    response = reset(None, SimpleNamespace(success=None, message=""))
    assert (response.success, response.message) == (False, "refused: not_faulted")
    status = _sub(node, "arm_status")[2]
    status(SimpleNamespace(fault=True, teaching=False))
    response = reset(None, SimpleNamespace(success=None, message=""))
    assert response.message == "refused: vendor_still_faulted"
    status(SimpleNamespace(fault=False, teaching=False))
    response = reset(None, SimpleNamespace(success=None, message=""))
    assert (response.success, response.message) == (True, "motion fault cleared")
    assert node.core.owner is MotionOwner.IDLE




def test_gripper_age_allows_one_shot_scheduling_jitter_only(fake_ros) -> None:
    node = _node()
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    _sub(node, "gripper_commands")[2](_msg(["joint7"], [0.04], age_s=0.4))
    assert len(node.sent) == 1
    assert node.sent[-1].position[-1] == 0.04

    _sub(node, "joint_commands")[2](_msg(*_MANUAL, age_s=0.4))
    _sub(node, "gripper_commands")[2](_msg(["joint7"], [0.0], age_s=0.8))
    assert len(node.sent) == 1


def test_a_gripper_input_carries_its_effort_to_the_vendor(fake_ros) -> None:
    node = _node()
    _sub(node, "joint_states")[2](_msg(*_MEASURED))
    _sub(node, "gripper_commands")[2](_msg(["joint7"], [0.04], effort=[0.8]))
    assert node.sent[-1].position == [0.0] * 6 + [0.04]
    assert node.sent[-1].effort == [0.0] * 6 + [0.8]


# ── vendor hooks ─────────────────────────────────────────────────────────────


def test_a_vendor_subclass_supplies_adapter_parameters_and_message(fake_ros) -> None:
    base = build_motion_arbiter_node_class(_spec())

    class VendorNode(base):  # type: ignore[misc, valid-type]
        def _vendor_adapter(self):
            self.declare_parameter("vendor_speed_percent", 0)
            percent = int(self.get_parameter("vendor_speed_percent").value)

            def adapt(command, *, held_gripper, gripper_effort):
                return AdaptationResult(
                    command=("speed", percent, dict(command.positions))
                )

            return adapt, 1 <= percent <= 100

        def _vendor_message(self, command):
            return command

    node = _node(node_class=VendorNode, extra=[_Param("vendor_speed_percent", 10)])
    _sub(node, "joint_commands")[2](_msg(*_MANUAL))
    assert node.sent[-1][:2] == ("speed", 10)

    inhibited = _node(node_class=VendorNode)
    inhibited.logger.error.assert_called()
    _sub(inhibited, "joint_commands")[2](_msg(*_MANUAL))
    assert inhibited.sent == []
    assert (
        inhibited.core.last_rejection is CommandRejection.VELOCITY_CONTRACT_UNRESOLVED
    )


# ── diagnostics ──────────────────────────────────────────────────────────────


def test_command_diagnostics_identify_source_target_age_and_bound_log_rate(
    fake_ros,
) -> None:
    node = _node()
    message = _msg(["joint7"], [0.0], age_s=0.4)
    report = node._report_command_outcome
    for _ in range(100):
        report(MotionSource.TRAJECTORY, message, None)
        report(MotionSource.GRIPPER, message, "stale_or_unstamped_command")
    node.logger.info.assert_called_once_with("trajectory commands accepted")
    drops = [
        call.args[0]
        for call in node.logger.warning.call_args_list
        if "DROPPED" in call.args[0]
    ]
    assert len(drops) == 1
    for detail in (
        "gripper command DROPPED",
        "stale_or_unstamped_command",
        "age_s=0.400",
        "max_age_s=0.75",
        "joint7",
    ):
        assert detail in drops[0]
    node.clock[0] += 1.1
    report(MotionSource.GRIPPER, message, "stale_or_unstamped_command")
    assert sum("DROPPED" in c.args[0] for c in node.logger.warning.call_args_list) == 2
    report(MotionSource.GRIPPER, message, None)
    assert node.logger.info.call_args.args[0] == "gripper commands accepted"


def test_gripper_effort_is_read_from_the_gripper_slot_only() -> None:
    names = [*ARM, "joint7"]
    assert (
        gripper_effort_from_message(
            SimpleNamespace(name=names, effort=[0.0] * 6 + [0.8]), SCOPES
        )
        == 0.8
    )
    assert (
        gripper_effort_from_message(
            SimpleNamespace(name=list(ARM), effort=[0.0] * 6), SCOPES
        )
        is None
    )
    assert (
        gripper_effort_from_message(SimpleNamespace(name=names, effort=[]), SCOPES)
        is None
    )
    assert (
        gripper_effort_from_message(
            SimpleNamespace(name=names, effort=[0.8] * 7), CommandScopes(arm=ARM)
        )
        is None
    )


def test_shared_ros_input_rejects_auxiliary_array_mismatches() -> None:
    manual = SimpleNamespace(
        name=[*ARM, "joint7"], position=[0.0] * 7, velocity=[], effort=[0.0] * 6 + [0.8]
    )
    assert command_arrays_valid(manual, MotionSource.MANUAL)
    manual.effort = [0.0] * 6
    assert not command_arrays_valid(manual, MotionSource.MANUAL)
    manual.effort = [0.0] * 6 + [math.nan]
    assert not command_arrays_valid(manual, MotionSource.MANUAL)
    manual.effort = []
    manual.velocity = [0.0] * 7
    assert not command_arrays_valid(manual, MotionSource.MANUAL)
    planned = SimpleNamespace(
        name=list(ARM), position=[0.0] * 6, velocity=[], effort=[]
    )
    assert command_arrays_valid(planned, MotionSource.TRAJECTORY)
    gripper = SimpleNamespace(
        name=["joint7"], position=[0.04], velocity=[], effort=[0.8]
    )
    assert command_arrays_valid(gripper, MotionSource.GRIPPER)
    planned.effort = [0.0] * 6
    assert not command_arrays_valid(planned, MotionSource.TRAJECTORY)
    planned.effort = []
    planned.position = [0.0] * 5
    assert not command_arrays_valid(planned, MotionSource.TRAJECTORY)


def test_importing_the_module_does_not_import_ros() -> None:
    import importlib

    module = importlib.import_module("cyberwave.driver.ros2.arbiter_node")
    assert "rclpy.node" not in vars(module)
