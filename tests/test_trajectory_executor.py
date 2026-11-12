"""TrajectoryExecutor: tracked FollowJointTrajectory goals, against fake ROS modules."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.driver.ros2.trajectory_executor import (
    OUTCOME_CANCELLED,
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    TrajectoryExecutor,
)

JOINTS = tuple(f"joint{i}" for i in range(1, 7))
START = dict.fromkeys(JOINTS, 0.0)
SUCCEEDED, CANCELED, ABORTED = 4, 5, 6


@dataclass
class _Trajectory:
    positions: list[dict[str, float]]
    velocities: list[dict[str, float]] = field(default_factory=list)
    accelerations: list[dict[str, float]] = field(default_factory=list)
    time_from_start_s: list[float] = field(default_factory=list)


def _trajectory(end: float = 0.5) -> _Trajectory:
    # Keys deliberately reversed: the wire order must come from joint_names.
    def point(value: float) -> dict[str, float]:
        return {name: value if name == "joint1" else 0.0 for name in reversed(JOINTS)}

    return _Trajectory(
        positions=[point(0.0), point(end / 2), point(end)],
        velocities=[point(0.0), point(0.3), point(0.0)],
        accelerations=[point(0.1), point(0.0), point(-0.1)],
        time_from_start_s=[0.0, 1.25, 2.5],
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
        "action_msgs.msg",
        GoalStatus=SimpleNamespace(
            STATUS_SUCCEEDED=SUCCEEDED, STATUS_CANCELED=CANCELED
        ),
    )
    install("builtin_interfaces")
    install("builtin_interfaces.msg", Duration=lambda **kw: SimpleNamespace(**kw))
    install("control_msgs")
    install(
        "control_msgs.action",
        FollowJointTrajectory=SimpleNamespace(
            Goal=SimpleNamespace, Result=SimpleNamespace(SUCCESSFUL=0)
        ),
    )
    install("control_msgs.msg", JointTolerance=lambda **kw: SimpleNamespace(**kw))
    if "rclpy" not in sys.modules:
        install("rclpy")
    action_client = MagicMock(name="ActionClient")
    install("rclpy.action", ActionClient=action_client)
    install("trajectory_msgs")
    install(
        "trajectory_msgs.msg",
        JointTrajectory=lambda: SimpleNamespace(joint_names=[], points=[]),
        JointTrajectoryPoint=lambda: SimpleNamespace(
            positions=[], velocities=[], accelerations=[]
        ),
    )
    return action_client


class _Client:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.goals: list = []
        self.future = MagicMock()

    def server_is_ready(self) -> bool:
        return self.ready

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.future

    def response(self, index: int = -1):
        return self.future.add_done_callback.call_args_list[index].args[0]


def _executor(client=None, **kwargs) -> TrajectoryExecutor:
    node = SimpleNamespace(get_logger=lambda: MagicMock())
    return TrajectoryExecutor(
        node,
        "/ns/arm_controller/follow_joint_trajectory",
        JOINTS,
        client=client or _Client(),
        **kwargs,
    )


def _accept(executor_client: _Client, index: int = -1) -> MagicMock:
    handle = MagicMock(accepted=True)
    executor_client.response(index)(SimpleNamespace(result=lambda: handle))
    return handle


def _result(
    handle: MagicMock, status: int, error_code: int = 0, error_string: str = ""
):
    completed = handle.get_result_async.return_value.add_done_callback.call_args.args[0]
    completed(
        SimpleNamespace(
            result=lambda: SimpleNamespace(
                status=status,
                result=SimpleNamespace(
                    error_code=error_code, error_string=error_string
                ),
            )
        )
    )


# ── the goal on the wire ─────────────────────────────────────────────────────


def test_the_planned_trajectory_is_sent_point_by_point_in_joint_order(fake_ros) -> None:
    client = _Client()
    executor = _executor(client)
    trajectory = _trajectory()
    assert executor.send(trajectory)
    goal = client.goals[0]
    assert goal.trajectory.joint_names == list(JOINTS)
    for point, positions, velocities, accelerations, when in zip(
        goal.trajectory.points,
        trajectory.positions,
        trajectory.velocities,
        trajectory.accelerations,
        trajectory.time_from_start_s,
        strict=True,
    ):
        assert point.positions == [positions[name] for name in JOINTS]
        assert point.velocities == [velocities[name] for name in JOINTS]
        assert point.accelerations == [accelerations[name] for name in JOINTS]
        assert point.time_from_start.sec + point.time_from_start.nanosec / 1e9 == when


def test_default_goal_keeps_the_controller_tolerances(fake_ros) -> None:
    client = _Client()
    assert _executor(client).send(_trajectory())
    goal = client.goals[0]
    assert not hasattr(goal, "path_tolerance")
    assert not hasattr(goal, "goal_time_tolerance")


def test_loose_tracking_vendors_can_disable_path_tolerance_and_goal_deadline(
    fake_ros,
) -> None:
    """One global vendor speed lags the timed curve; only the endpoint is checked."""
    client = _Client()
    executor = _executor(
        client, path_position_tolerance=-1.0, goal_time_tolerance_s=0.0
    )
    assert executor.send(_trajectory())
    goal = client.goals[0]
    assert [(t.name, t.position) for t in goal.path_tolerance] == [
        (name, -1.0) for name in JOINTS
    ]
    assert (goal.goal_time_tolerance.sec, goal.goal_time_tolerance.nanosec) == (0, 0)


def test_velocities_may_be_left_to_the_controller(fake_ros) -> None:
    client = _Client()
    trajectory = _trajectory()
    trajectory.velocities, trajectory.accelerations = [], []
    assert _executor(client).send(trajectory)
    assert client.goals[0].trajectory.points[1].velocities == []


# ── refusals: never pretend a trajectory was sent ────────────────────────────


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: setattr(t, "positions", []),
        lambda t: setattr(t, "time_from_start_s", [0.5, 1.0, 2.0]),
        lambda t: setattr(t, "velocities", t.velocities[:2]),
        lambda t: setattr(t, "time_from_start_s", [0.0, 1.0]),
    ],
)
def test_malformed_trajectories_are_refused(fake_ros, mutate) -> None:
    client = _Client()
    trajectory = _trajectory()
    mutate(trajectory)
    assert not _executor(client).send(trajectory)
    assert client.goals == []


def test_a_trajectory_that_does_not_start_at_the_measured_state_is_refused(
    fake_ros,
) -> None:
    client = _Client()
    measured = dict(START)
    executor = _executor(
        client, current_positions=lambda: measured, start_state_tolerance_rad=0.01
    )
    assert executor.send(_trajectory())
    measured["joint1"] = 0.5
    assert not executor.send(_trajectory())
    assert len(client.goals) == 1


def test_an_unready_server_is_refused_without_waiting(fake_ros) -> None:
    client = _Client(ready=False)
    log = MagicMock()
    node = SimpleNamespace(
        _managed_launch=SimpleNamespace(is_running=False, _log_path="/tmp/l.log")
    )
    executor = TrajectoryExecutor(node, "arm/fjt", JOINTS, client=client, log=log)
    assert not executor.send(_trajectory())
    assert "managed launch exited" in log.warning.call_args.args[0]
    assert "/tmp/l.log" in log.warning.call_args.args[0]


def test_without_the_ros_action_stack_send_refuses(monkeypatch) -> None:
    for name in ("control_msgs.action", "control_msgs.msg", "control_msgs"):
        monkeypatch.setitem(sys.modules, name, None)
    executor = _executor(_Client())
    assert executor.send(_trajectory()) is False


def test_a_transport_failure_is_refused_and_not_counted(fake_ros) -> None:
    client = _Client()
    client.send_goal_async = MagicMock(side_effect=RuntimeError("context invalid"))
    executor = _executor(client)
    assert not executor.send(_trajectory())
    assert executor.in_flight == 0


# ── tracking ─────────────────────────────────────────────────────────────────


def test_each_goal_is_tracked_until_its_outcome_is_reported(fake_ros) -> None:
    client = _Client()
    executor = _executor(client)
    done = MagicMock()
    assert executor.send(_trajectory(), on_done=done)
    assert executor.send(_trajectory(), on_done=done)
    assert executor.in_flight == 2 and executor.busy
    client.response(0)(SimpleNamespace(result=lambda: None))  # rejected
    assert executor.in_flight == 1
    done.assert_called_with(
        OUTCOME_FAILED, "joint trajectory controller rejected the goal"
    )
    client.response(1)(SimpleNamespace(result=MagicMock(side_effect=RuntimeError("x"))))
    assert executor.in_flight == 0
    assert done.call_count == 2


@pytest.mark.parametrize(
    "status,error_code,error_string,expected",
    [
        (SUCCEEDED, 0, "", (OUTCOME_SUCCEEDED, "")),
        (CANCELED, 0, "stopped", (OUTCOME_CANCELLED, "stopped")),
        (
            ABORTED,
            -4,
            "path tolerance violated",
            (OUTCOME_FAILED, "path tolerance violated"),
        ),
        (
            ABORTED,
            -5,
            "",
            (OUTCOME_FAILED, "FollowJointTrajectory status=6 error_code=-5"),
        ),
        (
            SUCCEEDED,
            -1,
            "",
            (OUTCOME_FAILED, "FollowJointTrajectory status=4 error_code=-1"),
        ),
    ],
)
def test_results_map_to_a_closed_outcome(
    fake_ros, status, error_code, error_string, expected
) -> None:
    client = _Client()
    executor = _executor(client)
    active, done = MagicMock(), MagicMock()
    assert executor.send(_trajectory(), active, done)
    handle = _accept(client)
    active.assert_called_once()
    _result(handle, status, error_code, error_string)
    done.assert_called_once_with(*expected)
    assert executor.in_flight == 0


def test_cancel_targets_accepted_goals_and_late_acceptances_only(fake_ros) -> None:
    """A cancel before acceptance must stop that late goal, never its successor."""
    client = _Client()
    executor = _executor(client)
    active = MagicMock()
    assert executor.send(_trajectory(), active)
    running = _accept(client)
    assert executor.send(_trajectory(), active)
    late_index = len(client.future.add_done_callback.call_args_list) - 1
    executor.cancel()
    running.cancel_goal_async.assert_called_once()
    assert executor.send(_trajectory(), active)
    late = _accept(client, late_index)
    fresh = _accept(client)
    late.cancel_goal_async.assert_called_once()
    fresh.cancel_goal_async.assert_not_called()
    assert active.call_count == 2  # the running goal and the fresh one


def test_finished_goals_are_no_longer_cancelled(fake_ros) -> None:
    client = _Client()
    executor = _executor(client)
    assert executor.send(_trajectory())
    handle = _accept(client)
    _result(handle, SUCCEEDED)
    executor.cancel()
    handle.cancel_goal_async.assert_not_called()


# ── startup wait ─────────────────────────────────────────────────────────────


def test_wait_for_server_creates_the_client_and_returns_when_ready(fake_ros) -> None:
    fake_ros.return_value.server_is_ready.return_value = True
    node = SimpleNamespace(get_logger=lambda: MagicMock())
    executor = TrajectoryExecutor(
        node, "/ns/arm_controller/follow_joint_trajectory", JOINTS
    )
    executor.wait_for_server(timeout_s=1.0)
    assert fake_ros.call_args.args[0] is node
    assert fake_ros.call_args.args[2] == "/ns/arm_controller/follow_joint_trajectory"
    assert executor.server_is_ready()


def test_wait_for_server_stops_at_an_exited_launch_and_reports_its_log(
    fake_ros,
) -> None:
    fake_ros.return_value.server_is_ready.return_value = False
    launch = SimpleNamespace(
        is_running=False,
        _log_path="/tmp/launch.log",
        _tail_log=lambda: "controller died",
    )
    node = SimpleNamespace(get_logger=lambda: MagicMock(), _managed_launch=launch)
    executor = TrajectoryExecutor(node, "arm/fjt", JOINTS)
    with pytest.raises(RuntimeError, match="controller died"):
        executor.wait_for_server(timeout_s=60.0)
