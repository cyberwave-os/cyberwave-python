"""Send timed joint trajectories to a ``FollowJointTrajectory`` action, tracked.

Every arm driver that plans locally (Cartesian jogs, IK moves, grasp
approaches) ends up executing the result the same way: one goal to the arm's
joint trajectory controller, an "it started" callback when the controller
accepts it, an "it finished" callback with a closed outcome, and a way to stop
it. :class:`TrajectoryExecutor` is that, once.

What it gets right that ad-hoc versions did not:

* **The trajectory is sent as planned.** Positions, velocities and
  accelerations are copied point by point; nothing is re-derived, so the
  derivative continuity the planner checked is what the controller executes.
* **It never waits.** :meth:`TrajectoryExecutor.send` refuses (returns False)
  when the action server is not ready instead of blocking the command path on
  a separate launch. Waiting for the server is an explicit, startup-only call
  (:meth:`TrajectoryExecutor.wait_for_server`).
* **Cancel targets only existing goals.** A cancel-all would also kill the goal
  that replaced them. Goals whose acceptance is still in flight when
  :meth:`TrajectoryExecutor.cancel` runs cancel themselves on acceptance (a
  cancel *generation*), so a late acceptance can never resume stopped motion.
* **Refuse a trajectory that does not start where the arm is.** When a
  current-state reader is configured, the first point is compared with
  measured joints before anything is sent.

Outcome strings passed to ``on_done``: ``"succeeded"``, ``"cancelled"``,
``"failed"`` (with a human-readable detail).

rclpy, ``control_msgs``, ``trajectory_msgs``, ``action_msgs`` and
``builtin_interfaces`` are imported lazily: off-robot (tests, manifest export)
the module imports fine and :meth:`send` refuses honestly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ..motion.guards import check_start_state

logger = logging.getLogger(__name__)

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_FAILED = "failed"

#: Startup wait for the action server (controller spawners can be slow).
DEFAULT_SERVER_WAIT_S = 120.0


class TimedJointTrajectory(Protocol):
    """What :meth:`TrajectoryExecutor.send` needs from a planned trajectory.

    ``positions[i]`` / ``velocities[i]`` / ``accelerations[i]`` map joint name
    to value at ``time_from_start_s[i]`` (seconds; the first must be 0.0, the
    start snapshot). ``velocities``/``accelerations`` may be empty to let the
    controller interpolate.
    """

    @property
    def positions(self) -> Sequence[Mapping[str, float]]: ...
    @property
    def velocities(self) -> Sequence[Mapping[str, float]]: ...
    @property
    def accelerations(self) -> Sequence[Mapping[str, float]]: ...
    @property
    def time_from_start_s(self) -> Sequence[float]: ...


class TrajectoryExecutor:
    """Tracked ``FollowJointTrajectory`` goals for one arm controller.

    Args:
        node: the rclpy node that owns the action client (usually the driver).
            When it has a ``_managed_launch`` its state is included in the
            "server unavailable" diagnostics.
        action_name: fully resolved action name, e.g.
            ``driver.namespaced("arm_controller/follow_joint_trajectory")``.
        joint_names: the controller's joints, in the order sent on the wire.
        path_position_tolerance: ``None`` (default) leaves ``path_tolerance``
            empty, i.e. the controller's configured path constraints. A number
            sets a per-joint position path tolerance; ``-1.0`` disables the
            in-flight position check while keeping the final goal tolerance --
            for vendors that track a timed curve only loosely (one global speed
            applied to position setpoints) but do converge on the endpoint.
        goal_time_tolerance_s: ``None`` (default) leaves the goal's
            ``goal_time_tolerance`` at its message default, which inherits the
            controller's ``goal_time``. A number sets it explicitly.
        current_positions: reads measured joints; with
            ``start_state_tolerance_rad`` it enables the start-state check.
        log: logger with ``info``/``warning`` taking one preformatted string
            (an rclpy logger works); defaults to ``node.get_logger()``.
        client: an existing action client to reuse (tests, or one created by
            :meth:`wait_for_server` elsewhere).
    """

    def __init__(
        self,
        node: Any,
        action_name: str,
        joint_names: Sequence[str],
        *,
        path_position_tolerance: float | None = None,
        goal_time_tolerance_s: float | None = None,
        current_positions: Callable[[], Mapping[str, float]] | None = None,
        start_state_tolerance_rad: float | None = None,
        log: Any = None,
        client: Any = None,
    ) -> None:
        self._node = node
        self.action_name = action_name
        self.joint_names: tuple[str, ...] = tuple(joint_names)
        self.path_position_tolerance = path_position_tolerance
        self.goal_time_tolerance_s = goal_time_tolerance_s
        self.current_positions = current_positions
        self.start_state_tolerance_rad = start_state_tolerance_rad
        self._log = log
        self.client = client
        self._goal_handles: list[Any] = []
        self._cancel_generation = 0
        self._in_flight = 0

    # ── observation ─────────────────────────────────────────────────────────

    @property
    def in_flight(self) -> int:
        """Goals sent whose outcome has not been reported yet."""
        return self._in_flight

    @property
    def busy(self) -> bool:
        return self._in_flight > 0

    def _logger(self) -> Any:
        if self._log is not None:
            return self._log
        get_logger = getattr(self._node, "get_logger", None)
        if callable(get_logger):
            try:
                return get_logger()
            except Exception:  # noqa: BLE001, S110 - logging must never be the failure
                pass
        return logger

    # ── action client ───────────────────────────────────────────────────────

    def ensure_client(self) -> Any:
        """The action client, created on first use (imports rclpy lazily)."""
        if self.client is None:
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient

            self.client = ActionClient(
                self._node, FollowJointTrajectory, self.action_name
            )
        return self.client

    def server_is_ready(self) -> bool:
        client = self.client
        return client is not None and bool(client.server_is_ready())

    def wait_for_server(
        self,
        timeout_s: float = DEFAULT_SERVER_WAIT_S,
        *,
        poll_s: float = 0.1,
        keep_waiting: Callable[[], bool] | None = None,
    ) -> None:
        """Block until the action server is ready, or raise ``RuntimeError``.

        Startup only (``connect_to_device``): Cartesian commands must not be
        activated before the action that executes them exists. Stops early when
        ``keep_waiting`` returns False, and by default when the node's managed
        launch has exited -- waiting two minutes on a dead launch only hides
        its log.
        """
        client = self.ensure_client()
        launch = getattr(self._node, "_managed_launch", None)
        if keep_waiting is None:

            def keep_waiting() -> bool:
                return launch is None or bool(launch.is_running)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not keep_waiting():
                break
            if client.server_is_ready():
                self._logger().info(f"arm trajectory action ready: {self.action_name}")
                return
            time.sleep(poll_s)
        detail = ""
        if launch is not None:
            tail = getattr(launch, "_tail_log", None)
            detail = (
                f" Managed launch log: {getattr(launch, '_log_path', '?')}\n"
                f"{tail() if callable(tail) else ''}"
            )
        raise RuntimeError(
            f"arm trajectory action {self.action_name} did not become available."
            f"{detail}"
        )

    def _unavailable_detail(self) -> str:
        launch = getattr(self._node, "_managed_launch", None)
        state = (
            "not managed"
            if launch is None
            else "running"
            if launch.is_running
            else "exited"
        )
        return (
            f"{self.action_name} is unavailable (managed launch {state}); inspect "
            f"{getattr(launch, '_log_path', '/tmp/cyberwave_managed_launch.log')}"
        )

    # ── execution ───────────────────────────────────────────────────────────

    def send(
        self,
        trajectory: TimedJointTrajectory,
        on_active: Callable[[], None] | None = None,
        on_done: Callable[[str, str], None] | None = None,
    ) -> bool:
        """Submit *trajectory* without rebuilding its derivatives.

        Returns False (and calls neither callback) when the trajectory has no
        start snapshot, does not start at the measured state, the ROS action
        stack is absent, the server is not ready, or submission raises.
        Otherwise exactly one ``on_done(outcome, detail)`` follows, and
        ``on_active()`` once the goal is accepted (unless it was cancelled
        before acceptance, in which case it is cancelled instead).
        """
        on_active = on_active or (lambda: None)
        on_done = on_done or (lambda _outcome, _detail: None)
        log = self._logger()
        if not trajectory.positions or trajectory.time_from_start_s[0] != 0.0:
            log.warning("refusing a trajectory without its start snapshot")
            return False
        count = len(trajectory.positions)
        if len(trajectory.time_from_start_s) != count or any(
            len(series) not in (0, count)
            for series in (trajectory.velocities, trajectory.accelerations)
        ):
            log.warning("refusing a trajectory whose point arrays differ in length")
            return False
        if (
            self.current_positions is not None
            and self.start_state_tolerance_rad is not None
        ):
            refusal = check_start_state(
                trajectory.positions[0],
                self.current_positions(),
                self.start_state_tolerance_rad,
            )
            if refusal is not None:
                log.warning(refusal.status_line())
                return False
        try:
            from action_msgs.msg import GoalStatus
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory
            from control_msgs.msg import JointTolerance
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

            client = self.ensure_client()
        except (ImportError, ModuleNotFoundError) as exc:
            # Off-robot, and during manifest export, the action stack is absent.
            # Refusing is the honest answer: nothing can execute a trajectory,
            # and the alternative would be to pretend one was sent.
            log.warning(
                f"trajectory execution unavailable ({exc}); Cartesian motion "
                "requires the ROS action stack"
            )
            return False
        if not client.server_is_ready():
            # Do not wait: the controller stack is a separate launch, and a
            # blocking wait here would stall the command path on it.
            log.warning(self._unavailable_detail())
            return False

        names = self.joint_names
        message = JointTrajectory()
        message.joint_names = list(names)
        velocities = list(trajectory.velocities)
        accelerations = list(trajectory.accelerations)
        for index in range(count):
            point = JointTrajectoryPoint()
            point.positions = [float(trajectory.positions[index][n]) for n in names]
            if velocities:
                point.velocities = [float(velocities[index][n]) for n in names]
            if accelerations:
                point.accelerations = [float(accelerations[index][n]) for n in names]
            nanoseconds = round(trajectory.time_from_start_s[index] * 1e9)
            point.time_from_start = Duration(
                sec=nanoseconds // 1_000_000_000,
                nanosec=nanoseconds % 1_000_000_000,
            )
            message.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = message
        if self.path_position_tolerance is not None:
            goal.path_tolerance = [
                JointTolerance(name=name, position=float(self.path_position_tolerance))
                for name in names
            ]
        if self.goal_time_tolerance_s is not None:
            total_ns = round(float(self.goal_time_tolerance_s) * 1e9)
            goal.goal_time_tolerance = Duration(
                sec=total_ns // 1_000_000_000, nanosec=total_ns % 1_000_000_000
            )
        cancel_generation = self._cancel_generation
        handles = self._goal_handles
        try:
            goal_future = client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001 - ROS action transport boundary
            log.warning(f"trajectory submission failed: {exc}")
            return False

        self._in_flight += 1

        def finish(outcome: str, detail: str) -> None:
            self._in_flight -= 1
            on_done(outcome, detail)

        def result_done(future: Any) -> None:
            try:
                wrapped = future.result()
                status = int(wrapped.status)
                result = wrapped.result
            except Exception as exc:  # noqa: BLE001 - ROS future boundary
                finish(OUTCOME_FAILED, f"trajectory result failed: {exc}")
                return
            if (
                status == GoalStatus.STATUS_SUCCEEDED
                and int(result.error_code) == FollowJointTrajectory.Result.SUCCESSFUL
            ):
                finish(OUTCOME_SUCCEEDED, "")
                return
            if status == GoalStatus.STATUS_CANCELED:
                finish(OUTCOME_CANCELLED, getattr(result, "error_string", ""))
                return
            detail = getattr(result, "error_string", "")
            if not detail:
                detail = (
                    f"FollowJointTrajectory status={status} "
                    f"error_code={getattr(result, 'error_code', '?')}"
                )
            finish(OUTCOME_FAILED, detail)

        def goal_response_done(future: Any) -> None:
            try:
                goal_handle = future.result()
            except Exception as exc:  # noqa: BLE001 - ROS future boundary
                finish(OUTCOME_FAILED, f"trajectory goal response failed: {exc}")
                return
            if goal_handle is None or not goal_handle.accepted:
                finish(OUTCOME_FAILED, "joint trajectory controller rejected the goal")
                return
            handles.append(goal_handle)
            if cancel_generation != self._cancel_generation:
                # Cancelled while its acceptance was in flight.
                goal_handle.cancel_goal_async()
            else:
                on_active()
            try:
                result_future = goal_handle.get_result_async()

                def completed(result: Any) -> None:
                    handles.remove(goal_handle)
                    result_done(result)

                result_future.add_done_callback(completed)
            except Exception as exc:  # noqa: BLE001 - ROS future boundary
                handles.remove(goal_handle)
                finish(OUTCOME_FAILED, f"trajectory result request failed: {exc}")

        goal_future.add_done_callback(goal_response_done)
        duration = float(trajectory.time_from_start_s[-1])
        timing = (
            "waiting for final joint tolerance without a time limit"
            if self.goal_time_tolerance_s == 0.0
            else "controller goal timing applies"
        )
        log.info(
            f"arm trajectory: {len(message.points)} point(s) over "
            f"{duration:.2f}s planned; {timing}"
        )
        return True

    def cancel(self) -> None:
        """Cancel every accepted goal and any whose acceptance is in flight.

        Targets existing goals only: a delayed cancel-all could kill the goal
        that replaced them. Fire-and-forget; outcomes still arrive via
        ``on_done`` (usually ``"cancelled"``).
        """
        self._cancel_generation += 1
        for handle in tuple(self._goal_handles):
            handle.cancel_goal_async()


__all__ = [
    "DEFAULT_SERVER_WAIT_S",
    "OUTCOME_CANCELLED",
    "OUTCOME_FAILED",
    "OUTCOME_SUCCEEDED",
    "TimedJointTrajectory",
    "TrajectoryExecutor",
]
