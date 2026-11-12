"""ROS 2 transport for :class:`~cyberwave.driver.control.arbiter.MotionArbiterCore`.

Transport only. Every decision lives in the pure core; this node moves
messages and calls injected callables. It is built by a factory
(:func:`build_motion_arbiter_node_class`) that imports rclpy and the message
packages lazily, so the core -- and every driver that merely imports this
module -- stays importable and testable without a ROS installation.

The node is meant to be embedded in the process that owns the hardware
transport (for example the vendor's CAN node), so an accepted command reaches
the vendor through an in-process ``vendor_send`` callable instead of another
topic hop that could be published to by anyone.

## Inputs

* ``command_topic`` -- manual (teleop) commands covering ``scopes.manual``.
* ``trajectory_command_topic`` -- trajectory setpoints covering ``scopes.arm``
  (``topic_based_ros2_control`` output, or a trajectory bridge).
* ``gripper_command_topic`` -- gripper-only commands covering ``scopes.gripper``.
* ``state_topic`` -- measured joint state.
* ``arm_status_topic`` -- optional vendor status, decoded by the injected
  ``is_fault``/``is_teaching`` predicates.
* ``{trajectory_action}/_action/status`` -- the action whose goals grant
  trajectory ownership. Configurable because the server that streams the
  setpoints is not always ``arm_controller``: a trajectory bridge can serve
  the same action name while publishing on ``trajectory_command_topic``.

A ``std_srvs/Trigger`` service (``reset_service``) clears a latched fault.

## Robot specifics

Static ones come from :class:`ArbiterNodeSpec`. The vendor wire format is a
subclass hook: override :meth:`_vendor_adapter` to return the adapter and
whether the velocity contract resolved (declare any vendor parameters there),
and :meth:`_vendor_message` if the outbound message is not a plain
``JointState`` built from
:class:`~cyberwave.driver.control.command_boundary.VendorJointCommand`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..control.arbiter import (
    DEFAULT_COMMAND_MAX_AGE_S,
    DEFAULT_FEEDBACK_TIMEOUT_S,
    DEFAULT_MANUAL_COMMAND_TIMEOUT_S,
    PREEMPT_CANCEL_GRACE_S,
    GoalState,
    MotionArbiterCore,
    MotionOwner,
)
from ..control.command_boundary import (
    CommandScopes,
    MotionSource,
    VendorAdapter,
    command_stamp_is_fresh,
)

#: Manual arm command input, relative to the node namespace.
COMMAND_TOPIC = "joint_commands"
#: Trajectory setpoints must never overwrite pending manual input.
TRAJECTORY_COMMAND_TOPIC = "trajectory_joint_commands"
#: Grip/release must not be overwritten by the arm setpoint stream.
GRIPPER_COMMAND_TOPIC = "gripper_commands"
# One-shot gripper commands can wait behind CAN callbacks for >0.25 s on Jetson.
# Keep the bound finite; arm trajectory and manual inputs retain their 0.25 s gate.
DEFAULT_GRIPPER_COMMAND_MAX_AGE_S = 0.75
#: The action whose goal status grants trajectory ownership.
TRAJECTORY_ACTION = "arm_controller/follow_joint_trajectory"
#: Explicit fault reset. A service, not a topic: a reset is a request with an
#: answer, and it must not be something a retained message can re-trigger.
RESET_FAULT_SERVICE = "reset_motion_fault"


@dataclass(frozen=True)
class ArbiterNodeSpec:
    """Static, per-robot configuration of the arbiter node.

    Topic and action names are *defaults* for same-named ROS parameters, so a
    launch can still remap them. ``node_name`` is deliberately not derived
    from a module path: it is a live graph identifier (``ros2 node list``,
    ``ros2 param``) and must not change when code moves.

    ``arm_status_type`` returns the vendor status message type (called lazily,
    inside the node, so the message package is only imported where ROS is).
    ``is_fault`` / ``is_teaching`` decode one status message; share them with
    the driver so the two can never disagree about whether the arm is faulted.
    """

    node_name: str
    scopes: CommandScopes
    command_topic: str = COMMAND_TOPIC
    trajectory_command_topic: str = TRAJECTORY_COMMAND_TOPIC
    gripper_command_topic: str = GRIPPER_COMMAND_TOPIC
    state_topic: str = "joint_states"
    trajectory_action: str = TRAJECTORY_ACTION
    reset_service: str = RESET_FAULT_SERVICE
    command_frame_id: str = ""
    arm_status_topic: str = "arm_status"
    arm_status_type: Callable[[], Any] | None = None
    is_fault: Callable[[Any], bool] | None = None
    is_teaching: Callable[[Any], bool] | None = None


def gripper_effort_from_message(message: Any, scopes: CommandScopes) -> float | None:
    """Gripper effort carried by an input ``JointState``, if it carries one.

    The effort slot of the first gripper joint; ``None`` when there is no
    gripper joint in the message or the effort array is not parallel to names.
    """
    if not scopes.gripper:
        return None
    names, efforts = list(message.name), list(message.effort)
    joint = scopes.gripper[0]
    if joint not in names or len(efforts) != len(names):
        return None
    return float(efforts[names.index(joint)])


def command_arrays_valid(message: Any, source: MotionSource) -> bool:
    """Accept only the manual/trajectory/gripper wire shapes before arbitration.

    Positions must be parallel to names; a velocity array is never accepted
    (the vendor speed has one writer); efforts are allowed only on manual and
    gripper input, parallel and finite.
    """
    names = list(message.name)
    if len(message.position) != len(names) or message.velocity:
        return False
    efforts = list(message.effort)
    return not efforts or (
        source in (MotionSource.MANUAL, MotionSource.GRIPPER)
        and len(efforts) == len(names)
        and all(math.isfinite(value) for value in efforts)
    )


def build_motion_arbiter_node_class(spec: ArbiterNodeSpec) -> type:
    """Build the arbiter ``rclpy`` node class for *spec* (imports ROS lazily)."""
    from action_msgs.msg import GoalStatus, GoalStatusArray
    from action_msgs.srv import CancelGoal
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_action_status_default,
    )
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    scopes = spec.scopes
    # State/status are sensor-like; each command source gets its own queue.
    latest = QoSProfile(
        depth=1,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    command_in = QoSProfile(
        # A live target stream. Queued setpoints become obsolete before the
        # hardware owner can send them and make motion stop and restart. Keep
        # the latest pending target while retaining reliable delivery.
        depth=1,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    active_statuses = {
        GoalStatus.STATUS_ACCEPTED,
        GoalStatus.STATUS_EXECUTING,
        GoalStatus.STATUS_CANCELING,
    }
    terminal_statuses = {
        GoalStatus.STATUS_SUCCEEDED: GoalState.SUCCEEDED,
        GoalStatus.STATUS_CANCELED: GoalState.CANCELED,
        GoalStatus.STATUS_ABORTED: GoalState.ABORTED,
    }

    class MotionArbiterNode(Node):
        """Arbitrates commands before the hardware owner's in-process send."""

        arbiter_spec = spec

        def __init__(
            self,
            *,
            vendor_send: Callable[[Any], None],
            parameter_overrides: list[Any] | None = None,
            namespace: str | None = None,
        ) -> None:
            super().__init__(
                spec.node_name,
                # A host process's process-wide __node remap (vendor launch)
                # must not rename this embedded node or silence its diagnostics.
                cli_args=["--ros-args", "-r", f"__node:={spec.node_name}"],
                **({"namespace": namespace} if namespace is not None else {}),
                parameter_overrides=parameter_overrides or [],
            )
            self.declare_parameter("command_topic", spec.command_topic)
            self.declare_parameter(
                "trajectory_command_topic", spec.trajectory_command_topic
            )
            self.declare_parameter("gripper_command_topic", spec.gripper_command_topic)
            self.declare_parameter("state_topic", spec.state_topic)
            self.declare_parameter("arm_status_topic", spec.arm_status_topic)
            self.declare_parameter("trajectory_action", spec.trajectory_action)
            self.declare_parameter("command_frame_id", spec.command_frame_id)
            self.declare_parameter("feedback_timeout_s", DEFAULT_FEEDBACK_TIMEOUT_S)
            self.declare_parameter("command_max_age_s", DEFAULT_COMMAND_MAX_AGE_S)
            self.declare_parameter(
                "gripper_command_max_age_s", DEFAULT_GRIPPER_COMMAND_MAX_AGE_S
            )
            self.declare_parameter(
                "manual_command_timeout_s", DEFAULT_MANUAL_COMMAND_TIMEOUT_S
            )
            # Declared by TYPE with no default. rclpy cannot infer an array
            # type from [], and there is no safe default to infer: the launch
            # must supply real limits (in scopes.manual order) or the arbiter
            # refuses everything rather than validating against a made-up range.
            self.declare_parameter("joint_limits_lower", Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter("joint_limits_upper", Parameter.Type.DOUBLE_ARRAY)
            # Whether the hardware gates are satisfied. Closed by default at the
            # ROS boundary; False inhibits the TRAJECTORY source only.
            self.declare_parameter("planned_motion_allowed", False)

            timeout = float(self.get_parameter("feedback_timeout_s").value)
            manual_timeout = float(self.get_parameter("manual_command_timeout_s").value)
            action = str(self.get_parameter("trajectory_action").value).strip("/")
            self._vendor_send = vendor_send
            group = ReentrantCallbackGroup()
            self._cancel_client = self.create_client(
                CancelGoal, f"{action}/_action/cancel_goal", callback_group=group
            )

            limits = self._limits()
            if len(limits) < len(scopes.manual):
                self.get_logger().error(
                    "motion arbiter started without complete joint limits "
                    f"(got {sorted(limits)}); every command will be refused. "
                    "The launch must pass joint_limits_lower/upper in "
                    f"{list(scopes.manual)} order."
                )
            planned_allowed = bool(self.get_parameter("planned_motion_allowed").value)
            if not planned_allowed:
                self.get_logger().warning(
                    "planned motion INHIBITED: the hardware gates are not "
                    "satisfied. Trajectory commands will be refused; manual "
                    "teleop is unaffected."
                )
            adapter, contract_resolved = self._vendor_adapter()
            if not contract_resolved:
                self.get_logger().error(
                    "ALL actuation INHIBITED: no honest vendor speed was resolved, "
                    "so there is no speed to command at. Manual and planned "
                    "commands are both refused (velocity_contract_unresolved); "
                    "measured state, ownership and fault diagnostics continue."
                )
            self._last_outcome: dict[MotionSource, str | None] = {}
            self._last_drop_log: dict[MotionSource, float] = {}
            self._core = MotionArbiterCore(
                limits=limits,
                scopes=scopes,
                planned_motion_allowed=planned_allowed,
                publish=self._publish_vendor_command,
                cancel_all_goals=self._request_cancel_all,
                adapt=adapter,
                velocity_contract_resolved=contract_resolved,
                feedback_timeout_s=timeout,
                manual_command_timeout_s=manual_timeout,
            )

            self.create_subscription(
                JointState, self._topic("command_topic"), self._on_command, command_in
            )
            self.create_subscription(
                JointState,
                self._topic("trajectory_command_topic"),
                self._on_trajectory_command,
                command_in,
            )
            if scopes.gripper:
                self.create_subscription(
                    JointState,
                    self._topic("gripper_command_topic"),
                    self._on_gripper_command,
                    command_in,
                )
            self.create_subscription(
                JointState, self._topic("state_topic"), self._on_state, latest
            )
            if spec.arm_status_type is not None and spec.is_fault is not None:
                self.create_subscription(
                    spec.arm_status_type(),
                    self._topic("arm_status_topic"),
                    self._on_arm_status,
                    latest,
                )
            self.create_subscription(
                GoalStatusArray,
                f"{action}/_action/status",
                self._on_goal_status,
                # Action status is not a best-effort sensor stream. Action
                # servers publish it RELIABLE + TRANSIENT_LOCAL so a late
                # subscriber still observes an already accepted goal. A VOLATILE
                # profile here is a race: the controller starts streaming
                # setpoints while the arbiter remains IDLE/MANUAL and suppresses
                # every one. Use the canonical rclpy contract.
                qos_profile_action_status_default,
            )
            self.create_service(
                Trigger, spec.reset_service, self._on_reset, callback_group=group
            )
            # The preemption grace, fault hold and manual lease are timer-driven.
            # The short tick lets an already-finished goal publish terminal
            # status before cancellation is requested.
            self.create_timer(
                min(timeout, manual_timeout, PREEMPT_CANCEL_GRACE_S), self._on_tick
            )

        # ── vendor hooks (override per robot) ───────────────────────────────

        def _vendor_adapter(self) -> tuple[VendorAdapter | None, bool]:
            """``(adapter, velocity_contract_resolved)``; None = plain joint state.

            Called once from ``__init__`` after the standard parameters are
            declared, so an override may declare and read its own (for example
            a vendor speed percentage) here.
            """
            return None, True

        def _vendor_message(self, command: Any) -> Any:
            """Build the outbound message for one adapted command.

            The default accepts a ``VendorJointCommand`` (or anything with
            the same ``names``/``positions``/``velocities``/``efforts`` fields)
            and returns a ``JointState`` stamped now with ``command_frame_id``.
            """
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(self.get_parameter("command_frame_id").value)
            message.name = list(command.names)
            message.position = list(command.positions)
            message.velocity = list(command.velocities)
            message.effort = list(command.efforts)
            return message

        # ── observation ─────────────────────────────────────────────────────

        @property
        def core(self) -> MotionArbiterCore:
            return self._core

        # ── helpers ─────────────────────────────────────────────────────────

        def _topic(self, parameter: str) -> str:
            """Relative by construction; absolute names escape the namespace."""
            return str(self.get_parameter(parameter).value).lstrip("/")

        def _limits(self) -> dict[str, tuple[float, float]]:
            """Validation limits from the launch, or nothing at all.

            An unset parameter yields {}, which makes every command fail
            validation with MISSING_LIMIT: a misconfigured arbiter must refuse
            motion, not invent a range.
            """
            try:
                lower = list(self.get_parameter("joint_limits_lower").value or [])
                upper = list(self.get_parameter("joint_limits_upper").value or [])
            except Exception:  # noqa: BLE001 - ParameterUninitializedException on Humble
                return {}
            joints = scopes.manual
            return {
                joints[i]: (float(lower[i]), float(upper[i]))
                for i in range(min(len(lower), len(upper), len(joints)))
            }

        @staticmethod
        def _now() -> float:
            return time.monotonic()

        # ── callbacks: bounded work only ────────────────────────────────────

        def _on_command(self, message: Any) -> None:
            self._submit_command(message, MotionSource.MANUAL)

        def _on_trajectory_command(self, message: Any) -> None:
            self._submit_command(message, MotionSource.TRAJECTORY)

        def _on_gripper_command(self, message: Any) -> None:
            self._submit_command(message, MotionSource.GRIPPER)

        def _command_max_age_s(self, source: MotionSource) -> float:
            name = (
                "gripper_command_max_age_s"
                if source is MotionSource.GRIPPER else "command_max_age_s"
            )
            return float(self.get_parameter(name).value)

        def _submit_command(self, message: Any, source: MotionSource) -> None:
            if not scopes.matches(source, list(message.name)):
                self._report_command_outcome(source, message, "invalid_joint_scope")
                return
            if not command_arrays_valid(message, source):
                self._report_command_outcome(source, message, "invalid_command_arrays")
                return
            stamp = message.header.stamp
            if not command_stamp_is_fresh(
                int(stamp.sec),
                int(stamp.nanosec),
                now_ns=self.get_clock().now().nanoseconds,
                max_age_s=self._command_max_age_s(source),
            ):
                self._report_command_outcome(
                    source, message, "stale_or_unstamped_command"
                )
                return
            before = self._core.owner
            accepted = self._core.submit(
                source,
                list(message.name),
                list(message.position),
                monotonic_now=self._now(),
                velocity_override=False,
                gripper_effort=gripper_effort_from_message(message, scopes),
            )
            reason: Any = (
                None if accepted else (self._core.last_rejection or "suppressed")
            )
            self._report_command_outcome(
                source, message, getattr(reason, "value", reason)
            )
            self._log_owner_if_changed(before)

        def _report_command_outcome(
            self, source: MotionSource, message: Any, reason: str | None
        ) -> None:
            """Bound diagnostics per source, including repeated stale drops."""
            previous = self._last_outcome.get(source, "unseen")
            now = self._now()
            if reason is None:
                if previous is not None:
                    self._last_outcome[source] = None
                    self.get_logger().info(f"{source.value} commands accepted")
                return
            if now - self._last_drop_log.get(source, -math.inf) < 1.0:
                return
            self._last_drop_log[source] = now
            self._last_outcome[source] = reason
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 10**9 + stamp.nanosec
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            targets = list(zip(message.name, message.position))[: len(scopes.manual)]
            self.get_logger().warning(
                f"{source.value} command DROPPED ({reason}); "
                f"owner={self._core.owner.value}; age_s={age_s:.3f}; "
                f"max_age_s={self._command_max_age_s(source)}; "
                f"stamp_s={stamp.sec}.{stamp.nanosec:09d}; targets={targets}"
            )

        def _on_state(self, message: Any) -> None:
            self._core.observe_measured_state(
                dict(zip(message.name, (float(p) for p in message.position))),
                monotonic_now=self._now(),
            )

        def _on_arm_status(self, message: Any) -> None:
            """Decode vendor status with the SAME predicates the driver uses."""
            before = self._core.owner
            if spec.is_teaching is not None:
                self._core.observe_teaching_mode(bool(spec.is_teaching(message)))
            if spec.is_fault is not None:
                self._core.observe_arm_fault(
                    bool(spec.is_fault(message)), monotonic_now=self._now()
                )
            self._log_owner_if_changed(before)

        def _on_goal_status(self, message: Any) -> None:
            """Ownership is driven by these events, never by command silence."""
            before = self._core.owner
            if not message.status_list:
                return
            # A historical CANCELING goal must not hide a newer terminal goal.
            # The controller accepts one active arm goal at a time, so the
            # newest goal's status is the ownership event that matters.
            entry = max(message.status_list, key=self._goal_stamp)
            now = self._now()
            if entry.status in active_statuses:
                self._core.begin_goal(monotonic_now=now, goal_id=self._goal_id(entry))
            elif entry.status in terminal_statuses:
                self._core.end_goal(
                    terminal_statuses[entry.status],
                    monotonic_now=now,
                    goal_id=self._goal_id(entry),
                )
            self._log_owner_if_changed(before)

        @staticmethod
        def _goal_id(entry: Any) -> str:
            return bytes(entry.goal_info.goal_id.uuid).hex()

        @staticmethod
        def _goal_stamp(entry: Any) -> tuple[int, int]:
            stamp = entry.goal_info.stamp
            return (int(stamp.sec), int(stamp.nanosec))

        def _on_tick(self) -> None:
            before = self._core.owner
            self._core.tick(monotonic_now=self._now())
            self._log_owner_if_changed(before)

        def _on_reset(self, _request: Any, response: Any) -> Any:
            outcome = self._core.reset_fault(monotonic_now=self._now())
            response.success = outcome.accepted
            response.message = (
                "motion fault cleared"
                if outcome.accepted
                else f"refused: {outcome.refusal.value if outcome.refusal else 'unknown'}"
            )
            self.get_logger().warning(f"{spec.reset_service} -> {response.message}")
            return response

        # ── outputs ─────────────────────────────────────────────────────────

        def _publish_vendor_command(self, command: Any) -> None:
            self._vendor_send(self._vendor_message(command))

        def _request_cancel_all(self) -> None:
            """Asynchronous by construction; a control callback never waits."""
            if not self._cancel_client.service_is_ready():
                self.get_logger().warning(
                    "cancel service unavailable; goals may run on"
                )
                return
            # Zero goal_id + zero stamp means "cancel everything".
            self._cancel_client.call_async(CancelGoal.Request())

        def _log_owner_if_changed(self, before: MotionOwner) -> None:
            if self._core.owner is before:
                return
            self.get_logger().warning(
                f"motion owner {before.value} -> {self._core.owner.value}"
            )

    return MotionArbiterNode


__all__ = [
    "COMMAND_TOPIC",
    "GRIPPER_COMMAND_TOPIC",
    "RESET_FAULT_SERVICE",
    "TRAJECTORY_ACTION",
    "TRAJECTORY_COMMAND_TOPIC",
    "ArbiterNodeSpec",
    "build_motion_arbiter_node_class",
    "command_arrays_valid",
    "gripper_effort_from_message",
]
