"""Arbitrate manual, trajectory and gripper commands before the vendor send.

Manual arm targets (teleop), trajectory setpoints (a joint trajectory
controller or a trajectory bridge streaming a ``FollowJointTrajectory`` goal)
and gripper targets arrive on separate inputs. :class:`MotionArbiterCore`
decides which of them may reach the vendor, and when -- the single writer in
front of the hardware.

It is a pure state machine: no ROS, no clock of its own, no I/O. Publishing,
goal cancellation and the vendor wire format are injected callables, so it runs
against recording fakes in tests and the ROS wrapper
(:mod:`cyberwave.driver.ros2.arbiter_node`) stays pure transport. Planning, IK,
perception, MQTT and task logic all live elsewhere; none of them can stall,
crash or block anything here.

## The rules, and why each one is the way it is

* **Ownership is event-based, never inferred from silence.** A long-running
  trajectory is silent between points. Treating a gap in command traffic as a
  release means teleop can interleave into the middle of an executing goal.
* **Manual input preempts a goal.** A complete valid manual command becomes the
  sole vendor command immediately and suppresses trajectory setpoints.
  Cancellation is requested after a short grace period so an already-succeeded
  goal can publish its terminal status; cancelling at that instant has been
  observed to leave a goal CANCELING indefinitely on hardware. Repeated active
  action status must not reclaim ownership before that goal is terminal.
* **An unpreempted goal releases only on terminal status *plus* measured
  stability.** The action result arrives while the arm is still decelerating.
* **Faults latch.** A vendor fault clearing is the vendor saying a condition
  went away, not an operator deciding it is safe to move. Resuming on a cleared
  status makes a transient fault self-clearing.
* **Feedback health is observational here.** Ingress owns command recency and
  ordering; hardware safety owns watchdog decisions. Feedback timing cannot
  seize command ownership from an accepted teleop command.
* **Nothing suppressed is ever replayed.** A command dropped during a goal
  described a world several seconds old by the time ownership releases.
* **A hold is only emitted from fresh state.** Holding at a posture we are no
  longer sure of is not a safe stop; it is a small commanded move to a guess.
* **The gripper is a standing command.** Grip/release/teleop set it; an
  arm-only trajectory carries it unchanged. Completing it from *measured*
  state re-commands a still-closing gripper to its mid-travel opening at every
  waypoint, which reopens it the instant a lift begins.

Bounded work only: no sleeping, no blocking on futures, no network or disk I/O,
no unbounded queue. Cancellation is fire-and-forget through an injected callable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .command_boundary import (
    CommandRejection,
    CommandScopes,
    MotionSource,
    VendorAdapter,
    command_stamp_is_fresh,
    joint_state_adapter,
    validate_command,
)

#: How long measured joints must stay put before a finished goal releases the arm.
DEFAULT_STABILITY_WINDOW_S = 0.25
#: Per-joint movement below which the arm counts as stationary.
DEFAULT_STABILITY_EPSILON_RAD = 1e-3
#: Manual teleop is a live stream, not a permanent ownership claim. A short
#: lease keeps sparse UI updates coherent while allowing later planned motion.
DEFAULT_MANUAL_COMMAND_TIMEOUT_S = 0.5
#: Maximum age of a stamped command on any arbiter input.
DEFAULT_COMMAND_MAX_AGE_S = 0.25
#: Delay between a manual preemption and the goal-cancel request, so a goal
#: that already succeeded can publish its terminal status first.
PREEMPT_CANCEL_GRACE_S = 0.1
#: How old measured state may be and still back a fault hold. Vendor ROS
#: wrappers can pause feedback for a few hundred ms during an enable service
#: call (473 ms observed on hardware); 750 ms keeps the hold honest without
#: making normal enable traffic look like lost state. It is NOT a teleop
#: admission timeout.
DEFAULT_FEEDBACK_TIMEOUT_S = 0.75


class MotionOwner(Enum):
    IDLE = "idle"
    MANUAL = "manual"
    TRAJECTORY = "trajectory"
    FAULT = "fault"


class GoalState(Enum):
    """Terminal ``FollowJointTrajectory`` outcomes the arbiter cares about."""

    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    ABORTED = "aborted"


class ResetRefusal(Enum):
    NOT_FAULTED = "not_faulted"
    GOAL_ACTIVE = "goal_active"
    VENDOR_STILL_FAULTED = "vendor_still_faulted"


@dataclass(frozen=True, slots=True)
class ResetOutcome:
    accepted: bool
    refusal: ResetRefusal | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Measured:
    positions: Mapping[str, float]
    monotonic_s: float


class MotionArbiterCore:
    """The single-writer state machine. No ROS, no clock of its own, no I/O.

    Robot specifics are injected:

    * ``scopes`` -- which joints each source must cover;
    * ``adapt`` -- the vendor wire conversion (default: a plain joint-state
      command, see :func:`~cyberwave.driver.control.command_boundary.joint_state_adapter`);
    * ``velocity_contract_resolved`` -- False when the vendor's speed could not
      be derived honestly; it inhibits ALL actuation while observation goes on;
    * ``planned_motion_allowed`` -- False inhibits the trajectory source only
      (hardware gates); teleop is how an arm gets commissioned.
    """

    def __init__(
        self,
        *,
        limits: Mapping[str, tuple[float, float]],
        scopes: CommandScopes,
        publish: Callable[[Any], None],
        cancel_all_goals: Callable[[], None],
        adapt: VendorAdapter | None = None,
        velocity_contract_resolved: bool = True,
        planned_motion_allowed: bool = True,
        feedback_timeout_s: float = DEFAULT_FEEDBACK_TIMEOUT_S,
        stability_window_s: float = DEFAULT_STABILITY_WINDOW_S,
        stability_epsilon_rad: float = DEFAULT_STABILITY_EPSILON_RAD,
        manual_command_timeout_s: float = DEFAULT_MANUAL_COMMAND_TIMEOUT_S,
        preempt_cancel_grace_s: float = PREEMPT_CANCEL_GRACE_S,
    ) -> None:
        self._limits = dict(limits)
        self._scopes = scopes
        self._publish = publish
        self._cancel_all_goals = cancel_all_goals
        self._adapt: VendorAdapter = adapt or joint_state_adapter(scopes)
        self._velocity_contract_resolved = bool(velocity_contract_resolved)
        self._planned_motion_allowed = bool(planned_motion_allowed)
        # Used only when deciding whether a vendor-fault hold can faithfully
        # reuse measured state; never to admit or suppress a command.
        self._feedback_timeout_s = float(feedback_timeout_s)
        self._stability_window_s = float(stability_window_s)
        self._stability_epsilon_rad = float(stability_epsilon_rad)
        self._manual_command_timeout_s = float(manual_command_timeout_s)
        self._preempt_cancel_grace_s = float(preempt_cancel_grace_s)

        self._owner = MotionOwner.IDLE
        self._measured: _Measured | None = None
        # The standing gripper command (see the module doc) and its effort.
        # None until something actually publishes one.
        self._commanded_gripper: dict[str, float] | None = None
        self._commanded_gripper_effort: float | None = None
        self._last_manual_arm: dict[str, float] | None = None
        self._last_trajectory_arm: dict[str, float] | None = None
        self._goal_active = False
        self._active_goal_id: str | None = None
        # A manual command can take the sole writer from an active goal before
        # the action server confirms cancellation. This guards against repeated
        # EXECUTING/CANCELING status messages taking it back.
        self._manual_preempting_goal = False
        self._manual_preempted_goal_id: str | None = None
        self._pending_cancel_at: float | None = None
        self._last_manual_command_at: float | None = None
        self._goal_ended_at: float | None = None
        self._vendor_faulted = False
        self._teaching_mode = False
        self._hold_emitted = False
        self._last_reject: CommandRejection | None = None
        # Stability tracking for the release condition.
        self._stable_since: float | None = None
        self._stable_reference: Mapping[str, float] | None = None

    # ── Observation ─────────────────────────────────────────────────────────

    @property
    def owner(self) -> MotionOwner:
        return self._owner

    @property
    def scopes(self) -> CommandScopes:
        return self._scopes

    @property
    def last_rejection(self) -> CommandRejection | None:
        return self._last_reject

    @property
    def goal_active(self) -> bool:
        return self._goal_active

    def observe_measured_state(
        self, positions: Mapping[str, float], *, monotonic_now: float
    ) -> None:
        """Record normalized measured state and its LOCAL receive time.

        Monotonic, and taken here rather than trusted from a bridge, because
        monotonic clocks are not comparable across processes.
        """
        previous = self._measured
        self._measured = _Measured(dict(positions), float(monotonic_now))
        self._track_stability(previous, monotonic_now)

    def observe_state_health(self, *, healthy: bool, monotonic_now: float) -> None:
        """Observe feedback health without using it as command authority.

        Letting a vendor-feedback timing gap decide whether a fresh accepted
        teleop command may move couples two independent responsibilities, and
        has latched teleop into FAULT during normal enable traffic. Kept for the
        observer interface; deliberately a no-op.
        """
        del healthy, monotonic_now

    def observe_arm_fault(self, faulted: bool, *, monotonic_now: float) -> None:
        """Vendor fault status. Setting it latches; clearing it does not release."""
        self._vendor_faulted = bool(faulted)
        if faulted:
            self._enter_fault("vendor_fault", monotonic_now=monotonic_now)

    def observe_teaching_mode(self, active: bool) -> None:
        """Hand-guiding / teaching mode: every command is refused while active."""
        self._teaching_mode = bool(active)

    def begin_goal(self, *, monotonic_now: float, goal_id: str | None = None) -> None:
        """A ``FollowJointTrajectory`` goal was accepted (or is still active)."""
        if self._owner is MotionOwner.FAULT:
            return
        if not self._goal_active or goal_id != self._active_goal_id:
            self._last_trajectory_arm = None
        if self._manual_preempting_goal and (
            goal_id is None or goal_id == self._manual_preempted_goal_id
        ):
            # Cancellation was already requested for this goal. Its status can
            # remain EXECUTING/CANCELING until the action server publishes a
            # terminal result; it must not retake the vendor writer meanwhile.
            return
        self._goal_active = True
        self._active_goal_id = goal_id
        if (
            self._owner is MotionOwner.MANUAL
            and self._last_manual_command_at is not None
            and monotonic_now - self._last_manual_command_at
            < self._manual_command_timeout_s
        ):
            # A new action acceptance is not permission to steal a live manual
            # stream. Cancel it, and suppress its repeated status until terminal.
            self._manual_preempting_goal = True
            self._manual_preempted_goal_id = goal_id
            self._pending_cancel_at = monotonic_now + self._preempt_cancel_grace_s
            return
        self._manual_preempting_goal = False
        self._manual_preempted_goal_id = None
        self._pending_cancel_at = None
        self._goal_ended_at = None
        self._owner = MotionOwner.TRAJECTORY

    def end_goal(
        self,
        state: GoalState,
        *,
        monotonic_now: float,
        goal_id: str | None = None,
    ) -> None:
        """Terminal action status. Not a release on its own -- see the module doc."""
        # Action servers publish the status array repeatedly. A terminal state
        # is an event, not a lease: replaying it must not restart the measured
        # stability window forever. It still clears an active goal while the
        # arbiter is faulted, which is required before an explicit fault reset.
        del state
        if not self._goal_active or (
            goal_id is not None
            and self._active_goal_id is not None
            and goal_id != self._active_goal_id
        ):
            return
        self._goal_active = False
        self._active_goal_id = None
        self._manual_preempting_goal = False
        self._manual_preempted_goal_id = None
        self._pending_cancel_at = None
        self._goal_ended_at = float(monotonic_now)
        self._stable_since = None
        self._stable_reference = None

    # ── Commanding ──────────────────────────────────────────────────────────

    @property
    def velocity_contract_resolved(self) -> bool:
        """Whether an honest vendor speed exists. False inhibits ALL actuation.

        Telemetry, ownership status and fault diagnostics keep running: an arm
        you cannot command is still an arm you need to be able to observe.
        """
        return self._velocity_contract_resolved

    def submit(
        self,
        source: MotionSource,
        names: list[str],
        positions: list[float],
        *,
        monotonic_now: float,
        velocity_override: bool = False,
        gripper_effort: float | None = None,
    ) -> bool:
        """Validate, arbitrate, convert and publish one command. Bounded work.

        ``gripper_effort`` is the gripper effort/torque limit a manual or
        gripper command asks for; None or non-finite keeps the previous one.
        """
        self._last_reject = None
        if self._owner is MotionOwner.FAULT:
            return False
        if self._teaching_mode:
            self._last_reject = CommandRejection.TEACHING_MODE
            return False
        if not self._velocity_contract_resolved:
            # Before validation, so the reported reason is the real one: the
            # command is fine, there is just no honest speed to send it at.
            self._last_reject = CommandRejection.VELOCITY_CONTRACT_UNRESOLVED
            return False
        if velocity_override:
            # A stale publisher still filling the vendor speed slot. Refused
            # rather than stripped: silently accepting it would leave two
            # apparent speed authorities and hide which one the arm obeyed.
            self._last_reject = CommandRejection.UNSUPPORTED_VELOCITY_OVERRIDE
            return False

        validated = validate_command(
            source, names, positions, limits=self._limits, scopes=self._scopes
        )
        if validated.command is None:
            self._last_reject = validated.reason
            return False

        if source is MotionSource.GRIPPER:
            gripper = self._scopes.gripper_of(validated.command.positions)
            assert gripper is not None  # validation guarantees the full scope
            return self._submit_gripper(gripper, gripper_effort, monotonic_now)

        held_gripper = self._commanded_gripper
        if held_gripper is None and self._measured is not None:
            # Nothing has commanded the gripper since startup: the measured
            # opening is the only honest value for an arm-only command.
            held_gripper = self._scopes.gripper_of(self._measured.positions)
        commanded_gripper = self._scopes.gripper_of(validated.command.positions)
        effort = self._commanded_gripper_effort
        if (
            commanded_gripper is not None
            and gripper_effort is not None
            and math.isfinite(gripper_effort)
        ):
            effort = float(gripper_effort)
        adapted = self._adapt(
            validated.command, held_gripper=held_gripper, gripper_effort=effort
        )
        if adapted.command is None:
            self._last_reject = adapted.reason
            return False

        if source is MotionSource.MANUAL and self._owner is MotionOwner.TRAJECTORY:
            # Validate and adapt before cancellation: malformed manual input
            # must not interrupt a valid planned motion. Once valid, manual is
            # the immediate override at the single-writer boundary. Trajectory
            # input stays suppressed until the action reports terminal status.
            self._manual_preempting_goal = True
            self._manual_preempted_goal_id = self._active_goal_id
            self._owner = MotionOwner.MANUAL
            self._pending_cancel_at = monotonic_now + self._preempt_cancel_grace_s

        if not self._source_may_command(source):
            # Dropped, not queued. Queueing is how a stale command executes later.
            return False

        if source is MotionSource.MANUAL and self._owner is MotionOwner.IDLE:
            self._owner = MotionOwner.MANUAL
        arm = {
            joint: float(validated.command.positions[joint])
            for joint in self._scopes.arm
        }
        if source is MotionSource.MANUAL:
            self._last_manual_command_at = float(monotonic_now)
            self._last_manual_arm = arm
        elif source is MotionSource.TRAJECTORY:
            self._last_trajectory_arm = arm
        if commanded_gripper is not None:
            # Latched only once actually published: a refused or dropped grip
            # must not become the gripper every later trajectory asserts.
            self._commanded_gripper = commanded_gripper
            self._commanded_gripper_effort = effort
        self._publish(adapted.command)
        return True

    def _submit_gripper(
        self,
        gripper: dict[str, float],
        gripper_effort: float | None,
        monotonic_now: float,
    ) -> bool:
        """Change the standing gripper target without taking arm ownership."""
        effort = self._commanded_gripper_effort
        if gripper_effort is not None and math.isfinite(gripper_effort):
            effort = float(gripper_effort)
        if self._owner is MotionOwner.TRAJECTORY and self._last_trajectory_arm is None:
            # Goal accepted but no setpoint yet. The first trajectory write will
            # carry this gripper target; measured arm state is not its plan.
            self._commanded_gripper = dict(gripper)
            self._commanded_gripper_effort = effort
            return True
        arm: Mapping[str, float]
        if (
            self._owner is MotionOwner.TRAJECTORY
            and self._last_trajectory_arm is not None
        ):
            # Reuse the last accepted arm setpoint without taking ownership.
            # Waiting for another setpoint can lose a grip at the final point.
            arm = self._last_trajectory_arm
        elif self._owner is MotionOwner.MANUAL and self._last_manual_arm is not None:
            arm = self._last_manual_arm
        elif self._measured is not None and self._feedback_is_fresh(monotonic_now):
            arm = self._measured.positions
        else:
            self._last_reject = CommandRejection.NO_MEASURED_ARM
            return False
        values = {joint: arm[joint] for joint in self._scopes.arm if joint in arm}
        if set(self._scopes.gripper) <= set(self._scopes.manual):
            # Otherwise (arm-only manual scope) the gripper rides on
            # ``held_gripper``; it was already validated as a GRIPPER command.
            values.update(gripper)
        checked = validate_command(
            MotionSource.MANUAL,
            list(values),
            list(values.values()),
            limits=self._limits,
            scopes=self._scopes,
        )
        if checked.command is None:
            self._last_reject = checked.reason
            return False
        adapted = self._adapt(
            checked.command, held_gripper=gripper, gripper_effort=effort
        )
        if adapted.command is None:
            self._last_reject = adapted.reason
            return False
        self._commanded_gripper = dict(gripper)
        self._commanded_gripper_effort = effort
        self._publish(adapted.command)
        return True

    def tick(self, *, monotonic_now: float) -> None:
        """Advance ownership release; feedback health cannot seize ownership."""
        if self._owner is MotionOwner.FAULT:
            return
        if (
            self._manual_preempting_goal
            and self._goal_active
            and self._pending_cancel_at is not None
            and monotonic_now >= self._pending_cancel_at
        ):
            self._pending_cancel_at = None
            self._cancel_all_goals()
        self._maybe_release_after_goal(monotonic_now)
        self._maybe_release_after_manual_timeout(monotonic_now)

    def reset_fault(self, *, monotonic_now: float) -> ResetOutcome:
        """Explicit operator action. Every precondition is checked, none inferred."""
        del monotonic_now
        if self._owner is not MotionOwner.FAULT:
            return ResetOutcome(False, ResetRefusal.NOT_FAULTED)
        if self._goal_active:
            return ResetOutcome(False, ResetRefusal.GOAL_ACTIVE)
        if self._vendor_faulted:
            return ResetOutcome(False, ResetRefusal.VENDOR_STILL_FAULTED)
        self._owner = MotionOwner.IDLE
        self._hold_emitted = False
        self._active_goal_id = None
        self._manual_preempting_goal = False
        self._manual_preempted_goal_id = None
        self._pending_cancel_at = None
        self._last_manual_command_at = None
        self._goal_ended_at = None
        return ResetOutcome(True)

    # ── Internals ───────────────────────────────────────────────────────────

    def _feedback_is_fresh(self, monotonic_now: float) -> bool:
        if self._measured is None:
            return False
        return (monotonic_now - self._measured.monotonic_s) <= self._feedback_timeout_s

    def _source_may_command(self, source: MotionSource) -> bool:
        """Who may write to the vendor right now.

        The trajectory source is gated on OWNING the arm, not merely on a
        command having arrived. ``topic_based_ros2_control`` publishes on every
        write() cycle once a controller is active -- including the controller's
        hold position while no goal is running -- so treating an inbound
        setpoint as intent would let the controller drive the arm outside any
        goal's tolerances, preemption and cancellation. Ownership is granted by
        goal acceptance (:meth:`begin_goal`) and by nothing else.

        Manual is deliberately asymmetric: teleop is a stateless stream with no
        goal to accept. A valid manual command may preempt a trajectory in
        :meth:`submit`; otherwise it commands from IDLE or MANUAL.
        """
        if source is MotionSource.TRAJECTORY:
            # The hardware gates apply HERE, not only on a driver's Cartesian
            # path. Anything can publish to the setpoint input; a gate that
            # only guards one producer is not a gate.
            if not self._planned_motion_allowed:
                return False
            return self._owner is MotionOwner.TRAJECTORY
        return self._owner in (MotionOwner.IDLE, MotionOwner.MANUAL)

    def _enter_fault(self, cause: str, *, monotonic_now: float) -> None:
        del cause
        if self._owner is MotionOwner.FAULT:
            return
        self._owner = MotionOwner.FAULT
        self._pending_cancel_at = None
        # Fire-and-forget: waiting on a cancel response would block the control
        # path on the very component that may have just stopped responding.
        #
        # The goal is NOT marked inactive: requesting cancellation is not the
        # goal having stopped, and treating it as such would let an operator
        # reset the fault latch while the controller was still driving the arm.
        self._cancel_all_goals()
        self._emit_hold_if_state_is_fresh(monotonic_now)

    def _emit_hold_if_state_is_fresh(self, monotonic_now: float) -> None:
        """At most one hold, and only at a posture we currently believe."""
        if self._hold_emitted or not self._feedback_is_fresh(monotonic_now):
            return
        if not self._velocity_contract_resolved:
            # No honest vendor command exists, and a hold IS a command.
            return
        assert self._measured is not None
        names = list(self._scopes.manual)
        if any(joint not in self._measured.positions for joint in names):
            return
        validated = validate_command(
            MotionSource.MANUAL,
            names,
            [float(self._measured.positions[j]) for j in names],
            limits=self._limits,
            scopes=self._scopes,
        )
        if validated.command is None:
            return
        adapted = self._adapt(
            validated.command,
            held_gripper=self._scopes.gripper_of(self._measured.positions),
            gripper_effort=self._commanded_gripper_effort,
        )
        if adapted.command is None:
            return
        self._hold_emitted = True
        self._publish(adapted.command)

    def _track_stability(
        self, previous: _Measured | None, monotonic_now: float
    ) -> None:
        measured = self._measured
        assert measured is not None
        if self._stable_reference is None or previous is None:
            self._stable_reference = dict(measured.positions)
            self._stable_since = monotonic_now
            return
        reference = self._stable_reference
        moved = any(
            abs(value - reference.get(joint, value)) > self._stability_epsilon_rad
            for joint, value in measured.positions.items()
        )
        if moved:
            self._stable_reference = dict(measured.positions)
            self._stable_since = monotonic_now

    def _maybe_release_after_goal(self, monotonic_now: float) -> None:
        if self._owner is not MotionOwner.TRAJECTORY or self._goal_active:
            return
        if self._goal_ended_at is None or self._stable_since is None:
            return
        settled_for = monotonic_now - max(self._stable_since, self._goal_ended_at)
        if settled_for >= self._stability_window_s:
            self._owner = MotionOwner.IDLE

    def _maybe_release_after_manual_timeout(self, monotonic_now: float) -> None:
        """Release a stopped teleop stream without guessing a trajectory state."""
        if (
            self._owner is not MotionOwner.MANUAL
            or self._manual_preempting_goal
            or self._last_manual_command_at is None
        ):
            return
        if (
            monotonic_now - self._last_manual_command_at
            >= self._manual_command_timeout_s
        ):
            self._owner = MotionOwner.IDLE


__all__ = [
    "DEFAULT_COMMAND_MAX_AGE_S",
    "DEFAULT_FEEDBACK_TIMEOUT_S",
    "DEFAULT_MANUAL_COMMAND_TIMEOUT_S",
    "DEFAULT_STABILITY_EPSILON_RAD",
    "DEFAULT_STABILITY_WINDOW_S",
    "PREEMPT_CANCEL_GRACE_S",
    "GoalState",
    "MotionArbiterCore",
    "MotionOwner",
    "ResetOutcome",
    "ResetRefusal",
    "command_stamp_is_fresh",
]
