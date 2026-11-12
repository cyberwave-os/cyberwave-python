"""MotionArbiterCore: the single-writer state machine before the vendor send.

Everything is driven through injected fakes: a clock (explicit ``monotonic_now``),
a recording publisher and a goal canceler. No ROS, no bus, no sleeping. The
robot is a 6-joint arm with a one-joint gripper and the default joint-state
adapter, so ``published[-1].positions`` is ``[joint1..joint6, joint7]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberwave.driver.control.arbiter import (
    GoalState,
    MotionArbiterCore,
    MotionOwner,
    ResetRefusal,
)
from cyberwave.driver.control.command_boundary import (
    CommandRejection,
    CommandScopes,
    MotionSource,
)

ARM_SCOPE = tuple(f"joint{i}" for i in range(1, 7))
SCOPES = CommandScopes(arm=ARM_SCOPE, gripper=("joint7",))
MANUAL_SCOPE = SCOPES.manual

_LIMITS = {f"joint{i}": (-2.0, 2.0) for i in range(1, 7)}
_LIMITS["joint7"] = (0.0, 0.05)

_ARM = (list(ARM_SCOPE), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
_MANUAL = (list(MANUAL_SCOPE), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02])
_MEASURED = {f"joint{i}": 0.0 for i in range(1, 8)}


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list = []

    def __call__(self, command) -> None:
        self.published.append(command)


class RecordingCanceler:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def __call__(self) -> None:
        self.cancel_calls += 1


@pytest.fixture
def arbiter():
    publisher, canceler = RecordingPublisher(), RecordingCanceler()
    core = MotionArbiterCore(
        limits=_LIMITS,
        scopes=SCOPES,
        publish=publisher,
        cancel_all_goals=canceler,
        feedback_timeout_s=0.1,
    )
    core.publisher, core.canceler = publisher, canceler  # test handles
    return core


def _feed(core, *, now=10.0, positions=None):
    core.observe_measured_state(positions or _MEASURED, monotonic_now=now)


# ── Startup: nothing moves until feedback exists ─────────────────────────────


def test_it_starts_idle_and_owns_nothing() -> None:
    core = MotionArbiterCore(
        limits=_LIMITS,
        scopes=SCOPES,
        publish=lambda c: None,
        cancel_all_goals=lambda: None,
    )
    assert core.owner is MotionOwner.IDLE


def test_a_complete_manual_command_needs_no_feedback_admission_gate(arbiter) -> None:
    """Teleop recency is owned by Cyberwave ingress, not measured-state timing."""
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0) is True
    assert arbiter.publisher.published


def test_teaching_mode_refuses_commands_and_release_restores_manual_control(
    arbiter,
) -> None:
    arbiter.observe_teaching_mode(True)
    assert not arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0)
    assert arbiter.last_rejection is CommandRejection.TEACHING_MODE
    assert not arbiter.publisher.published
    arbiter.observe_teaching_mode(False)
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.1)


def test_default_timeout_tolerates_vendor_enable_jitter_before_teleop() -> None:
    """Normal 473 ms enable gaps must not pre-latch the manual path's fault."""
    publisher = RecordingPublisher()
    core = MotionArbiterCore(
        limits=_LIMITS,
        scopes=SCOPES,
        publish=publisher,
        cancel_all_goals=lambda: None,
    )
    _feed(core, now=10.0)
    core.tick(monotonic_now=10.473)
    assert core.owner is MotionOwner.IDLE
    assert core.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.473) is True
    assert publisher.published


# ── The vendor speed slot survives the whole path ────────────────────────────


def test_a_manual_message_carrying_its_own_velocity_array_is_refused(arbiter) -> None:
    """A stale publisher, not an override. Refused rather than silently stripped."""
    _feed(arbiter)

    accepted = arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, velocity_override=True
    )

    assert accepted is False
    assert arbiter.last_rejection is CommandRejection.UNSUPPORTED_VELOCITY_OVERRIDE
    assert arbiter.publisher.published == []


def test_a_trajectory_command_completes_its_gripper_from_measured_state_before_any_grip(
    arbiter,
) -> None:
    """Until something commands the gripper, measured is the only honest value."""
    _feed(arbiter, positions={**_MEASURED, "joint7": 0.031})
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.01)
    assert arbiter.publisher.published[-1].positions[6] == 0.031


def test_a_trajectory_command_keeps_the_last_commanded_gripper_not_the_measured_opening(
    arbiter,
) -> None:
    """A grasp is a standing command. A lift must not re-command the gripper to
    wherever the still-closing fingers happened to be when the trajectory started."""
    _feed(arbiter, positions={**_MEASURED, "joint7": 0.035})
    # grip: manual 7-joint command closing the gripper (joint7 -> 0.0)
    assert (
        arbiter.submit(
            MotionSource.MANUAL,
            list(_MEASURED),
            [*_ARM[1][:6], 0.0],
            monotonic_now=10.0,
        )
        is True
    )
    # Feedback reports the fingers still closing when the lift is dispatched.
    _feed(arbiter, now=10.5, positions={**_MEASURED, "joint7": 0.021})
    arbiter.tick(monotonic_now=10.6)
    arbiter.begin_goal(monotonic_now=12.0)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=12.01) is True

    assert arbiter.publisher.published[-1].positions[6] == 0.0


def test_a_refused_gripper_command_does_not_become_the_held_target(arbiter) -> None:
    _feed(arbiter, positions={**_MEASURED, "joint7": 0.031})
    arbiter.begin_goal(monotonic_now=10.0)
    # Manual while trajectory owns: allowed (preempts). Use an out-of-limit gripper
    # value instead so validation refuses it before anything is latched.
    assert (
        arbiter.submit(
            MotionSource.MANUAL,
            list(_MEASURED),
            [*_ARM[1][:6], 9.0],
            monotonic_now=10.01,
        )
        is False
    )
    arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.02)
    assert arbiter.publisher.published[-1].positions[6] == 0.031


def test_a_manual_gripper_effort_reaches_the_vendor_command(arbiter) -> None:
    """The effort the driver resolved is the gripper's torque limit on the wire;
    dropping it here sent every grip at the 3.0 N.m maximum."""
    arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, gripper_effort=0.8
    )
    assert arbiter.publisher.published[-1].efforts[6] == 0.8


def test_a_lift_keeps_the_grip_effort_not_the_maximum(arbiter) -> None:
    """Effort is part of the standing grip command, like its position: an
    arm-only trajectory must not squeeze harder than the grip asked for."""
    _feed(arbiter)
    arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, gripper_effort=0.8
    )
    arbiter.tick(monotonic_now=11.0)
    arbiter.begin_goal(monotonic_now=12.0)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=12.01) is True
    assert arbiter.publisher.published[-1].efforts[6] == 0.8


@pytest.mark.parametrize("effort", [None, float("nan"), float("inf")])
def test_a_missing_or_non_finite_effort_keeps_the_previous_one(arbiter, effort) -> None:
    arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, gripper_effort=0.8
    )
    arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.1, gripper_effort=effort
    )
    assert arbiter.publisher.published[-1].efforts[6] == 0.8


def test_a_refused_command_does_not_latch_its_effort(arbiter) -> None:
    arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, gripper_effort=0.8
    )
    names, positions = _MANUAL
    assert (
        arbiter.submit(
            MotionSource.MANUAL,
            names,
            [*positions[:6], 9.0],
            monotonic_now=10.1,
            gripper_effort=2.5,
        )
        is False
    )
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.2)
    assert arbiter.publisher.published[-1].efforts[6] == 0.8


def test_a_trajectory_command_uses_the_last_observed_gripper_without_freshness_gating(
    arbiter,
) -> None:
    """The arbiter does not own the feedback watchdog or command time policy."""
    _feed(arbiter, now=10.0)
    arbiter.begin_goal(monotonic_now=10.0)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.5) is True
    assert arbiter.publisher.published


# ── Event-based ownership ────────────────────────────────────────────────────


def test_an_accepted_goal_takes_ownership(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_a_valid_manual_command_preempts_trajectory_and_cancels_the_goal(
    arbiter,
) -> None:
    """Manual writes immediately; a still-active goal is cancelled after grace."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01) is True
    assert arbiter.owner is MotionOwner.MANUAL
    assert arbiter.publisher.published[-1].positions == _MANUAL[1]
    arbiter.tick(monotonic_now=10.09)
    assert arbiter.canceler.cancel_calls == 0
    arbiter.tick(monotonic_now=10.12)
    assert arbiter.canceler.cancel_calls == 1


def test_gripper_update_during_trajectory_sends_immediately_without_canceling_goal(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.01)
    before = len(arbiter.publisher.published)
    assert arbiter.submit(
        MotionSource.GRIPPER,
        ["joint7"],
        [0.04],
        monotonic_now=10.02,
        gripper_effort=0.8,
    )
    assert arbiter.owner is MotionOwner.TRAJECTORY
    assert len(arbiter.publisher.published) == before + 1
    assert arbiter.publisher.published[-1].positions[:6] == _ARM[1]
    assert arbiter.publisher.published[-1].positions[-1] == pytest.approx(0.04)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.03)
    assert arbiter.publisher.published[-1].positions[-1] == pytest.approx(0.04)
    assert arbiter.publisher.published[-1].efforts[-1] == pytest.approx(0.8)
    arbiter.tick(monotonic_now=10.2)
    assert arbiter.canceler.cancel_calls == 0


def test_gripper_queued_before_first_goal_setpoint_is_carried_by_it(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="first")
    before = len(arbiter.publisher.published)
    assert arbiter.submit(MotionSource.GRIPPER, ["joint7"], [0.0], monotonic_now=10.01)
    assert len(arbiter.publisher.published) == before
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.02)
    assert arbiter.publisher.published[-1].positions[-1] == pytest.approx(0.0)


def test_gripper_update_without_goal_holds_measured_arm(arbiter) -> None:
    _feed(arbiter)
    assert arbiter.submit(MotionSource.GRIPPER, ["joint7"], [0.04], monotonic_now=10.01)
    assert arbiter.owner is MotionOwner.IDLE
    assert arbiter.publisher.published[-1].positions == [0.0] * 6 + [0.04]


def test_gripper_only_without_fresh_arm_feedback_is_refused(arbiter) -> None:
    assert not arbiter.submit(
        MotionSource.GRIPPER, ["joint7"], [0.04], monotonic_now=10.01
    )
    assert arbiter.last_rejection is CommandRejection.NO_MEASURED_ARM


def test_gripper_update_during_manual_stream_preserves_commanded_arm(arbiter) -> None:
    _feed(arbiter)
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0)
    assert arbiter.submit(MotionSource.GRIPPER, ["joint7"], [0.04], monotonic_now=10.01)
    assert arbiter.owner is MotionOwner.MANUAL
    assert arbiter.publisher.published[-1].positions == _MANUAL[1][:6] + [0.04]


def test_manual_preemption_avoids_canceling_an_already_finished_goal(arbiter) -> None:
    """The trajectory success/cancel race left a hardware goal CANCELING indefinitely."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="goal")
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.end_goal(GoalState.SUCCEEDED, monotonic_now=10.05, goal_id="goal")
    arbiter.tick(monotonic_now=10.12)
    assert arbiter.canceler.cancel_calls == 0


def test_manual_override_suppresses_trajectory_until_the_preempted_goal_is_terminal(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="old")
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.begin_goal(
        monotonic_now=10.02, goal_id="old"
    )  # repeated EXECUTING/CANCELING status
    assert arbiter.owner is MotionOwner.MANUAL
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.03) is False


def test_repeated_manual_input_does_not_repeat_the_goal_cancellation(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.02)
    arbiter.tick(monotonic_now=10.12)
    arbiter.tick(monotonic_now=10.22)
    assert arbiter.canceler.cancel_calls == 1
    assert len(arbiter.publisher.published) == 2


def test_invalid_manual_input_does_not_interrupt_an_active_planned_goal(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    assert (
        arbiter.submit(
            MotionSource.MANUAL, _MANUAL[0][:-1], _MANUAL[1][:-1], monotonic_now=10.01
        )
        is False
    )
    assert arbiter.owner is MotionOwner.TRAJECTORY
    assert arbiter.canceler.cancel_calls == 0


def test_command_silence_never_changes_ownership(arbiter) -> None:
    """A long-running goal is silent between points. That is not a release."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    for tick in range(100):
        _feed(arbiter, now=10.0 + tick * 0.05)
        arbiter.tick(monotonic_now=10.0 + tick * 0.05)
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_terminal_status_alone_does_not_release_ownership(arbiter) -> None:
    """The arm is still decelerating when the action result arrives."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.end_goal(GoalState.SUCCEEDED, monotonic_now=10.5)
    moving = {**_MEASURED, "joint1": 0.5}
    _feed(arbiter, now=10.51, positions=moving)
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_ownership_releases_after_terminal_status_plus_measured_stability(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.end_goal(GoalState.SUCCEEDED, monotonic_now=10.5)
    for tick in range(10):
        _feed(arbiter, now=10.5 + tick * 0.05)
        arbiter.tick(monotonic_now=10.5 + tick * 0.05)
    assert arbiter.owner is MotionOwner.IDLE


def test_repeated_terminal_status_does_not_restart_the_stability_window(
    arbiter,
) -> None:
    """The action server republishes its status array; terminal is idempotent."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    for tick in range(10):
        now = 10.5 + tick * 0.05
        arbiter.end_goal(GoalState.SUCCEEDED, monotonic_now=now)
        _feed(arbiter, now=now)
        arbiter.tick(monotonic_now=now)
    assert arbiter.owner is MotionOwner.IDLE


def test_manual_preemption_does_not_replay_or_resume_the_preempted_goal(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="old")
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.end_goal(GoalState.CANCELED, monotonic_now=10.1, goal_id="old")
    for tick in range(10):
        _feed(arbiter, now=10.1 + tick * 0.05)
        arbiter.tick(monotonic_now=10.1 + tick * 0.05)
    assert arbiter.owner is MotionOwner.IDLE
    assert len(arbiter.publisher.published) == 1


def test_a_later_goal_can_take_ownership_after_manual_preemption_is_terminal(
    arbiter,
) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="old")
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.end_goal(GoalState.CANCELED, monotonic_now=10.1, goal_id="old")
    arbiter.begin_goal(monotonic_now=11.0, goal_id="new")
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_an_old_terminal_status_cannot_end_a_newer_goal(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0, goal_id="old")
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.01)
    arbiter.begin_goal(monotonic_now=10.6, goal_id="new")
    arbiter.end_goal(GoalState.CANCELED, monotonic_now=10.7, goal_id="old")
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_new_goal_cannot_steal_live_manual_stream_or_resume_after_it(arbiter) -> None:
    _feed(arbiter)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.4)
    arbiter.begin_goal(monotonic_now=10.6, goal_id="late")
    assert arbiter.owner is MotionOwner.MANUAL
    assert not arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.61)
    arbiter.tick(monotonic_now=10.71)
    assert arbiter.canceler.cancel_calls == 1
    arbiter.begin_goal(monotonic_now=11.0, goal_id="late")
    assert arbiter.owner is MotionOwner.MANUAL
    assert not arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=11.01)
    arbiter.end_goal(GoalState.CANCELED, monotonic_now=11.02, goal_id="late")
    arbiter.begin_goal(monotonic_now=11.1, goal_id="fresh")
    assert arbiter.owner is MotionOwner.TRAJECTORY


def test_manual_ownership_expires_after_the_live_teleop_stream_stops(arbiter) -> None:
    _feed(arbiter)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0)
    arbiter.tick(monotonic_now=10.49)
    assert arbiter.owner is MotionOwner.MANUAL
    arbiter.tick(monotonic_now=10.5)
    assert arbiter.owner is MotionOwner.IDLE


# ── Faults latch ─────────────────────────────────────────────────────────────


def test_a_vendor_fault_cancels_goals_and_latches(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=10.1)
    assert arbiter.owner is MotionOwner.FAULT
    assert arbiter.canceler.cancel_calls == 1


def test_a_fault_emits_at_most_one_hold_from_fresh_state(arbiter) -> None:
    _feed(arbiter, now=10.05)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=10.1)
    holds = [c for c in arbiter.publisher.published]
    assert len(holds) == 1
    arbiter.tick(monotonic_now=10.11)
    assert len(arbiter.publisher.published) == 1


def test_a_fault_on_stale_state_emits_no_hold_at_all(arbiter) -> None:
    """Holding at a posture we are no longer sure of is not a safe stop."""
    _feed(arbiter, now=10.0)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=11.0)
    assert arbiter.publisher.published == []


def test_both_sources_are_rejected_while_faulted(arbiter) -> None:
    _feed(arbiter)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    before = len(arbiter.publisher.published)
    _feed(arbiter, now=10.02)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.02)
    arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.02)
    assert len(arbiter.publisher.published) == before


def test_clearing_the_vendor_status_does_not_resume_motion(arbiter) -> None:
    """The latch is the point. A cleared fault is not a decision to move."""
    _feed(arbiter)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    arbiter.observe_arm_fault(False, monotonic_now=10.5)
    assert arbiter.owner is MotionOwner.FAULT


def test_stale_feedback_does_not_seize_motion_ownership(arbiter) -> None:
    _feed(arbiter, now=10.0)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.tick(monotonic_now=10.5)
    assert arbiter.owner is MotionOwner.TRAJECTORY
    assert arbiter.canceler.cancel_calls == 0


def test_unhealthy_state_health_is_observational(arbiter) -> None:
    _feed(arbiter)
    arbiter.observe_state_health(healthy=False, monotonic_now=10.01)
    assert arbiter.owner is MotionOwner.IDLE


# ── Reset is explicit ────────────────────────────────────────────────────────


def test_reset_does_not_depend_on_feedback_freshness(arbiter) -> None:
    _feed(arbiter, now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    arbiter.observe_arm_fault(False, monotonic_now=10.02)
    outcome = arbiter.reset_fault(monotonic_now=20.0)
    assert outcome.accepted is True
    assert arbiter.owner is MotionOwner.IDLE


def test_reset_is_refused_while_a_goal_is_still_active(arbiter) -> None:
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    _feed(arbiter, now=10.02)
    outcome = arbiter.reset_fault(monotonic_now=10.02)
    assert outcome.accepted is False
    assert outcome.refusal is ResetRefusal.GOAL_ACTIVE


def test_reset_is_refused_while_the_vendor_still_reports_a_fault(arbiter) -> None:
    _feed(arbiter)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    _feed(arbiter, now=10.02)
    outcome = arbiter.reset_fault(monotonic_now=10.02)
    assert outcome.accepted is False
    assert outcome.refusal is ResetRefusal.VENDOR_STILL_FAULTED


def test_an_explicit_reset_with_fresh_state_and_no_goal_succeeds(arbiter) -> None:
    _feed(arbiter)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    arbiter.end_goal(GoalState.CANCELED, monotonic_now=10.02)
    arbiter.observe_arm_fault(False, monotonic_now=10.03)
    _feed(arbiter, now=10.04)
    outcome = arbiter.reset_fault(monotonic_now=10.04)
    assert outcome.accepted is True
    assert arbiter.owner is MotionOwner.IDLE


def test_a_reset_does_not_replay_anything_that_was_dropped(arbiter) -> None:
    """The fault hold is expected; the dropped command reappearing is not."""
    _feed(arbiter)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    after_hold = len(arbiter.publisher.published)
    arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.02)
    assert len(arbiter.publisher.published) == after_hold  # dropped, not queued
    arbiter.observe_arm_fault(False, monotonic_now=10.03)
    _feed(arbiter, now=10.04)
    arbiter.reset_fault(monotonic_now=10.04)
    arbiter.tick(monotonic_now=10.05)
    assert len(arbiter.publisher.published) == after_hold


# ── Sole writer / isolation ──────────────────────────────────────────────────


#: Files allowed to name the vendor command topic, and why. Anything else
#: naming it is a second writer until proven otherwise, which is exactly the
#: condition this contract exists to prevent.


# ── Startup: never having had state is not the same as losing it ─────────────


def test_startup_without_feedback_does_not_latch_a_fault(arbiter) -> None:
    """Found by running the processes, not by reading them.

    The bridge correctly reports no_data before the vendor node has published
    anything. Latching on that means the arbiter enters FAULT ~100 ms after every
    boot and needs an explicit operator reset before the arm can ever move.

    A latch must mean "we had good state and lost it". Nothing unsafe is allowed
    by staying IDLE here: commands are still refused for want of fresh feedback.
    """
    arbiter.observe_state_health(healthy=False, monotonic_now=10.0)
    assert arbiter.owner is MotionOwner.IDLE


def test_a_complete_manual_command_is_accepted_during_that_startup_window(
    arbiter,
) -> None:
    """A normal startup health report cannot block timestamp-validated teleop."""
    arbiter.observe_state_health(healthy=False, monotonic_now=10.0)
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0) is True
    assert arbiter.publisher.published


def test_losing_state_after_having_it_is_observational_to_the_arbiter(arbiter) -> None:
    _feed(arbiter, now=10.0)
    arbiter.observe_state_health(healthy=False, monotonic_now=10.01)
    assert arbiter.owner is MotionOwner.IDLE


def test_a_state_health_report_never_suppresses_a_later_vendor_fault(arbiter) -> None:
    """Vendor faults remain the arbiter's independent, latched fault source."""
    arbiter.observe_state_health(healthy=False, monotonic_now=9.0)
    _feed(arbiter, now=10.0)
    arbiter.observe_arm_fault(True, monotonic_now=10.01)
    assert arbiter.owner is MotionOwner.FAULT


# ── Planned commands require an accepted goal ────────────────────────────────


def test_a_planned_command_in_idle_is_refused(arbiter) -> None:
    """topic_based_ros2_control streams commands continuously once a controller
    is active -- including the controller's hold position with no goal running.
    So "a planned command arrived" is not evidence that anything asked the arm to
    move, and accepting it in IDLE lets the controller drive the arm outside any
    goal's tolerances, preemption or cancellation.
    """
    _feed(arbiter)
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.0) is False
    assert arbiter.publisher.published == []
    assert arbiter.owner is MotionOwner.IDLE


def test_a_planned_command_does_not_grant_itself_ownership(arbiter) -> None:
    """Otherwise the first stray command silently becomes the owner."""
    _feed(arbiter)
    arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.0)
    assert arbiter.owner is not MotionOwner.TRAJECTORY


def test_a_planned_command_is_refused_again_after_the_goal_ends(arbiter) -> None:
    """The controller keeps streaming its hold after a goal completes."""
    _feed(arbiter)
    arbiter.begin_goal(monotonic_now=10.0)
    arbiter.end_goal(GoalState.SUCCEEDED, monotonic_now=10.1)
    for tick in range(10):  # release: terminal + stability
        _feed(arbiter, now=10.1 + tick * 0.05)
        arbiter.tick(monotonic_now=10.1 + tick * 0.05)
    assert arbiter.owner is MotionOwner.IDLE
    arbiter.publisher.published.clear()
    assert arbiter.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.7) is False
    assert arbiter.publisher.published == []


def test_manual_is_still_accepted_from_idle(arbiter) -> None:
    """The restriction is on the planned source, not a freeze of the arm."""
    _feed(arbiter)
    assert arbiter.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0) is True


def _gated(**kwargs):
    publisher, canceler = RecordingPublisher(), RecordingCanceler()
    core = MotionArbiterCore(
        limits=_LIMITS,
        scopes=SCOPES,
        publish=publisher,
        cancel_all_goals=canceler,
        feedback_timeout_s=0.1,
        **kwargs,
    )
    core.publisher, core.canceler = publisher, canceler
    return core


def test_planned_commands_are_refused_when_the_hardware_gates_fail() -> None:
    """Anything can publish to joint_commands. A gate that guards only the
    driver's Cartesian path is not a gate -- it is a convention.
    """
    core = _gated(planned_motion_allowed=False)
    _feed(core)
    core.begin_goal(monotonic_now=10.0)
    assert core.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.0) is False
    assert core.publisher.published == []


def test_teleop_still_works_while_planned_motion_is_inhibited() -> None:
    """Teleop is how an arm gets commissioned; inhibiting it would deadlock the
    very process that clears the gates."""
    core = _gated(planned_motion_allowed=False)
    _feed(core)
    assert core.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0) is True


def test_an_arbiter_with_no_limits_refuses_everything() -> None:
    """The failure mode of a launch that forgets to pass them."""
    publisher = RecordingPublisher()
    core = MotionArbiterCore(
        limits={},
        scopes=SCOPES,
        publish=publisher,
        cancel_all_goals=lambda: None,
    )
    core.observe_measured_state(_MEASURED, monotonic_now=10.0)
    assert core.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0) is False
    assert publisher.published == []


def test_an_unresolved_velocity_contract_inhibits_all_actuation() -> None:
    """No honest speed means no command at all -- manual included.

    Observation is untouched: this is exactly the state an operator has to be
    able to diagnose, and an arm you cannot command is still one you must watch.
    """
    core = _gated(velocity_contract_resolved=False)
    _feed(core)

    assert core.velocity_contract_resolved is False
    for source, command in (
        (MotionSource.MANUAL, _MANUAL),
        (MotionSource.TRAJECTORY, _ARM),
    ):
        assert core.submit(source, *command, monotonic_now=10.0) is False
        assert core.last_rejection is CommandRejection.VELOCITY_CONTRACT_UNRESOLVED
    assert core.publisher.published == []
    assert core.owner is MotionOwner.IDLE
    # A fault still latches, but no hold is emitted: a hold IS a command.
    core.observe_arm_fault(True, monotonic_now=10.01)
    assert core.owner is MotionOwner.FAULT
    assert core.publisher.published == []


def test_a_velocity_override_is_refused_not_stripped(arbiter) -> None:
    _feed(arbiter)
    assert not arbiter.submit(
        MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0, velocity_override=True
    )
    assert arbiter.last_rejection is CommandRejection.UNSUPPORTED_VELOCITY_OVERRIDE
    assert arbiter.publisher.published == []


def test_the_injected_vendor_adapter_shapes_every_published_command() -> None:
    """The vendor wire format is the adapter's; the core never builds one."""
    seen = []

    def adapter(command, *, held_gripper, gripper_effort):
        from cyberwave.driver.control.command_boundary import AdaptationResult

        seen.append((command.source, dict(held_gripper or {}), gripper_effort))
        return AdaptationResult(command=("vendor", dict(command.positions)))

    core = _gated(adapt=adapter)
    _feed(core, positions={**_MEASURED, "joint7": 0.031})
    core.begin_goal(monotonic_now=10.0)
    assert core.submit(MotionSource.TRAJECTORY, *_ARM, monotonic_now=10.01)
    assert core.publisher.published[-1][0] == "vendor"
    assert seen[-1] == (MotionSource.TRAJECTORY, {"joint7": 0.031}, None)


def test_an_adapter_refusal_is_reported_and_nothing_is_published() -> None:
    from cyberwave.driver.control.command_boundary import AdaptationResult

    core = _gated(
        adapt=lambda command, **_: AdaptationResult(
            reason=CommandRejection.INVALID_SPEED
        )
    )
    assert not core.submit(MotionSource.MANUAL, *_MANUAL, monotonic_now=10.0)
    assert core.last_rejection is CommandRejection.INVALID_SPEED
    assert core.owner is MotionOwner.IDLE
    assert core.publisher.published == []


def test_an_arm_only_manual_scope_carries_the_gripper_separately() -> None:
    """Arm-only teleop with its own gripper channel keeps the same semantics."""
    scopes = CommandScopes(arm=ARM_SCOPE, gripper=("joint7",), manual=ARM_SCOPE)
    publisher = RecordingPublisher()
    core = MotionArbiterCore(
        limits=_LIMITS, scopes=scopes, publish=publisher, cancel_all_goals=lambda: None
    )
    core.observe_measured_state(_MEASURED, monotonic_now=10.0)
    assert core.submit(MotionSource.MANUAL, *_ARM, monotonic_now=10.0)
    assert publisher.published[-1].positions == _ARM[1] + [0.0]
    assert core.submit(MotionSource.GRIPPER, ["joint7"], [0.04], monotonic_now=10.01)
    assert publisher.published[-1].positions == _ARM[1] + [0.04]
    assert core.submit(MotionSource.MANUAL, *_ARM, monotonic_now=10.02)
    assert publisher.published[-1].positions == _ARM[1] + [0.04]


def test_the_core_module_is_free_of_ros_mqtt_planning_and_blocking() -> None:
    import cyberwave.driver.control.arbiter as module

    source = Path(module.__file__).read_text()
    for forbidden in (
        "rclpy",
        "mqtt",
        "pinocchio",
        "moveit",
        "time.sleep",
        "spin_until_future_complete",
        ".result()",
        "requests.",
        "open(",
    ):
        assert forbidden not in source, f"arbiter core must not use {forbidden}"
