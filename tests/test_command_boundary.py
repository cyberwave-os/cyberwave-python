"""Source-scoped joint-command validation (``cyberwave.driver.control.command_boundary``).

Pure: no ROS, no clock, no bus. A 6-joint arm with a one-joint gripper is used
throughout because that is the shape that exposes every scope distinction.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cyberwave.driver.control.command_boundary import (
    LIMIT_BOUNDARY_TOLERANCE_RAD,
    CommandRejection,
    CommandScopes,
    MotionSource,
    VendorJointCommand,
    command_stamp_is_fresh,
    joint_state_adapter,
    source_for_names,
    validate_command,
)

ARM = tuple(f"joint{i}" for i in range(1, 7))
SCOPES = CommandScopes(arm=ARM, gripper=("joint7",))
LIMITS = {**{j: (-2.0, 2.0) for j in ARM}, "joint7": (0.0, 0.05)}
ARM_POS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MANUAL_NAMES = [*ARM, "joint7"]
MANUAL_POS = [*ARM_POS, 0.02]


def _validate(source, names, positions, limits=LIMITS):
    return validate_command(source, names, positions, limits=limits, scopes=SCOPES)


# ── scopes ────────────────────────────────────────────────────────────────────


def test_manual_scope_defaults_to_arm_plus_gripper() -> None:
    assert SCOPES.manual == (*ARM, "joint7")
    assert SCOPES.for_source(MotionSource.TRAJECTORY) == ARM
    assert SCOPES.for_source(MotionSource.GRIPPER) == ("joint7",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm": ()},
        {"arm": ("a", "a")},
        {"arm": ("a",), "gripper": ("a",)},
        {"arm": ("a", "b"), "manual": ("a",)},  # manual must contain the arm
        {"arm": ("a",), "manual": ("a", "z")},  # and nothing foreign
    ],
)
def test_incoherent_scopes_are_refused_at_construction(kwargs) -> None:
    with pytest.raises(ValueError):
        CommandScopes(**kwargs)


def test_command_scopes_classify_only_exact_joint_sets() -> None:
    assert source_for_names(list(reversed(ARM)), SCOPES) is MotionSource.TRAJECTORY
    assert source_for_names(list(reversed(MANUAL_NAMES)), SCOPES) is MotionSource.MANUAL
    assert source_for_names(["joint7"], SCOPES) is MotionSource.GRIPPER
    for names in ([], list(ARM[:-1]), [*ARM[:-1], "joint1"], [*MANUAL_NAMES, "joint8"]):
        assert source_for_names(names, SCOPES) is None


def test_an_arm_without_gripper_still_matches_each_input_by_its_own_source() -> None:
    scopes = CommandScopes(arm=("a", "b"))
    assert scopes.matches(MotionSource.MANUAL, ["b", "a"])
    assert scopes.matches(MotionSource.TRAJECTORY, ["a", "b"])
    assert not scopes.matches(MotionSource.GRIPPER, [])
    assert scopes.gripper_of({"a": 1.0}) is None


# ── validation ───────────────────────────────────────────────────────────────


def test_a_valid_trajectory_command_preserves_arm_positions() -> None:
    command = _validate(MotionSource.TRAJECTORY, list(ARM), ARM_POS).command
    assert command is not None
    assert [command.positions[j] for j in ARM] == ARM_POS


def test_a_trajectory_command_carrying_a_gripper_is_out_of_scope() -> None:
    result = _validate(MotionSource.TRAJECTORY, MANUAL_NAMES, MANUAL_POS)
    assert result.command is None
    assert result.reason is CommandRejection.UNKNOWN_JOINT


@pytest.mark.parametrize(
    "names,positions,reason",
    [
        ([], [], CommandRejection.EMPTY),
        (list(ARM[:3]), ARM_POS[:3], CommandRejection.INCOMPLETE_SCOPE),
        (list(ARM), ARM_POS[:-1], CommandRejection.LENGTH_MISMATCH),
        (["joint1"] * 6, ARM_POS, CommandRejection.DUPLICATE_JOINT),
        ([*ARM[:5], "elbow"], ARM_POS, CommandRejection.UNKNOWN_JOINT),
        (
            list(ARM),
            [0.1, float("nan"), 0.3, 0.4, 0.5, 0.6],
            CommandRejection.NON_FINITE,
        ),
        (list(ARM), [0.1, math.inf, 0.3, 0.4, 0.5, 0.6], CommandRejection.NON_FINITE),
        (list(ARM), [99.0, 0.2, 0.3, 0.4, 0.5, 0.6], CommandRejection.OUT_OF_LIMIT),
    ],
)
def test_malformed_commands_are_rejected(names, positions, reason) -> None:
    result = _validate(MotionSource.TRAJECTORY, names, positions)
    assert result.command is None
    assert result.reason is reason


def test_a_joint_with_no_declared_limit_fails_closed() -> None:
    limits = {k: v for k, v in LIMITS.items() if k != "joint3"}
    result = _validate(MotionSource.TRAJECTORY, list(ARM), ARM_POS, limits=limits)
    assert result.reason is CommandRejection.MISSING_LIMIT
    assert "joint3" in result.detail
    assert (
        _validate(MotionSource.TRAJECTORY, list(ARM), ARM_POS, limits={}).command
        is None
    )


def test_sensor_noise_outside_a_bound_is_clamped_into_range_not_refused() -> None:
    limits = {**LIMITS, "joint2": (0.0, 2.0)}
    for offset in (-1e-6, -1e-4, -LIMIT_BOUNDARY_TOLERANCE_RAD):
        positions = [0.1, offset, 0.3, 0.4, 0.5, 0.6]
        command = _validate(
            MotionSource.TRAJECTORY, list(ARM), positions, limits
        ).command
        assert command is not None
        assert command.positions["joint2"] == 0.0
    beyond = [0.1, -2 * LIMIT_BOUNDARY_TOLERANCE_RAD, 0.3, 0.4, 0.5, 0.6]
    assert (
        _validate(MotionSource.TRAJECTORY, list(ARM), beyond, limits).reason
        is CommandRejection.OUT_OF_LIMIT
    )
    upper = [0.1, 2.0 + 1e-4, 0.3, 0.4, 0.5, 0.6]
    command = _validate(MotionSource.TRAJECTORY, list(ARM), upper, limits).command
    assert command is not None and command.positions["joint2"] == 2.0


def test_validated_commands_are_immutable() -> None:
    command = _validate(MotionSource.MANUAL, MANUAL_NAMES, MANUAL_POS).command
    assert command is not None
    with pytest.raises(TypeError):
        command.positions["joint1"] = 9.0  # type: ignore[index]


def test_command_input_rejects_stale_unstamped_and_future_samples() -> None:
    now = 10_000_000_000
    assert command_stamp_is_fresh(9, 900_000_000, now_ns=now, max_age_s=0.25)
    assert not command_stamp_is_fresh(9, 700_000_000, now_ns=now, max_age_s=0.25)
    assert not command_stamp_is_fresh(0, 0, now_ns=now, max_age_s=0.25)
    assert not command_stamp_is_fresh(10, 100_000_000, now_ns=now, max_age_s=0.25)


# ── default adapter ──────────────────────────────────────────────────────────


def _adapt(source, names, positions, *, held=None, effort=None):
    command = _validate(source, names, positions).command
    assert command is not None
    return joint_state_adapter(SCOPES)(
        command, held_gripper=held, gripper_effort=effort
    )


def test_default_adapter_emits_arm_then_gripper_in_scope_order() -> None:
    result = _adapt(
        MotionSource.MANUAL, list(reversed(MANUAL_NAMES)), list(reversed(MANUAL_POS))
    )
    assert result.command == VendorJointCommand(
        names=list(MANUAL_NAMES), positions=MANUAL_POS, velocities=[], efforts=[]
    )


def test_an_arm_only_command_carries_the_held_gripper_never_a_default() -> None:
    held = _adapt(MotionSource.TRAJECTORY, list(ARM), ARM_POS, held={"joint7": 0.031})
    assert held.command.positions[-1] == 0.031
    missing = _adapt(MotionSource.TRAJECTORY, list(ARM), ARM_POS)
    assert missing.command is None
    assert missing.reason is CommandRejection.NO_MEASURED_GRIPPER
    nan = _adapt(MotionSource.TRAJECTORY, list(ARM), ARM_POS, held={"joint7": math.nan})
    assert nan.reason is CommandRejection.NO_MEASURED_GRIPPER


def test_gripper_only_input_requires_the_arbiter_to_supply_the_arm() -> None:
    result = _adapt(MotionSource.GRIPPER, ["joint7"], [0.04], held={"joint7": 0.04})
    assert result.reason is CommandRejection.NO_MEASURED_ARM


def test_a_gripper_effort_rides_in_the_gripper_effort_slot() -> None:
    result = _adapt(MotionSource.MANUAL, MANUAL_NAMES, MANUAL_POS, effort=0.8)
    assert result.command.efforts == [0.0] * 6 + [0.8]


def test_the_boundary_module_is_free_of_ros_driver_and_transport_imports() -> None:
    import cyberwave.driver.control.command_boundary as module

    source = Path(module.__file__).read_text()
    for forbidden in ("rclpy", "mqtt", "pinocchio", "import time", "moveit"):
        assert forbidden not in source
