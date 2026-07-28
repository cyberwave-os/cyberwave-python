"""joint_command: scalar→vector resolution with override precedence and clamp."""

from __future__ import annotations

import pytest

from cyberwave.driver.control.command_vectors import resolve_command_vector, resolve_effort_vector

_VEL_DEFAULTS = [50.0] * 7
_EFF_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]


def test_vector_full_override_wins_and_is_truncated() -> None:
    out = resolve_command_vector([1, 2, 3, 4, 5, 6, 7, 8], None, defaults=_VEL_DEFAULTS, count=7)
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_vector_scalar_broadcast() -> None:
    out = resolve_command_vector(None, 12.5, defaults=_VEL_DEFAULTS, count=7)
    assert out == [12.5] * 7


def test_vector_defaults_fallback() -> None:
    assert resolve_command_vector(None, None, defaults=_VEL_DEFAULTS, count=7) == _VEL_DEFAULTS


def test_effort_full_override_verbatim() -> None:
    out = resolve_effort_vector(
        [1, 1, 1, 1, 1, 1, 1], None, None,
        defaults=_EFF_DEFAULTS, gripper_index=6, gripper_clamp=(0.5, 3.0),
    )
    assert out == [1.0] * 7  # override is used verbatim, clamp not applied


def test_effort_arm_scalar_leaves_gripper_index() -> None:
    out = resolve_effort_vector(
        None, None, 2.0,
        defaults=_EFF_DEFAULTS, gripper_index=6, gripper_clamp=(0.5, 3.0),
    )
    assert out[:6] == [2.0] * 6
    assert out[6] == pytest.approx(3.0)  # gripper index untouched by arm_scalar


def test_effort_gripper_clamped_both_bounds() -> None:
    hi = resolve_effort_vector(
        None, 99.0, None, defaults=_EFF_DEFAULTS, gripper_index=6, gripper_clamp=(0.5, 3.0)
    )
    assert hi[6] == pytest.approx(3.0)
    lo = resolve_effort_vector(
        None, 0.01, None, defaults=_EFF_DEFAULTS, gripper_index=6, gripper_clamp=(0.5, 3.0)
    )
    assert lo[6] == pytest.approx(0.5)
