"""Joint command-vector resolution — scalar→vector broadcast with precedence.

Pure helpers (no ROS, no numpy) for building per-joint velocity/effort command
vectors. Precedence: a full-array override wins; else a single scalar broadcast;
else the arm's built-in defaults. Parameterized by joint layout so any arm reuses
them; the vendor joint-name ordering and message construction stay in the driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def resolve_command_vector(
    override: Any,
    scalar: float | None,
    *,
    defaults: Sequence[float],
    count: int,
) -> list[float]:
    """Full-array *override* (len >= count) wins, truncated to *count*; else *scalar*
    broadcast to *count* joints; else ``list(defaults)``."""
    if isinstance(override, (list, tuple)) and len(override) >= count:
        return [float(v) for v in override[:count]]
    if scalar is not None:
        return [float(scalar)] * count
    return list(defaults)


def resolve_effort_vector(
    override: Any,
    gripper_effort: float | None,
    arm_scalar: float | None,
    *,
    defaults: Sequence[float],
    gripper_index: int,
    gripper_clamp: tuple[float, float],
) -> list[float]:
    """Full-array *override* (len >= len(defaults)) is used verbatim. Otherwise start
    from *defaults*, replace the arm joints (indices < *gripper_index*) with
    *arm_scalar* when given, then set the gripper index to a clamped *gripper_effort*
    when given."""
    count = len(defaults)
    if isinstance(override, (list, tuple)) and len(override) >= count:
        return [float(v) for v in override[:count]]
    efforts = [float(v) for v in defaults]
    if arm_scalar is not None:
        for i in range(gripper_index):
            efforts[i] = float(arm_scalar)
    if gripper_effort is not None:
        lo, hi = gripper_clamp
        efforts[gripper_index] = max(lo, min(hi, float(gripper_effort)))
    return efforts
