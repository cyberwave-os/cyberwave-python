"""Proto mirror: robot_state_message.proto — discriminated union stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .space.cartesian import CartesianPose

RobotStateBranch = Literal["cartesian_pose", "joint_state", "unknown"]


@dataclass(frozen=True, slots=True)
class RobotStateMessage:
    message_type: RobotStateBranch
    cartesian_pose: CartesianPose | None = None
    joint_positions: dict[str, float] | None = None
    raw: dict[str, Any] | None = None

    def which(self) -> RobotStateBranch:
        return self.message_type
