"""ArmKinematicsConfig — the full description of an arm for kinematics + commands.

Pure data (numpy only). No pinocchio, no ROS, no cloud imports. The command
mixin gates which arm commands to register on the ``supports_*`` properties, all
of which read only cheap static fields (never the URDF), so this is safe to
evaluate before ``configure()`` during manifest export.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


def _immutable(mapping: Mapping[str, object] | None) -> Mapping:
    return MappingProxyType(dict(mapping)) if mapping else MappingProxyType({})


@dataclass(frozen=True)
class GripperConfig:
    """Description of a (usually binary) end-effector for command + telemetry handling."""

    names: tuple[str, ...]                 # commanded gripper joint(s), e.g. ("joint7",)
    open: float = 0.0                      # fully-open / released position
    closed: float = 0.0                    # fully-closed / engaged position
    mimic: Mapping[str, float] = field(default_factory=dict)  # passive mirrors, e.g. {"joint8": -1.0}
    binarize: bool = True                  # snap measured + commanded values to open/closed

    def __post_init__(self) -> None:
        object.__setattr__(self, "mimic", _immutable(self.mimic))


@dataclass(frozen=True)
class ArmKinematicsConfig:
    """Frozen description of a robot arm for FK/IK and standard arm commands."""

    # --- kinematics model (required for ee_* commands) ---
    urdf_path: str
    ee_frame: str
    arm_joints: tuple[str, ...]
    locked_joints: tuple[str, ...] = ()

    # --- tool/base axis mapping (required for ee_move / ee_rotate) ---
    tool_axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    base_axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    rot_axes: Mapping[str, np.ndarray] = field(default_factory=dict)

    # --- gripper / end-effector (enables grip/release when set) ---
    gripper: GripperConfig | None = None

    # --- base joint nudge (enables base_rotate_* when set) ---
    base_joint: str | None = None
    base_joint_limits: tuple[float, float] | None = None
    """Optional (lower, upper) limit for ``base_joint``, in radians. When set,
    ``base_rotate`` uses it directly instead of building kinematics (pinocchio +
    URDF parse) just to read the joint limit from the model."""

    # --- IK tuning ---
    ik_eps: float = 1e-4
    ik_max_iter: int = 1000
    ik_dt: float = 1e-1
    ik_damp: float = 1e-6

    def __post_init__(self) -> None:
        # Coerce mappings to immutable views so a frozen instance stays a value.
        object.__setattr__(self, "tool_axes", _immutable(self.tool_axes))
        object.__setattr__(self, "base_axes", _immutable(self.base_axes))
        object.__setattr__(self, "rot_axes", _immutable(self.rot_axes))

    @property
    def supports_ee(self) -> bool:
        return bool(self.ee_frame and self.arm_joints and self.tool_axes and self.rot_axes)

    @property
    def supports_gripper(self) -> bool:
        return self.gripper is not None

    @property
    def supports_base_rotate(self) -> bool:
        return self.base_joint is not None

    @property
    def supports_home(self) -> bool:
        return True  # home is always available; drivers declare _home_position
