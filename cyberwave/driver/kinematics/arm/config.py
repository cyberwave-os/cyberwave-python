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
    binarize: bool = True                  # master switch for open/closed snapping
    binarize_measured: bool = True
    """Snap **measured** gripper values to the open/closed endpoints. Set False
    to preserve continuous width, which pick verification needs: a grasp is
    confirmed by the opening after closure differing from the empty-close
    position, and snapping destroys exactly that signal. Defaults True to keep
    existing drivers' telemetry unchanged."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mimic", _immutable(self.mimic))


def tool_offset_from_xyz_rpy(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """4x4 TCP offset from translation (m) + fixed-axis RPY (rad).

    The result is expressed in the ``ee_frame`` local frame: it moves the
    solved/reported point from the model frame to the physical grasp centre.
    Convention matches URDF ``rpy``: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = [x, y, z]
    return out


def _validated_tool_offset(value: object) -> np.ndarray:
    """Coerce *value* to a frozen 4x4 **rigid** transform, or raise ``ValueError``.

    A shape check alone is not enough: ``base.py`` inverts this matrix and
    composes it into every FK/IK result, so a non-rigid value corrupts geometry
    silently or fails much later inside the solver. Rejects non-finite entries, a
    non-homogeneous bottom row, and any 3x3 block that is not a proper rotation
    (reflections, scaling, and singular matrices included). Orthonormality with
    ``det == +1`` also guarantees invertibility, so no separate check is needed.
    """
    offset = np.array(value, dtype=float)
    if offset.shape != (4, 4):
        raise ValueError(
            f"tool_offset must be a 4x4 homogeneous matrix, got shape {offset.shape}"
        )
    if not np.all(np.isfinite(offset)):
        raise ValueError("tool_offset must be finite (no NaN or inf entries)")
    if not np.allclose(offset[3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(
            f"tool_offset bottom row must be [0, 0, 0, 1], got "
            f"{offset[3, :].tolist()}"
        )
    rot = offset[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6):
        raise ValueError(
            "tool_offset rotation block must be orthonormal (no scaling, shear, "
            "or singular matrices)"
        )
    det = float(np.linalg.det(rot))
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"tool_offset rotation block must be right-handed (det == +1), got "
            f"det == {det:.6f}; a reflection is not a rigid transform"
        )
    offset.flags.writeable = False
    return offset


@dataclass(frozen=True)
class ArmKinematicsConfig:
    """Frozen description of a robot arm for FK/IK and standard arm commands."""

    # --- kinematics model (required for ee_* commands) ---
    urdf_path: str
    ee_frame: str
    arm_joints: tuple[str, ...]
    locked_joints: tuple[str, ...] = ()

    base_frame: str | None = None
    """Frame that ``fk`` poses are expressed in, and the label published with
    them. ``None`` (the default) means the URDF root link, which is the frame
    pinocchio measures ``oMf`` from. Set it to publish poses under another name;
    it MUST then name the URDF root or a frame fixed to it at identity, because
    a ``base_frame`` that sits elsewhere would report poses in one frame under
    the name of another.
    :class:`~cyberwave.driver.kinematics.arm.base.BaseKinematicsManipulator`
    validates an explicit name at construction rather than trusting it."""

    tool_offset: np.ndarray | None = None
    """Optional 4x4 TCP offset in the ``ee_frame`` local frame. When set, ``fk``
    returns and ``ik`` accepts the **tool centre point**, not ``ee_frame``.
    Build it with :func:`tool_offset_from_xyz_rpy`. Note that ``tool_axes`` are
    then interpreted in the TCP frame."""

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
        if self.tool_offset is not None:
            offset = _validated_tool_offset(self.tool_offset)
            object.__setattr__(self, "tool_offset", offset)

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
