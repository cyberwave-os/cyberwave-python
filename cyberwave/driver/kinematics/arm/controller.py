"""ArmController — pure joint-target math for robot arms.

Extracted from ``ArmCapabilityMixin`` so single-arm and dual-arm paths share one
engine. This class **computes joint targets and never applies them**; it knows
nothing about ``self``, MQTT, seams, or ROS. Kinematics (pinocchio via
``BaseKinematicsManipulator``) is built lazily on the first EE/pose call, so
drivers that never issue EE/pose commands never import pinocchio.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .config import ArmKinematicsConfig


class EEPoseLike(Protocol):
    """Structural type for an absolute end-effector pose (no hard import).

    Members are read-only properties so a frozen dataclass (``EEPose``) satisfies
    the protocol.
    """

    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def z(self) -> float: ...
    @property
    def roll(self) -> float: ...
    @property
    def pitch(self) -> float: ...
    @property
    def yaw(self) -> float: ...


def pose_to_matrix(pose: EEPoseLike) -> np.ndarray:
    """4x4 homogeneous transform from x,y,z + roll,pitch,yaw.

    Standard fixed-axis (extrinsic XYZ) RPY convention: R = Rz(yaw) @ Ry(pitch)
    @ Rx(roll), matching ROS/URDF ``rpy`` semantics.
    """
    cr, sr = np.cos(pose.roll), np.sin(pose.roll)
    cp, sp = np.cos(pose.pitch), np.sin(pose.pitch)
    cy, sy = np.cos(pose.yaw), np.sin(pose.yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = [pose.x, pose.y, pose.z]
    return out


class ArmController:
    """Config-driven pure math: computes joint targets, never applies them."""

    def __init__(
        self,
        config: ArmKinematicsConfig,
        *,
        build_kinematics: Callable[[ArmKinematicsConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._build_kinematics = build_kinematics
        self._kin: Any = None

    @property
    def config(self) -> ArmKinematicsConfig:
        return self._config

    def _kinematics(self) -> Any:
        if self._kin is None:
            build = self._build_kinematics
            if build is None:
                from .base import BaseKinematicsManipulator  # lazy: imports pinocchio

                build = BaseKinematicsManipulator
            self._kin = build(self._config)
        return self._kin

    def _start(self, current: dict[str, float]) -> dict[str, float]:
        return {n: current.get(n, 0.0) for n in self._config.arm_joints}

    # ── EE motion (FK -> transform -> IK) ────────────────────────────────────

    def translate(
        self,
        current: dict[str, float],
        components: dict[str, float],
        frame: str = "tool",
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        start = self._start(current)
        target = kin.apply_translation(kin.fk(start), components, frame=frame)
        return kin.ik(start, target)

    def rotate(
        self, current: dict[str, float], axis: str, angle: float, frame: str = "tool"
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        start = self._start(current)
        target = kin.apply_rotation(kin.fk(start), axis, angle, frame=frame)
        return kin.ik(start, target)

    def pose_to_joints(
        self, current: dict[str, float], pose: EEPoseLike
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        return kin.ik(self._start(current), pose_to_matrix(pose))

    # ── base joint nudge (no IK) ─────────────────────────────────────────────

    def base_rotate(self, current: dict[str, float], angle: float) -> dict[str, float]:
        base = self._config.base_joint
        assert base is not None, "base_rotate requires config.base_joint"
        cur = current.get(base, 0.0)
        if self._config.base_joint_limits is not None:
            lo, hi = self._config.base_joint_limits
        else:
            lo, hi = (
                self._kinematics()
                .joint_limits()
                .get(base, (float("-inf"), float("inf")))
            )
        return {base: max(lo, min(hi, cur + angle))}

    # ── gripper command + feedback ───────────────────────────────────────────

    def gripper_target(self, force: float) -> dict[str, float]:
        g = self._config.gripper
        if g is None:
            return {}
        force = max(0.0, min(1.0, force))
        target = g.open + force * (g.closed - g.open)
        positions: dict[str, float] = {name: target for name in g.names}
        for mimic, factor in g.mimic.items():
            positions[mimic] = factor * target
        return positions

    def classify_gripper(self, value: float) -> str:
        g = self._config.gripper
        assert g is not None, "classify_gripper requires a gripper config"
        return "open" if abs(value - g.open) <= abs(value - g.closed) else "closed"

    def binarize(self, payload: dict[str, float]) -> dict[str, float]:
        g = self._config.gripper
        if g is None or not g.binarize:
            return payload
        result = payload
        for name in g.names:
            if name in payload:
                snapped = (
                    g.open
                    if self.classify_gripper(payload[name]) == "open"
                    else g.closed
                )
                result = {**result, name: snapped}
                for mimic, factor in g.mimic.items():
                    if mimic in result:
                        result = {**result, mimic: factor * snapped}
        return result

    def hold(self, latched_state: str | None) -> dict[str, float]:
        g = self._config.gripper
        if g is None:
            return {}
        value = g.open if (latched_state or "open") == "open" else g.closed
        return {name: value for name in g.names}
