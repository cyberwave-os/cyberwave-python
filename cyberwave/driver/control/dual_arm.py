"""DualArmMixin — one-twin bimanual capability (one URDF, two end-effectors).

Targets robots like OpenArm: a single URDF with two arms whose joints are
prefixed (``left_*`` / ``right_*``) and combined into shared topics. Exposes
per-arm ``left_*`` / ``right_*`` commands routed to per-arm
:class:`ArmController`s, plus a conjunct ``home`` that fans out to both arms.

Because the joints are already prefixed, per-arm controllers return prefixed
joint names, so the ``_apply_joint_targets`` seam stays arm-agnostic — the names
disambiguate which arm. Only ``_apply_ee_pose`` needs an explicit ``arm`` (a pose
has no joint names to disambiguate it).

Telemetry is unchanged and unified: the driver forwards one combined JointState
(all ``left_*`` + ``right_*`` joints) to a single ``joint/update`` payload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import TELEOP_MODES
from ..kinematics import ArmKinematicsConfig
from ..kinematics.arm.controller import ArmController
from .arm import (
    _BASE_ROTATIONS,
    _EE_ROTATIONS,
    _EE_TRANSLATE,
    _ArmSeams,
    execute_arm_command,
)
from .cartesian import EE_POSE_ARGS, EEPose

logger = logging.getLogger(__name__)

_ARMS: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class DualArmConfig:
    """Two arm configs for a single bimanual twin (may share ``urdf_path``)."""

    left: ArmKinematicsConfig
    right: ArmKinematicsConfig


class DualArmMixin(_ArmSeams):
    """Registers + dispatches per-arm ``left_*``/``right_*`` commands + conjunct ``home``."""

    use_dual_arm: bool = False

    # ── author-supplied seams ────────────────────────────────────────────────

    def dual_arm_config(self) -> DualArmConfig | None:
        """Return the dual-arm config, or None to disable. Safe before configure()."""
        return None

    def _apply_arm_home(self, arm: str) -> None:
        """Home a single arm (write seam). Stub by default."""
        raise NotImplementedError(f"per-arm home is not implemented for {arm!r}")

    # ── controllers ──────────────────────────────────────────────────────────

    @property
    def _arms(self) -> dict[str, ArmController]:
        arms = getattr(self, "_arms_cache", None)
        if arms is None:
            cfg = self.dual_arm_config()
            if cfg is None:
                raise RuntimeError("dual_arm_config() returned None at dispatch time")
            arms = {
                "left": ArmController(
                    cfg.left, build_kinematics=self._build_arm_kinematics
                ),
                "right": ArmController(
                    cfg.right, build_kinematics=self._build_arm_kinematics
                ),
            }
            self._arms_cache = arms
        return arms

    # ── registration ─────────────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if not self.use_dual_arm:
            return
        cfg = self.dual_arm_config()
        if cfg is None:
            return
        cmd_topic = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            description="Dual-arm per-arm and conjunct commands",
        )
        for name, args in self._dual_arm_command_specs(cfg):
            iface.add_listener(
                cmd_topic,
                CallbackGroup(callback=self._on_dual_arm_command),
                command=CommandArgs(name=name, args=tuple(args)),
                operation_modes=TELEOP_MODES,
            )

    def _dual_arm_command_specs(
        self, cfg: DualArmConfig
    ) -> list[tuple[str, list[CommandArg]]]:
        dist = [
            CommandArg("forward", 0.0, "m"),
            CommandArg("up", 0.0, "m"),
            CommandArg("left", 0.0, "m"),
            CommandArg("frame", "tool"),
        ]
        angle = [CommandArg("angle", 0.0, "rad")]
        specs: list[tuple[str, list[CommandArg]]] = []
        arm_cfgs = {"left": cfg.left, "right": cfg.right}
        cartesian = getattr(type(self), "use_cartesian_pose", False)
        for prefix in _ARMS:
            arm_cfg = arm_cfgs[prefix]
            if arm_cfg.supports_ee:
                specs.append((f"{prefix}_{_EE_TRANSLATE}", list(dist)))
                for name in _EE_ROTATIONS:
                    specs.append((f"{prefix}_{name}", list(angle)))
                if cartesian:
                    specs.append((f"{prefix}_ee_pose", list(EE_POSE_ARGS)))
            if arm_cfg.supports_base_rotate:
                for name in _BASE_ROTATIONS:
                    specs.append((f"{prefix}_{name}", list(angle)))
            if arm_cfg.supports_gripper:
                specs.append((f"{prefix}_grip", [CommandArg("force", 1.0)]))
                specs.append((f"{prefix}_release", []))
            if arm_cfg.supports_home:
                specs.append((f"{prefix}_home", []))
        specs.append(("home", []))  # conjunct — fans out to both arms
        return specs

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _on_dual_arm_command(self, payload: dict[str, Any]) -> None:
        command = payload.get("command")
        data = payload.get("data") or {}
        try:
            if command == "home":
                request_home = getattr(self, "request_home", None)
                if callable(request_home):
                    request_home()
                else:
                    self._home_pending = True
                return
            if command is None or "_" not in command:
                logger.warning("DualArm: unknown command %r", command)
                return
            prefix, _, sub = command.partition("_")
            if prefix not in self._arms:
                logger.warning("DualArm: unknown command %r", command)
                return
            controller = self._arms[prefix]
            if sub == "home":
                self._apply_arm_home(prefix)
                return
            if sub == "ee_pose":
                pose = EEPose.from_payload(data)
                self._dispatch_ee_pose(pose, controller=controller, arm=prefix)  # type: ignore[attr-defined]
                return
            if execute_arm_command(
                controller,
                sub,
                data,
                self._current_joint_positions(),
                apply_targets=self._apply_joint_targets,
                on_failed=self._on_arm_command_failed,
            ):
                return
            logger.warning("DualArm: unknown command %r", command)
        except Exception as exc:  # stub / solver errors must not crash dispatch
            self._on_arm_command_failed(command, str(exc))
