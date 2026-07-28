"""ArmCapabilityMixin — auto-wire standard arm commands from a config.

Combine with any BaseDriver subclass via multiple inheritance::

    class MyArmDriver(ArmCapabilityMixin, BaseROS2Driver):
        use_base_commands = True
        def arm_config(self): ...
        def _apply_joint_targets(self, positions): ...
        def _current_joint_positions(self): ...

The mixin owns command registration and dispatch; all math is delegated to a
single ``self._arm`` :class:`~cyberwave.driver.kinematics.arm.controller.ArmController`
so single-arm and dual-arm paths never diverge. Kinematics (pinocchio) is built
lazily on the first EE command.

``_on_arm_command_failed`` defaults to logging plus a throttled ``ik_failed``
twin alert (``IK_ALERT_THROTTLE_S``, default 5 s) via ``create_twin_alert`` when
present — override only to change that behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import TELEOP_MODES
from ..kinematics import ArmKinematicsConfig
from ..kinematics.arm.controller import ArmController
from ..support.throttle import Throttle

logger = logging.getLogger(__name__)

# command name -> rot axis + sign for IK-driven orientation commands.
_EE_ROTATIONS: dict[str, tuple[str, float]] = {
    "ee_rotate_left": ("yaw", 1.0),
    "ee_rotate_right": ("yaw", -1.0),
    "ee_rotate_up": ("pitch", 1.0),
    "ee_rotate_down": ("pitch", -1.0),
}
# base-joint yaw nudge (no IK).
_BASE_ROTATIONS: dict[str, float] = {
    "base_rotate_left": 1.0,
    "base_rotate_right": -1.0,
}
_EE_TRANSLATE = "ee_move"


def execute_arm_command(
    controller: ArmController,
    command: str,
    data: dict[str, Any],
    current: dict[str, float],
    *,
    apply_targets: Callable[[dict[str, float]], None],
    on_failed: Callable[[str | None, str], None],
) -> bool:
    """Run one motion/gripper command against *controller*.

    Handles ``ee_move``/``ee_rotate_*``/``base_rotate_*``/``grip``/``release``.
    Returns ``True`` if the command was recognized and handled, ``False`` if not
    (the caller handles ``home``/``ee_pose``). IK misses are routed to
    *on_failed* and still count as handled.
    """
    if command == "grip":
        positions = controller.gripper_target(float(data.get("force", 1.0)))
        if positions:
            apply_targets(positions)
        return True
    if command == "release":
        positions = controller.gripper_target(0.0)
        if positions:
            apply_targets(positions)
        return True
    if command in _BASE_ROTATIONS:
        angle = float(data.get("angle", 0.0)) * _BASE_ROTATIONS[command]
        apply_targets(controller.base_rotate(current, angle))
        return True
    if command == _EE_TRANSLATE or command in _EE_ROTATIONS:
        frame = str(data.get("frame", "tool"))
        if command == _EE_TRANSLATE:
            components = {k: float(data.get(k, 0.0)) for k in ("forward", "up", "left")}
            solution = controller.translate(current, components, frame=frame)
        else:
            axis, sign = _EE_ROTATIONS[command]
            solution = controller.rotate(
                current, axis, float(data.get("angle", 0.0)) * sign, frame=frame
            )
        if solution is None:
            on_failed(command, "IK did not converge / out of reach")
            return True
        apply_targets(solution)
        return True
    return False


class _ArmSeams:
    """Shared author-supplied seams + failure hook for arm-command mixins."""

    IK_ALERT_THROTTLE_S: float = 5.0
    _ALERT_AUTO_RESOLVE_S: float = 5.0

    def _apply_joint_targets(self, positions: dict[str, float]) -> None:
        """Apply solved joint targets to hardware (write seam)."""
        raise NotImplementedError

    def _current_joint_positions(self) -> dict[str, float]:
        """Return the arm's current joint positions (read seam)."""
        return super()._current_joint_positions()  # type: ignore[misc]

    def _build_arm_kinematics(self, config: ArmKinematicsConfig) -> Any:
        from ..kinematics import BaseKinematicsManipulator

        return BaseKinematicsManipulator(config)

    def _on_arm_command_failed(self, command: str | None, reason: str) -> None:
        """IK/out-of-reach failure hook: log + throttled ``ik_failed`` twin alert."""
        logger.warning("Arm command %r failed: %s", command, reason)
        throttle = getattr(self, "_ik_alert_throttle", None)
        if throttle is None:
            throttle = Throttle(type(self).IK_ALERT_THROTTLE_S)
            self._ik_alert_throttle = throttle
        if not throttle.ready():
            return
        create_alert = getattr(self, "create_twin_alert", None)
        if callable(create_alert):
            create_alert(
                "ik_failed",
                description=f"Arm command {command!r} could not be executed: {reason}",
                alert_type="ik_failed",
                severity="warning",
                auto_resolve_after=type(self)._ALERT_AUTO_RESOLVE_S,
            )


class ArmCapabilityMixin(_ArmSeams):
    """Mixin that registers + dispatches standard arm commands from a config."""

    use_base_commands: bool = False

    # ── author-supplied seam ─────────────────────────────────────────────────

    def arm_config(self) -> ArmKinematicsConfig | None:
        """Return the arm config, or None to disable all arm commands.

        MUST be safe to call before ``configure()`` (manifest export calls
        ``define_interface`` early) and MUST NOT require loading the URDF.
        """
        return None

    # ── controller (all math lives here) ─────────────────────────────────────

    def _resolved_arm_config(self) -> ArmKinematicsConfig:
        cfg = getattr(self, "_arm_cfg_cached", None)
        if cfg is None:
            cfg = self.arm_config()
            if cfg is None:
                raise RuntimeError("arm_config() returned None at dispatch time")
            self._arm_cfg_cached = cfg
        return cfg

    @property
    def _arm(self) -> ArmController:
        ctrl = getattr(self, "_arm_ctrl", None)
        if ctrl is None:
            ctrl = ArmController(
                self._resolved_arm_config(), build_kinematics=self._build_arm_kinematics
            )
            self._arm_ctrl = ctrl
        return ctrl

    # ── command registration ─────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if not self.use_base_commands:
            return
        cfg = self.arm_config()
        if cfg is None:
            return
        cmd_topic = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            description="Standard arm end-effector, gripper, and base commands",
        )
        for name, args in self._arm_command_specs(cfg):
            iface.add_listener(
                cmd_topic,
                CallbackGroup(callback=self._on_arm_command),
                command=CommandArgs(name=name, args=tuple(args)),
                operation_modes=TELEOP_MODES,
            )

    def _arm_command_specs(
        self, cfg: ArmKinematicsConfig
    ) -> list[tuple[str, list[CommandArg]]]:
        dist = [
            CommandArg("forward", 0.0, "m"),
            CommandArg("up", 0.0, "m"),
            CommandArg("left", 0.0, "m"),
            CommandArg("frame", "tool"),
        ]
        angle = [CommandArg("angle", 0.0, "rad")]
        specs: list[tuple[str, list[CommandArg]]] = []
        if cfg.supports_ee:
            specs.append((_EE_TRANSLATE, dist))
            for name in _EE_ROTATIONS:
                specs.append((name, list(angle)))
        if cfg.supports_base_rotate:
            for name in _BASE_ROTATIONS:
                specs.append((name, list(angle)))
        if cfg.supports_gripper:
            specs.append(("grip", [CommandArg("force", 1.0)]))
            specs.append(("release", []))
        if cfg.supports_home:
            specs.append(("home", []))
        return specs

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _on_arm_command(self, payload: dict[str, Any]) -> None:
        command = payload.get("command")
        data = payload.get("data") or {}
        try:
            if command == "home":
                # Prefer JointCommandBufferMixin.request_home when composed; fall
                # back to the bare flag for arm drivers without the buffer mixin.
                request_home = getattr(self, "request_home", None)
                if callable(request_home):
                    request_home()
                else:
                    self._home_pending = True
                return
            if command is not None and execute_arm_command(
                self._arm,
                command,
                data,
                self._current_joint_positions(),
                apply_targets=self._apply_joint_targets,
                on_failed=self._on_arm_command_failed,
            ):
                return
            logger.warning("Arm: unknown command %r", command)
        except Exception as exc:  # solver/build failures must not crash dispatch
            self._on_arm_command_failed(command, str(exc))

    # ── binary end-effector feedback (delegates to the controller) ───────────

    def latch_gripper(self, positions: dict[str, float]) -> None:
        """Record the commanded binary end-effector state from *positions* (no-op if the
        gripper joint is absent, no gripper is configured, or binarize is off)."""
        g = self._resolved_arm_config().gripper
        if g is None or not g.binarize:
            return
        for name in g.names:
            if name in positions:
                self._gripper_latched = self._arm.classify_gripper(positions[name])
                return

    def binarize_gripper(self, payload: dict[str, float]) -> dict[str, float]:
        """Snap the gripper joint(s) + any present mimic joints in *payload* to the open/
        closed endpoints (for telemetry). Returns *payload* unchanged when no gripper /
        binarize is off."""
        return self._arm.binarize(payload)

    def gripper_hold(self) -> dict[str, float]:
        """The latched commanded end-effector target to re-assert every control cycle —
        commanded joint(s) only (mimics are passive). Empty when no gripper."""
        return self._arm.hold(getattr(self, "_gripper_latched", None))
