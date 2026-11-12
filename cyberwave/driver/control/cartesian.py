"""Cartesian command semantics for arms — what a cartesian command *means*.

This module owns the meaning of every cartesian command: the argument tables
(:func:`cartesian_command_specs`), the solve for relative jogs
(:func:`solve_cartesian_jog`), and the absolute-pose command mixin
(:class:`EECartesianPoseMixin`). ``arm.py`` and ``dual_arm.py`` own how a
command reaches *hardware* — the speed seam, applying targets, alerting on
failure — and both build their cartesian rows from here.

One source for the tables is deliberate: they were previously hand-rolled in
each place and drifted, which is how ``speed`` reached ``ee_move`` but silently
skipped ``left_ee_move``.

The relative jogs (``ee_move`` / ``ee_rotate_*``) and the absolute pose
(``ee_pose``) are the same kind of thing — cartesian targets solved through the
arm's :class:`~cyberwave.driver.kinematics.arm.controller.ArmController`. The
read half lives in :mod:`~cyberwave.driver.control.cartesian_state`.

``ee_pose`` specifics: by default (``process_joints_ik=False``) the raw pose is
forwarded to the ``_apply_ee_pose`` write seam — the native-cartesian path (e.g.
a vendor ``/pos_cmd`` topic), no IK imposed. With ``process_joints_ik=True`` the
pose is solved locally and applied as joint targets. The wire payload is a single
6-element ``pose`` array ``[x, y, z, roll, pitch, yaw]``; robot-specific mode
selectors are mixin config (``pose_mode1`` / ``pose_mode2``), read by the driver
inside ``_apply_ee_pose``, so nothing robot-specific leaks onto the wire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import TELEOP_MODES
from ..kinematics.arm.controller import ArmController

if TYPE_CHECKING:
    from ..kinematics import ArmKinematicsConfig

logger = logging.getLogger(__name__)

# The wire payload is a single 6-element pose array [x, y, z, roll, pitch, yaw].
EE_POSE_ARGS: tuple[CommandArg, ...] = (
    CommandArg("pose", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "[x,y,z,roll,pitch,yaw]"),
)

#: Per-command speed scale (0-1), honoured by drivers that override
#: ``_set_command_speed``. Only cartesian commands carry it — ``base_rotate_*``,
#: ``grip``, ``release`` and ``home`` run no IK.
SPEED_ARG: CommandArg = CommandArg("speed", 0.5)

#: ``ee_move``: a translation delta in the tool or base frame.
EE_JOG_TRANSLATE_ARGS: tuple[CommandArg, ...] = (
    CommandArg("forward", 0.0, "m"),
    CommandArg("up", 0.0, "m"),
    CommandArg("left", 0.0, "m"),
    CommandArg("frame", "tool"),
    SPEED_ARG,
)

#: ``ee_rotate_*``: a rotation delta about a named axis.
EE_JOG_ROTATE_ARGS: tuple[CommandArg, ...] = (
    CommandArg("angle", 0.0, "rad"),
    SPEED_ARG,
)

#: Relative-jog command name -> (rot axis, sign). ``ee_move`` is handled apart.
EE_ROTATE_AXES: dict[str, tuple[str, float]] = {
    "ee_rotate_left": ("yaw", 1.0),
    "ee_rotate_right": ("yaw", -1.0),
    "ee_rotate_up": ("pitch", 1.0),
    "ee_rotate_down": ("pitch", -1.0),
}

EE_TRANSLATE = "ee_move"


def is_cartesian_jog(command: str) -> bool:
    """True for the relative cartesian commands (``ee_move`` / ``ee_rotate_*``)."""
    return command == EE_TRANSLATE or command in EE_ROTATE_AXES


def cartesian_command_specs(
    cfg: "ArmKinematicsConfig", *, include_pose: bool = False
) -> list[tuple[str, tuple[CommandArg, ...]]]:
    """Cartesian ``(name, args)`` rows for *cfg*, or ``[]`` when EE is unsupported.

    The single source of truth for both the single-arm and dual-arm surfaces;
    dual-arm prefixes the names. *include_pose* adds ``ee_pose``, which is gated
    separately on ``use_cartesian_pose``.
    """
    if not cfg.supports_ee:
        return []
    specs: list[tuple[str, tuple[CommandArg, ...]]] = [
        (EE_TRANSLATE, EE_JOG_TRANSLATE_ARGS)
    ]
    specs.extend((name, EE_JOG_ROTATE_ARGS) for name in EE_ROTATE_AXES)
    if include_pose:
        specs.append(("ee_pose", EE_POSE_ARGS))
    return specs


def solve_cartesian_jog(
    controller: "ArmController",
    command: str,
    data: dict[str, Any],
    current: dict[str, float],
) -> dict[str, float] | None:
    """Solve a relative cartesian jog to joint targets. Applies nothing.

    ``None`` when IK misses. Callers own the failure and speed policy — this is
    pure semantics: FK the current state, transform, solve.
    """
    frame = str(data.get("frame", "tool"))
    if command == EE_TRANSLATE:
        components = {k: float(data.get(k, 0.0)) for k in ("forward", "up", "left")}
        return controller.translate(current, components, frame=frame)
    axis, sign = EE_ROTATE_AXES[command]
    return controller.rotate(
        current, axis, float(data.get("angle", 0.0)) * sign, frame=frame
    )


@dataclass(frozen=True)
class EEPose:
    """Absolute end-effector pose in the robot base frame."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "EEPose":
        """Parse the 6-element ``pose`` array [x, y, z, roll, pitch, yaw]."""
        pose = list(data.get("pose") or ())
        if len(pose) != 6:
            raise ValueError(f"ee_pose expects a 6-element pose array, got {pose!r}")
        x, y, z, roll, pitch, yaw = (float(v) for v in pose)
        return cls(x, y, z, roll, pitch, yaw)


class EECartesianPoseMixin:
    """Registers + dispatches the absolute ``ee_pose`` command."""

    use_cartesian_pose: bool = False
    use_dual_arm: bool = False  # set by DualArmMixin; suppresses the bare ee_pose
    pose_mode1: int = 0
    pose_mode2: int = 0

    if TYPE_CHECKING:
        # Seams provided by the composed base (ArmCapabilityMixin / DualArmMixin
        # via _ArmSeams). Declared here for type-checkers only; never define them
        # at runtime so they don't shadow the concrete driver's implementations.
        def _current_joint_positions(self) -> dict[str, float]: ...
        def _apply_joint_targets(self, positions: dict[str, float]) -> None: ...
        def _on_arm_command_failed(self, command: str | None, reason: str) -> None: ...

    def __init__(self, *a: Any, process_joints_ik: bool = False, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._process_joints_ik = process_joints_ik

    # ── write seam (STUB) ────────────────────────────────────────────────────

    def _apply_ee_pose(self, pose: EEPose, *, arm: str | None = None) -> None:
        """Send the raw pose to hardware (native cartesian). Stub by default."""
        raise NotImplementedError("ee_pose send is not implemented for this driver")

    # ── registration ─────────────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if not self.use_cartesian_pose or self.use_dual_arm:
            # DualArmMixin registers left_/right_ee_pose instead of the bare command.
            return
        cmd_topic = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            description="Absolute end-effector cartesian pose command",
        )
        iface.add_listener(
            cmd_topic,
            CallbackGroup(callback=self._on_ee_pose),
            command=CommandArgs(name="ee_pose", args=EE_POSE_ARGS),
            operation_modes=TELEOP_MODES,
        )

    # ── dispatch ─────────────────────────────────────────────────────────────

    def plan_ee_pose(
        self, pose: EEPose, *, controller: ArmController | None = None
    ) -> dict[str, float] | None:
        """Solve *pose* against the current joint state without moving.

        Returns the joint solution, or ``None`` when IK does not converge or the
        target is out of reach. Raises no alert and applies nothing — callers use
        this to validate a pre-grasp, release, or retreat pose before committing
        the arm to it.

        *controller* defaults to the composed single-arm ``self._arm``; dual-arm
        callers pass the specific arm's controller (``DualArmMixin`` has no
        ``_arm``, only a left/right ``_arms`` map).
        """
        if controller is None:
            controller = getattr(self, "_arm", None)
        if controller is None:
            raise RuntimeError("plan_ee_pose requires a composed arm controller")
        return controller.pose_to_joints(self._current_joint_positions(), pose)

    def _on_ee_pose(self, payload: dict[str, Any]) -> None:
        pose = EEPose.from_payload(payload.get("data") or {})
        self._dispatch_ee_pose(pose, controller=getattr(self, "_arm", None), arm=None)

    def _dispatch_ee_pose(
        self, pose: EEPose, *, controller: ArmController | None, arm: str | None
    ) -> None:
        """Reusable core: IK-on solves joints, IK-off forwards the raw pose.

        Reused by DualArmMixin's per-arm dispatch (which passes the arm's
        controller + the ``left``/``right`` arm label).
        """
        try:
            if self._process_joints_ik:
                solution = self.plan_ee_pose(pose, controller=controller)
                if solution is None:
                    self._fail_ee_pose("IK did not converge / out of reach")
                    return
                self._apply_joint_targets(solution)
            else:
                self._apply_ee_pose(pose, arm=arm)
        except Exception as exc:  # stub NotImplementedError / solver errors
            self._fail_ee_pose(str(exc))

    def _fail_ee_pose(self, reason: str) -> None:
        on_failed = getattr(self, "_on_arm_command_failed", None)
        if callable(on_failed):
            on_failed("ee_pose", reason)
        else:
            logger.warning("ee_pose failed: %s", reason)
