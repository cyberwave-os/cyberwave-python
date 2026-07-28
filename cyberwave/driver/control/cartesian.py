"""EECartesianPoseMixin — absolute end-effector pose command (``ee_pose``).

By default (``process_joints_ik=False``) the raw pose is forwarded to the
``_apply_ee_pose`` write seam — the native-cartesian path (e.g. a vendor
``/pos_cmd`` topic), no IK imposed. With ``process_joints_ik=True`` the pose
is solved locally via the composed arm's :class:`ArmController` and applied
as joint targets.

The wire payload is a single 6-element ``pose`` array ``[x, y, z, roll, pitch,
yaw]``. Robot-specific mode selectors are mixin config (``pose_mode1`` /
``pose_mode2``), read by the driver inside ``_apply_ee_pose`` — nothing
robot-specific leaks onto the wire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import TELEOP_MODES
from ..kinematics.arm.controller import ArmController

logger = logging.getLogger(__name__)

# The wire payload is a single 6-element pose array [x, y, z, roll, pitch, yaw].
EE_POSE_ARGS: tuple[CommandArg, ...] = (
    CommandArg("pose", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "[x,y,z,roll,pitch,yaw]"),
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
                if controller is None:
                    raise RuntimeError(
                        "process_joints_ik=True requires a composed arm controller"
                    )
                solution = controller.pose_to_joints(
                    self._current_joint_positions(), pose
                )
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
