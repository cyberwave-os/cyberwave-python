"""CartesianStateMixin — broadcast the measured end-effector pose.

The read half of the cartesian surface, and a thin adapter by design: it owns no
kinematics and runs no forward kinematics itself. It asks the arm's
:class:`~cyberwave.driver.kinematics.arm.controller.ArmController` for
``ee_pose()`` and wraps the result for the wire, so exactly one pinocchio model
exists per arm and the command and state paths can never disagree about where
the tool is.

Publishes on ``cyberwave/twin/{uuid}/state/kinematics`` as the ``cartesian_pose``
branch that
:func:`~cyberwave.data.state_representation.decode_kinematics_protobuf_stub`
parses and ``twin.pose`` consumes. This is ``T_base_ee(t)`` — the first term of
any camera-to-base transform chain.

Two properties come from the arm config rather than from this layer: the pose is
the **tool centre point** whenever ``tool_offset`` is set, and the reference
frame is ``config.base_frame`` (the URDF root link when unset), which
:class:`~cyberwave.driver.kinematics.arm.base.BaseKinematicsManipulator`
validates against the URDF root. A driver therefore cannot label poses with a
frame they were not measured in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..interface import CallbackGroup, PublisherArgs, TopicSpec
from ..interface.args import default_operation_modes
from ..motion.poses import transform_to_pose

if TYPE_CHECKING:
    from ..kinematics import ArmKinematicsConfig

logger = logging.getLogger(__name__)


class CartesianStateMixin:
    """Registers the ``state/kinematics`` end-effector pose publisher.

    Requires a composed arm controller (``ArmCapabilityMixin``). Dual-arm twins
    are not supported: the ``cartesian_pose`` payload carries a single pose and
    :class:`~cyberwave.driver.control.dual_arm.DualArmMixin` exposes a left/right
    pair, so the publisher stays silent rather than picking an arm.
    """

    use_cartesian_state: bool = False
    CARTESIAN_STATE_RATE_HZ: float = 10.0

    if TYPE_CHECKING:
        # Seams supplied by the composed driver (ArmCapabilityMixin + a joint
        # feedback mixin). Declared for type-checkers only.
        def arm_config(self) -> "ArmKinematicsConfig | None": ...
        def _current_joint_positions(self) -> dict[str, float]: ...

    # ── registration ─────────────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if not self.use_cartesian_state:
            return
        cfg = self.arm_config()
        if cfg is None or not cfg.supports_ee:
            return
        iface.add_publisher(
            TopicSpec(
                namespace="state",
                leaf="kinematics",
                payload_schema_ref="TwinKinematicsPayload",
                description=(
                    "Measured end-effector (TCP) pose in the robot base frame, "
                    "derived from joint feedback by forward kinematics"
                ),
            ),
            CallbackGroup(callback=self._cartesian_state_publisher_callback),
            publisher=PublisherArgs(rate_hz=type(self).CARTESIAN_STATE_RATE_HZ),
            operation_modes=default_operation_modes(management=True),
        )

    # ── payload + publish ────────────────────────────────────────────────────

    def _cartesian_state_payload(self) -> dict[str, Any] | None:
        """Decoder-compatible ``cartesian_pose`` payload, or None if not ready.

        Requires a reported position for **every** configured arm joint. A
        partial message — a startup ``JointState`` carrying only the gripper, or
        one arm joint — would otherwise be completed with 0.0 for the rest by
        ``ArmController``, and the resulting neutral-configuration pose would be
        published as if it were measured. Downstream perception and planning
        treat this topic as ground truth, so silence is the only safe output
        until the arm has fully reported.
        """
        controller = getattr(self, "_arm", None)
        if controller is None:
            return None  # no single-arm controller (e.g. a dual-arm twin)
        positions = self._current_joint_positions()
        arm_joints = controller.config.arm_joints
        if not arm_joints or any(name not in positions for name in arm_joints):
            return None
        pose = controller.ee_pose(positions)
        x, y, z, w = transform_to_pose(pose).orientation
        return {
            "message_type": "cartesian_pose",
            "cartesian_pose": {
                "reference_frame": controller.base_frame,
                "position": {
                    "x": float(pose[0, 3]),
                    "y": float(pose[1, 3]),
                    "z": float(pose[2, 3]),
                },
                "orientation": {"x": x, "y": y, "z": z, "w": w},
            },
        }

    def _cartesian_state_publisher_callback(self) -> dict[str, Any] | None:
        """Registry publisher tick: the TCP pose payload; the registry publishes it.

        Returned rather than sent through ``twin.publish_kinematics``, which
        reconnects to MQTT synchronously: during a broker outage that blocked
        the driver loop for seconds every tick. The registry's publish returns
        immediately when the broker is offline, like every other publisher.
        """
        if getattr(self, "_twin", None) is None:
            return None
        try:
            return self._cartesian_state_payload()
        except Exception:
            logger.exception("Cartesian state: FK failed; skipping this tick")
            return None
