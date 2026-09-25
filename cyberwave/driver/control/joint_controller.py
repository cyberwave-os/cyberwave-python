"""JointControllerMixin — joint targets shaped by a JointController.

Compose BEFORE the arm mixins so the managed ``_apply_joint_targets`` wins the
MRO over ``_ArmSeams``'s raise-stub::

    class MyDriver(JointControllerMixin, ArmCapabilityMixin,
                   JointCommandBufferMixin, BaseROS2Driver):
        use_joint_controller = True
        def joint_controller_config(self): ...
        def joint_controller_after_process(self):
            return lambda cmd: self.submit_joint_targets(cmd.positions)

With a config present, every arm command (``ee_move``, ``grip``, ...) and every
MQTT ``joint/update`` target flows through ``JointController.plan``: shaped
moves stream via an asyncio pacer task; superseding cancels only streams whose
joint sets overlap (dual-arm safe). Without a config the mixin is inert and
``_apply_joint_targets`` falls through to the driver's own override.

Adopting drivers MUST NOT register their own ``joint/update`` listener — the
interface registry appends listeners, so both callbacks would fire.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from .controller import JointController, waypoint_command
from .motion import trapezoidal_motion
from .types import (
    JointCommand,
    JointControllerConfig,
    MotionGenerator,
    TrajectoryPlan,
)
from ..interface import CallbackGroup, ProtocolArgs, TopicSpec
from ..interface.args import TELEOP_MODES
from .joint_teleop import coerce_joint_timestamp

logger = logging.getLogger(__name__)

_TARGET_SOURCE_TYPES = ("tele",)


class _Stream:
    """One in-flight waypoint stream; cancellation is a cooperative flag."""

    __slots__ = ("joints", "cancelled")

    def __init__(self, joints: frozenset[str]) -> None:
        self.joints = joints
        self.cancelled = False


class JointControllerMixin:
    """Registers + shapes joint targets through a config-driven JointController."""

    use_joint_controller: bool = False

    # ── author-supplied seams ────────────────────────────────────────────────

    def joint_controller_config(self) -> JointControllerConfig | None:
        """Return the controller config, or None to disable the mixin entirely.
        MUST be safe before configure() and MUST NOT load a URDF."""
        return None

    def joint_controller_after_process(self) -> Callable[[JointCommand], None]:
        """The raw write seam (e.g. ``lambda cmd: self.submit_joint_targets(
        cmd.positions)``). Runs on the driver asyncio loop."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement joint_controller_after_process()"
        )

    def joint_controller_motion(self) -> MotionGenerator | None:
        """Override to swap the generator or return None for pure forwarding."""
        return trapezoidal_motion

    def _current_joint_positions(self) -> dict[str, float]:
        """Read seam (shared with the arm mixins / Ros2JointFeedbackMixin)."""
        return super()._current_joint_positions()  # type: ignore[misc]

    # ── opportunistic kinematics providers (never force a URDF load) ─────────

    def joint_controller_velocity_limits(self) -> Mapping[str, float]:
        kin = self._built_arm_kinematics()
        if kin is None:
            return {}
        limits = getattr(kin, "velocity_limits", None)
        return limits() if callable(limits) else {}

    def joint_controller_position_limits(self) -> Mapping[str, tuple[float, float]]:
        kin = self._built_arm_kinematics()
        if kin is None:
            return {}
        limits = getattr(kin, "joint_limits", None)
        return limits() if callable(limits) else {}

    def _built_arm_kinematics(self) -> Any:
        """The ALREADY-BUILT arm kinematics instance, or None. Reads the lazy
        caches of ArmCapabilityMixin without triggering a build."""
        ctrl = getattr(self, "_arm_ctrl", None)
        return getattr(ctrl, "_kin", None)

    # ── lazily-built controller ──────────────────────────────────────────────

    @property
    def _joint_controller(self) -> JointController | None:
        if not type(self).use_joint_controller:
            return None
        cached = getattr(self, "_joint_ctrl_cached", None)
        if cached is None:
            cfg = self.joint_controller_config()
            if cfg is None:
                return None
            cached = JointController(
                cfg,
                motion=self.joint_controller_motion(),
                after_process=self.joint_controller_after_process(),
                velocity_limits=self.joint_controller_velocity_limits,
                position_limits=self.joint_controller_position_limits,
            )
            self._joint_ctrl_cached = cached
        return cached

    @property
    def _active_streams(self) -> list[_Stream]:
        streams = getattr(self, "_joint_streams", None)
        if streams is None:
            streams = []
            self._joint_streams = streams
        return streams

    # ── registration ─────────────────────────────────────────────────────────

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        if not type(self).use_joint_controller:
            return
        if self.joint_controller_config() is None:
            return
        iface.add_listener(
            TopicSpec(
                namespace="joint",
                leaf="update",
                payload_schema_ref="JointUpdatePayload",
                description="Joint position targets (shaped by JointController)",
            ),
            CallbackGroup(callback=self._on_joint_target),
            protocol=ProtocolArgs(source_types=list(_TARGET_SOURCE_TYPES)),
            operation_modes=TELEOP_MODES,
        )

    # ── dispatch (driver asyncio loop) ───────────────────────────────────────

    async def _on_joint_target(self, payload: dict[str, Any]) -> None:
        try:
            ctrl = self._joint_controller
            if ctrl is None:
                return
            ts = coerce_joint_timestamp(payload.get("timestamp"))
            if ts is not None:
                watermark = getattr(self, "_joint_target_watermark", None)
                if watermark is not None and ts < watermark:
                    return  # stale: never planned (post-plan waypoints are untimestamped)
                self._joint_target_watermark = ts
            from cyberwave.data.state_representation import parse_joint_mqtt_payload

            names = set(ctrl.config.joints) if ctrl.config.joints else None
            parsed = parse_joint_mqtt_payload(payload, controllable_names=names)
            target = parsed.target_positions or parsed.positions
            if not target:
                log_ignored = getattr(self, "log_tele_ignored", None)
                if callable(log_ignored):
                    log_ignored(payload)
                return
            log_rx = getattr(self, "log_tele_rx", None)
            if callable(log_rx):
                log_rx(payload, target)
            plan = ctrl.plan(
                self._current_joint_positions(),
                target,
                velocities=parsed.target_velocities or None,
            )
            self._dispatch_plan(plan)
        except Exception:
            logger.exception("joint target dispatch failed")

    # ── default arm-command wiring ───────────────────────────────────────────

    def _apply_joint_targets(self, positions: dict[str, float]) -> None:
        ctrl = self._joint_controller
        if ctrl is None:
            return super()._apply_joint_targets(positions)  # type: ignore[misc]
        try:
            self._dispatch_plan(
                ctrl.plan(self._current_joint_positions(), positions)
            )
        except Exception:
            logger.exception("joint target planning failed")

    # ── stream execution & superseding ───────────────────────────────────────

    def _dispatch_plan(self, plan: TrajectoryPlan) -> None:
        ctrl = self._joint_controller
        if ctrl is None or not plan.waypoints:
            return
        self.cancel_motion(plan.joints)  # supersede overlapping streams only
        after = ctrl.after_process
        assert after is not None
        if len(plan.waypoints) == 1:
            after(waypoint_command(plan.waypoints[0], ctrl.config))
            return
        stream = _Stream(plan.joints)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (unusual sync context): degrade to the final target.
            logger.warning("no running event loop; applying final target directly")
            after(waypoint_command(plan.waypoints[-1], ctrl.config))
            return
        self._active_streams.append(stream)
        loop.create_task(self._run_stream(plan, stream, after))

    async def _run_stream(
        self,
        plan: TrajectoryPlan,
        stream: _Stream,
        after: Callable[[JointCommand], None],
    ) -> None:
        ctrl = self._joint_controller
        assert ctrl is not None
        start = time.monotonic()
        try:
            for wp in plan.waypoints:
                delay = start + wp.time_from_start - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if stream.cancelled:
                    return
                after(waypoint_command(wp, ctrl.config))
        except Exception:
            logger.exception("joint stream failed")
        finally:
            try:
                self._active_streams.remove(stream)
            except ValueError:
                pass

    def cancel_motion(self, joints: frozenset[str] | None = None) -> None:
        """Cancel all in-flight streams (or only those intersecting *joints*)."""
        for stream in self._active_streams:
            if joints is None or stream.joints & joints:
                stream.cancelled = True

    # ── lifecycle (cooperative overrides) ────────────────────────────────────

    async def on_exit_operation(self) -> None:
        self.cancel_motion()
        parent = getattr(super(), "on_exit_operation", None)
        if parent is not None:
            await parent()

    def request_home(self) -> None:
        self.cancel_motion()
        parent = getattr(super(), "request_home", None)
        if parent is not None:
            parent()
        else:
            self._home_pending = True
