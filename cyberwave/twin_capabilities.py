"""Capability mixins and specialized twin implementations."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Type, Tuple, List

from .compact_api import CompactTwin, _await_result, _run_async_in_jupyter, _safe_float

logger = logging.getLogger(__name__)


class _TwinCommandMixin:
    """Shared helpers for capability mixins."""

    def _twins_api(self):
        client = getattr(self, "_client", None)
        twins_api = getattr(client, "twins", None) if client else None
        if twins_api is None:
            raise RuntimeError("Twin command API not available")
        return twins_api

    def _ensure_uuid(self) -> Optional[str]:
        twin_uuid = getattr(self, "_twin_uuid", None)
        if not twin_uuid:
            try:
                _await_result(self._ensure_twin_exists())
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Unable to ensure twin exists: %s", exc)
            twin_uuid = getattr(self, "_twin_uuid", None)
        return twin_uuid

    def _dispatch_command(self, name: str, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(payload or {})
        try:
            twin_uuid = self._ensure_uuid()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Command '%s' aborted: %s", name, exc)
            return {"success": False, "error": str(exc)}

        if not twin_uuid:
            return {"success": False, "offline": True, "command": name}
        if str(twin_uuid).startswith("local-"):
            logger.debug("Twin %s running locally; command '%s' skipped", twin_uuid, name)
            return {"success": False, "offline": True, "command": name}

        try:
            result = self._twins_api().command(twin_uuid, name, payload)
            if inspect.isawaitable(result) or isinstance(result, (asyncio.Future, asyncio.Task)):
                return _await_result(result) or {"success": True}
            return result or {"success": True}
        except Exception as exc:
            logger.warning("Command '%s' failed: %s", name, exc)
            return {"success": False, "error": str(exc)}

    def _call_async(self, coro):
        try:
            return _await_result(coro)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Twin async call failed: %s", exc)
            return None


class ManipulationCapabilities(_TwinCommandMixin):
    """High-level manipulation helpers built on the joint controller."""

    def move_joints(self, joint_positions: Mapping[str, float]) -> Dict[str, float]:
        if not isinstance(joint_positions, Mapping):
            raise TypeError("joint_positions must be a mapping of alias -> value")
        self.joints.set_many(joint_positions)
        return self.joints.all()

    def move_joint(self, alias: str, position: float) -> float:
        states = self.move_joints({alias: position})
        return states.get(alias, position)

    def move_pose(self, pose: Mapping[str, float]) -> Dict[str, Any]:
        if not isinstance(pose, Mapping):
            raise TypeError("pose must be a mapping")
        return self._dispatch_command("move_pose", {"pose": dict(pose)})

    def move_to(self, position: Mapping[str, float], orientation: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"position": dict(position)}
        if orientation:
            payload["orientation"] = dict(orientation)
        return self._dispatch_command("move_to", payload)

    def calibrate(self) -> Dict[str, Any]:
        return self._dispatch_command("calibrate")

    def home(self) -> Dict[str, Any]:
        return self._dispatch_command("home")


class GripperCapabilities(_TwinCommandMixin):
    """Convenience methods for gripper control."""

    def open_gripper(self) -> Dict[str, Any]:
        return self._dispatch_command("open_gripper")

    def close_gripper(self) -> Dict[str, Any]:
        return self._dispatch_command("close_gripper")

    def set_gripper(self, opening_mm: float) -> Dict[str, Any]:
        return self._dispatch_command("set_gripper", {"position": _safe_float(opening_mm)})


class TeleoperationCapabilities(_TwinCommandMixin):
    """Leader-follower teleoperation controls."""

    def start_teleoperation(self) -> Dict[str, Any]:
        return self._dispatch_command("start_leader_follower")

    def stop_teleoperation(self) -> Dict[str, Any]:
        return self._dispatch_command("stop_leader_follower")

    def calibrate_leader(self) -> Dict[str, Any]:
        return self._dispatch_command("calibrate_leader")

    def calibrate_follower(self) -> Dict[str, Any]:
        return self._dispatch_command("calibrate_follower")

    def sync_arms(self) -> Dict[str, Any]:
        return self._dispatch_command("sync_arms")


class SafetyCapabilities(_TwinCommandMixin):
    """Safety and torque management helpers."""

    def emergency_stop(self) -> Dict[str, Any]:
        return self._dispatch_command("emergency_stop")

    def enable_torque(self) -> Dict[str, Any]:
        return self._dispatch_command("enable_torque")

    def disable_torque(self) -> Dict[str, Any]:
        return self._dispatch_command("disable_torque")

    def get_safety_status(self) -> Optional[Dict[str, Any]]:
        result = self._dispatch_command("get_safety_status")
        return result if isinstance(result, dict) else None


class TelemetryCapabilities(_TwinCommandMixin):
    """Read-only helpers for joint and pose telemetry."""

    def joints_live(self) -> Dict[str, float]:
        return self.joints.all()

    def joint_states(self) -> Optional[Dict[str, Any]]:
        twin_uuid = self._ensure_uuid()
        if not twin_uuid or str(twin_uuid).startswith("local-"):
            return None
        return self._call_async(self._twins_api().get_joint_states(twin_uuid))

    def pose(self) -> Optional[Dict[str, Any]]:
        twin_uuid = self._ensure_uuid()
        if not twin_uuid or str(twin_uuid).startswith("local-"):
            return None
        return self._dispatch_command("get_position")

    def gripper_state(self) -> Optional[Dict[str, Any]]:
        result = self._dispatch_command("get_gripper_state")
        return result if isinstance(result, dict) else None


class FlightCapabilities(_TwinCommandMixin):
    """Common drone flight controls."""

    _DIRECTION_COMMANDS = {
        "forward": "forward",
        "back": "back",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
    }

    def takeoff(self) -> Dict[str, Any]:
        return self._dispatch_command("takeoff")

    def land(self) -> Dict[str, Any]:
        return self._dispatch_command("land")

    def emergency_stop(self) -> Dict[str, Any]:
        return self._dispatch_command("emergency")

    def set_speed(self, speed_cm_s: float) -> Dict[str, Any]:
        return self._dispatch_command("speed", {"value": _safe_float(speed_cm_s)})

    def move(self, direction: str, distance_cm: float = 50.0) -> Dict[str, Any]:
        direction_key = direction.lower()
        command = self._DIRECTION_COMMANDS.get(direction_key)
        if not command:
            raise ValueError(f"Unsupported flight direction '{direction}'")
        return self._dispatch_command(command, {"distance": _safe_float(distance_cm)})

    def rotate(self, degrees: float) -> Dict[str, Any]:
        command = "cw" if degrees >= 0 else "ccw"
        return self._dispatch_command(command, {"degrees": abs(float(degrees))})

    def navigate_to(self, x_cm: float, y_cm: float, z_cm: float, speed_cm_s: float = 50.0) -> Dict[str, Any]:
        payload = {
            "x": _safe_float(x_cm),
            "y": _safe_float(y_cm),
            "z": _safe_float(z_cm),
            "speed": _safe_float(speed_cm_s),
        }
        return self._dispatch_command("go", payload)


class CameraStreamingCapabilities(_TwinCommandMixin):
    """Streaming helpers for video-capable devices."""

    def start_video_stream(self) -> Dict[str, Any]:
        return self._dispatch_command("streamon")

    def stop_video_stream(self) -> Dict[str, Any]:
        return self._dispatch_command("streamoff")

    def get_camera_feed(self) -> Dict[str, Any]:
        return self._dispatch_command("get_camera_feed")


class QuadrupedMobilityCapabilities(_TwinCommandMixin):
    """Mobility helpers for legged robots."""

    def walk(self, speed_m_s: float = 0.5) -> Dict[str, Any]:
        return self._dispatch_command("walk", {"speed": _safe_float(speed_m_s)})

    def turn(self, yaw_degrees: float) -> Dict[str, Any]:
        return self._dispatch_command("turn", {"degrees": float(yaw_degrees)})

    def sit(self) -> Dict[str, Any]:
        return self._dispatch_command("sit")

    def stand(self) -> Dict[str, Any]:
        return self._dispatch_command("stand")

    def dance(self) -> Dict[str, Any]:
        return self._dispatch_command("dance")

    def navigate_to(self, position: Mapping[str, float]) -> Dict[str, Any]:
        return self._dispatch_command("navigate_to", dict(position))


class SO101Twin(
    ManipulationCapabilities,
    GripperCapabilities,
    TeleoperationCapabilities,
    SafetyCapabilities,
    TelemetryCapabilities,
    CompactTwin,
):
    """Specialized twin for the SO-101 robotic arm."""

    def __init__(
        self,
        registry_id: str = "cyberwave/so101",
        name: Optional[str] = None,
        environment_id: Optional[str] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        environment_name: Optional[str] = None,
    ) -> None:
        super().__init__(registry_id, name, environment_id, project_id, project_name, environment_name)
        self._configure_joint_aliases()

    def pick(
        self,
        approach_pose: Mapping[str, float],
        grasp_pose: Mapping[str, float],
        lift_pose: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """Scripted pick routine composed from capability primitives."""

        steps: Iterable[Dict[str, Any]] = [
            self.move_pose(approach_pose),
            self.open_gripper(),
            self.move_pose(grasp_pose),
            self.close_gripper(),
        ]
        results = list(steps)
        if lift_pose:
            results.append(self.move_pose(lift_pose))
        success = all(res.get("success", True) if isinstance(res, dict) else True for res in results)
        return {"success": success, "steps": results}

    def place(
        self,
        approach_pose: Mapping[str, float],
        release_pose: Mapping[str, float],
        retreat_pose: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """Scripted place routine composed from capability primitives."""

        steps: list[Dict[str, Any]] = [
            self.move_pose(approach_pose),
            self.move_pose(release_pose),
            self.open_gripper(),
        ]
        if retreat_pose:
            steps.append(self.move_pose(retreat_pose))
        success = all(res.get("success", True) if isinstance(res, dict) else True for res in steps)
        return {"success": success, "steps": steps}

    # ------------------------------------------------------------------
    # SO-101 specific helpers
    # ------------------------------------------------------------------

    SO101_JOINT_LABELS: Dict[int, str] = {
        1: "base_rotation",
        2: "shoulder_lift",
        3: "elbow",
        4: "wrist_pitch",
        5: "wrist_yaw",
        6: "gripper",
    }

    def _configure_joint_aliases(self) -> None:
        controller = self.joints
        for index, label in self.SO101_JOINT_LABELS.items():
            controller.register_alias(label, index=index)
        controller.register_alias("joint6", index=6)
        controller.register_alias("gripper", index=6)

    def joint_labels(self) -> Dict[int, str]:
        """Return the human-friendly joint labels for SO-101."""
        return dict(self.SO101_JOINT_LABELS)

    def resolve_joint(self, identifier: Any) -> float:
        """Fetch a joint position by alias, backend name, or index."""
        return self.joints.get(identifier)

    def set_joint(self, identifier: Any, value: float) -> None:
        """Update a joint by alias, backend name, or index."""
        self.joints.set(identifier, value)

    def grip_percent(self, percent: float) -> float:
        """Set the gripper opening in percent (0=open, 100=closed)."""
        clamped = max(0.0, min(100.0, float(percent)))
        self.joints.set("gripper", clamped)
        return clamped

    def joints_state(self) -> List[Dict[str, Any]]:
        """Return a list summarising joint index, alias and position."""
        return self.joints.describe()


_DYNAMIC_TWIN_CACHE: Dict[Tuple[str, Tuple[Type[CompactTwin], ...]], Type[CompactTwin]] = {}

CAPABILITY_MIXIN_REGISTRY: Dict[str, Tuple[Type[CompactTwin], ...]] = {
    "manipulation": (ManipulationCapabilities,),
    "gripper": (GripperCapabilities,),
    "teleoperation": (TeleoperationCapabilities,),
    "safety": (SafetyCapabilities,),
    "telemetry": (TelemetryCapabilities,),
    "flight": (FlightCapabilities,),
    "video_streaming": (CameraStreamingCapabilities,),
    "camera": (CameraStreamingCapabilities,),
    "perception": (CameraStreamingCapabilities,),
    "mobility": (QuadrupedMobilityCapabilities,),
    "navigation": (QuadrupedMobilityCapabilities,),
}

CATEGORY_MIXIN_REGISTRY: Dict[str, Tuple[Type[CompactTwin], ...]] = {
    "robotic_arm": (
        ManipulationCapabilities,
        GripperCapabilities,
        TelemetryCapabilities,
    ),
    "drone": (
        FlightCapabilities,
        CameraStreamingCapabilities,
        TelemetryCapabilities,
    ),
    "quadruped": (
        QuadrupedMobilityCapabilities,
        TelemetryCapabilities,
    ),
}


def mixins_for_spec(spec: Any) -> List[Type[CompactTwin]]:
    """Return an ordered list of mixins matching a device specification."""

    mixins: List[Type[CompactTwin]] = []
    category = getattr(spec, "category", "") or ""
    for cls in CATEGORY_MIXIN_REGISTRY.get(category, ()):  # type: ignore[arg-type]
        mixins.append(cls)

    capabilities = getattr(spec, "capabilities", None) or []
    for capability in capabilities:
        name = getattr(capability, "name", "") or ""
        for mixin in CAPABILITY_MIXIN_REGISTRY.get(name.lower(), ()):  # type: ignore[arg-type]
            mixins.append(mixin)

    # Always ensure telemetry at least once
    if TelemetryCapabilities not in mixins:
        mixins.append(TelemetryCapabilities)

    # Deduplicate preserving order
    ordered: List[Type[CompactTwin]] = []
    seen: set[Type[CompactTwin]] = set()
    for mixin in mixins:
        if mixin not in seen:
            ordered.append(mixin)
            seen.add(mixin)
    return ordered


def compose_dynamic_twin_class(spec: Any, base_cls: Type[CompactTwin]) -> Type[CompactTwin]:
    """Compose a twin subclass using mixins inferred from a spec."""

    mixins = mixins_for_spec(spec)
    if not mixins:
        return base_cls

    cache_key = (getattr(spec, "id", "dynamic"), tuple(mixins + [base_cls]))
    cached = _DYNAMIC_TWIN_CACHE.get(cache_key)
    if cached:
        return cached

    dynamic = type(
        f"{getattr(spec, 'model', getattr(spec, 'name', 'Dynamic'))}Twin",
        tuple(mixins) + (base_cls,),
        {},
    )
    _DYNAMIC_TWIN_CACHE[cache_key] = dynamic
    return dynamic


__all__ = [
    "ManipulationCapabilities",
    "GripperCapabilities",
    "TeleoperationCapabilities",
    "SafetyCapabilities",
    "TelemetryCapabilities",
    "FlightCapabilities",
    "CameraStreamingCapabilities",
    "QuadrupedMobilityCapabilities",
    "SO101Twin",
    "CAPABILITY_MIXIN_REGISTRY",
    "CATEGORY_MIXIN_REGISTRY",
    "mixins_for_spec",
    "compose_dynamic_twin_class",
]
