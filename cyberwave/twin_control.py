"""Backend-dispatched twin control helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .exceptions import CyberwaveError
from .locomotion_contracts import build_locomotion_velocity_command

if TYPE_CHECKING:
    from .resources import PolicyRefPayload
    from .twin import Twin

_RelativeTranslation = List[float] | tuple[float, float, float] | Dict[str, float]


class TwinControlHandle:
    """Backend-dispatched control commands for any twin.

    The methods in this handle intentionally reuse the Control Agent dispatch
    API. SDK callers and agents therefore execute the same server-side
    contracts for navigation, policy selection, runtime routing, and live
    safety checks.
    """

    def __init__(self, twin: "Twin"):
        self._twin = twin

    def relative_move(
        self,
        relative_translation: _RelativeTranslation,
        *,
        frame: str = "body",
        mode: Optional[str] = None,
        simulation_backend: Optional[str] = None,
        confirmed: Optional[bool] = None,
        sync: bool = False,
        timeout: float = 120.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Move in a relative body/world frame through backend navigation."""
        translation = self._normalize_translation(relative_translation)
        action = {
            "kind": "navigation_command",
            "target_twin_uuid": self._twin.uuid,
            "payload": {
                "command": "relative_move",
                "relative_translation": translation,
                "frame": frame,
                "metadata": self._movement_metadata(metadata),
            },
        }
        return self._dispatch(
            action,
            mode=mode,
            simulation_backend=simulation_backend,
            confirmed=confirmed,
            sync=sync,
            timeout=timeout,
        )

    move_relative = relative_move

    def forward(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move forward in the twin body frame."""
        return self.relative_move([float(distance), 0.0, 0.0], **kwargs)

    move_forward = forward

    def backward(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move backward in the twin body frame."""
        return self.relative_move([-float(distance), 0.0, 0.0], **kwargs)

    move_backward = backward

    def left(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move left in the twin body frame."""
        return self.relative_move([0.0, float(distance), 0.0], **kwargs)

    strafe_left = left

    def right(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move right in the twin body frame."""
        return self.relative_move([0.0, -float(distance), 0.0], **kwargs)

    strafe_right = right

    def up(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move up in the twin body frame."""
        return self.relative_move([0.0, 0.0, float(distance)], **kwargs)

    ascend = up

    def down(self, distance: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Move down in the twin body frame."""
        return self.relative_move([0.0, 0.0, -float(distance)], **kwargs)

    descend = down

    def stop(
        self,
        *,
        mode: Optional[str] = None,
        simulation_backend: Optional[str] = None,
        confirmed: Optional[bool] = None,
        sync: bool = False,
        timeout: float = 120.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Stop backend navigation for this twin."""
        action = {
            "kind": "stop_navigation",
            "target_twin_uuid": self._twin.uuid,
            "payload": {"metadata": self._movement_metadata(metadata)},
        }
        return self._dispatch(
            action,
            mode=mode,
            simulation_backend=simulation_backend,
            confirmed=confirmed,
            sync=sync,
            timeout=timeout,
        )

    def set_velocity(
        self,
        linear_x: Optional[float] = None,
        *,
        linear: Optional[float] = None,
        linear_y: float = 0.0,
        angular_z: Optional[float] = None,
        angular: Optional[float] = None,
        duration_ms: int = 500,
        gait: str = "walk",
        origin: str = "teleop",
        mode: Optional[str] = None,
        simulation_backend: Optional[str] = None,
        policy_ref: Optional["PolicyRefPayload"] = None,
        controller_policy_uuid: Optional[str] = None,
        confirmed: Optional[bool] = None,
        sync: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Dispatch a backend-owned locomotion velocity policy."""
        dispatch_mode = self._normalize_dispatch_mode(mode)
        runtime_kind = "physical" if dispatch_mode == "live" else "simulation"
        linear_value = linear_x if linear_x is not None else linear
        if linear_value is None:
            linear_value = 0.0
        angular_value = angular_z if angular_z is not None else angular
        if angular_value is None:
            angular_value = 0.0
        command = build_locomotion_velocity_command(
            linear_x=linear_value,
            linear_y=linear_y,
            angular_z=angular_value,
            duration_ms=duration_ms,
            gait=gait,
            origin=origin,
        ).to_payload()
        payload: dict[str, Any] = {
            "velocity_command": command,
            "runtime_kind": runtime_kind,
        }
        if runtime_kind == "simulation" and simulation_backend:
            payload["simulation_backend"] = simulation_backend
        if policy_ref:
            payload["policy_ref"] = dict(policy_ref)
        if controller_policy_uuid:
            payload["controller_policy_uuid"] = controller_policy_uuid

        action: dict[str, Any] = {
            "kind": "controller_policy_execute",
            "target_twin_uuid": self._twin.uuid,
            "payload": payload,
        }
        if policy_ref:
            action["policy_ref"] = dict(policy_ref)
        if controller_policy_uuid:
            action["controller_policy_uuid"] = controller_policy_uuid

        return self._dispatch(
            action,
            mode=dispatch_mode,
            simulation_backend=simulation_backend
            if runtime_kind == "simulation"
            else None,
            confirmed=confirmed,
            sync=sync,
            timeout=timeout,
        )

    move_velocity = set_velocity

    def drive_forward(
        self,
        speed: float = 0.25,
        *,
        duration_ms: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Drive forward using a resolved velocity policy."""
        return self.set_velocity(linear=float(speed), duration_ms=duration_ms, **kwargs)

    def drive_backward(
        self,
        speed: float = 0.25,
        *,
        duration_ms: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Drive backward using a resolved velocity policy."""
        return self.set_velocity(
            linear=-float(speed), duration_ms=duration_ms, **kwargs
        )

    def turn_left(
        self,
        angular: float = 0.5,
        *,
        duration_ms: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Turn left using a resolved velocity policy."""
        return self.set_velocity(
            angular=float(angular),
            duration_ms=duration_ms,
            **kwargs,
        )

    def turn_right(
        self,
        angular: float = 0.5,
        *,
        duration_ms: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Turn right using a resolved velocity policy."""
        return self.set_velocity(
            angular=-float(angular),
            duration_ms=duration_ms,
            **kwargs,
        )

    def stop_velocity(self, **kwargs: Any) -> dict[str, Any]:
        """Send a zero velocity command through the resolved policy."""
        kwargs["duration_ms"] = 0
        return self.set_velocity(linear=0.0, angular=0.0, gait="stand", **kwargs)

    def _dispatch(
        self,
        action: dict[str, Any],
        *,
        mode: Optional[str],
        simulation_backend: Optional[str],
        confirmed: Optional[bool],
        sync: bool,
        timeout: float,
    ) -> dict[str, Any]:
        environment_uuid = self._environment_uuid()
        dispatch_mode = self._normalize_dispatch_mode(mode)
        control = self._control_client()
        response = control.dispatch(
            environment_uuid,
            action,
            mode=dispatch_mode,
            simulation_backend=simulation_backend,
            confirmed=confirmed if confirmed is not None else dispatch_mode == "live",
        )
        if sync and isinstance(response, dict) and response.get("action_id"):
            actions = getattr(self._twin.client, "actions", None)
            if actions is None or not hasattr(actions, "wait"):
                raise CyberwaveError("sync=True requires a client actions.wait helper.")
            return actions.wait(
                response["action_id"],
                twin_uuid=self._twin.uuid,
                timeout=timeout,
            )
        return response

    def _environment_uuid(self) -> str:
        data = self._twin._data
        raw = getattr(data, "environment_uuid", None)
        if raw is None and isinstance(data, dict):
            raw = data.get("environment_uuid")
        if not raw:
            raw = getattr(
                getattr(self._twin.client, "config", None),
                "environment_id",
                "",
            )
        environment_uuid = str(raw or "")
        if not environment_uuid:
            raise CyberwaveError(
                "Backend control commands require the twin environment_uuid or "
                "a client environment_id."
            )
        return environment_uuid

    def _control_client(self) -> Any:
        control = getattr(self._twin.client, "control", None)
        if control is None or not hasattr(control, "dispatch"):
            raise CyberwaveError(
                "Backend control commands require a client with cw.control.dispatch()."
            )
        return control

    def _normalize_dispatch_mode(self, mode: Optional[str]) -> str:
        raw = (
            str(
                mode
                or getattr(
                    getattr(self._twin.client, "config", None),
                    "runtime_mode",
                    "live",
                )
            )
            .strip()
            .lower()
        )
        if raw in {"simulation", "sim", "sim_tele"}:
            return "simulation"
        if raw in {"live", "real-world", "real", "tele", "teleoperation"}:
            return "live"
        if raw in {"preview", "playground", "kinematic"}:
            return "preview"
        raise CyberwaveError(
            f"Unknown control mode '{mode}'. Use 'live', 'simulation', or 'preview'."
        )

    def _normalize_translation(
        self,
        relative_translation: _RelativeTranslation,
    ) -> List[float]:
        if isinstance(relative_translation, dict):
            values = [
                relative_translation.get("x", 0.0),
                relative_translation.get("y", 0.0),
                relative_translation.get("z", 0.0),
            ]
        else:
            values = list(relative_translation)
        if len(values) != 3:
            raise CyberwaveError("relative_translation must contain exactly 3 values.")
        return [float(values[0]), float(values[1]), float(values[2])]

    def _movement_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"units": "meters", "source": "sdk_simple_command"}
        if metadata:
            merged.update(metadata)
        return merged
