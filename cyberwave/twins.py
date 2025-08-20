from __future__ import annotations

from typing import Any, Dict, Optional, List

from .http import HttpClient


class TwinsAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def command(self, twin_uuid: str, name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._h.post(f"/api/v1/twins/{twin_uuid}/commands", {"name": name, "payload": payload or {}})

    # New: move twin pose (partial state)
    def set_state(self, twin_uuid: str, *, position: Optional[list[float]] = None, rotation: Optional[list[float]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if position is not None:
            if len(position) != 3:
                raise ValueError("position must be [x,y,z]")
            body.update({
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "position_z": float(position[2]),
            })
        if rotation is not None:
            if len(rotation) != 4:
                raise ValueError("rotation must be [w,x,y,z]")
            body.update({
                "rotation_w": float(rotation[0]),
                "rotation_x": float(rotation[1]),
                "rotation_y": float(rotation[2]),
                "rotation_z": float(rotation[3]),
            })
        # Use PATCH state endpoint for partial updates
        return self._h.patch(f"/api/v1/twins/{twin_uuid}/state", body)

    # New: update one joint (normalized position in [-100, 100])
    def set_joint(self, twin_uuid: str, joint_name: str, position: float) -> Dict[str, Any]:
        return self._h.put(f"/api/v1/twins/{twin_uuid}/joints/{joint_name}/state", {"position": float(position)})

    # New: bulk joints update { name: {position: float} }
    def set_joints(self, twin_uuid: str, joint_states: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        return self._h.put(f"/api/v1/twins/{twin_uuid}/joint_states", {"joint_states": joint_states})

    # New: discover joints/limits
    def get_kinematics(self, twin_uuid: str) -> Dict[str, Any]:
        return self._h.get(f"/api/v1/twins/{twin_uuid}/kinematics")

    # Minimal ergonomic wrapper
    def as_robotic_arm(self, twin_uuid: str) -> "RoboticArmTwin":
        return RoboticArmTwin(self, twin_uuid)


class RoboticArmTwin:
    """Thin helper around a twin that behaves like a robotic arm.

    Methods normalize units and delegate to TwinsAPI.
    """

    def __init__(self, api: TwinsAPI, twin_uuid: str):
        self._api = api
        self.uuid = twin_uuid

    # Discovery
    def list_joints(self) -> List[str]:
        kin = self._api.get_kinematics(self.uuid) or {}
        joints = kin.get("joints") or []
        return [j.get("name") for j in joints if j.get("name")]

    def get_limits(self, joint_name: str) -> Dict[str, Any]:
        kin = self._api.get_kinematics(self.uuid) or {}
        for j in kin.get("joints") or []:
            if j.get("name") == joint_name:
                return j.get("limits") or j.get("limit") or {}
        return {}

    # Motion
    def move_joint(self, joint_name: str, position: float) -> Dict[str, Any]:
        return self._api.set_joint(self.uuid, joint_name, position)

    def move_joints(self, joint_positions: Dict[str, float]) -> Dict[str, Any]:
        payload = {name: {"position": float(pos)} for name, pos in joint_positions.items()}
        return self._api.set_joints(self.uuid, payload)

    def move_pose(self, *, position: Optional[List[float]] = None, rotation: Optional[List[float]] = None) -> Dict[str, Any]:
        return self._api.set_state(self.uuid, position=position, rotation=rotation)


