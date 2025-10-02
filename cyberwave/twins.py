from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Optional, List, Union, Awaitable, Sequence, Tuple, TYPE_CHECKING, Mapping

from .async_http import AsyncHttpClient
from .http import HttpClient


if TYPE_CHECKING:
    from .assets_api import AssetsAPI


class TwinsAPI:
    """Twins management API following proper segregation of competence"""

    def __init__(
        self,
        http: AsyncHttpClient,
        sync_http: Optional[HttpClient] = None,
        assets_api: Optional["AssetsAPI"] = None,
    ):
        self._h = http
        self._sync_http = sync_http
        self._assets = assets_api

    @staticmethod
    def _with_leading_slash(path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    @staticmethod
    def _loop_running() -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        return loop.is_running()

    @staticmethod
    def _normalize_position(position: Union[Sequence[float], Mapping[str, Any]]) -> Tuple[float, float, float]:
        if isinstance(position, Mapping):
            x = position.get("x", position.get("position_x"))
            y = position.get("y", position.get("position_y"))
            z = position.get("z", position.get("position_z"))
            if x is None or y is None or z is None:
                raise ValueError("position mapping must include x, y, and z components")
            return float(x), float(y), float(z)

        if len(position) != 3:
            raise ValueError("position must be [x, y, z]")
        return float(position[0]), float(position[1]), float(position[2])

    @staticmethod
    def _normalize_rotation(
        rotation: Union[Sequence[float], Dict[str, float]],
    ) -> Tuple[float, float, float, float]:
        if isinstance(rotation, dict):
            return (
                float(rotation.get("w", rotation.get("rotation_w", 1.0))),
                float(rotation.get("x", rotation.get("rotation_x", 0.0))),
                float(rotation.get("y", rotation.get("rotation_y", 0.0))),
                float(rotation.get("z", rotation.get("rotation_z", 0.0))),
            )

        if len(rotation) == 4:
            w, x, y, z = rotation
            return float(w), float(x), float(y), float(z)

        if len(rotation) == 3:
            roll, pitch, yaw = [math.radians(float(angle)) for angle in rotation]
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return w, x, y, z

        raise ValueError("rotation must be a quaternion [w,x,y,z] or Euler angles [roll,pitch,yaw]")

    async def _resolve_asset_uuid(self, body: Dict[str, Any]) -> None:
        asset_uuid = body.get("asset_uuid")
        if asset_uuid:
            return

        registry_id = body.pop("registry_id", None)
        if not registry_id:
            registry_id = body.pop("asset_registry_id", None)

        if not registry_id:
            return

        asset: Optional[Dict[str, Any]] = None
        if self._assets:
            asset = await self._assets.find_by_registry_id(registry_id)
        else:
            try:
                result = await self._h.get("assets", params={"registry_id": registry_id})
                if isinstance(result, list) and result:
                    asset = result[0]
                elif isinstance(result, dict):
                    items = result.get("results")
                    if isinstance(items, list) and items:
                        asset = items[0]
            except Exception:
                asset = None

        if not asset:
            raise ValueError(f"Asset with registry_id '{registry_id}' not found.")

        body["asset_uuid"] = asset.get("uuid")

    async def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a twin from a high-level payload."""

        body = dict(payload or {})

        await self._resolve_asset_uuid(body)

        position = body.pop("position", None)
        rotation = body.pop("rotation", None)

        if position is not None:
            px, py, pz = self._normalize_position(position)
            body.update({
                "position_x": px,
                "position_y": py,
                "position_z": pz,
            })

        if rotation is not None:
            rw, rx, ry, rz = self._normalize_rotation(rotation)
            body.update({
                "rotation_w": rw,
                "rotation_x": rx,
                "rotation_y": ry,
                "rotation_z": rz,
            })

        return await self._h.post("twins", body)

    async def _command_async(self, twin_uuid: str, name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send command to twin"""
        return await self._h.post(f"twins/{twin_uuid}/commands", {"name": name, "payload": payload or {}})

    def command_sync(self, twin_uuid: str, name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"name": name, "payload": payload or {}}
        if self._sync_http:
            return self._sync_http.post(self._with_leading_slash(f"twins/{twin_uuid}/commands"), json=body)
        return asyncio.run(self._command_async(twin_uuid, name, payload))

    def command_dispatch(self, twin_uuid: str, name: str, payload: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        if self._sync_http and not self._loop_running():
            return self.command_sync(twin_uuid, name, payload)
        return self._command_async(twin_uuid, name, payload)

    def command(self, twin_uuid: str, name: str, payload: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """Send a command to a twin.

        Returns the response directly for synchronous callers while
        preserving awaitable behaviour for async contexts.
        """
        return self.command_dispatch(twin_uuid, name, payload)

    async def update_state(
        self,
        twin_uuid: str,
        *,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Patch the pose of an existing twin."""

        body: Dict[str, Any] = {}

        if position is not None:
            px, py, pz = self._normalize_position(position)
            body.update({
                "position_x": px,
                "position_y": py,
                "position_z": pz,
            })

        if rotation is not None:
            rw, rx, ry, rz = self._normalize_rotation(rotation)
            body.update({
                "rotation_w": rw,
                "rotation_x": rx,
                "rotation_y": ry,
                "rotation_z": rz,
            })

        return await self._h.patch(f"twins/{twin_uuid}/state", body)

    async def set_state(
        self,
        twin_uuid: str,
        *,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Update twin pose (partial state)."""

        return await self.update_state(twin_uuid, position=position, rotation=rotation)

    async def set_joint(self, twin_uuid: str, joint_name: str, position: float) -> Dict[str, Any]:
        """Update one joint (normalized position in [-100, 100])"""
        return await self._h.put(f"twins/{twin_uuid}/joints/{joint_name}/state", {"position": float(position)})

    async def set_joints(self, twin_uuid: str, joint_states: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Bulk joints update { name: {position: float} }"""
        return await self._h.put(f"twins/{twin_uuid}/joint_states", {"joint_states": joint_states})

    async def get_joint_states(self, twin_uuid: str) -> Dict[str, Any]:
        """Return ROS-style joint states (name + position arrays)."""
        return await self._h.get(f"twins/{twin_uuid}/joint_states")

    async def get_kinematics(self, twin_uuid: str) -> Dict[str, Any]:
        """Discover joints/limits"""
        return await self._h.get(f"twins/{twin_uuid}/kinematics")

    # Minimal ergonomic wrapper
    def as_robotic_arm(self, twin_uuid: str) -> "RoboticArmTwin":
        return RoboticArmTwin(self, twin_uuid)

    # Ergonomic handle that exposes sync-like helpers
    def get(self, twin_uuid: str) -> "TwinHandle":
        """Return a lightweight twin handle with convenient sync helpers.

        Example:
            my_twin = client.twins.get(twin_uuid="...")
            my_twin.set_joint("5", 25)
        """
        return TwinHandle(self, twin_uuid)


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


class TwinHandle:
    """Lightweight convenience wrapper offering synchronous twin operations.

    Designed for simple scripts where `await` is undesirable. Methods block
    using `asyncio.run` when no loop is running.
    """

    def __init__(self, api: TwinsAPI, twin_uuid: str):
        self._api = api
        self.uuid = twin_uuid

    # Pose helpers
    def set_state(
        self,
        *,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        if self._api._loop_running():
            # In a running loop, return an awaitable for advanced users
            return asyncio.get_event_loop().create_task(
                self._api.set_state(self.uuid, position=position, rotation=rotation)
            )  # type: ignore[return-value]
        return asyncio.run(self._api.set_state(self.uuid, position=position, rotation=rotation))

    # Joint helpers
    def set_joint(self, joint_name: str, position: float) -> Dict[str, Any]:
        if self._api._loop_running():
            return asyncio.get_event_loop().create_task(
                self._api.set_joint(self.uuid, joint_name, position)
            )  # type: ignore[return-value]
        return asyncio.run(self._api.set_joint(self.uuid, joint_name, position))

    def set_joints(self, joint_states: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        if self._api._loop_running():
            return asyncio.get_event_loop().create_task(
                self._api.set_joints(self.uuid, joint_states)
            )  # type: ignore[return-value]
        return asyncio.run(self._api.set_joints(self.uuid, joint_states))

    # Discovery helpers
    def get_joint_states(self) -> Dict[str, Any]:
        if self._api._loop_running():
            return asyncio.get_event_loop().create_task(
                self._api.get_joint_states(self.uuid)
            )  # type: ignore[return-value]
        return asyncio.run(self._api.get_joint_states(self.uuid))

    def get_kinematics(self) -> Dict[str, Any]:
        if self._api._loop_running():
            return asyncio.get_event_loop().create_task(
                self._api.get_kinematics(self.uuid)
            )  # type: ignore[return-value]
        return asyncio.run(self._api.get_kinematics(self.uuid))
