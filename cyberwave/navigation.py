"""
Navigation planning utilities for waypoint-based navigation.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence


class NavigationPlan:
    """Builder for waypoint-based navigation plans."""

    def __init__(
        self,
        *,
        plan_id: Optional[str] = None,
        name: Optional[str] = None,
        controller_policy_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._id = plan_id or _make_id("nav")
        self._name = name
        self._controller_policy_uuid = controller_policy_uuid
        self._metadata = metadata or {}
        self._waypoints: List[Dict[str, Any]] = []

    @property
    def plan_id(self) -> str:
        return self._id

    @property
    def waypoints(self) -> List[Dict[str, Any]]:
        return list(self._waypoints)

    def set_name(self, value: str) -> "NavigationPlan":
        self._name = value
        return self

    def with_controller(self, policy_uuid: str) -> "NavigationPlan":
        """Set a default navigation controller policy UUID."""
        self._controller_policy_uuid = policy_uuid
        return self

    def set_metadata(self, **metadata: Any) -> "NavigationPlan":
        self._metadata.update(metadata)
        return self

    def waypoint(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        *,
        position: Optional[Sequence[float] | Dict[str, Any]] = None,
        rotation: Optional[Sequence[float] | Dict[str, Any]] = None,
        yaw: Optional[float] = None,
        waypoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NavigationPlan":
        """Add a waypoint to the plan."""
        if position is None:
            if x is None or y is None or z is None:
                raise ValueError("waypoint requires position or x,y,z")
            position = [x, y, z]
        waypoint = _normalize_waypoint({
            "id": waypoint_id,
            "position": position,
            "rotation": rotation,
            "yaw": yaw,
            "metadata": metadata or {},
        })
        self._append_waypoint(waypoint)
        return self

    def extend(self, waypoints: Iterable[Any]) -> "NavigationPlan":
        """Add multiple waypoints to the plan."""
        for item in waypoints:
            waypoint = _normalize_waypoint(item)
            self._append_waypoint(waypoint)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the navigation plan as a dictionary."""
        return {
            "id": self._id,
            "name": self._name,
            "controller_policy_uuid": self._controller_policy_uuid,
            "waypoints": list(self._waypoints),
            "metadata": dict(self._metadata),
        }

    def to_mission(
        self,
        *,
        twin_uuid: str,
        mission_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        created_at: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Convert the plan to a mission payload."""
        mission_id = mission_id or self._id
        mission_name = name or self._name or mission_id
        created_at = created_at or datetime.utcnow().isoformat()

        plan_payload = self.build()
        plan_payload["id"] = mission_id
        plan_payload["name"] = mission_name

        payload: Dict[str, Any] = {
            "id": mission_id,
            "name": mission_name,
            "description": description or "",
            "twinUuid": twin_uuid,
            "createdAt": created_at,
            "navigation": plan_payload,
            "waypoints": plan_payload.get("waypoints", []),
        }
        if is_active is not None:
            payload["isActive"] = bool(is_active)
        return payload

    def _append_waypoint(self, waypoint: Dict[str, Any]) -> None:
        if not waypoint.get("id"):
            waypoint["id"] = _make_id("waypoint", len(self._waypoints) + 1)
        if "metadata" not in waypoint or waypoint["metadata"] is None:
            waypoint["metadata"] = {}
        self._waypoints.append(waypoint)


def _normalize_waypoint(item: Any) -> Dict[str, Any]:
    """Normalize a waypoint to a standard dictionary format."""
    if isinstance(item, dict):
        if "position" in item:
            position = item.get("position")
        elif all(k in item for k in ("x", "y", "z")):
            position = {"x": item.get("x"), "y": item.get("y"), "z": item.get("z")}
        else:
            raise ValueError("waypoint dict requires position or x,y,z")
        rotation = _normalize_rotation(item.get("rotation"), item.get("yaw"))
        waypoint = {
            "id": item.get("id"),
            "position": _normalize_position(position),
            "metadata": item.get("metadata") or {},
        }
        if rotation is not None:
            waypoint["rotation"] = rotation
        return waypoint

    if isinstance(item, (list, tuple)):
        return {
            "id": None,
            "position": _normalize_position(item),
            "metadata": {},
        }

    raise ValueError("unsupported waypoint format")


def _normalize_position(position: Sequence[float] | Dict[str, Any]) -> Dict[str, float]:
    """Normalize a position to {x, y, z} format."""
    if isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
        z = position.get("z")
    else:
        if len(position) != 3:
            raise ValueError("position must be [x,y,z]")
        x, y, z = position
    if x is None or y is None or z is None:
        raise ValueError("position requires x,y,z values")
    return {"x": float(x), "y": float(y), "z": float(z)}


def _normalize_rotation(
    rotation: Optional[Sequence[float] | Dict[str, Any]],
    yaw: Optional[float],
) -> Optional[List[float]]:
    if rotation is not None and yaw is not None:
        raise ValueError("waypoint rotation and yaw are mutually exclusive")
    if rotation is None and yaw is None:
        return None
    if rotation is not None:
        if isinstance(rotation, dict):
            w = rotation.get("w")
            x = rotation.get("x")
            y = rotation.get("y")
            z = rotation.get("z")
        else:
            if len(rotation) != 4:
                raise ValueError("rotation must be [w, x, y, z]")
            w, x, y, z = rotation
        if w is None or x is None or y is None or z is None:
            raise ValueError("rotation requires w, x, y, z values")
        return [float(w), float(x), float(y), float(z)]
    half = float(yaw) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _make_id(prefix: str, suffix: Optional[int] = None) -> str:
    """Generate a unique ID with a timestamp."""
    stamp = int(time.time() * 1000)
    if suffix is None:
        return f"{prefix}_{stamp}"
    return f"{prefix}_{stamp}_{suffix}"



