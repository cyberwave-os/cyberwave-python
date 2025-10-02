from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


@dataclass
class World:
    assets: List[Dict[str, Any]] = field(default_factory=list)
    placements: List[Dict[str, Any]] = field(default_factory=list)

    def asset(self, asset_id: str, alias: str) -> "World":
        self.assets.append({"asset_id": asset_id, "alias": alias})
        return self

    def place(self, alias: str, pose: List[float]) -> "World":
        self.placements.append({"alias": alias, "pose": pose})
        return self


@dataclass
class Mission:
    key: str
    version: int = 1
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    world_setup: World = field(default_factory=World)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    workflow: Optional[Dict[str, Any]] = None

    def world(self) -> World:
        return self.world_setup

    def goal_object_in_zone(self, obj: str, zone: str, tolerance_m: float = 0.05, hold_s: float = 2.0) -> "Mission":
        self.goals.append({"type": "object_in_zone", "object": obj, "zone": zone, "tolerance_m": tolerance_m, "hold_s": hold_s})
        return self

    def goal_coverage_pct(self, target: str, zones: List[str], min_pct: float) -> "Mission":
        self.goals.append({"type": "coverage_pct", "target": target, "zones": zones, "min_pct": min_pct})
        return self

    def to_payload(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "name": self.name or self.key,
            "description": self.description,
            "parameters": self.parameters,
            "world_setup": {
                "assets": self.world_setup.assets,
                "placements": self.world_setup.placements,
            },
            "goals": self.goals,
            "workflow": self.workflow,
        }


class MissionsAPI:
    """Missions management API - exemplar of proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    def define(self, key: str, version: int = 1, name: Optional[str] = None, description: Optional[str] = None) -> Mission:
        """Define a new mission (local operation)"""
        return Mission(key=key, version=version, name=name, description=description)

    async def register(self, mission: Mission) -> Dict[str, Any]:
        """Register mission with backend"""
        return await self._h.post("missions", mission.to_payload())

    async def list(self) -> List[Dict[str, Any]]:
        """List all missions"""
        return await self._h.get("missions")

    async def get(self, key: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get mission by key and version"""
        q = f"?version={version}" if version is not None else ""
        return await self._h.get(f"missions/{key}{q}")

    async def by_key(self, key: str, version: Optional[int] = None) -> Mission:
        """Get mission object by key"""
        data = await self.get(key, version)
        m = Mission(key=data["key"], version=int(data.get("version", 1)), name=data.get("name"), description=data.get("description"))
        return m


