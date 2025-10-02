from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class ProjectsAPI:
    """Projects management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def list(self) -> List[Dict[str, Any]]:
        """List all projects"""
        return await self._h.get("projects")

    async def get(self, uuid: str) -> Dict[str, Any]:
        """Get project by UUID"""
        return await self._h.get(f"projects/{uuid}")

    async def create(self, name: str, description: str = "", team_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Create a new project"""
        payload: Dict[str, Any] = {"name": name, "description": description}
        if team_uuid:
            payload["team_uuid"] = team_uuid
        return await self._h.post("projects", payload)

    async def get_or_create_by_name(self, name: str, description: str = "", team_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Get existing project by name or create new one"""
        name_l = name.strip().lower()
        projects = await self.list()
        for p in projects:
            if (p.get("name") or "").strip().lower() == name_l:
                return p
        return await self.create(name=name, description=description, team_uuid=team_uuid)


