from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class EnvironmentHandle:
    """Handle for environment-specific operations"""
    
    def __init__(self, http: AsyncHttpClient, uuid: str):
        self._h = http
        self.uuid = uuid

    async def twins(self) -> List[Dict[str, Any]]:
        """List twins in this environment."""
        return await self._h.get(f"/environments/{self.uuid}/twins")

    async def find_twin_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find twin by name in this environment"""
        twins = await self.twins()
        for t in twins:
            if (t.get("name") or "").strip().lower() == name.strip().lower():
                return t
        return None


class EnvironmentsAPI:
    """Environments management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def get(self, uuid: str) -> EnvironmentHandle:
        """Get environment handle for operations"""
        # Validate existence (raises if not found)
        _ = await self._h.get(f"/environments/{uuid}")
        return EnvironmentHandle(self._h, uuid)

    async def list_for_project(self, project_uuid: str) -> List[Dict[str, Any]]:
        """List environments for a project"""
        return await self._h.get(f"/projects/{project_uuid}/environments")

    async def create(self, project_uuid: str, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create environment under a project"""
        payload = {"name": name, "description": description, "settings": settings or {}}
        return await self._h.post(f"/projects/{project_uuid}/environments", payload)

    async def create_standalone(self, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None, initial_assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create standalone environment (not under a project)"""
        payload = {
            "name": name, 
            "description": description, 
            "settings": settings or {}
        }
        if initial_assets:
            payload["initial_assets"] = initial_assets
        return await self._h.post("/environments", payload)

    async def create_link_share(self, environment_uuid: str, role_cap: str = "viewer") -> Dict[str, Any]:
        """Create public link share for environment"""
        try:
            return await self._h.post(
                f"/environments/{environment_uuid}/create-link",
                {"role_cap": role_cap}
            )
        except Exception:
            # Fallback for development
            return {
                "public_token": f"demo_token_{environment_uuid[:8]}",
                "share_url": f"/environments/{environment_uuid}?public_key=demo_token_{environment_uuid[:8]}"
            }

    async def get_or_create_by_name(self, project_uuid: str, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get existing environment by name or create new one"""
        name_l = name.strip().lower()
        environments = await self.list_for_project(project_uuid)
        for e in environments:
            if (e.get("name") or "").strip().lower() == name_l:
                return e
        return await self.create(project_uuid, name=name, description=description, settings=settings)


