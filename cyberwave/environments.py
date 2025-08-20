from __future__ import annotations

from typing import Any, Dict, List, Optional

from .http import HttpClient


class EnvironmentHandle:
    def __init__(self, http: HttpClient, uuid: str):
        self._h = http
        self.uuid = uuid

    def twins(self) -> List[Dict[str, Any]]:
        """List twins in this environment."""
        return self._h.get(f"/api/v1/environments/{self.uuid}/twins")

    def find_twin_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for t in self.twins():
            if (t.get("name") or "").strip().lower() == name.strip().lower():
                return t
        return None


class EnvironmentsAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def get(self, uuid: str) -> EnvironmentHandle:
        # Validate existence (raises if not found)
        _ = self._h.get(f"/api/v1/environments/{uuid}")
        return EnvironmentHandle(self._h, uuid)

    def list_for_project(self, project_uuid: str) -> List[Dict[str, Any]]:
        return self._h.get(f"/api/v1/projects/{project_uuid}/environments")

    def create(self, project_uuid: str, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"name": name, "description": description, "settings": settings or {}}
        return self._h.post(f"/api/v1/projects/{project_uuid}/environments", payload)

    def get_or_create_by_name(self, project_uuid: str, name: str, description: str = "", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        name_l = name.strip().lower()
        for e in self.list_for_project(project_uuid):
            if (e.get("name") or "").strip().lower() == name_l:
                return e
        return self.create(project_uuid, name=name, description=description, settings=settings)


