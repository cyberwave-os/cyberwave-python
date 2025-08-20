from __future__ import annotations

from typing import Any, Dict, List, Optional

from .http import HttpClient


class ProjectsAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def list(self) -> List[Dict[str, Any]]:
        return self._h.get("/api/v1/projects")

    def get(self, uuid: string) -> Dict[str, Any]:  # type: ignore[name-defined]
        return self._h.get(f"/api/v1/projects/{uuid}")

    def create(self, name: str, description: str = "", team_uuid: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": name, "description": description}
        if team_uuid:
            payload["team_uuid"] = team_uuid
        return self._h.post("/api/v1/projects", payload)

    def get_or_create_by_name(self, name: str, description: str = "", team_uuid: Optional[str] = None) -> Dict[str, Any]:
        name_l = name.strip().lower()
        for p in self.list():
            if (p.get("name") or "").strip().lower() == name_l:
                return p
        return self.create(name=name, description=description, team_uuid=team_uuid)


