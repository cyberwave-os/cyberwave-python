from __future__ import annotations

from typing import Any, Dict, Optional

from .http import HttpClient


class SensorsAPI:
    def __init__(self, http: HttpClient):
        self._h = http

    def create(
        self,
        *,
        environment_uuid: str,
        name: str,
        sensor_type: str = "camera",
        description: str = "",
        twin_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "description": description,
            "sensor_type": sensor_type,
            "twin_uuid": twin_uuid,
            "environment_uuid": environment_uuid,
            "metadata": metadata or {},
        }
        return self._h.post("/api/v1/sensors", payload)

    def send_frame(self, sensor_uuid: str, frame_bytes: bytes, content_type: str = "image/jpeg") -> Dict[str, Any]:
        return self._h.post_bytes(f"/api/v1/sensors/{sensor_uuid}/video", data=frame_bytes, content_type=content_type)


