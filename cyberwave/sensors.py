from __future__ import annotations

from typing import Any, Dict, Optional

from .async_http import AsyncHttpClient


class SensorsAPI:
    """Sensors management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def create(
        self,
        *,
        environment_uuid: str,
        name: str,
        sensor_type: str = "camera",
        description: str = "",
        twin_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new sensor"""
        payload = {
            "name": name,
            "description": description,
            "sensor_type": sensor_type,
            "twin_uuid": twin_uuid,
            "environment_uuid": environment_uuid,
            "metadata": metadata or {},
        }
        return await self._h.post("sensors", payload)

    async def send_frame(self, sensor_uuid: str, frame_bytes: bytes, content_type: str = "image/jpeg") -> Dict[str, Any]:
        """Send sensor frame data (Note: post_bytes not implemented in AsyncHttpClient yet)"""
        # TODO: Implement binary data upload in AsyncHttpClient
        raise NotImplementedError("Binary data upload not yet implemented in AsyncHttpClient")


