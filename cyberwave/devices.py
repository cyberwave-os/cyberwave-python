from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class DevicesAPI:
    """Devices management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def list(self, device_type: Optional[str] = None, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all devices with optional filters"""
        params = {}
        if device_type:
            params["device_type"] = device_type
        if capability:
            params["capability"] = capability
        return await self._h.get("devices/", params=params)

    async def get(self, uuid: str) -> Dict[str, Any]:
        """Get device by UUID"""
        return await self._h.get(f"devices/{uuid}/")

    async def create(
        self,
        name: str,
        device_type: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        connection_string: str = "",
        connection_type: str = "network",
        serial_number: str = "",
        manufacturer: str = "",
        model: str = "",
        project_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new device"""
        payload: Dict[str, Any] = {
            "name": name,
            "device_type": device_type,
            "description": description,
            "capabilities": capabilities or [],
            "metadata": metadata or {},
            "connection_string": connection_string,
            "connection_type": connection_type,
            "serial_number": serial_number,
            "manufacturer": manufacturer,
            "model": model
        }
        
        if project_id:
            payload["project_id"] = project_id
            
        return await self._h.post("devices/", payload)

    async def update(self, uuid: str, **kwargs) -> Dict[str, Any]:
        """Update device by UUID"""
        return await self._h.patch(f"devices/{uuid}", kwargs)

    async def delete(self, uuid: str) -> None:
        """Delete device by UUID"""
        await self._h.delete(f"devices/{uuid}")

    async def get_events(self, uuid: str, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get events for a specific device"""
        params = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return await self._h.get(f"devices/{uuid}/events", params=params)

    async def send_event(self, uuid: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send event to device webhook"""
        return await self._h.post(f"devices/{uuid}/events/webhook", event_data)

    async def update_capabilities(self, uuid: str, capabilities: List[str]) -> Dict[str, Any]:
        """Update device capabilities"""
        return await self._h.patch(f"devices/{uuid}/capabilities", {"capabilities": capabilities})
