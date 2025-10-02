from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class NodesAPI:
    """Nodes management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def list(self) -> List[Dict[str, Any]]:
        """List all nodes"""
        response = await self._h.get("nodes")
        return response.get("nodes", [])

    async def get(self, uuid: str) -> Dict[str, Any]:
        """Get node by UUID"""
        return await self._h.get(f"nodes/{uuid}")

    async def create(
        self,
        node_id: str,
        name: str,
        hostname: str,
        platform: str,
        capabilities: List[str],
        metadata: Dict[str, Any],
        description: str = "",
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new node"""
        payload: Dict[str, Any] = {
            "node_id": node_id,
            "name": name,
            "hostname": hostname,
            "platform": platform,
            "capabilities": capabilities,
            "metadata": metadata,
            "description": description,
            "status": "online"
        }
        
        if project_id:
            payload["project_id"] = project_id
        if environment_id:
            payload["environment_id"] = environment_id
            
        return await self._h.post("nodes", payload)

    async def update(self, uuid: str, **kwargs) -> Dict[str, Any]:
        """Update node by UUID"""
        return await self._h.patch(f"nodes/{uuid}", kwargs)

    async def delete(self, uuid: str) -> None:
        """Delete node by UUID"""
        await self._h.delete(f"nodes/{uuid}")

    async def get_or_create_by_node_id(
        self,
        node_id: str,
        name: str,
        hostname: str,
        platform: str,
        capabilities: List[str],
        metadata: Dict[str, Any],
        description: str = "",
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get existing node by node_id or create new one"""
        try:
            nodes = await self.list()
            for node in nodes:
                if node.get("name") == node_id:
                    return node
        except Exception:
            pass
            
        return await self.create(
            node_id=node_id,
            name=name,
            hostname=hostname,
            platform=platform,
            capabilities=capabilities,
            metadata=metadata,
            description=description,
            project_id=project_id,
            environment_id=environment_id,
        )

    async def register_model_runtime(
        self,
        node_uuid: str,
        *,
        runtime_name: str,
        runtime_type: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Register a model runtime with a node."""

        payload = {
            "runtime_name": runtime_name,
            "runtime_type": runtime_type,
            "config": config or {},
            "enabled": enabled,
        }
        response = await self._h.post(f"nodes/{node_uuid}/model-runtimes", payload)
        if isinstance(response, dict):
            return response.get("model_runtime", response)
        return response

    async def update_model_runtime(
        self,
        node_uuid: str,
        runtime_name: str,
        *,
        updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update an existing model runtime on a node."""

        response = await self._h.put(
            f"nodes/{node_uuid}/model-runtimes/{runtime_name}", updates or {}
        )
        if isinstance(response, dict):
            return response.get("model_runtime", response)
        return response

    async def unregister_model_runtime(self, node_uuid: str, runtime_name: str) -> Dict[str, Any]:
        """Remove a model runtime registration from a node."""

        response = await self._h.delete(f"nodes/{node_uuid}/model-runtimes/{runtime_name}")
        if isinstance(response, dict):
            return response.get("removed_model_runtime", response)
        return response

    async def list_backend_model_runtimes(self) -> List[Dict[str, Any]]:
        """List backend-available ML model runtimes."""

        payload = await self._h.get("nodes/backend-model-runtimes")
        if isinstance(payload, dict):
            runtimes = payload.get("backend_model_runtimes")
            if runtimes is None:
                runtimes = payload.get("model_runtimes")
            if isinstance(runtimes, list):
                return runtimes
        return []

    async def assign_backend_model_runtime(
        self,
        runtime_name: str,
        assignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assign a backend model runtime to process specific nodes or sensors."""

        return await self._h.post(
            f"nodes/backend-model-runtimes/{runtime_name}/assign", assignment
        )
