from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class WorkflowsAPI:
    """Event workflows management API following proper segregation of competence"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def list(
        self, 
        device_id: Optional[str] = None,
        enabled_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filters"""
        if device_id:
            # Use device-specific endpoint
            params = {}
            if enabled_only:
                params["enabled"] = "true"
            return await self._h.get(f"devices/{device_id}/workflows", params=params)
        else:
            # Use global workflows endpoint (if available)
            params = {}
            if enabled_only:
                params["enabled"] = "true"
            return await self._h.get("workflows/", params=params)

    async def get(self, workflow_id: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get workflow by ID"""
        if device_id:
            return await self._h.get(f"devices/{device_id}/workflows/{workflow_id}")
        else:
            return await self._h.get(f"workflows/{workflow_id}/")

    async def create(
        self,
        name: str,
        device_id: str,
        trigger_type: str = "event",
        trigger_config: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        is_active: bool = True,
        cooldown_seconds: int = 60,
        max_executions_per_hour: int = 100
    ) -> Dict[str, Any]:
        """Create a new workflow for a device"""
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "is_active": is_active,
            "trigger_type": trigger_type,
            "trigger_config": trigger_config or {},
            "cooldown_seconds": cooldown_seconds,
            "max_executions_per_hour": max_executions_per_hour,
            "actions": actions or []
        }
        
        return await self._h.post(f"devices/{device_id}/workflows", payload)

    async def create_motion_workflow(
        self,
        name: str,
        device_id: str,
        event_type: str = "motion",
        confidence_threshold: float = 0.1,
        action_type: str = "call_agent",
        reasoner_type: str = "security",
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a motion detection workflow with standard configuration"""
        trigger_config = {
            "event_type": event_type,
            "confidence_threshold": confidence_threshold,
            "device_id": device_id
        }
        
        action_config = {}
        if action_type == "call_agent":
            action_config["reasoner_type"] = reasoner_type
        elif action_type == "notify_user":
            action_config["message_template"] = f"Motion detected on device {device_id}"
        elif action_type == "log_event":
            action_config["log_level"] = "info"
        
        actions = [{
            "action_type": action_type,
            "name": f"{action_type}_{name.lower().replace(' ', '_')}",
            "description": f"Auto-generated {action_type} action for {name}",
            "order": 0,
            "is_active": True,
            "config": action_config,
            "condition_config": {}
        }]
        
        return await self.create(
            name=name,
            device_id=device_id,
            trigger_type="event",
            trigger_config=trigger_config,
            actions=actions,
            description=description or f"Motion detection workflow for {name}",
            is_active=True
        )

    async def update(
        self, 
        workflow_id: str, 
        device_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Update workflow by ID"""
        return await self._h.patch(f"devices/{device_id}/workflows/{workflow_id}", kwargs)

    async def delete(self, workflow_id: str, device_id: str) -> None:
        """Delete workflow by ID"""
        await self._h.delete(f"devices/{device_id}/workflows/{workflow_id}")

    async def enable(self, workflow_id: str, device_id: str) -> Dict[str, Any]:
        """Enable a workflow"""
        return await self.update(workflow_id, device_id, is_active=True)

    async def disable(self, workflow_id: str, device_id: str) -> Dict[str, Any]:
        """Disable a workflow"""
        return await self.update(workflow_id, device_id, is_active=False)

    async def get_executions(
        self, 
        workflow_id: str, 
        device_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        params = {"limit": limit}
        return await self._h.get(f"devices/{device_id}/workflows/{workflow_id}/executions", params=params)
