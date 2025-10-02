from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class AgentsAPI:
    """Unified software-defined agents management API"""
    
    def __init__(self, http: AsyncHttpClient):
        self._h = http

    async def get_templates(
        self,
        category: Optional[str] = None,
        context_scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get available agent templates"""
        params = {}
        if category:
            params["category"] = category
        if context_scope:
            params["context_scope"] = context_scope
            
        return await self._h.get("agents/templates", params=params)

    async def list(
        self,
        agent_category: Optional[str] = None,
        context_scope: Optional[str] = None,
        is_active: Optional[bool] = None,
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List agents with optional filters"""
        params = {}
        if agent_category:
            params["agent_category"] = agent_category
        if context_scope:
            params["context_scope"] = context_scope
        if is_active is not None:
            params["is_active"] = "true" if is_active else "false"
        if project_id:
            params["project_id"] = project_id
        if environment_id:
            params["environment_id"] = environment_id
        if device_id:
            params["device_id"] = device_id
            
        return await self._h.get("agents/", params=params)

    async def get(self, agent_id: str) -> Dict[str, Any]:
        """Get agent by ID"""
        return await self._h.get(f"agents/{agent_id}")

    async def create(
        self,
        name: str,
        agent_template: str,
        context_scope: str = "global",
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        device_id: Optional[str] = None,
        description: str = "",
        is_active: bool = False,
        priority: int = 0,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new agent from template"""
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "agent_template": agent_template,
            "context_scope": context_scope,
            "is_active": is_active,
            "priority": priority,
            "config_overrides": config_overrides or {}
        }
        
        if project_id:
            payload["project_id"] = project_id
        if environment_id:
            payload["environment_id"] = environment_id
        if device_id:
            payload["device_id"] = device_id
            
        return await self._h.post("agents/", payload)

    async def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        priority: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update agent"""
        payload: Dict[str, Any] = {}
        
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if is_active is not None:
            payload["is_active"] = is_active
        if priority is not None:
            payload["priority"] = priority
        if config is not None:
            payload["config"] = config
            
        return await self._h.put(f"agents/{agent_id}", payload)

    async def delete(self, agent_id: str) -> Dict[str, Any]:
        """Delete agent"""
        return await self._h.delete(f"agents/{agent_id}")

    async def activate(self, agent_id: str) -> Dict[str, Any]:
        """Activate agent"""
        return await self._h.post(f"agents/{agent_id}/activate", {})

    async def deactivate(self, agent_id: str) -> Dict[str, Any]:
        """Deactivate agent"""
        return await self._h.post(f"agents/{agent_id}/deactivate", {})

    async def chat(
        self, 
        agent_id: str, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Chat with an interactive agent"""
        payload = {
            "message": message,
            "context": context or {}
        }
        return await self._h.post(f"agents/{agent_id}/chat", payload)

    async def test(self, agent_id: str) -> Dict[str, Any]:
        """Test agent with sample data"""
        return await self._h.post(f"agents/{agent_id}/test", {})

    async def get_overview(self) -> List[Dict[str, Any]]:
        """Get system-wide agent overview"""
        return await self._h.get("agents/overview")

    async def get_stats(self) -> Dict[str, Any]:
        """Get system-wide agent statistics"""
        return await self._h.get("agents/stats")

    # Convenience methods for creating specific agent types

    async def create_security_monitor(
        self,
        environment_id: str,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create a security monitoring agent for an environment"""
        return await self.create(
            name=name or "Security Monitor",
            agent_template="security_monitor",
            context_scope="environment",
            environment_id=environment_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_home_automation_agent(
        self,
        environment_id: str,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create a home automation agent for an environment"""
        return await self.create(
            name=name or "Home Automation Assistant",
            agent_template="home_automation",
            context_scope="environment",
            environment_id=environment_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_environment_live_assistant(
        self,
        environment_id: str,
        name: Optional[str] = None,
        is_active: bool = True,
        **settings
    ) -> Dict[str, Any]:
        """Create a live environment assistant"""
        return await self.create(
            name=name or "Live Environment Assistant",
            agent_template="environment_live",
            context_scope="environment",
            environment_id=environment_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_environment_editor_assistant(
        self,
        environment_id: str,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create an environment editor assistant"""
        return await self.create(
            name=name or "Environment Editor Assistant",
            agent_template="environment_editor",
            context_scope="environment",
            environment_id=environment_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_project_assistant(
        self,
        project_id: str,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create a project management assistant"""
        return await self.create(
            name=name or "Project Assistant",
            agent_template="project_assistant",
            context_scope="project",
            project_id=project_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_global_assistant(
        self,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create a global platform assistant"""
        return await self.create(
            name=name or "Global Assistant",
            agent_template="global_assistant",
            context_scope="global",
            is_active=is_active,
            config_overrides={"settings": settings}
        )

    async def create_analytics_agent(
        self,
        context_scope: str,
        context_id: Optional[str] = None,
        name: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create an analytics agent"""
        kwargs = {
            "name": name or "Analytics Agent",
            "agent_template": "analytics_agent",
            "context_scope": context_scope,
            "is_active": is_active,
            "config_overrides": {"settings": settings}
        }
        
        if context_scope == "project" and context_id:
            kwargs["project_id"] = context_id
        elif context_scope == "environment" and context_id:
            kwargs["environment_id"] = context_id
        elif context_scope == "device" and context_id:
            kwargs["device_id"] = context_id
            
        return await self.create(**kwargs)

    # Legacy compatibility methods (for backward compatibility with old reasoner API)

    async def create_legacy_reasoner(
        self,
        name: str,
        reasoner_type: str = "security",
        environment_id: Optional[str] = None,
        is_active: bool = False,
        **settings
    ) -> Dict[str, Any]:
        """Create agent using legacy reasoner parameters (for backward compatibility)"""
        # Map old reasoner types to new templates
        template_mapping = {
            'security': 'reasoner_security',
            'home_automation': 'reasoner_home_automation',
            'health_wellness': 'reasoner_general',
            'general_assistant': 'reasoner_general',
            'custom': 'reasoner_general'
        }
        
        template = template_mapping.get(reasoner_type, 'reasoner_general')
        
        return await self.create(
            name=name,
            agent_template=template,
            context_scope="environment" if environment_id else "global",
            environment_id=environment_id,
            is_active=is_active,
            config_overrides={"settings": settings}
        )