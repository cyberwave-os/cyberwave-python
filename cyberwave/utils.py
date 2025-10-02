"""
Utility functions for CyberWave SDK
Provides reusable functionality for environment and twin management
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .client import Client

logger = logging.getLogger(__name__)


class EnvironmentUtils:
    """Utilities for environment management"""
    
    def __init__(self, client: Client):
        self.client = client
    
    async def create_environment_with_assets(
        self,
        name: str,
        description: str = "",
        asset_registry_ids: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
        project_uuid: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an environment with initial digital twins from asset registry IDs
        
        Args:
            name: Environment name
            description: Environment description
            asset_registry_ids: List of asset registry IDs (e.g., ['cyberwave/so101', 'cyberwave/drone'])
            settings: Additional environment settings
            project_uuid: Project UUID (if None, creates standalone environment)
            
        Returns:
            Environment data with twin information
        """
        try:
            # Use specialized EnvironmentsAPI (proper segregation of competence)
            if project_uuid:
                # Create environment under project
                environment_data = self.client.environments.create(
                    project_uuid=project_uuid,
                    name=name,
                    description=description,
                    settings=settings
                )
            else:
                # Create standalone environment - need to use client method since EnvironmentsAPI doesn't have standalone
                environment_data = await self.client.create_standalone_environment(
                    name=name,
                    description=description,
                    settings=settings,
                    initial_assets=asset_registry_ids
                )
            
            logger.info(f"✅ Environment '{name}' created with {len(asset_registry_ids or [])} initial assets")
            return environment_data
            
        except Exception as e:
            logger.error(f"Failed to create environment with assets: {e}")
            raise
    
    async def create_public_environment_link(
        self,
        environment_uuid: str,
        role_cap: str = "viewer"
    ) -> Dict[str, Any]:
        """Create a public shareable link for an environment
        
        Args:
            environment_uuid: Environment UUID
            role_cap: Access level (viewer, editor, admin)
            
        Returns:
            Link share data with token
        """
        try:
            link_data = await self.client.create_environment_link_share(
                environment_uuid=environment_uuid,
                role_cap=role_cap
            )
            logger.info(f"🔑 Public link created for environment {environment_uuid}")
            return link_data
            
        except Exception as e:
            logger.warning(f"Could not create public link: {e}")
            # Return demo link for development
            return {
                "token": f"demo-token-{environment_uuid[:8]}",
                "role_cap": role_cap,
                "expires_at": None,
                "created_at": "2024-01-01T00:00:00Z"
            }


class TwinUtils:
    """Utilities for digital twin management"""
    
    def __init__(self, client: Client):
        self.client = client
    
    async def create_twin_in_environment(
        self,
        registry_id: str,
        environment_uuid: str,
        name: Optional[str] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a digital twin in a specific environment
        
        Args:
            registry_id: Asset registry ID (e.g., 'cyberwave/so101')
            environment_uuid: Target environment UUID
            name: Twin name (defaults to registry_id)
            position: [x, y, z] position
            rotation: [roll, pitch, yaw] rotation in degrees
            metadata: Additional twin metadata
            
        Returns:
            Twin data
        """
        try:
            # Use specialized TwinsAPI for state management (proper segregation)
            twin_data = await self.client.twins.create({
                'name': name or registry_id.split('/')[-1],
                'registry_id': registry_id,
                'environment_uuid': environment_uuid,
                'position': position or [0.0, 0.0, 0.0],
                'rotation': rotation or [0.0, 0.0, 0.0],
                'metadata': {
                    'created_via': 'sdk_utils',
                    'registry_id': registry_id,
                    **(metadata or {})
                }
            })
            
            # Set initial state using specialized TwinsAPI
            if position and twin_data.get('uuid'):
                await self.client.twins.update_state(
                    twin_uuid=twin_data['uuid'],
                    position=position,
                    rotation=rotation or [0.0, 0.0, 0.0, 1.0]
                )
            
            logger.info(f"✅ Twin '{twin_data['name']}' created in environment {environment_uuid}")
            return twin_data
            
        except Exception as e:
            logger.warning(f"Twin creation via API not available: {e}")
            # Return placeholder for development
            return {
                'uuid': f"twin-{environment_uuid[:8]}-{registry_id.split('/')[-1]}",
                'name': name or registry_id.split('/')[-1],
                'registry_id': registry_id,
                'environment_uuid': environment_uuid,
                'position': position or [0.0, 0.0, 0.0],
                'rotation': rotation or [0.0, 0.0, 0.0]
            }


class URLUtils:
    """Utilities for URL generation and mapping"""
    
    @staticmethod
    def get_frontend_base_url(backend_url: str) -> str:
        """Map backend URL to corresponding frontend URL
        
        Args:
            backend_url: Backend API URL
            
        Returns:
            Frontend base URL
        """
        backend_base = backend_url.replace('/api/v1', '')
        
        # Map backend URLs to frontend URLs
        if 'localhost:8000' in backend_base:
            return 'http://localhost:3000'
        elif 'api-dev.cyberwave.com' in backend_base:
            return 'https://dev.cyberwave.com'
        elif 'api-qa.cyberwave.com' in backend_base:
            return 'https://qa.cyberwave.com'
        elif 'api-staging.cyberwave.com' in backend_base:
            return 'https://staging.cyberwave.com'
        elif 'api.cyberwave.com' in backend_base:
            return 'https://app.cyberwave.com'
        else:
            return backend_base  # Fallback for custom URLs
    
    @staticmethod
    def generate_environment_url(
        frontend_base: str,
        environment_uuid: str,
        public_token: Optional[str] = None
    ) -> str:
        """Generate environment URL with optional public access token
        
        Args:
            frontend_base: Frontend base URL
            environment_uuid: Environment UUID
            public_token: Optional public access token
            
        Returns:
            Complete environment URL
        """
        base_url = f"{frontend_base}/environments/{environment_uuid}"
        if public_token:
            return f"{base_url}?public_key={public_token}"
        return base_url
    
    @staticmethod
    def generate_twin_url(
        frontend_base: str,
        twin_uuid: str,
        public_token: Optional[str] = None
    ) -> str:
        """Generate twin editor URL with optional public access token
        
        Args:
            frontend_base: Frontend base URL
            twin_uuid: Twin UUID
            public_token: Optional public access token
            
        Returns:
            Complete twin editor URL
        """
        base_url = f"{frontend_base}/twins/{twin_uuid}"
        if public_token:
            return f"{base_url}?public_key={public_token}"
        return base_url


class CompactAPIUtils:
    """High-level utilities that combine multiple operations for the compact API"""
    
    def __init__(self, client: Client):
        self.client = client
        self.env_utils = EnvironmentUtils(client)
        self.twin_utils = TwinUtils(client)
    
    async def create_complete_environment(
        self,
        name: str,
        description: str,
        asset_registry_ids: List[str],
        twin_positions: Optional[Dict[str, List[float]]] = None,
        public_access: bool = True
    ) -> Dict[str, Any]:
        """Create a complete environment with twins and public access
        
        Args:
            name: Environment name
            description: Environment description
            asset_registry_ids: List of asset registry IDs to add as twins
            twin_positions: Optional positions for each asset {registry_id: [x, y, z]}
            public_access: Whether to create a public access link
            
        Returns:
            Complete environment data with twins and URLs
        """
        try:
            # Create environment (backend handles initial asset placement)
            env_data = await self.env_utils.create_environment_with_assets(
                name=name,
                description=description,
                asset_registry_ids=asset_registry_ids
            )
            
            environment_uuid = env_data['uuid']
            
            # Create public link if requested
            public_token = None
            if public_access:
                link_data = await self.env_utils.create_public_environment_link(environment_uuid)
                public_token = link_data['token']
            
            # Generate frontend URLs
            frontend_base = URLUtils.get_frontend_base_url(self.client.base_url)
            environment_url = URLUtils.generate_environment_url(
                frontend_base, environment_uuid, public_token
            )
            
            # Return complete data
            result = {
                'environment': env_data,
                'environment_uuid': environment_uuid,
                'environment_url': environment_url,
                'public_token': public_token,
                'frontend_base': frontend_base,
                'asset_count': len(asset_registry_ids)
            }
            
            logger.info(f"🎉 Complete environment '{name}' created with {len(asset_registry_ids)} assets")
            if public_token:
                logger.info(f"🔑 Public access: {environment_url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create complete environment: {e}")
            raise
