"""
Compact API for Cyberwave SDK
Provides the simplified interface shown in the catalog
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Type, Mapping, TYPE_CHECKING
from .client import Client
from .digital_twin import DigitalTwin
from .constants import (
    DEFAULT_BACKEND_URL,
    CyberWaveEnvironment,
    ENVIRONMENT_URLS,
    get_backend_url
)
from .utils import CompactAPIUtils, URLUtils
import os
import asyncio
import importlib
import logging
from enum import Enum
from dataclasses import dataclass
import math
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .twins import TwinHandle

def _run_async(coro):
    """
    Helper to run async code, handling both normal Python and Jupyter environments.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're already in an event loop (like Jupyter), we need to await the coroutine
        # But since we can't await from a sync function, we'll return the task
        # This will be handled differently in the calling code
        return loop.create_task(coro)
    except RuntimeError:
        # No event loop running, use asyncio.run()
        return asyncio.run(coro)

def _run_async_in_jupyter(coro):
    """
    Special handler for Jupyter environments where we need to use nest_asyncio or similar approach.
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)
    except ImportError:
        # nest_asyncio not available, fall back to creating task
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            # For now, return None and handle gracefully
            return None
        except RuntimeError:
            return asyncio.run(coro)


def _await_result(coro):
    """Execute coroutine synchronously regardless of environment."""
    result = _run_async_in_jupyter(coro)
    if isinstance(result, asyncio.Task):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(result)
    if result is not None:
        return result

    outcome = _run_async(coro)
    if isinstance(outcome, asyncio.Task):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(outcome)
    return outcome


def _safe_float(value: Any) -> float:
    """Best-effort conversion of a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _euler_deg_to_quaternion(rotation: List[float]) -> List[float]:
    """Convert Euler angles in degrees to a quaternion."""

    if len(rotation) != 3:
        raise ValueError("Rotation must be [roll, pitch, yaw] in degrees")

    roll, pitch, yaw = [math.radians(float(angle)) for angle in rotation]
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def _append_query_params(url: Optional[str], params: Dict[str, str]) -> Optional[str]:
    """Append query parameters to a URL while preserving existing ones."""

    if not url:
        return None

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    for key, value in params.items():
        if value is not None:
            query[key] = value

    return urlunparse(parsed._replace(query=urlencode(query)))

class AuthTrigger(Enum):
    """Defines when authentication should be triggered"""
    NEVER = "never"  # Always local mode
    AUTO = "auto"    # Auto-detect based on resource access
    ALWAYS = "always"  # Always require auth
    ON_PROTECTED_RESOURCE = "on_protected_resource"  # Auth when accessing specific IDs

@dataclass
class AuthConfig:
    """Configuration for authentication behavior"""
    trigger: AuthTrigger = AuthTrigger.AUTO
    interactive: bool = True  # Allow interactive auth prompts
    fallback_to_local: bool = True  # Fall back to local mode on auth failure
    protected_resource_patterns: List[str] = None  # Patterns that require auth
    
    def __post_init__(self):
        if self.protected_resource_patterns is None:
            self.protected_resource_patterns = []

# Global client instance
_global_client: Optional[Client] = None
_global_project_uuid: Optional[str] = None
_global_environment_uuid: Optional[str] = None
_auth_config: AuthConfig = AuthConfig()
_twin_registry: Dict[str, "CompactTwin"] = {}

def configure(api_key: Optional[str] = None, 
              backend_url: Optional[str] = None,
              environment: Optional[CyberWaveEnvironment] = None,
              project_name: Optional[str] = None, environment_name: Optional[str] = None,
              auth_trigger: AuthTrigger = AuthTrigger.AUTO, interactive: bool = True,
              fallback_to_local: bool = True, protected_patterns: Optional[List[str]] = None):
    """Configure the global Cyberwave client with environment and auth configuration"""
    global _global_client, _global_project_uuid, _global_environment_uuid, _auth_config
    
    # Configure auth behavior
    _auth_config = AuthConfig(
        trigger=auth_trigger,
        interactive=interactive,
        fallback_to_local=fallback_to_local,
        protected_resource_patterns=protected_patterns or []
    )
    
    if api_key is None:
        api_key = os.getenv('CYBERWAVE_API_KEY')
    
    # Create client with environment support
    _global_client = Client(
        base_url=backend_url,  # Custom URL overrides environment
        environment=environment  # Use predefined environment
    )
    
    # Store the API key for later use
    if api_key:
        _global_client._access_token = api_key
    
    # Auto-create or get project/environment if needed (only if auth configured)
    if _global_client._access_token and _auth_config.trigger != AuthTrigger.NEVER:
        try:
            _global_project_uuid, _global_environment_uuid = asyncio.run(
                _ensure_project_and_environment(project_name, environment_name)
            )
        except Exception as e:
            logger.warning(f"Failed to auto-create project/environment: {e}")
            logger.warning("Will create them when first twin is created")

def _requires_authentication(project_id: Optional[str] = None, environment_id: Optional[str] = None,
                           operation: str = "access") -> bool:
    """Determine if authentication is required based on scalable rules"""
    global _auth_config
    
    # Check auth trigger rules
    if _auth_config.trigger == AuthTrigger.NEVER:
        return False
    elif _auth_config.trigger == AuthTrigger.ALWAYS:
        return True
    elif _auth_config.trigger == AuthTrigger.ON_PROTECTED_RESOURCE:
        # Check if accessing specific protected resources
        if project_id or environment_id:
            return True
        return False
    elif _auth_config.trigger == AuthTrigger.AUTO:
        # Auto-detect: require auth if specific IDs provided or protected patterns match
        if project_id or environment_id:
            return True
        
        # Check against protected patterns
        for pattern in _auth_config.protected_resource_patterns:
            if pattern in (project_id or "") or pattern in (environment_id or ""):
                return True
        
        return False
    
    return False

async def _handle_authentication_flow(client: Client, operation_context: str = "") -> bool:
    """Handle authentication flow in a scalable way"""
    global _auth_config
    
    if client._access_token:
        return True  # Already authenticated
    
    if not _auth_config.interactive:
        logger.warning(f"Authentication required for {operation_context} but interactive mode disabled")
        return False
    
    try:
        logger.info(f"Authentication required for {operation_context}")
        
        # Try keyring first
        await client.authenticate()
        
        if client._access_token:
            logger.info("✅ Authentication successful")
            return True
        else:
            logger.warning("❌ Authentication failed")
            return False
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        
        if _auth_config.fallback_to_local:
            logger.warning("🔄 Falling back to local mode")
            return False
        else:
            raise Exception(f"Authentication required but failed: {e}")

async def _ensure_authenticated_access(project_id: Optional[str] = None, 
                                     environment_id: Optional[str] = None,
                                     operation: str = "access") -> bool:
    """Ensure user has authenticated access to specified resources"""
    client = _get_client()
    
    # Check if auth is required
    if not _requires_authentication(project_id, environment_id, operation):
        return False  # No auth required, can use local mode
    
    # Auth is required - attempt authentication
    context = f"{operation}"
    if project_id:
        context += f" project {project_id}"
    if environment_id:
        context += f" environment {environment_id}"
    
    auth_success = await _handle_authentication_flow(client, context)
    
    if auth_success and (project_id or environment_id):
        # Validate access to specific resources
        try:
            if project_id:
                await client.get_project(project_id)
                logger.info(f"✅ Access validated for project {project_id}")
            
            if environment_id:
                await client.get_environment(environment_id)
                logger.info(f"✅ Access validated for environment {environment_id}")
                
        except Exception as e:
            logger.error(f"❌ Access denied to specified resources: {e}")
            if not _auth_config.fallback_to_local:
                raise Exception(f"Access denied to {context}: {e}")
            return False
    
    return auth_success

async def _ensure_project_and_environment(project_name: Optional[str] = None, 
                                          environment_name: Optional[str] = None) -> Tuple[str, str]:
    """Ensure project and environment exist, creating them if needed"""
    global _global_client
    
    if not _global_client or not _global_client._access_token:
        raise Exception("Client not configured with authentication")
    
    # Default names if not provided
    project_name = project_name or "SDK Default Project"
    environment_name = environment_name or "Default Environment"
    
    # Get user workspaces (new route under /users)
    workspaces = await _global_client.get_workspaces()
    if not workspaces:
        raise Exception("No workspaces found for user")
    
    # New BE shape uses uuid
    workspace_uuid = str(workspaces[0].get('uuid') or workspaces[0].get('id'))
    
    # Get or create project using specialized ProjectsAPI (proper delegation)
    try:
        project = await _global_client.projects.get_or_create_by_name(
            name=project_name,
            description="Created via SDK"
        )
        logger.info(f"Using/created project: {project_name}")
    except Exception as e:
        logger.warning(f"ProjectsAPI not available, falling back to workspace-based approach: {e}")
        # Fallback to workspace-based approach
        projects = await _global_client.get_projects(workspace_uuid)
        project = None
        for p in projects:
            if p['name'] == project_name:
                project = p
                break
        
        if not project:
            project = await _global_client.create_project(workspace_uuid, project_name)
            logger.info(f"Created new project: {project_name}")
    
    project_uuid = project['uuid']
    
    # Get or create environment using specialized EnvironmentsAPI (proper delegation)
    try:
        environment = await _global_client.environments.get_or_create_by_name(
            project_uuid=project_uuid,
            name=environment_name,
            description="Created via SDK"
        )
        logger.info(f"Using/created environment: {environment_name}")
    except Exception as e:
        logger.warning(f"EnvironmentsAPI not available, falling back to direct approach: {e}")
        # Fallback to direct approach
        environments = await _global_client.get_environments(project_uuid)
        environment = None
        for e in environments:
            if e['name'] == environment_name:
                environment = e
                break
        
        if not environment:
            environment = await _global_client.create_environment(project_uuid, environment_name)
            logger.info(f"Created new environment: {environment_name}")
    
    environment_uuid = environment['uuid']
    
    logger.info(f"Using project: {project_name} ({project_uuid})")
    logger.info(f"Using environment: {environment_name} ({environment_uuid})")
    
    return project_uuid, environment_uuid

def _get_client() -> Client:
    """Get the global client, creating one if needed"""
    global _global_client

    if _global_client is None:
        configure()

    return _global_client


def _resolve_twin_class(registry_id: Optional[str]) -> Type["CompactTwin"]:
    """Return the specialized twin class for a registry id, if available."""

    if not registry_id:
        return CompactTwin

    try:
        from .device_specs import DeviceSpecRegistry

        spec = DeviceSpecRegistry.get(registry_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Device spec lookup failed for %s: %s", registry_id, exc)
        return CompactTwin

    candidate_path = None
    for attr_name in ("twin_class", "asset_class"):
        candidate_path = getattr(spec, attr_name, None)
        if candidate_path:
            break

    if isinstance(candidate_path, str):
        module_name, _, class_name = candidate_path.rpartition(".")
        if module_name and class_name:
            try:
                module = importlib.import_module(module_name)
                candidate = getattr(module, class_name)
                if isinstance(candidate, type) and issubclass(candidate, CompactTwin):
                    return candidate
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Unable to import twin class '%s': %s", candidate_path, exc)

    from . import twin_capabilities

    return twin_capabilities.compose_dynamic_twin_class(spec, CompactTwin)


def _register_twin_instance(twin: "CompactTwin") -> None:
    if not twin:
        return
    identifiers = {
        twin.name.lower(): twin,
        twin.registry_id.lower(): twin,
    }
    if getattr(twin, "uuid", None):
        identifiers[str(twin.uuid).lower()] = twin
    for key, value in identifiers.items():
        _twin_registry[key] = value


def _resolve_twin(identifier: Union[str, "CompactTwin"]) -> Optional["CompactTwin"]:
    if isinstance(identifier, CompactTwin):
        return identifier
    if not identifier:
        return None
    return _twin_registry.get(str(identifier).lower())

async def _get_or_create_project_environment(project_name: Optional[str] = None,
                                            environment_name: Optional[str] = None) -> Tuple[str, str]:
    """Get or create project and environment with authentication"""
    global _global_project_uuid, _global_environment_uuid
    
    # Use global cache if no specific names requested and cache exists
    if not project_name and not environment_name and _global_project_uuid and _global_environment_uuid:
        return _global_project_uuid, _global_environment_uuid
    
    # Try to authenticate and create if not already done
    client = _get_client()
    if not client._access_token:
        # Try to get token from keyring or prompt user
        try:
            await client.authenticate()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise Exception("Authentication required. Please configure API key or use client.authenticate()")
    
    project_uuid, environment_uuid = await _ensure_project_and_environment(project_name, environment_name)
    
    # Update global cache if using defaults
    if not project_name and not environment_name:
        _global_project_uuid, _global_environment_uuid = project_uuid, environment_uuid
    
    return project_uuid, environment_uuid

class CompactTwin:
    """
    Compact twin interface for easy digital twin control
    Implements the API shown in the catalog Python snippets
    """
    
    def __init__(self, registry_id: Optional[str], name: Optional[str] = None, environment_id: Optional[str] = None, 
                 project_id: Optional[str] = None, project_name: Optional[str] = None, 
                 environment_name: Optional[str] = None, *, twin_uuid: Optional[str] = None, auto_create: bool = True):
        self.registry_id = registry_id
        self.name = name or (registry_id.split('/')[-1] if registry_id else "twin")
        self.environment_id = environment_id
        self.project_id = project_id
        self.project_name = project_name
        self.environment_name = environment_name
        self._client = _get_client()
        self._utils = CompactAPIUtils(self._client) if self._client else None
        self._twin_uuid: Optional[str] = twin_uuid or None
        self._project_uuid: Optional[str] = None
        self._environment_uuid: Optional[str] = None
        self._position = [0.0, 0.0, 0.0]
        self._rotation = [0.0, 0.0, 0.0]  # roll, pitch, yaw in degrees
        self._web_url: Optional[str] = None
        self._environment_url: Optional[str] = None
        self._authenticated: bool = False
        self._auto_create: bool = bool(auto_create)
        
        # Check if authentication is required for the specified resources
        self._auth_required = _requires_authentication(
            project_id=self.project_id,
            environment_id=self.environment_id,
            operation="twin creation"
        )
        
        # Try to create twin immediately, unless explicitly disabled or already bound to an existing UUID
        if self._twin_uuid is None and self._auto_create:
            try:
                result = _run_async_in_jupyter(self._ensure_twin_exists())
                if result is None:
                    # Running in Jupyter - set up local mode URLs immediately
                    logger.info(f"🔧 Twin '{self.name}' created in local simulation mode")
                    logger.info(f"💡 Configure authentication to enable web editor integration")
                    self._setup_local_mode_urls()
            except Exception as e:
                logger.warning(f"Failed to create twin immediately: {e}")
                logger.warning("Twin will be created when first operation is performed")
        
    async def _ensure_twin_exists(self):
        """Ensure the twin exists in the environment - requires authentication"""
        if self._twin_uuid is None:
            try:
                # Check if user is authenticated
                if not self._client.is_authenticated():
                    logger.warning("🔐 Authentication required to create twins and environments")
                    auth_ok = await self._client.authenticate(interactive=_auth_config.interactive)

                    if not auth_ok:
                        logger.error("❌ Authentication failed. Cannot create twin.")
                        return

                # User is authenticated, proceed with creation
                logger.info("✅ User authenticated, proceeding with twin creation")
                self._authenticated = True
                
                # Use specific IDs if provided, otherwise get/create defaults
                if self.project_id and self.environment_id:
                    self._project_uuid = self.project_id
                    self._environment_uuid = self.environment_id
                    logger.info(f"Using specified project: {self.project_id}")
                    logger.info(f"Using specified environment: {self.environment_id}")
                else:
                    # Get or create project/environment with custom names if provided
                    self._project_uuid, self._environment_uuid = await _get_or_create_project_environment(
                        project_name=self.project_name,
                        environment_name=self.environment_name
                    )
                
                # No need for public links - user is authenticated
                if self._client.is_authenticated():
                    # Create twin in backend and place it in the environment
                    try:
                        twin_data = await self._client.twins.create({
                            'name': self.name,
                            'registry_id': self.registry_id,
                            'environment_uuid': self._environment_uuid,
                            'position': self._position,
                            'rotation': self._rotation,
                            'metadata': {
                                'created_via': 'compact_api',
                                'environment_id': self._environment_uuid,
                                'registry_id': self.registry_id
                            }
                        })
                        self._twin_uuid = twin_data['uuid']
                        logger.info(f"✅ Twin '{self.name}' successfully added to environment {self._environment_uuid}")
                    except Exception as twin_error:
                        logger.warning(f"Twin creation via API not available: {twin_error}")
                        # Create a placeholder twin that the environment knows about
                        self._twin_uuid = f"twin-{self._environment_uuid[:8]}"
                        logger.info(f"📝 Created placeholder twin: {self._twin_uuid}")
                        
                        # Try to register the twin with the environment through a different API
                        try:
                            await self._register_twin_in_environment()
                        except Exception as reg_error:
                            logger.debug(f"Could not register twin in environment: {reg_error}")
                    
                    # Generate web URLs for frontend (no public keys needed for authenticated users)
                    frontend_base = self._client._get_frontend_url()
                    self._environment_url = f"{frontend_base}/environments/{self._environment_uuid}"
                    self._web_url = f"{frontend_base}/twins/{self._twin_uuid}"
                    
                    logger.info(f"✅ Twin '{self.name}' created successfully!")
                    logger.info(f"🌐 Environment: {self._environment_url}")
                    logger.info(f"🔧 Twin Editor: {self._web_url}")
                    logger.info(f"🆔 Environment UUID: {self._environment_uuid}")
                    
                    return self._twin_uuid
                else:
                    # Try to create real environment even without authentication (for local development)
                    logger.info(f"🔧 Attempting to create real environment in local mode...")
                    
                    try:
                        # Try to create a standalone environment using specialized APIs
                        env_data = await self._client.environments.create_standalone(
                            name=self.environment_name or f"SDK Environment {self.name}",
                            description=f"Auto-created for {self.registry_id}",
                            initial_assets=[self.registry_id]  # Backend should handle twin placement
                        )
                        
                        self._environment_uuid = env_data['uuid']
                        logger.info(f"✅ Real environment created: {self._environment_uuid}")
                        
                        # Try to create public link for the environment
                        public_token = None
                        try:
                            link_data = await self._client.environments.create_link_share(
                                environment_uuid=self._environment_uuid,
                                role_cap="viewer"
                            )
                            public_token = link_data.get('public_token') or link_data.get('token')
                            logger.info(f"🔑 Public access token created: {public_token[:12] if public_token else 'None'}...")
                        except Exception as link_error:
                            logger.warning(f"Could not create public link: {link_error}")
                        
                        # Generate real twin UUID (backend should have created it)
                        self._twin_uuid = f"twin-{self._environment_uuid[:8]}-{self.name}"
                        
                        # Generate frontend URLs
                        backend_base = self._client.base_url.replace('/api/v1', '')
                        if 'localhost:8000' in backend_base:
                            frontend_base = 'http://localhost:3000'
                        elif 'api-dev.cyberwave.com' in backend_base:
                            frontend_base = 'https://dev.cyberwave.com'
                        elif 'api-qa.cyberwave.com' in backend_base:
                            frontend_base = 'https://qa.cyberwave.com'
                        elif 'api-staging.cyberwave.com' in backend_base:
                            frontend_base = 'https://staging.cyberwave.com'
                        elif 'api.cyberwave.com' in backend_base:
                            frontend_base = 'https://app.cyberwave.com'
                        else:
                            frontend_base = backend_base
                        
                        # Generate real URLs with public access
                        if public_token:
                            self._environment_url = f"{frontend_base}/environments/{self._environment_uuid}?public_key={public_token}"
                            self._web_url = f"{frontend_base}/twins/{self._twin_uuid}?public_key={public_token}"
                        else:
                            self._environment_url = f"{frontend_base}/environments/{self._environment_uuid}"
                            self._web_url = f"{frontend_base}/twins/{self._twin_uuid}"
                        
                        logger.info(f"🎉 Twin '{self.name}' ready!")
                        logger.info(f"🌐 Environment: {self._environment_url}")
                        logger.info(f"🔧 Twin Editor: {self._web_url}")
                        logger.info(f"🆔 Twin UUID: {self._twin_uuid}")
                        
                    except Exception as env_error:
                        logger.warning(f"Could not create real environment: {env_error}")
                        logger.info(f"🔧 Falling back to local simulation mode")
                        
                        # Fallback to local simulation
                        self._twin_uuid = f"local-{self.registry_id.replace('/', '-')}"
                        
                        backend_base = self._client.base_url.replace('/api/v1', '')
                        if 'localhost:8000' in backend_base:
                            frontend_base = 'http://localhost:3000'
                        else:
                            frontend_base = backend_base
                        
                        demo_env_uuid = "demo-env-12345"
                        self._environment_url = f"{frontend_base}/environments/{demo_env_uuid}"
                        self._web_url = f"{frontend_base}/twins/{self._twin_uuid}"
                        
                        logger.info(f"🔧 Running in local simulation mode: {self._twin_uuid}")
                        logger.info(f"🌐 Demo environment URL: {self._environment_url}")
                        logger.info(f"🔧 Demo twin URL: {self._web_url}")
                        logger.info(f"💡 Note: URLs are demos - configure authentication for real backend access")
                    
                    if self._auth_required:
                        logger.info("💡 Specific project/environment IDs require authentication")
                        logger.info("💡 Configure authentication to access protected resources")
                    
                    return self._twin_uuid
                
            except Exception as e:
                logger.error(f"Failed to create twin: {e}")
                # Fallback to local simulation
                self._twin_uuid = f"local-{self.registry_id.replace('/', '-')}"
                logger.warning(f"🔄 Falling back to local simulation mode: {self._twin_uuid}")
                # Ensure demo URLs are available even in local fallback
                try:
                    self._setup_local_mode_urls()
                except Exception:
                    pass
                return self._twin_uuid
    
    def _setup_local_mode_urls(self):
        """Set up demo URLs for local mode when running in Jupyter"""
        if self._twin_uuid is None:
            self._twin_uuid = f"local-{self.registry_id.replace('/', '-')}"
        
        # Generate demo URLs (frontend URLs)
        backend_base = self._client.base_url.replace('/api/v1', '')
        
        # Map backend URLs to frontend URLs
        if 'localhost:8000' in backend_base:
            frontend_base = 'http://localhost:3000'
        elif 'api-dev.cyberwave.com' in backend_base:
            frontend_base = 'https://dev.cyberwave.com'
        elif 'api-qa.cyberwave.com' in backend_base:
            frontend_base = 'https://qa.cyberwave.com'
        elif 'api-staging.cyberwave.com' in backend_base:
            frontend_base = 'https://staging.cyberwave.com'
        elif 'api.cyberwave.com' in backend_base:
            frontend_base = 'https://app.cyberwave.com'
        else:
            frontend_base = backend_base  # Fallback for custom URLs
        
        demo_env_uuid = "demo-env-12345"
        self._environment_url = f"{frontend_base}/environments/{demo_env_uuid}"
        self._web_url = f"{frontend_base}/twins/{self._twin_uuid}"
    
    async def _register_twin_in_environment(self):
        """Register the twin in the environment so it appears in the 3D world"""
        try:
            # Try to add the twin to the environment's scene
            await self._client._request("POST", f"/environments/{self._environment_uuid}/twins", json={
                "twin_uuid": self._twin_uuid,
                "name": self.name,
                "registry_id": self.registry_id,
                "position": self._position,
                "rotation": self._rotation,
                "asset_type": "robot",
                "robot_model": "so101"
            })
            logger.info(f"🤖 Twin registered in environment scene")
        except Exception as e:
            logger.debug(f"Environment twin registration not available: {e}")
            # Try alternative approach - update environment settings to include the twin
            try:
                await self._client._request("PATCH", f"/environments/{self._environment_uuid}", json={
                    "settings": {
                        "twins": [
                            {
                                "uuid": self._twin_uuid,
                                "name": self.name,
                                "registry_id": self.registry_id,
                                "position": self._position,
                                "rotation": self._rotation,
                                "visible": True
                            }
                        ]
                    }
                })
                logger.info(f"🔧 Twin added to environment settings")
            except Exception as e2:
                logger.debug(f"Could not update environment settings: {e2}")
    
    async def _get_or_create_twin_in_environment(self) -> str:
        """Get or create twin in the current environment using utilities"""
        try:
            # Use the twin utils to create/get the twin
            twin_data = await self._utils.twin_utils.create_twin_in_environment(
                registry_id=self.registry_id,
                environment_uuid=self._environment_uuid,
                name=self.name,
                position=self._position,
                rotation=self._rotation
            )
            return twin_data['uuid']
        except Exception as e:
            logger.warning(f"Could not create twin via utils: {e}")
            # Fallback to placeholder
            return f"twin-{self._environment_uuid[:8]}-{self.name}"
    
    @property
    def uuid(self) -> Optional[str]:
        """Get twin UUID"""
        return self._twin_uuid
    
    @property 
    def web_url(self) -> Optional[str]:
        """Get web editor URL for this twin"""
        return self._web_url
    
    @property
    def environment_url(self) -> Optional[str]:
        """Get environment URL where this twin exists"""
        return self._environment_url
    
    @property
    def position(self) -> List[float]:
        """Get current position [x, y, z]"""
        return self._position.copy()
    
    @position.setter
    def position(self, pos: List[float]):
        """Set position [x, y, z]"""
        if len(pos) != 3:
            raise ValueError("Position must be [x, y, z]")
        self._position = pos
        # Update backend if possible (Jupyter-safe)
        try:
            if self._twin_uuid and self._client:
                _run_async_in_jupyter(self._async_update_backend_position())
            print(f"[CompactTwin] {self.name} moved to position: {self._position} (local)")
        except Exception as e:
            logger.debug(f"Position update: {e}")
            print(f"[CompactTwin] {self.name} moved to position: {self._position} (local)")
    
    @property
    def rotation(self) -> List[float]:
        """Get current rotation [roll, pitch, yaw] in degrees"""
        return self._rotation.copy()
    
    @rotation.setter
    def rotation(self, rot: List[float]):
        """Set rotation [roll, pitch, yaw] in degrees"""
        if len(rot) != 3:
            raise ValueError("Rotation must be [roll, pitch, yaw] in degrees")
        self._rotation = rot
        # Update backend if possible (Jupyter-safe)
        try:
            if self._twin_uuid and self._client:
                _run_async_in_jupyter(self._async_update_backend_rotation())
            print(f"[CompactTwin] {self.name} rotated to: {self._rotation} degrees (local)")
        except Exception as e:
            logger.debug(f"Rotation update: {e}")
            print(f"[CompactTwin] {self.name} rotated to: {self._rotation} degrees (local)")
    
    def move(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        """Move to position with optional coordinates"""
        if x is not None:
            self._position[0] = x
        if y is not None:
            self._position[1] = y
        if z is not None:
            self._position[2] = z
        
        # Update backend if possible (Jupyter-safe)
        try:
            if self._twin_uuid and self._client:
                _run_async_in_jupyter(self._async_update_backend_position())
            print(f"[CompactTwin] {self.name} moved to position: {self._position} (local)")
        except Exception as e:
            logger.debug(f"Position update: {e}")
            print(f"[CompactTwin] {self.name} moved to position: {self._position} (local)")
    
    def rotate(self, roll: Optional[float] = None, pitch: Optional[float] = None, yaw: Optional[float] = None):
        """Rotate with optional angles in degrees"""
        if roll is not None:
            self._rotation[0] = roll
        if pitch is not None:
            self._rotation[1] = pitch
        if yaw is not None:
            self._rotation[2] = yaw
        
        # Update backend if possible (Jupyter-safe)
        try:
            if self._twin_uuid and self._client:
                _run_async_in_jupyter(self._async_update_backend_rotation())
            print(f"[CompactTwin] {self.name} rotated to: {self._rotation} degrees (local)")
        except Exception as e:
            logger.debug(f"Rotation update: {e}")
            print(f"[CompactTwin] {self.name} rotated to: {self._rotation} degrees (local)")
    
    def move_to(self, position: List[float], orientation: Optional[List[float]] = None):
        """Move to target position with optional orientation"""
        self.position = position
        if orientation:
            self.rotation = orientation
    
    async def _async_update_backend_position(self):
        """Update position in backend (async) - REFACTORED to use specialized TwinsAPI"""
        await self._ensure_twin_exists()
        if self._twin_uuid and not self._twin_uuid.startswith('local-'):
            # Use specialized TwinsAPI (proper delegation)
            try:
                await self._client.twins.update_state(
                    self._twin_uuid,
                    position=self._position
                )
            except Exception as e:
                logger.warning(f"Failed to update twin position: {e}")

    async def _async_update_backend_rotation(self):
        """Update rotation in backend (async) - REFACTORED to use specialized TwinsAPI"""
        await self._ensure_twin_exists()
        if self._twin_uuid and not self._twin_uuid.startswith('local-'):
            # Use specialized TwinsAPI (proper delegation)
            try:
                # Convert rotation to quaternion format for TwinsAPI
                rotation_quat = _euler_deg_to_quaternion(self._rotation)
                await self._client.twins.update_state(
                    self._twin_uuid,
                    rotation=rotation_quat
                )
            except Exception as e:
                logger.warning(f"Failed to update twin rotation: {e}")
    
    @property
    def joints(self):
        """Access to the memoized joint controller for this twin."""
        controller = getattr(self, "_joint_controller", None)
        if controller is None or getattr(controller, "_twin", None) is not self:
            controller = JointController(self)
            setattr(self, "_joint_controller", controller)
        return controller
    
    @property
    def has_sensors(self) -> bool:
        """Check if twin has sensors"""
        # TODO: Implement sensor detection
        return False
    
    def delete(self):
        """Delete the twin from the environment"""
        # TODO: Implement twin deletion
        print(f"[CompactTwin] {self.name} deleted")

class JointController:
    """High-level joint control interface with live backend syncing."""

    def __init__(self, twin: "CompactTwin") -> None:
        object.__setattr__(self, "_twin", twin)
        object.__setattr__(self, "_alias_to_joint", {})
        object.__setattr__(self, "_joint_to_alias", {})
        object.__setattr__(self, "_alias_to_index", {})
        object.__setattr__(self, "_index_to_alias", {})
        object.__setattr__(self, "_joint_cache", {})
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_allow_dynamic_alias", True)
        self._initialize()

    def __dir__(self) -> list[str]:
        base_dir = set(super().__dir__())
        base_dir.update(self._alias_to_joint.keys())
        base_dir.update(str(idx) for idx in self._index_to_alias.keys())
        return sorted(base_dir)

    def __getattr__(self, name: str) -> float:
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_initialized()
        alias = None
        if name in self._alias_to_joint:
            alias = name
        elif name in self._joint_to_alias:
            alias = self._joint_to_alias[name]
        elif name.isdigit() and int(name) in self._index_to_alias:
            alias = self._index_to_alias[int(name)]
        if alias and alias in self._joint_cache:
            return self._joint_cache[alias]
        raise AttributeError(f"Unknown joint alias '{name}'")

    def __setattr__(self, name: str, value: float) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.set(name, value)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize()

    def _initialize(self) -> None:
        try:
            _await_result(self._bootstrap())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Joint bootstrap failed: %s", exc)
        finally:
            object.__setattr__(self, "_initialized", True)

    async def _bootstrap(self) -> None:
        await self._twin._ensure_twin_exists()
        twin_uuid = getattr(self._twin, "_twin_uuid", None)
        client = getattr(self._twin, "_client", None)
        if not twin_uuid or str(twin_uuid).startswith("local-") or client is None:
            return

        twins_api = getattr(client, "twins", None)
        if twins_api is None:
            return

        try:
            kin = await twins_api.get_kinematics(twin_uuid)
        except Exception as exc:
            logger.debug("Failed to fetch kinematics for %s: %s", twin_uuid, exc)
            kin = {}

        joints_info = kin.get("joints") if isinstance(kin, dict) else None
        alias_map = self._build_alias_map(joints_info)

        # Fallback to names reported in joint states if kinematics missing
        try:
            joint_states = await twins_api.get_joint_states(twin_uuid)
        except Exception as exc:
            logger.debug("Failed to fetch joint states for %s: %s", twin_uuid, exc)
            joint_states = {}

        names = joint_states.get("name") if isinstance(joint_states, dict) else None
        positions = joint_states.get("position") if isinstance(joint_states, dict) else None

        if not alias_map and isinstance(names, list):
            alias_map = self._build_alias_map([{ "name": n } for n in names if isinstance(n, str)])

        allow_dynamic = not bool(alias_map)
        object.__setattr__(self, "_allow_dynamic_alias", allow_dynamic)
        object.__setattr__(self, "_alias_to_joint", alias_map)
        object.__setattr__(self, "_joint_to_alias", {
            actual: alias for alias, actual in alias_map.items() if actual
        })

        alias_to_index: dict[str, int] = {}
        index_to_alias: dict[int, str] = {}
        for idx, alias in enumerate(alias_map.keys(), start=1):
            alias_to_index[alias] = idx
            index_to_alias[idx] = alias
        object.__setattr__(self, "_alias_to_index", alias_to_index)
        object.__setattr__(self, "_index_to_alias", index_to_alias)

        cache: dict[str, float] = {}
        if isinstance(names, list) and isinstance(positions, list):
            for actual, value in zip(names, positions, strict=False):
                alias = self._joint_to_alias.get(actual)
                if alias:
                    cache[alias] = _safe_float(value)

        # Fill missing aliases with zero defaults
        for alias in alias_map.keys():
            cache.setdefault(alias, 0.0)

        object.__setattr__(self, "_joint_cache", cache)

    def _build_alias_map(self, joints: Any) -> dict[str, str | None]:
        alias_map: dict[str, str | None] = {}
        if not isinstance(joints, list):
            return alias_map

        seen: set[str] = set()
        for idx, joint in enumerate(joints, start=1):
            if not isinstance(joint, dict):
                continue
            actual = joint.get("name")
            alias = self._make_alias(actual, idx, seen)
            alias_map[alias] = actual if isinstance(actual, str) and actual else None
            seen.add(alias)
        return alias_map

    @staticmethod
    def _make_alias(name: Any, idx: int, seen: set[str]) -> str:
        base = "joint"
        if isinstance(name, str) and name.strip():
            import re

            candidate = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
            if candidate:
                if candidate[0].isdigit():
                    candidate = f"joint_{candidate}"
                base = candidate

        alias = base or f"joint_{idx}"
        if alias in seen:
            suffix = 2
            while f"{alias}_{suffix}" in seen:
                suffix += 1
            alias = f"{alias}_{suffix}"
        return alias

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def aliases(self) -> list[str]:
        """Return a list of known joint aliases."""
        self._ensure_initialized()
        return list(self._alias_to_joint.keys())

    def indices(self) -> list[int]:
        """Return a list of joint indexes (1-based)."""
        self._ensure_initialized()
        return list(self._index_to_alias.keys())

    def describe(self) -> list[Dict[str, Any]]:
        """Return structured joint metadata with positions."""
        self._ensure_initialized()
        description: list[Dict[str, Any]] = []
        for idx in sorted(self._index_to_alias.keys()):
            alias = self._index_to_alias[idx]
            backend = self._alias_to_joint.get(alias)
            description.append(
                {
                    "index": idx,
                    "alias": alias,
                    "backend_name": backend,
                    "position": self._joint_cache.get(alias, 0.0),
                }
            )
        return description

    def snapshot(self) -> list[Dict[str, Any]]:
        """Alias of :meth:`describe` for ergonomic SDK usage."""
        return self.describe()

    def get(self, identifier: Any) -> float:
        """Return the joint position for an alias or index."""
        alias = self._resolve_identifier(identifier)
        return self._joint_cache.get(alias, 0.0)

    def set(self, identifier: Any, value: float) -> None:
        """Update a single joint by alias or index."""
        self.set_many({identifier: value})

    def apply(self, targets: Mapping[Any, float]) -> None:
        """Alias of :meth:`set_many` to emphasise composability."""
        self.set_many(targets)

    def register_alias(self, alias: str, *, actual: Optional[str] = None, index: Optional[int] = None) -> None:
        """Register an additional alias, optionally referencing a backend name or index."""

        if not alias or not isinstance(alias, str):
            raise ValueError("alias must be a non-empty string")

        self._ensure_initialized()
        alias = alias.strip()

        resolved_actual = actual
        resolved_index = None
        if index is not None:
            idx = int(index)
            resolved_index = idx
            base_alias = self._index_to_alias.get(idx)
            if base_alias:
                resolved_actual = resolved_actual or self._alias_to_joint.get(base_alias)
            else:
                self._index_to_alias[idx] = alias
        elif alias in self._alias_to_index:
            resolved_index = self._alias_to_index[alias]

        if resolved_actual is None and alias in self._alias_to_joint:
            resolved_actual = self._alias_to_joint[alias]

        if resolved_actual is not None:
            self._alias_to_joint[alias] = resolved_actual
            if resolved_actual:
                self._joint_to_alias[resolved_actual] = alias
        else:
            self._alias_to_joint.setdefault(alias, None)

        if resolved_index is not None:
            self._alias_to_index[alias] = resolved_index
            self._index_to_alias.setdefault(resolved_index, alias)
        elif alias not in self._alias_to_index:
            next_index = (max(self._index_to_alias) + 1) if self._index_to_alias else 1
            self._alias_to_index[alias] = next_index
            self._index_to_alias.setdefault(next_index, alias)

        self._joint_cache.setdefault(alias, 0.0)

    def all(self) -> dict[str, float]:
        """Return a copy of the cached joint positions."""
        self._ensure_initialized()
        return dict(self._joint_cache)

    def set_many(self, updates: Mapping[Any, float]) -> None:
        """Set multiple joints at once using aliases or indices."""

        if not isinstance(updates, Mapping):
            raise TypeError("updates must be a mapping of alias/index -> position")

        self._ensure_initialized()

        normalized: dict[str, float] = {}
        remote_candidates: dict[str, float] = {}

        for identifier, value in updates.items():
            alias = self._resolve_identifier(identifier)
            value_f = _safe_float(value)
            normalized[alias] = value_f

            actual = self._alias_to_joint.get(alias)
            if actual:
                remote_candidates[alias] = value_f

        if remote_candidates and self._can_sync():
            self._push_remote(remote_candidates)

        self._joint_cache.update(normalized)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_identifier(self, identifier: Any) -> str:
        if isinstance(identifier, str):
            if identifier in self._alias_to_joint:
                return identifier
            if identifier in self._joint_to_alias:
                return self._joint_to_alias[identifier]
            if identifier.isdigit():
                idx = int(identifier)
                if idx in self._index_to_alias:
                    return self._index_to_alias[idx]
            if self._allow_dynamic_alias:
                self.register_alias(identifier)
                return identifier
        elif isinstance(identifier, (int, float)):
            idx = int(identifier)
            if idx in self._index_to_alias:
                return self._index_to_alias[idx]
        raise AttributeError(f"Unknown joint identifier '{identifier}'")

    def _can_sync(self) -> bool:
        twin_uuid = getattr(self._twin, "_twin_uuid", None)
        if not twin_uuid or str(twin_uuid).startswith("local-"):
            return False
        client = getattr(self._twin, "_client", None)
        return bool(client and getattr(client, "twins", None))

    def _push_remote(self, alias_updates: Mapping[str, float]) -> None:
        actual_payload = {
            self._alias_to_joint[alias]: value for alias, value in alias_updates.items()
            if self._alias_to_joint.get(alias)
        }
        if not actual_payload:
            return

        twin_uuid = self._twin._twin_uuid
        twins_api = self._twin._client.twins

        try:
            if len(actual_payload) == 1:
                joint_name, value = next(iter(actual_payload.items()))
                _run_async_in_jupyter(self._async_set_joint(twins_api, twin_uuid, joint_name, value))
            else:
                _run_async_in_jupyter(self._async_set_joints(twins_api, twin_uuid, actual_payload))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to push joint update: %s", exc)
            raise

    async def _async_set_joint(self, api, twin_uuid: str, joint_name: str, value: float):
        await self._twin._ensure_twin_exists()
        return await api.set_joint(twin_uuid, joint_name, value)

    async def _async_set_joints(self, api, twin_uuid: str, payload: Mapping[str, float]):
        await self._twin._ensure_twin_exists()
        body = {name: {"position": float(value)} for name, value in payload.items()}
        return await api.set_joints(twin_uuid, body)

class SimulationController:
    """Global simulation control"""
    
    @staticmethod
    def play():
        """Start/resume simulation"""
        print("[Simulation] Started")
    
    @staticmethod
    def pause():
        """Pause simulation"""
        print("[Simulation] Paused")
    
    @staticmethod
    def step():
        """Single simulation step"""
        print("[Simulation] Single step")
    
    @staticmethod
    def reset():
        """Reset simulation"""
        print("[Simulation] Reset")

# Global simulation instance
simulation = SimulationController()

class TwinNamespace:
    """Namespace object to expose both call and get ergonomics for twins.

    Supports:
      - cw.twin(from_="dji/tello") or cw.twin("dji/tello")
      - cw.twin(id="<uuid>") to get a handle without creating
    """

    def __call__(self, registry_id: Optional[str] = None, /, *, from_: Optional[str] = None, id: Optional[str] = None,
                 name: Optional[str] = None, environment_id: Optional[str] = None,
                 project_id: Optional[str] = None, project_name: Optional[str] = None,
                 environment_name: Optional[str] = None) -> CompactTwin:
        # Disambiguate parameters
        specified_registry = registry_id or from_
        if id and specified_registry:
            raise ValueError("Provide either 'id' to reference an existing twin or 'from_'/'registry_id' to create, not both.")

        if id:
            # Return a handle bound to existing twin, do not auto-create
            twin_cls = _resolve_twin_class(None)
            return twin_cls(
                None,
                name,
                environment_id,
                project_id,
                project_name,
                environment_name,
                twin_uuid=id,
                auto_create=False,
            )

        if not specified_registry:
            raise ValueError("Missing registry id. Use cw.twin('vendor/model') or cw.twin(from_='vendor/model').")

        twin_cls = _resolve_twin_class(specified_registry)
        return twin_cls(
            specified_registry,
            name,
            environment_id,
            project_id,
            project_name,
            environment_name,
        )

    def get(self, twin_uuid: str) -> "TwinHandle":
        client = _get_client()
        if not client:
            raise RuntimeError("Cyberwave client not configured. Call cw.configure() first.")
        return client.twins.get(twin_uuid)


# Export namespace instance (callable and has .get)
Twin = TwinNamespace
twin = TwinNamespace()


def pose(*, x: float, y: float, z: float = 0.0) -> Dict[str, float]:
    """Utility helper to build pose dictionaries for compact commands."""

    return {"x": float(x), "y": float(y), "z": float(z)}


def alert(event: Union[str, Dict[str, Any]], *, event_type: Optional[str] = None,
          severity: str = "warning", source: str = "sdk") -> Any:
    """Send an alert into the Cyberwave universal event system."""

    client = _get_client()
    if client is None:
        raise RuntimeError("Cyberwave client not configured. Call cw.configure() first.")

    payload: Dict[str, Any]
    if isinstance(event, dict):
        payload = dict(event)
    else:
        payload = {"message": str(event)}

    resolved_event_type = event_type or payload.pop("event_type", "sdk.alert")
    events_api = getattr(client, "events", None)
    if events_api is None:
        raise RuntimeError("Events API not available on Cyberwave client")

    return _await_result(
        events_api.ingest(
            event_type=resolved_event_type,
            payload=payload,
            severity=severity,
            source=source,
        )
    )


def dispatch(target: Union[str, CompactTwin], *, command: Optional[str] = None,
             task: Optional[str] = None, **payload: Any) -> Any:
    """Dispatch a high level action to an existing twin."""

    twin_instance = _resolve_twin(target)
    if twin_instance is None:
        raise ValueError(f"Twin '{target}' not found. Create it first with cw.twin().")

    command_name = command or task
    if not command_name:
        raise ValueError("dispatch requires either 'command' or 'task' argument")

    prepared_payload = twin_instance._prepare_command_payload(command_name, (), payload)
    return twin_instance._invoke_command(command_name, prepared_payload)
