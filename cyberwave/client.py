"""
Main Cyberwave client that integrates REST and MQTT APIs
"""

import os
from typing import Optional

from cyberwave.rest import DefaultApi, ApiClient, Configuration

from cyberwave.config import CyberwaveConfig
from cyberwave.mqtt_client import CyberwaveMQTTClient
from cyberwave.resources import (
    WorkspaceManager,
    ProjectManager,
    EnvironmentManager,
    AssetManager,
    TwinManager,
)
from cyberwave.twin import Twin
from cyberwave.exceptions import (
    CyberwaveError,
    CyberwaveAPIError,
    UnauthorizedException,
)


class Cyberwave:
    """
    Main client for the Cyberwave Digital Twin Platform.

    This client provides access to both REST and MQTT APIs, along with
    high-level abstractions for working with digital twins.

    Example:
        >>> client = Cyberwave(base_url="http://localhost:8000", token="your_token")
        >>> workspaces = client.workspaces.list()
        >>> twin = client.twin("cyberwave/so101")

    Args:
        base_url: Base URL of the Cyberwave backend
        token: Bearer token for authentication
        api_key: API key for authentication (alternative to token)
        mqtt_host: MQTT broker host (optional, defaults to base_url host)
        mqtt_port: MQTT broker port (default: 1883)
        environment_id: Default environment ID
        workspace_id: Default workspace ID
        **config_kwargs: Additional configuration options
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        mqtt_host: Optional[str] = "mqtt.cyberwave.com",
        mqtt_port: int = 1883,
        **config_kwargs,
    ):
        # Grab an env var for the base URL if unspecified
        if not base_url:
            base_url = os.getenv("CYBERWAVE_BASE_URL", "https://api.cyberwave.com")

        if token is None:
            token = os.getenv("CYBERWAVE_TOKEN", None)

        if api_key is None:
            api_key = os.getenv("CYBERWAVE_API_KEY", None)

        # Create configuration
        self.config = CyberwaveConfig(
            base_url=base_url,
            token=token,
            api_key=api_key,
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port,
            # environment_id=os.getenv("CYBERWAVE_ENVIRONMENT_ID", None),
            workspace_id=os.getenv("CYBERWAVE_WORKSPACE_ID", None),
            **config_kwargs,
        )

        # Initialize REST API client
        self._setup_rest_client()

        # Initialize MQTT client (lazy - only connects when needed)
        self._mqtt_client: Optional[CyberwaveMQTTClient] = None

        # Initialize resource managers
        self.workspaces = WorkspaceManager(self.api)
        self.projects = ProjectManager(self.api)
        self.environments = EnvironmentManager(self.api)
        self.assets = AssetManager(self.api)
        self.twins = TwinManager(self.api)

    def _setup_rest_client(self):
        """Setup the REST API client with authentication"""
        configuration = Configuration(host=self.config.base_url)

        # Set authentication - the backend uses CustomTokenAuthentication which expects
        # the token in the Authorization header with Bearer or Token prefix
        if self.config.token:
            # Use CustomTokenAuthentication with Bearer prefix
            configuration.api_key["CustomTokenAuthentication"] = self.config.token
            configuration.api_key_prefix["CustomTokenAuthentication"] = "Bearer"
        elif self.config.api_key:
            # Use CustomTokenAuthentication with Bearer prefix for API keys too
            configuration.api_key["CustomTokenAuthentication"] = self.config.api_key
            configuration.api_key_prefix["CustomTokenAuthentication"] = "Bearer"

        # Set other configuration
        configuration.verify_ssl = self.config.verify_ssl

        # Create API client
        api_client = ApiClient(configuration)

        # Monkey-patch the response_deserialize method to add request headers to exceptions
        original_response_deserialize = api_client.response_deserialize
        last_request_headers = {}

        def response_deserialize_with_headers(response_data, response_types_map=None):
            try:
                return original_response_deserialize(response_data, response_types_map)
            except Exception as e:
                # Add request headers to the exception if available
                if hasattr(e, "__dict__") and not hasattr(e, "request_headers"):
                    e.request_headers = last_request_headers.copy()
                raise

        original_call_api = api_client.call_api

        def call_api_with_header_tracking(
            method,
            url,
            header_params=None,
            body=None,
            post_params=None,
            _request_timeout=None,
        ):
            # Store the request headers for later use in exception handling
            last_request_headers.clear()
            if header_params:
                last_request_headers.update(header_params)
            return original_call_api(
                method, url, header_params, body, post_params, _request_timeout
            )

        api_client.response_deserialize = response_deserialize_with_headers
        api_client.call_api = call_api_with_header_tracking

        self.api = DefaultApi(api_client)
        self._api_client = api_client

        # Wrap the API client to intercept authentication errors
        self._wrap_api_methods()

    def _wrap_api_methods(self):
        """Wrap API methods to provide better error messages for authentication failures"""
        # Get all methods from the DefaultApi class
        for attr_name in dir(self.api):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self.api, attr_name)
            if callable(attr):
                # Wrap the method
                wrapped = self._create_wrapped_method(attr)
                setattr(self.api, attr_name, wrapped)

    def _create_wrapped_method(self, method):
        """Create a wrapped version of an API method that handles auth errors"""

        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except UnauthorizedException as e:
                # Provide a helpful error message for authentication failures
                error_msg = "Authentication failed: Invalid or missing credentials.\n\n"

                if self.config.token:
                    error_msg += "Your token appears to be invalid or expired.\n"
                elif self.config.api_key:
                    error_msg += "Your API key appears to be invalid or expired.\n"
                else:
                    error_msg += "No authentication credentials were provided.\n"

                error_msg += "\To start using the SDK:\n"
                error_msg += "  1. Add a token at https://cyberwave.com/profile\n"
                error_msg += "  2. Copy it to your clipboard\n"
                error_msg += "  3. Set the environment variable:\n\nexport CYBERWAVE_TOKEN=your_token\n"
                error_msg += "  4. Run your script again!\n"

                # Show what was sent (without revealing the full token)
                if hasattr(e, "request_headers") and e.request_headers:
                    auth_header = e.request_headers.get("Authorization", "Not present")
                    if auth_header and auth_header != "Not present":
                        # Mask the token for security
                        parts = auth_header.split(" ")
                        if len(parts) == 2:
                            token_preview = (
                                parts[1][:8] + "..." if len(parts[1]) > 8 else parts[1]
                            )
                            error_msg += (
                                f"Authorization header: {parts[0]} {token_preview}\n"
                            )
                    else:
                        error_msg += "Authorization header: Not present\n"

                raise CyberwaveAPIError(
                    error_msg,
                    status_code=401,
                    response_data=e.body if hasattr(e, "body") else None,
                ) from e

        return wrapped

    @property
    def mqtt(self) -> CyberwaveMQTTClient:
        """
        Get MQTT client instance (lazy initialization)

        Returns:
            CyberwaveMQTTClient instance
        """
        if self._mqtt_client is None:
            self._mqtt_client = CyberwaveMQTTClient(self.config)
        return self._mqtt_client

    def twin(
        self, asset_key: str, environment_id: Optional[str] = None, **kwargs
    ) -> Twin:
        """
        Create or get a twin instance (compact API)

        This is a convenience method for quickly creating twins.

        Args:
            asset_key: Asset identifier (e.g., "cyberwave/so101")
            environment_id: Environment ID (uses default if not provided)
            **kwargs: Additional twin creation parameters

        Returns:
            Twin instance

        Example:
            >>> robot = client.twin("cyberwave/so101")
            >>> robot.move(x=1, y=0, z=0.5)
        """
        env_id = environment_id or self.config.environment_id
        if not env_id:
            # check if the user has a project. If not, create a new project and environment.
            projects = self.projects.list()
            if not projects:
                project_id = self.projects.create(
                    name="Quickstart Project",
                ).uuid
                self.config.project_id = project_id
            else:
                project_id = projects[0].uuid
            # create a new environment
            env_id = self.environments.create(
                name="Quickstart Environment",
                project_id=project_id,
            ).uuid
            self.config.environment_id = env_id

        # Search for asset by key or name
        assets = self.assets.search(asset_key)
        if not assets:
            raise CyberwaveError(f"Asset '{asset_key}' not found")

        asset = assets[0]

        # Check if twin already exists in environment
        try:
            existing_twins = self.twins.list(environment_id=env_id)
            for twin_data in existing_twins:
                if hasattr(twin_data, "asset") and twin_data.asset == asset.uuid:
                    # Return existing twin
                    return Twin(self, twin_data)
        except Exception:
            # If listing fails, just create a new twin
            pass

        # Create new twin
        twin_data = self.twins.create(
            asset_id=asset.uuid, environment_id=env_id, **kwargs
        )

        return Twin(self, twin_data)

    def configure(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        environment_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Update client configuration

        Args:
            base_url: Base URL of the Cyberwave backend
            token: Bearer token for authentication
            api_key: API key for authentication
            environment_id: Default environment ID
            workspace_id: Default workspace ID
            **kwargs: Additional configuration options
        """
        if base_url:
            self.config.base_url = base_url
        if token:
            self.config.token = token
        if api_key:
            self.config.api_key = api_key
        if environment_id:
            self.config.environment_id = environment_id
        if workspace_id:
            self.config.workspace_id = workspace_id

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Recreate REST client with new config
        self._setup_rest_client()

        # Reset MQTT client to use new config
        if self._mqtt_client:
            self._mqtt_client.disconnect()
            self._mqtt_client = None

    def disconnect(self):
        """Disconnect all connections (REST and MQTT)"""
        if self._mqtt_client:
            self._mqtt_client.disconnect()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
