"""
Main Cyberwave client that integrates REST and MQTT APIs
"""

import os
from typing import Optional

from cyberwave.rest import DefaultApi, ApiClient, Configuration
from cyberwave.config import (
    CyberwaveConfig,
    DEFAULT_BASE_URL,
    DEFAULT_MQTT_HOST,
    DEFAULT_MQTT_PORT,
)
from cyberwave.controller import EdgeController
from cyberwave.mqtt_client import CyberwaveMQTTClient
from cyberwave.resources import (
    WorkspaceManager,
    ProjectManager,
    EnvironmentManager,
    AssetManager,
    TwinManager,
)
from cyberwave.twin import Twin, create_twin
from cyberwave.utils import TimeReference
from cyberwave.exceptions import (
    CyberwaveError,
    CyberwaveAPIError,
    UnauthorizedException,
)

# Import CameraStreamer with optional dependency handling
try:
    from cyberwave.camera import CameraStreamer

    _has_camera = True
except ImportError:
    _has_camera = False
    CameraStreamer = None


class Cyberwave:
    """
    Main client for the Cyberwave Digital Twin Platform.

    This client provides access to both REST and MQTT APIs, along with
    high-level abstractions for working with digital twins.

    Example:
        >>> client = Cyberwave(base_url="http://localhost:8000", token="your_token")
        >>> workspaces = client.workspaces.list()
        >>> twin = client.twin("the-robot-studio/so101")

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
        mqtt_host: Optional[str] = None,
        mqtt_port: int | None = None,
        mqtt_username: Optional[str] = None,
        mqtt_password: Optional[str] = None,
        **config_kwargs,
    ):
        if not base_url:
            base_url = os.getenv("CYBERWAVE_BASE_URL", DEFAULT_BASE_URL)

        if token is None:
            token = os.getenv("CYBERWAVE_TOKEN", None)

        if api_key is None:
            api_key = os.getenv("CYBERWAVE_API_KEY", None)

        if api_key is None and token is None:
            raise ValueError(
                "No CYBERWAVE_API_KEY found! Get yours at https://cyberwave.com/profile"
            )

        self.config = CyberwaveConfig(
            base_url=base_url,
            token=token,
            api_key=api_key,
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port or DEFAULT_MQTT_PORT,
            mqtt_username=mqtt_username,
            mqtt_password=mqtt_password,
            environment_id=os.getenv("CYBERWAVE_ENVIRONMENT_ID", None),
            workspace_id=os.getenv("CYBERWAVE_WORKSPACE_ID", None),
            **config_kwargs,
        )

        self._setup_rest_client()
        self._mqtt_client: Optional[CyberwaveMQTTClient] = None

        self.workspaces = WorkspaceManager(self.api)
        self.projects = ProjectManager(self.api)
        self.environments = EnvironmentManager(self.api)
        self.assets = AssetManager(self.api)
        self.twins = TwinManager(self.api)

    def _setup_rest_client(self):
        """Setup the REST API client with authentication"""
        configuration = Configuration(host=self.config.base_url)

        if self.config.token:
            configuration.api_key["CustomTokenAuthentication"] = self.config.token
            configuration.api_key_prefix["CustomTokenAuthentication"] = "Bearer"
        elif self.config.api_key:
            configuration.api_key["CustomTokenAuthentication"] = self.config.api_key
            configuration.api_key_prefix["CustomTokenAuthentication"] = "Bearer"

        configuration.verify_ssl = self.config.verify_ssl

        api_client = ApiClient(configuration)

        original_response_deserialize = api_client.response_deserialize
        last_request_headers = {}

        def response_deserialize_with_headers(response_data, response_types_map=None):
            try:
                return original_response_deserialize(response_data, response_types_map)
            except Exception as e:
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

        self._wrap_api_methods()

    def _wrap_api_methods(self):
        """Wrap API methods to provide better error messages for authentication failures"""
        for attr_name in dir(self.api):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self.api, attr_name)
            if callable(attr):
                wrapped = self._create_wrapped_method(attr)
                setattr(self.api, attr_name, wrapped)

    def _create_wrapped_method(self, method):
        """Create a wrapped version of an API method that handles auth errors"""

        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except UnauthorizedException as e:
                error_msg = "Authentication failed: Invalid or missing credentials.\n\n"

                if self.config.token:
                    error_msg += "Your token appears to be invalid or expired.\n"
                elif self.config.api_key:
                    error_msg += "Your API key appears to be invalid or expired.\n"
                else:
                    error_msg += "No authentication credentials were provided.\n"

                error_msg += "  1. Add a token at https://cyberwave.com/profile\n"
                error_msg += "  2. Copy it to your clipboard\n"
                error_msg += "  3. Set the environment variable:\n\nexport CYBERWAVE_TOKEN=your_token\n"
                error_msg += "  4. Run your script again!\n"

                if hasattr(e, "request_headers") and e.request_headers:
                    auth_header = e.request_headers.get("Authorization", "Not present")
                    if auth_header and auth_header != "Not present":
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
        """Get MQTT client instance (lazy initialization)"""
        if self._mqtt_client is None:
            self._mqtt_client = CyberwaveMQTTClient(self.config)
        return self._mqtt_client

    def twin(
        self, asset_key: str, environment_id: Optional[str] = None, twin_id: Optional[str] = None, **kwargs
    ) -> Twin:
        """
        Get or create a twin instance (compact API)

        This is a convenience method for quickly creating twins. The returned
        twin will be an appropriate subclass based on the asset's capabilities:
        
        - CameraTwin: For assets with RGB sensors (has start_streaming(), etc.)
        - DepthCameraTwin: For assets with depth sensors (has get_point_cloud(), etc.)
        - FlyingTwin: For drones/UAVs (has takeoff(), land(), hover())
        - GripperTwin: For manipulators (has grip(), release())
        - Twin: Base class for assets without special capabilities

        Args:
            asset_key: Asset identifier (e.g., "the-robot-studio/so101")
            environment_id: Environment ID (uses default if not provided)
            twin_id: Existing twin ID to fetch (skips creation)
            **kwargs: Additional twin creation parameters

        Returns:
            Twin instance (or appropriate subclass based on capabilities)

        Example:
            >>> robot = client.twin("unitree/go2")  # Returns CameraTwin
            >>> robot.start_streaming(fps=15)  # Available because of RGB sensor
            >>> robot.move(x=1, y=0, z=0.5)
        """
        if twin_id:
            twin_data = self.twins.get(twin_id)
            return create_twin(self, twin_data, registry_id=asset_key)
        
        env_id = environment_id or self.config.environment_id
        if not env_id:
            projects = self.projects.list()
            if not projects:
                project_id = self.projects.create(
                    name="Quickstart Project",
                ).uuid
                self.config.project_id = project_id
            else:
                project_id = projects[0].uuid
            env_id = self.environments.create(
                name="Quickstart Environment",
                project_id=project_id,
            ).uuid
            self.config.environment_id = env_id

        assets = self.assets.search(asset_key)
        if not assets:
            raise CyberwaveError(f"Asset '{asset_key}' not found")
        asset = assets[0]
        
        # Get registry_id for capability lookup
        registry_id = getattr(asset, "registry_id", None) or asset_key
        
        try:
            existing_twins = self.twins.list(environment_id=env_id)
            for twin_data in existing_twins:
                if twin_data.asset_uuid == asset.uuid:
                    return create_twin(self, twin_data, registry_id=registry_id)
            
            twin_data = self.twins.create(
                asset_id=asset.uuid, environment_id=env_id, **kwargs
            )
            return create_twin(self, twin_data, registry_id=registry_id)
        except Exception:
            return create_twin(self, twin_data, registry_id=registry_id)

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

        self._setup_rest_client()

        if self._mqtt_client:
            self._mqtt_client.disconnect()
            self._mqtt_client = None

    def video_stream(
        self,
        twin_uuid: str,
        camera_id: int = 0,
        fps: int = 10,
        turn_servers: Optional[list] = None,
        time_reference: TimeReference = None,
    ) -> "CameraStreamer":
        """
        Create a camera streamer for the specified twin.

        This method creates a CameraStreamer instance that's pre-configured with
        the client's MQTT connection, providing a seamless experience for streaming
        video to digital twins.

        Args:
            twin_uuid: UUID of the digital twin to stream to
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
            turn_servers: Optional list of TURN server configurations

        Returns:
            CameraStreamer instance ready to start streaming

        Example:
            >>> client = Cyberwave(token="your_token")
            >>> streamer = client.video_stream(twin_uuid="your_twin_uuid")
            >>> await streamer.start()

        Raises:
            ImportError: If camera dependencies are not installed (install with: pip install cyberwave[camera])
        """
        if not _has_camera:
            raise ImportError(
                "Camera streaming requires additional dependencies. "
                "Install them with: pip install cyberwave[camera]"
            )

        if self._mqtt_client is None:
            self.mqtt.connect()

        self.mqtt.connect()
        self.mqtt._client._handle_twin_update_with_telemetry(twin_uuid)
        return CameraStreamer(
            client=self.mqtt,
            camera_id=camera_id,
            fps=fps,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
        )

    def controller(
        self,
        twin_uuid: str,
    ) -> "EdgeController":
        """
        Create an edge controller for the specified twin.

        This method creates an EdgeController instance that's pre-configured with
        the client's MQTT connection, providing a seamless experience for sending
        commands to edge devices.

        Args:
            twin_uuid: UUID of the digital twin to control

        Returns:
            EdgeController instance ready to start

        Example:
            >>> client = Cyberwave(token="your_token")
            >>> controller = client.controller(twin_uuid="your_twin_uuid")
            >>> await controller.start()

        """
        if self._mqtt_client is None:
            self.mqtt.connect()

        return EdgeController(
            client=self.mqtt,
            twin_uuid=twin_uuid,
        )

    def disconnect(self):
        """Disconnect all connections (REST and MQTT)"""
        if self._mqtt_client:
            self._mqtt_client.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
