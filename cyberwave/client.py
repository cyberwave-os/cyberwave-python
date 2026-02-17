"""
Main Cyberwave client that integrates REST and MQTT APIs
"""

import os
from typing import Optional

from cyberwave.rest import DefaultApi, ApiClient, Configuration
from cyberwave.config import (
    CyberwaveConfig,
    DEFAULT_BASE_URL,
    DEFAULT_MQTT_PORT,
)
from cyberwave.controller import EdgeController
from cyberwave.mqtt_client import CyberwaveMQTTClient
from cyberwave.resources import (
    WorkspaceManager,
    ProjectManager,
    EnvironmentManager,
    AssetManager,
    EdgeManager,
    TwinManager,
)
from cyberwave.twin import Twin, create_twin
from cyberwave.utils import TimeReference
from cyberwave.exceptions import (
    CyberwaveError,
    CyberwaveAPIError,
    UnauthorizedException,
)
from cyberwave.constants import SOURCE_TYPE_EDGE

# Import camera streamers with optional dependency handling
try:
    from cyberwave.sensor import CV2CameraStreamer as CameraStreamer

    _has_camera = True
except ImportError:
    _has_camera = False
    CameraStreamer = None

try:
    from cyberwave.sensor import RealSenseStreamer

    _has_realsense = True
except ImportError:
    _has_realsense = False
    RealSenseStreamer = None

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
        topic_prefix: Optional[str] = None,
        source_type: Optional[str] = SOURCE_TYPE_EDGE,
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
            topic_prefix=topic_prefix,
            environment_id=os.getenv("CYBERWAVE_ENVIRONMENT_ID", None),
            workspace_id=os.getenv("CYBERWAVE_WORKSPACE_ID", None),
            source_type=os.getenv("CYBERWAVE_SOURCE_TYPE", SOURCE_TYPE_EDGE),
            **config_kwargs,
        )

        self._setup_rest_client()
        self._mqtt_client: Optional[CyberwaveMQTTClient] = None

        self.workspaces = WorkspaceManager(self.api)
        self.projects = ProjectManager(self.api)
        self.environments = EnvironmentManager(self.api)
        self.assets = AssetManager(self.api)
        self.edges = EdgeManager(self.api)
        self.twins = TwinManager(self.api, client=self)

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
        self,
        asset_key: Optional[str] = None,
        environment_id: Optional[str] = None,
        twin_id: Optional[str] = None,
        **kwargs,
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
        - LocomoteTwin: For assets that can locomote (has move(), etc.)

        Args:
            asset_key: Asset identifier (e.g., "the-robot-studio/so101"). Required for creation, optional when twin_id is provided.
            environment_id: Environment ID (uses default if not provided)
            twin_id: Existing twin ID to fetch (skips creation)
            **kwargs: Additional twin creation parameters

        Returns:
            Twin instance (or appropriate subclass based on capabilities)

        Example:
            >>> robot = client.twin("unitree/go2")  # Create new twin
            >>> robot = client.twin(twin_id="uuid")  # Fetch existing twin by ID
            >>> robot.start_streaming(fps=15)  # Available because of RGB sensor
            >>> robot.edit_position(x=1, y=0, z=0.5)
        """
        if twin_id:
            twin_data = self.twins.get_raw(twin_id)
            return create_twin(self, twin_data, registry_id=asset_key)

        # asset_key is required for twin creation
        if not asset_key:
            raise CyberwaveError("asset_key is required when creating a new twin (twin_id not provided)")

        twin_name = kwargs.get("name", None)

        env_id = environment_id or self.config.environment_id
        if not env_id:
            projects = self.projects.list()
            if not projects:
                workspace_id = self.config.workspace_id
                if not workspace_id:
                    workspaces = self.workspaces.list()
                    if not workspaces:
                        # create a new workspace
                        workspace_id = self.workspaces.create(
                            name="Quickstart Workspace",
                        ).uuid
                        self.config.workspace_id = workspace_id
                    workspace_id = workspaces[0].uuid
                project_id = self.projects.create(
                    name="Quickstart Project",
                    workspace_id=self.config.workspace_id,
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
                if twin_data.asset_uuid == asset.uuid and (
                    not twin_name or twin_data.name == twin_name
                ):
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

    @property
    def time_reference(self) -> TimeReference:
        """Get a shared time reference for synchronization."""
        if not hasattr(self, "_time_reference"):
            self._time_reference = TimeReference()
        return self._time_reference

    def video_stream(
        self,
        twin_uuid: str,
        camera_type: str = "cv2",
        camera_id: int | str = 0,
        fps: int = 30,
        resolution: Optional[tuple] = None,
        enable_depth: bool = False,
        depth_fps: int = 30,
        depth_resolution: Optional[tuple] = None,
        auto_detect: bool = True,
        turn_servers: Optional[list] = None,
        time_reference: Optional[TimeReference] = None,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[callable] = None,
    ):
        """
        Create a camera streamer for the specified twin.

        This method creates a camera streamer instance that's pre-configured with
        the client's MQTT connection, providing a seamless experience for streaming
        video to digital twins.

        Supports:
        - Local cameras: camera_id=0, camera_id=1 (device index)
        - IP cameras: camera_id="http://192.168.1.100/snapshot.jpg"
        - RTSP streams: camera_id="rtsp://192.168.1.100:554/stream"
        - Intel RealSense: camera_type="realsense"

        Args:
            twin_uuid: UUID of the digital twin to stream to
            camera_type: Camera type - "cv2" for USB/webcam/IP, "realsense" for Intel RealSense
            camera_id: Camera device ID (int) or stream URL (str) (default: 0)
            fps: Frames per second (default: 30)
            resolution: Video resolution as (width, height) tuple (default: 640x480)
            enable_depth: Enable depth streaming for RealSense (default: False)
            depth_fps: Depth stream FPS for RealSense (default: 30)
            depth_resolution: Depth resolution as (width, height) tuple (default: same as color)
            auto_detect: Auto-detect RealSense capabilities (default: True)
            turn_servers: Optional list of TURN server configurations
            time_reference: Optional time reference for synchronization
            keyframe_interval: Force a keyframe every N frames for better streaming start.
                If None, uses CYBERWAVE_KEYFRAME_INTERVAL env var, or disables forced keyframes.
                Recommended: fps * 2 (e.g., 60 for 30fps = keyframe every 2 seconds)
            frame_callback: Optional callback for each frame (ML inference, etc.).
                Signature: callback(frame: np.ndarray, frame_count: int) -> None

        Returns:
            Camera streamer instance (CV2CameraStreamer or RealSenseStreamer)

        Example:
            >>> client = Cyberwave(token="your_token")
            >>>
            >>> # Local USB camera
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="cv2",
            ...     camera_id=0,
            ...     resolution=(1280, 720),
            ...     fps=30
            ... )
            >>>
            >>> # IP camera / RTSP stream
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="cv2",
            ...     camera_id="rtsp://192.168.1.100:554/stream",
            ...     fps=15
            ... )
            >>>
            >>> # RealSense camera with depth
            >>> streamer = client.video_stream(
            ...     twin_uuid="your_twin_uuid",
            ...     camera_type="realsense",
            ...     resolution=(1280, 720),
            ...     enable_depth=True,
            ...     auto_detect=True
            ... )
            >>>
            >>> await streamer.start()

        Raises:
            ImportError: If camera dependencies are not installed
            CyberwaveError: If camera type is not supported
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

        # Use shared time reference if not provided
        if time_reference is None:
            time_reference = self.time_reference

        # Default resolution
        if resolution is None:
            resolution = (640, 480)

        camera_type_lower = camera_type.lower()

        if camera_type_lower == "cv2":
            return CameraStreamer(
                client=self.mqtt,
                camera_id=camera_id,
                fps=fps,
                resolution=resolution,
                turn_servers=turn_servers,
                twin_uuid=twin_uuid,
                time_reference=time_reference,
                keyframe_interval=keyframe_interval,
                frame_callback=frame_callback,
            )
        elif camera_type_lower == "realsense":
            if not _has_realsense:
                raise ImportError(
                    "RealSense camera support requires additional dependencies. "
                    "Install them with: pip install cyberwave[realsense]"
                )

            # Import Resolution for RealSense
            from cyberwave.sensor import Resolution

            # Convert tuple to Resolution enum for from_device
            def to_resolution(res):
                if isinstance(res, tuple):
                    return Resolution.from_size(res[0], res[1]) or Resolution.closest(res[0], res[1])
                return res

            # Depth resolution defaults to color resolution
            if depth_resolution is None:
                depth_resolution = resolution

            if auto_detect:
                return RealSenseStreamer.from_device(
                    client=self.mqtt,
                    prefer_resolution=to_resolution(resolution),
                    prefer_fps=fps,
                    enable_depth=enable_depth,
                    turn_servers=turn_servers,
                    twin_uuid=twin_uuid,
                    time_reference=time_reference,
                )
            else:
                return RealSenseStreamer(
                    client=self.mqtt,
                    color_fps=fps,
                    depth_fps=depth_fps,
                    color_resolution=resolution,
                    depth_resolution=depth_resolution,
                    enable_depth=enable_depth,
                    turn_servers=turn_servers,
                    twin_uuid=twin_uuid,
                    time_reference=time_reference,
                )
        else:
            raise CyberwaveError(
                f"Unsupported camera type: {camera_type}. "
                "Supported types: 'cv2', 'realsense'"
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
