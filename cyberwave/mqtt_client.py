"""
MQTT client wrapper for real-time communication with Cyberwave platform.

This module provides a compatibility layer that adapts the CyberwaveMQTTClient
from the mqtt module to work with the CyberwaveConfig object used by the main client.
"""

import logging
from typing import Callable, Optional, Dict, Any

from .config import CyberwaveConfig
from .mqtt import CyberwaveMQTTClient as BaseMQTTClient

logger = logging.getLogger(__name__)


class CyberwaveMQTTClient:
    """
    Wrapper for MQTT communication with the Cyberwave platform.

    This class adapts the BaseMQTTClient to work with CyberwaveConfig objects,
    providing a compatibility layer for the main Cyberwave client.

    Provides high-level methods for publishing and subscribing to twin updates,
    joint states, and other real-time events.
    """

    def __init__(self, config: CyberwaveConfig):
        """
        Initialize MQTT client from a CyberwaveConfig object.

        Args:
            config: Cyberwave configuration object containing MQTT settings
        """
        self.config = config

        # Determine the password/token to use for MQTT authentication
        mqtt_password = config.mqtt_password or "mqttcyb231"

        # Initialize the base MQTT client with extracted config values
        self._client = BaseMQTTClient(
            mqtt_password=mqtt_password,
        )

    @property
    def connected(self) -> bool:
        """Check if the client is connected to the MQTT broker."""
        return self._client.connected

    def connect(self):
        """Connect to the MQTT broker."""
        if not self.connected:
            self._client.connect()

    def disconnect(self):
        """Disconnect from the MQTT broker."""
        self._client.disconnect()

    # Delegate all methods to the base client
    def subscribe_twin(self, twin_uuid: str, on_update: Optional[Callable] = None):
        """
        Subscribe to twin updates via MQTT.

        Args:
            twin_uuid: UUID of the twin to monitor
            on_update: Callback function for updates
        """
        return self._client.subscribe_twin(twin_uuid, on_update)

    def subscribe_twin_position(
        self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to twin position updates.

        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when position updates are received
        """
        # The base client expects a handler that receives the full message data
        return self._client.subscribe(f"cyberwave/twin/{twin_uuid}/position", callback)

    def subscribe_twin_rotation(
        self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to twin rotation updates.

        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when rotation updates are received
        """
        return self._client.subscribe(f"cyberwave/twin/{twin_uuid}/rotation", callback)

    def subscribe_joint_states(
        self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to joint state updates.

        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when joint state updates are received
        """
        return self._client.subscribe(f"cyberwave/joint/{twin_uuid}/+", callback)

    def update_twin_position(self, twin_uuid: str, position: Dict[str, float]):
        """
        Update twin position via MQTT.

        Args:
            twin_uuid: UUID of the twin
            position: Dictionary with x, y, z coordinates
        """
        return self._client.update_twin_position(twin_uuid, position)

    def publish_twin_position(self, twin_uuid: str, x: float, y: float, z: float):
        """
        Publish twin position update (backward compatibility method).

        Args:
            twin_uuid: UUID of the twin
            x, y, z: Position coordinates
        """
        return self._client.update_twin_position(twin_uuid, {"x": x, "y": y, "z": z})

    def update_twin_rotation(self, twin_uuid: str, rotation: Dict[str, float]):
        """
        Update twin rotation via MQTT.

        Args:
            twin_uuid: UUID of the twin
            rotation: Dictionary with rotation values (quaternion or euler)
        """
        return self._client.update_twin_rotation(twin_uuid, rotation)

    def publish_twin_rotation(
        self, twin_uuid: str, x: float, y: float, z: float, w: float
    ):
        """
        Publish twin rotation update as quaternion (backward compatibility method).

        Args:
            twin_uuid: UUID of the twin
            x, y, z, w: Quaternion components
        """
        return self._client.update_twin_rotation(
            twin_uuid, {"x": x, "y": y, "z": z, "w": w}
        )

    def update_twin_scale(self, twin_uuid: str, scale: Dict[str, float]):
        """
        Update twin scale via MQTT.

        Args:
            twin_uuid: UUID of the twin
            scale: Dictionary with scale values
        """
        return self._client.update_twin_scale(twin_uuid, scale)

    def update_joint_state(
        self,
        twin_uuid: str,
        joint_name: str,
        position: Optional[float] = None,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ):
        """
        Update joint state via MQTT.

        Args:
            twin_uuid: UUID of the twin
            joint_name: Name of the joint
            position: Joint position (radians for revolute, meters for prismatic)
            velocity: Joint velocity
            effort: Joint effort/torque
        """
        return self._client.update_joint_state(
            twin_uuid, joint_name, position, velocity, effort
        )

    def subscribe_environment(
        self, environment_uuid: str, on_update: Optional[Callable] = None
    ):
        """
        Subscribe to environment updates via MQTT.

        Args:
            environment_uuid: UUID of the environment
            on_update: Callback function for updates
        """
        return self._client.subscribe_environment(environment_uuid, on_update)

    def publish_environment_update(
        self, environment_uuid: str, update_type: str, data: Dict[str, Any]
    ):
        """
        Publish environment update via MQTT.

        Args:
            environment_uuid: UUID of the environment
            update_type: Type of update
            data: Update data
        """
        return self._client.publish_environment_update(
            environment_uuid, update_type, data
        )

    def subscribe_video_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to video stream via MQTT."""
        return self._client.subscribe_video_stream(twin_uuid, on_frame)

    def subscribe_depth_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to depth stream via MQTT."""
        return self._client.subscribe_depth_stream(twin_uuid, on_frame)

    def subscribe_pointcloud_stream(
        self, twin_uuid: str, on_pointcloud: Optional[Callable] = None
    ):
        """Subscribe to point cloud stream via MQTT."""
        return self._client.subscribe_pointcloud_stream(twin_uuid, on_pointcloud)

    def publish_depth_frame(self, twin_uuid: str, depth_data: Dict[str, Any]):
        """Publish depth frame data via MQTT."""
        return self._client.publish_depth_frame(twin_uuid, depth_data)

    def publish_webrtc_message(self, twin_uuid: str, webrtc_data: Dict[str, Any]):
        """Publish WebRTC signaling message via MQTT."""
        return self._client.publish_webrtc_message(twin_uuid, webrtc_data)

    def subscribe_webrtc_messages(
        self, twin_uuid: str, on_message: Optional[Callable] = None
    ):
        """Subscribe to WebRTC signaling messages via MQTT."""
        return self._client.subscribe_webrtc_messages(twin_uuid, on_message)

    def ping(self, resource_uuid: str):
        """Send ping message to test connectivity."""
        return self._client.ping(resource_uuid)

    def subscribe_pong(self, resource_uuid: str, on_pong: Optional[Callable] = None):
        """Subscribe to pong responses."""
        return self._client.subscribe_pong(resource_uuid, on_pong)
