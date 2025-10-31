"""
MQTT Client for Cyberwave Platform

This module provides a high-level MQTT client for real-time communication with the Cyberwave platform.
It uses paho-mqtt (2.1.0+) for reliable MQTT connectivity.
"""

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion  # type: ignore
# Try to import CallbackAPIVersion for paho-mqtt 2.x, fallback for older versions

logger = logging.getLogger(__name__)


class CyberwaveMQTTClient:
    """
    Client for Cyberwave MQTT API interactions.

    This client provides methods for publishing and subscribing to MQTT topics
    for digital twin updates, joint states, sensor streams, and more.

    Args:
        mqtt_broker: MQTT broker hostname or IP address
        mqtt_port: MQTT broker port (default: 1883)
        mqtt_username: MQTT username (default: "cyberwave")
        mqtt_password: MQTT password (API token)
        api_token: Cyberwave API token (used as MQTT password if mqtt_password not provided)
        client_id: Custom MQTT client ID (auto-generated if not provided)
        topic_prefix: Prefix for MQTT topics (default: "")
        auto_connect: Automatically connect on initialization (default: True)
    """

    def __init__(
        self,
        mqtt_broker: str = "mqtt.cyberwave.com",
        mqtt_port: int = 1883,
        mqtt_username: str = "mqttcyb",
        mqtt_password: str = "mqttcyb231",
        api_token: Optional[str] = None,
        client_id: Optional[str] = None,
        topic_prefix: str = "",
        auto_connect: bool = False,
    ):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username

        # Use mqtt_password if provided, otherwise use api_token
        self.mqtt_password = mqtt_password

        if not self.mqtt_password:
            raise ValueError("Either mqtt_password or api_token is required")

        # Topic prefix (empty by default, can be set for custom deployments)
        self.topic_prefix = topic_prefix

        # Generate unique client ID
        self.client_id = client_id or f"sdk_{uuid.uuid4().hex[:8]}"

        # MQTT client (compatible with paho-mqtt 1.x and 2.x)
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=self.client_id,  # type: ignore
        )
        self.client.username_pw_set(
            username=self.mqtt_username, password=self.mqtt_password
        )

        # Connection state
        self.connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

        # Event handlers
        self._handlers: Dict[str, List[Callable]] = {}

        # Position tracking to avoid duplicate updates
        self._last_positions: Dict[str, Dict[str, float]] = {}

        # Rotation tracking to avoid duplicate updates
        self._last_rotations: Dict[str, Dict[str, float]] = {}

        # Rate limiting
        self._last_update_times: Dict[str, float] = {}
        self._min_update_interval = 0.025  # 40 Hz max

        # Setup MQTT callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Auto-connect if requested
        if auto_connect:
            self.connect()

    def _positions_equal(
        self, pos1: Dict[str, float], pos2: Dict[str, float], tolerance: float = 1e-6
    ) -> bool:
        """Compare two position dictionaries with floating point tolerance."""
        if set(pos1.keys()) != set(pos2.keys()):
            return False

        for key in pos1:
            if abs(pos1[key] - pos2[key]) > tolerance:
                return False
        return True

    def _is_rate_limited(self, key: str) -> bool:
        """Check if this update is being sent too frequently."""
        current_time = time.time()
        last_time = self._last_update_times.get(key, 0)

        if current_time - last_time < self._min_update_interval:
            return True

        self._last_update_times[key] = current_time
        return False

    def _add_handler(self, topic: str, handler: Callable):
        """Add event handler for a specific topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    def _trigger_handlers(self, topic: str, data: Any):
        """Trigger all handlers for a specific topic."""
        if topic in self._handlers:
            for handler in self._handlers[topic]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in handler for {topic}: {e}")

    def _on_connect(self, client, userdata, flags, rc, *args, **kwargs):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            logger.info(
                f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}"
            )
            self.connected = True
            self._reconnect_attempts = 0

            # Resubscribe to all topics
            for topic in self._handlers.keys():
                client.subscribe(topic)
                logger.debug(f"Subscribed to topic: {topic}")
        else:
            logger.error(
                f"Failed to connect to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}, return code: {rc}"
            )
            self.connected = False

    def _on_disconnect(self, client, userdata, rc, *args, **kwargs):
        """Callback when disconnected from MQTT broker."""
        self.connected = False

        # In paho-mqtt 2.x with CallbackAPIVersion.VERSION2, rc is a DisconnectFlags object
        # Check if this is an unexpected disconnection
        is_unexpected = False

        # Try to check if rc is a DisconnectFlags object (paho-mqtt 2.x)
        if hasattr(rc, "is_disconnect_packet_from_server"):
            # Normal client-initiated disconnections have is_disconnect_packet_from_server=False
            # Server disconnections or abnormal disconnections have it as True
            is_unexpected = rc.is_disconnect_packet_from_server
        elif isinstance(rc, int):
            # Fallback for paho-mqtt 1.x where rc is an integer (0 = normal, non-0 = unexpected)
            is_unexpected = rc != 0

        if is_unexpected:
            logger.warning(
                f"Unexpected MQTT disconnection - rc: {rc}, reason: {kwargs.get('reason_code', 'Unknown')}, broker: {self.mqtt_broker}:{self.mqtt_port}, client_id: {self.client_id}"
            )
            self._reconnect_attempts += 1
            if self._reconnect_attempts < self._max_reconnect_attempts:
                logger.info(
                    f"Attempting to reconnect ({self._reconnect_attempts}/{self._max_reconnect_attempts})..."
                )
            else:
                logger.error("Max reconnection attempts reached")
        else:
            logger.debug(
                f"Normal MQTT disconnection from {self.mqtt_broker}:{self.mqtt_port}"
            )

    def _on_message(self, client, userdata, msg):
        """Callback when a message is received."""
        try:
            topic = msg.topic
            payload = msg.payload.decode("utf-8")

            # Try to parse as JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload

            logger.debug(f"Received message on topic {topic}")

            # Trigger handlers for this topic
            self._trigger_handlers(topic, data)

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def connect(self):
        """Connect to MQTT broker."""
        try:
            logger.debug(
                f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}"
            )
            self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.client.loop_start()

            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.connected:
                raise Exception("Failed to connect to MQTT broker within timeout")

            logger.debug("Successfully connected to MQTT broker")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def publish(self, topic: str, message: Dict[str, Any], qos: int = 0):
        """Publish message to MQTT topic."""
        if not self.connected:
            logger.warning(f"Cannot publish to {topic}: not connected to MQTT broker")
            return

        try:
            payload = json.dumps(message) if isinstance(message, dict) else message
            result = self.client.publish(topic, payload, qos=qos)

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to publish to {topic}: {result.rc}")
            else:
                logger.debug(f"Published to {topic}")
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")

    def subscribe(self, topic: str, handler: Optional[Callable] = None, qos: int = 0):
        """Subscribe to MQTT topic."""
        if handler:
            self._add_handler(topic, handler)

        if self.connected:
            result = self.client.subscribe(topic, qos=qos)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Subscribed to topic: {topic}")
            else:
                logger.error(f"Failed to subscribe to {topic}: {result[0]}")
        else:
            logger.warning(f"Cannot subscribe to {topic}: not connected to MQTT broker")

    # Environment MQTT methods
    def subscribe_environment(
        self, environment_uuid: str, on_update: Optional[Callable] = None
    ):
        """Subscribe to environment updates via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/environment/{environment_uuid}/+"
        self.subscribe(topic, on_update)

    def publish_environment_update(
        self, environment_uuid: str, update_type: str, data: Dict[str, Any]
    ):
        """Publish environment update via MQTT."""
        topic = (
            f"{self.topic_prefix}cyberwave/environment/{environment_uuid}/{update_type}"
        )
        message = {"type": update_type, "data": data, "timestamp": time.time()}
        self.publish(topic, message)

    # Twin MQTT methods
    def subscribe_twin(self, twin_uuid: str, on_update: Optional[Callable] = None):
        """Subscribe to twin updates via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/+"
        self.subscribe(topic, on_update)

    def update_twin_position(self, twin_uuid: str, position: Dict[str, float]):
        """Update twin position via MQTT."""
        # Check if this position is the same as the last one sent
        if twin_uuid in self._last_positions:
            if self._positions_equal(self._last_positions[twin_uuid], position):
                # Position hasn't changed, skip the update
                logger.debug(f"Position hasn't changed for twin {twin_uuid}")
                return

        # Check rate limiting
        rate_key = f"twin:{twin_uuid}:position"
        if self._is_rate_limited(rate_key):
            logger.warning(f"Rate limited for twin {twin_uuid}")
            return

        # Store the new position
        self._last_positions[twin_uuid] = position.copy()

        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/position"
        message = {
            "type": "position",
            "position": position,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def update_twin_rotation(self, twin_uuid: str, rotation: Dict[str, float]):
        """Update twin rotation via MQTT."""
        # Check if this rotation is the same as the last one sent
        if twin_uuid in self._last_rotations:
            if self._positions_equal(self._last_rotations[twin_uuid], rotation):
                # Rotation hasn't changed, skip the update
                logger.debug(f"Rotation hasn't changed for twin {twin_uuid}")
                return

        # Check rate limiting
        rate_key = f"twin:{twin_uuid}:rotation"
        if self._is_rate_limited(rate_key):
            logger.warning(f"Rate limited for twin {twin_uuid}")
            return

        # Store the new rotation
        self._last_rotations[twin_uuid] = rotation.copy()

        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/rotation"
        message = {
            "type": "rotation",
            "rotation": rotation,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def update_twin_scale(self, twin_uuid: str, scale: Dict[str, float]):
        """Update twin scale via MQTT."""
        # Check rate limiting
        rate_key = f"twin:{twin_uuid}:scale"
        if self._is_rate_limited(rate_key):
            return

        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/scale"
        message = {
            "type": "scale",
            "scale": scale,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    # Joint state MQTT methods
    def subscribe_twin_joint_states(
        self, twin_uuid: str, on_update: Optional[Callable] = None
    ):
        """Subscribe to twin joint states via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/joint/{twin_uuid}/+"
        self.subscribe(topic, on_update)

    def update_joint_state(
        self,
        twin_uuid: str,
        joint_name: str,
        position: Optional[float] = None,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ):
        """Update joint state via MQTT."""
        # Check rate limiting
        rate_key = f"joint:{twin_uuid}:{joint_name}"
        if self._is_rate_limited(rate_key):
            return

        joint_state = {}
        if position is not None:
            joint_state["position"] = position
        if velocity is not None:
            joint_state["velocity"] = velocity
        if effort is not None:
            joint_state["effort"] = effort

        topic = f"{self.topic_prefix}cyberwave/joint/{twin_uuid}/update"
        message = {
            "type": "joint_state",
            "joint_name": joint_name,
            "joint_state": joint_state,
            "timestamp": time.time(),
        }
        logger.info(
            f"Publishing joint state for {twin_uuid} {joint_name}: {joint_state}"
        )
        self.publish(topic, message)

    # Sensor stream MQTT methods
    def subscribe_video_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to video stream via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/video"
        self.subscribe(topic, on_frame)

    def subscribe_depth_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to depth stream via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/depth"
        self.subscribe(topic, on_frame)

    def subscribe_pointcloud_stream(
        self, twin_uuid: str, on_pointcloud: Optional[Callable] = None
    ):
        """Subscribe to colored point cloud via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/pointcloud"
        self.subscribe(topic, on_pointcloud)

    def publish_depth_frame(self, twin_uuid: str, depth_data: Dict[str, Any]):
        """Publish depth frame data via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/depth"
        message = {
            "type": "depth_data",
            "data": depth_data,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def publish_webrtc_message(self, twin_uuid: str, webrtc_data: Dict[str, Any]):
        """Publish WebRTC signaling message via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc"
        self.publish(topic, webrtc_data)

    def subscribe_webrtc_messages(
        self, twin_uuid: str, on_message: Optional[Callable] = None
    ):
        """Subscribe to WebRTC signaling messages via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc"
        self.subscribe(topic, on_message)

    # Utility methods
    def ping(self, resource_uuid: str):
        """Send ping message to test connectivity."""
        topic = f"{self.topic_prefix}cyberwave/ping/{resource_uuid}/request"
        message = {"type": "ping", "timestamp": time.time()}
        self.publish(topic, message)

    def subscribe_pong(self, resource_uuid: str, on_pong: Optional[Callable] = None):
        """Subscribe to pong responses."""
        topic = f"{self.topic_prefix}cyberwave/pong/{resource_uuid}/response"
        self.subscribe(topic, on_pong)

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            logger.info("Disconnecting from MQTT broker")
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False


# Export the main client class
__all__ = ["CyberwaveMQTTClient"]
