"""
MQTT Client for Cyberwave Platform

This module provides a high-level MQTT client for real-time communication with the Cyberwave platform.
It uses paho-mqtt (2.1.0+) for reliable MQTT connectivity.
"""

import json
import logging
import threading
import time
import uuid
import re
import ssl
from typing import Any, Callable, Dict, List, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion  # type: ignore
# Try to import CallbackAPIVersion for paho-mqtt 2.x, fallback for older versions

from ..constants import SOURCE_TYPE_EDGE, SOURCE_TYPES

logger = logging.getLogger(__name__)
SOURCE_TYPES_DISPLAY = ", ".join(SOURCE_TYPES)


class CyberwaveMQTTClient:
    """
    Client for Cyberwave MQTT API interactions.

    This client provides methods for publishing and subscribing to MQTT topics
    for digital twin updates, joint states, sensor streams, and more.

    Args:
        mqtt_broker: MQTT broker hostname or IP address
        mqtt_port: MQTT broker port (default: 8883)
        mqtt_username: MQTT username placeholder (default: "mqttcyb")
        api_key: Cyberwave API key used for MQTT authN/authZ
        mqtt_password: Explicit MQTT password (overrides api_key when provided)
        client_id: Custom MQTT client ID (auto-generated if not provided)
        client_id_prefix: Prefix for auto-generated MQTT client IDs
        use_tls: Enable TLS transport for MQTT
        tls_ca_cert: Path to CA certificate bundle for broker verification
        topic_prefix: Prefix for MQTT topics (default: "")
        auto_connect: Automatically connect on initialization (default: True)
        protocol: MQTT protocol version (default: ``mqtt.MQTTv311``).
            Pass ``mqtt.MQTTv5`` to use MQTT v5 features when your broker supports it.
    """

    def __init__(
        self,
        mqtt_broker: str = "mqtt.cyberwave.com",
        mqtt_port: int = 8883,
        mqtt_username: str = "mqttcyb",
        api_key: Optional[str] = None,
        mqtt_password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_id_prefix: str = "sdk_",
        use_tls: bool = False,
        tls_ca_cert: Optional[str] = None,
        topic_prefix: str = "",
        auto_connect: bool = False,
        twin_uuids: Optional[List[str]] = None,
        source_type: Optional[str] = SOURCE_TYPE_EDGE,
        protocol: Optional[int] = None,
    ):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username

        self.api_key = api_key
        self.mqtt_password = mqtt_password

        auth_password = self.mqtt_password or self.api_key
        if not auth_password:
            raise ValueError(
                "api_key or mqtt_password is required for MQTT authentication. "
                "Set CYBERWAVE_API_KEY or pass mqtt_password explicitly"
            )

        # Topic prefix (empty by default, can be set for custom deployments)
        self.topic_prefix = topic_prefix

        # Generate unique client ID
        self.client_id = client_id or f"{client_id_prefix}{uuid.uuid4().hex[:8]}"

        self._protocol = protocol if protocol is not None else mqtt.MQTTv311
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=self.client_id,  # type: ignore
            protocol=self._protocol,
        )

        self.client.username_pw_set(username=self.mqtt_username, password=auth_password)
        # Port 8883 is the conventional MQTT-over-TLS port.
        self.use_tls = use_tls or self.mqtt_port == 8883
        self.tls_ca_cert = tls_ca_cert
        if self.use_tls:
            self.client.tls_set(
                ca_certs=tls_ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
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
        self.client.on_subscribe = self._on_subscribe

        self.twin_uuids = twin_uuids or []
        self.twin_uuids_with_telemetry_start: List[str] = []
        self._telemetry_lock = threading.Lock()  # Thread safety for telemetry tracking
        self._subscription_lock = threading.Lock()
        self._pending_subscriptions: Dict[int, str] = {}
        self.source_type = source_type

        # Auto-connect if requested (must happen after all state is initialized)
        if auto_connect:
            self.connect()

    def _get_effective_source_type(self, source_type: Optional[str]) -> str:
        """Resolve and validate the source type for outgoing MQTT messages."""
        effective_source_type = source_type or self.source_type or SOURCE_TYPE_EDGE
        if effective_source_type not in SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {effective_source_type}. Must be one of: "
                f"{SOURCE_TYPES_DISPLAY}"
            )
        return effective_source_type

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

    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from an MQTT topic and remove all its handlers.

        Idempotent — safe to call even if the topic was never subscribed.
        """
        self._handlers.pop(topic, None)
        if self.connected:
            self.client.unsubscribe(topic)

    def _match_mqtt_pattern(self, pattern: str, topic: str) -> bool:
        """Match MQTT topic against MQTT pattern (supports + and # wildcards)."""
        # Convert MQTT pattern to regex
        # + matches a single level (any characters except /)
        # # matches zero or more levels (must be at end)

        # Escape special regex characters except + and #
        pattern_escaped = re.escape(pattern)
        # Replace escaped \+ with regex for single level
        pattern_escaped = pattern_escaped.replace(r"\+", r"[^/]+")
        # Replace escaped \# with regex for multi-level (only at end)
        if pattern_escaped.endswith(r"\#"):
            pattern_escaped = pattern_escaped[:-2] + r".*"
        elif r"\#" in pattern_escaped:
            # # can only be at the end in MQTT
            return False

        # Match the pattern
        return bool(re.match(f"^{pattern_escaped}$", topic))

    def _trigger_handlers(self, topic: str, data: Any):
        """Trigger all handlers for a specific topic."""
        # First, try exact match
        if topic in self._handlers:
            for handler in self._handlers[topic]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in handler for {topic}: {e}")

        # Then, try pattern matches (for wildcard subscriptions)
        for pattern, handlers in self._handlers.items():
            if pattern != topic and ("+" in pattern or "#" in pattern):
                if self._match_mqtt_pattern(pattern, topic):
                    for handler in handlers:
                        try:
                            # Pass both topic and data to handler if it accepts 2 args
                            import inspect

                            sig = inspect.signature(handler)
                            if len(sig.parameters) >= 2:
                                handler(topic, data)
                            else:
                                handler(data)
                        except Exception as e:
                            logger.error(f"Error in handler for pattern {pattern}: {e}")

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
                result = client.subscribe(topic)
                if result[0] == mqtt.MQTT_ERR_SUCCESS:
                    with self._subscription_lock:
                        self._pending_subscriptions[result[1]] = topic
                    logger.debug(
                        "Resubscribe request sent for topic %s (mid=%s)",
                        topic,
                        result[1],
                    )
                else:
                    logger.error("Failed to resubscribe to %s: %s", topic, result[0])
        else:
            logger.error(
                f"Failed to connect to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}, return code: {rc}"
            )
            self.connected = False

    def _on_subscribe(self, client, userdata, mid, reason_codes=None, properties=None):
        """Callback when the broker acknowledges a subscription."""
        del client, userdata, properties

        with self._subscription_lock:
            topic = self._pending_subscriptions.pop(mid, f"<unknown mid={mid}>")

        if reason_codes is None:
            codes = []
        elif isinstance(reason_codes, (list, tuple)):
            codes = list(reason_codes)
        else:
            codes = [reason_codes]

        statuses = []
        failed = False
        for code in codes:
            value = getattr(code, "value", code if isinstance(code, int) else None)
            label = str(code)
            statuses.append(f"{label} ({value})" if isinstance(value, int) else label)
            if isinstance(value, int):
                failed = failed or value >= 128

        status_text = ", ".join(statuses) if statuses else "no reason codes"
        if failed:
            logger.error(
                "SUBACK rejected subscription for %s (mid=%s): %s",
                topic,
                mid,
                status_text,
            )
        else:
            logger.info(
                "SUBACK accepted subscription for %s (mid=%s): %s",
                topic,
                mid,
                status_text,
            )

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

    def _handle_twin_update_with_telemetry(
        self, twin_uuid: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Handle telemetry start for a twin, ensuring it's only sent once.

        Thread-safe: Uses a lock to prevent duplicate telemetry_start messages
        when called from multiple threads (e.g., main loop + camera worker).
        """
        with self._telemetry_lock:
            if twin_uuid not in self.twin_uuids:
                self.twin_uuids.append(twin_uuid)

            already_started = twin_uuid in self.twin_uuids_with_telemetry_start
            logger.debug(
                "_handle_twin_update_with_telemetry: twin=%s already_started=%s "
                "current_tracking_list=%s",
                twin_uuid,
                already_started,
                self.twin_uuids_with_telemetry_start,
            )
            if not already_started:
                self.twin_uuids_with_telemetry_start.append(twin_uuid)
                self._publish_connect_message(twin_uuid)
                self._publish_telemetry_start_message(twin_uuid, metadata)

    def _publish_connect_message(self, twin_uuid: str):
        """Publish connect message to MQTT broker."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/telemetry"
        message = {
            "type": "connected",
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def _publish_disconnect_message(self, twin_uuid: str):
        """Publish disconnect message to MQTT broker."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/telemetry"
        message = {
            "type": "disconnected",
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def connect(self):
        """Connect to MQTT broker."""
        try:
            logger.warning(
                "MQTT connection settings: tls=%s, broker=%s, port=%s, custom_ca=%s",
                self.use_tls,
                self.mqtt_broker,
                self.mqtt_port,
                bool(self.tls_ca_cert),
            )
            logger.debug(
                f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}"
            )
            self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.client.loop_start()

            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.5)

            if not self.connected:
                raise Exception("Failed to connect to MQTT broker within timeout")

            logger.debug("Successfully connected to MQTT broker")

            # Send telemetry start message only for twins that haven't received one yet
            # This prevents duplicate telemetry_start messages on reconnection
            # Thread-safe: Uses lock to coordinate with _handle_twin_update_with_telemetry
            with self._telemetry_lock:
                for twin_uuid in self.twin_uuids:
                    if twin_uuid not in self.twin_uuids_with_telemetry_start:
                        self.twin_uuids_with_telemetry_start.append(twin_uuid)
                        self._publish_connect_message(twin_uuid)
                        self._publish_telemetry_start_message(twin_uuid, None)
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker."""

        for twin_uuid in self.twin_uuids:
            self._publish_disconnect_message(twin_uuid)
            self.publish_telemetry_end(twin_uuid)
        if self.connected:
            logger.info("Disconnecting from MQTT broker")
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False

    def publish(self, topic: str, message: Dict[str, Any], qos: int = 0):
        """Publish message to MQTT topic."""
        if not self.connected:
            logger.warning(f"Cannot publish to {topic}: not connected to MQTT broker")
            return

        try:
            if isinstance(message, dict):
                message.setdefault("session_id", self.client_id)
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
                with self._subscription_lock:
                    self._pending_subscriptions[result[1]] = topic
                logger.info(
                    "Subscribe request sent for topic: %s (mid=%s), awaiting SUBACK",
                    topic,
                    result[1],
                )
            else:
                logger.error(f"Failed to subscribe to {topic}: {result[0]}")
        else:
            logger.warning(f"Cannot subscribe to {topic}: not connected to MQTT broker")

    # Telemetry MQTT methods
    def _publish_telemetry_start_message(
        self, twin_uuid: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build and publish telemetry_start message to MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/telemetry"
        message: Dict[str, Any] = {
            "type": "telemetry_start",
            "timestamp": time.time(),
        }
        if metadata is not None:
            if "fps" in metadata:
                message["fps"] = metadata["fps"]
            if "observations" in metadata:
                message["observations"] = metadata["observations"]
            if "camera_participants" in metadata:
                message["camera_participants"] = metadata["camera_participants"]
        logger.info(
            f"Publishing telemetry start message for twin {twin_uuid}: {message}"
        )
        self.publish(topic, message)

    def publish_telemetry_start_message(
        self, twin_uuid: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publish telemetry_start message unconditionally (no already_started check).

        Registers the twin and publishes connect + telemetry_start. Caller is
        responsible for ensuring this is only called once (e.g. scripts manage
        their own "already started" state).

        Args:
            twin_uuid: UUID of the twin
            metadata: Optional dict (e.g. fps, observations, camera_participants)
        """
        with self._telemetry_lock:
            if twin_uuid not in self.twin_uuids:
                self.twin_uuids.append(twin_uuid)
            if twin_uuid not in self.twin_uuids_with_telemetry_start:
                self.twin_uuids_with_telemetry_start.append(twin_uuid)
            self._publish_connect_message(twin_uuid)
            self._publish_telemetry_start_message(twin_uuid, metadata)

    def publish_telemetry_start(
        self, twin_uuid: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publish telemetry start message via MQTT.

        Registers the twin so no duplicate telemetry_start is sent when joint
        updates or other twin updates trigger _handle_twin_update_with_telemetry.

        Args:
            twin_uuid: UUID of the twin
            metadata: Optional dict (e.g. {"fps": 100, "observations": {"edge_leader": {...}, "edge_follower": {...}}})
        """
        self._handle_twin_update_with_telemetry(twin_uuid, metadata)

    def publish_telemetry_end(
        self,
        twin_uuid: str,
        sensor: str | None = None,
        stream_source: str | None = None,
        stream_instance_id: str | None = None,
    ):
        """Publish telemetry end message via MQTT.

        Also clears the telemetry tracking state for this twin, allowing
        subsequent publish_telemetry_start calls to work properly when
        a new operation (teleoperate/remoteoperate) is started.

        This method is idempotent: if telemetry_end was already published for
        this twin (i.e., twin is no longer in tracking list), this call is a
        no-op to avoid sending duplicate telemetry_end messages.
        """
        # Check and clear tracking state atomically to ensure idempotency.
        # Only publish if the twin was still being tracked (telemetry_start was sent).
        with self._telemetry_lock:
            was_in_list = twin_uuid in self.twin_uuids_with_telemetry_start
            if was_in_list:
                self.twin_uuids_with_telemetry_start.remove(twin_uuid)
            logger.info(
                "publish_telemetry_end: twin %s was_in_tracking_list=%s, "
                "remaining_tracked_twins=%s",
                twin_uuid,
                was_in_list,
                self.twin_uuids_with_telemetry_start,
            )

        # Skip publishing if already ended (idempotent behavior)
        if not was_in_list:
            logger.debug(
                "publish_telemetry_end: skipping duplicate for twin %s (already ended)",
                twin_uuid,
            )
            return

        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/telemetry"
        message = {
            "type": "telemetry_end",
            "timestamp": time.time(),
        }
        if sensor:
            message["sensor"] = sensor
        if stream_source:
            message["stream_source"] = stream_source
        if stream_instance_id:
            message["stream_instance_id"] = stream_instance_id
        self.publish(topic, message)

    def publish_connected(self, twin_uuid: str):
        """Publish connected message via MQTT.

        Call this when starting an operation (teleoperate, remoteoperate) to
        indicate the twin is now connected/online.
        """
        self._publish_connect_message(twin_uuid)

    def publish_disconnected(self, twin_uuid: str):
        """Publish disconnected message via MQTT.

        Call this when stopping an operation to indicate the twin is no longer
        actively streaming telemetry.
        """
        self._publish_disconnect_message(twin_uuid)

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
        self._handle_twin_update_with_telemetry(twin_uuid)

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
            "source_type": self.source_type,
            "type": "position",
            "position": position,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def update_twin_rotation(self, twin_uuid: str, rotation: Dict[str, float]):
        """Update twin rotation via MQTT."""
        # Check if this rotation is the same as the last one sent

        self._handle_twin_update_with_telemetry(twin_uuid)
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
            "source_type": self.source_type,
            "type": "rotation",
            "rotation": rotation,
            "timestamp": time.time(),
        }
        self.publish(topic, message)

    def update_twin_scale(self, twin_uuid: str, scale: Dict[str, float]):
        """Update twin scale via MQTT."""

        self._handle_twin_update_with_telemetry(twin_uuid)
        # Check rate limiting
        rate_key = f"twin:{twin_uuid}:scale"
        if self._is_rate_limited(rate_key):
            return

        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/scale"
        message = {
            "source_type": self.source_type,
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
        timestamp: Optional[float] = None,
        source_type: Optional[str] = None,
    ):
        """
        Update joint state via MQTT.

        Args:
            twin_uuid: UUID of the twin
            joint_name: Name of the joint
            position: Joint position (radians for revolute, meters for prismatic)
            velocity: Joint velocity
            effort: Joint effort/torque
            timestamp: Unix timestamp (defaults to current time)
            source_type: Source type for the message. Must be one of:
                SOURCE_TYPE_EDGE, SOURCE_TYPE_TELE, SOURCE_TYPE_EDIT, SOURCE_TYPE_SIM.
                Defaults to SOURCE_TYPE_EDGE (SDKs run on edge devices by default).
                Users can override this to use any source type they need.
        """
        effective_source_type = self._get_effective_source_type(source_type)

        self._handle_twin_update_with_telemetry(twin_uuid)
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
            "source_type": effective_source_type,
            "type": "joint_state",
            "joint_name": joint_name,
            "joint_state": joint_state,
            "timestamp": timestamp or time.time(),
        }
        logger.debug(
            f"Publishing joint state for {twin_uuid} {joint_name}: {joint_state} (source_type: {effective_source_type})"
        )

        self.publish(topic, message)

    def update_joints_state(
        self,
        twin_uuid: str,
        joint_positions: Dict[str, float],
        source_type: Optional[str] = None,
        velocities: Optional[Dict[str, float]] = None,
        efforts: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
        source_subtype: Optional[str] = None,
        workload_uuid: Optional[str] = None,
        session_id: Optional[str] = None,
        camera_frame_counters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Update multiple joints at once via MQTT.

        Supports two formats based on provided parameters:

        1. **Flat format** (joint_positions only, no velocities/efforts/timestamp):
           Sends positions directly as top-level keys. Simple and lightweight.
           ```json
           {"source_type": "edge", "_1": 0.5, "_2": 0.3}
           ```

        2. **Aggregated format** (with velocities, efforts, timestamp, or metadata):
           Sends structured payload with nested objects. Supports full joint state
           and additional metadata for telemetry tracking.
           ```json
           {
               "source_type": "edge_follower",
               "positions": {"_1": 0.5, "_2": 0.3},
               "velocities": {"_1": 0.0, "_2": 0.0},
               "efforts": {"_1": 0.0, "_2": 0.0},
               "timestamp": 1709123456.789,
               "source_subtype": "openvla",
               "workload_uuid": "uuid-here",
               "session_id": "session-id",
               "camera_frame_counters": {
                   "<track_id>": {"frame_count": 1234, "sensor_id": "wrist"}
               }
           }
           ```

        Both formats are parsed by Vector into individual joint_state_update telemetry events.

        Args:
            twin_uuid: UUID of the twin
            joint_positions: Dict of joint names to positions (e.g., {"_1": 0.5, "_2": 0.3})
            source_type: SOURCE_TYPE_EDGE (default), SOURCE_TYPE_TELE, SOURCE_TYPE_EDIT, etc.
            velocities: Optional dict of joint names to velocities (triggers aggregated format)
            efforts: Optional dict of joint names to efforts (triggers aggregated format)
            timestamp: Optional timestamp in seconds (triggers aggregated format)
            source_subtype: Optional subtype (e.g., "openvla" for inference workloads)
            workload_uuid: Optional UUID of the workload generating this update
            session_id: Optional session ID for grouping related updates
            camera_frame_counters: Optional dict mapping camera track_id to frame info.
                Each value is a dict with "frame_count" (int) and "sensor_id" (str).
                Used for robot-camera synchronization. Only included in aggregated format.
        """
        effective_source_type = self._get_effective_source_type(source_type)

        if not joint_positions:
            raise ValueError("joint_positions cannot be empty")

        self._handle_twin_update_with_telemetry(twin_uuid)

        topic = f"{self.topic_prefix}cyberwave/joint/{twin_uuid}/update"

        # Determine format: use aggregated if any extended parameters are provided
        use_aggregated = (
            velocities is not None
            or efforts is not None
            or timestamp is not None
            or source_subtype is not None
            or workload_uuid is not None
            or session_id is not None
            or camera_frame_counters is not None
        )

        if use_aggregated:
            message: Dict[str, Any] = {
                "source_type": effective_source_type,
                "positions": joint_positions,
                "timestamp": timestamp if timestamp is not None else time.time(),
            }
            if velocities:
                message["velocities"] = velocities
            if efforts:
                message["efforts"] = efforts
            if source_subtype:
                message["source_subtype"] = source_subtype
            if workload_uuid:
                message["workload_uuid"] = workload_uuid
            if session_id:
                message["session_id"] = session_id
            if camera_frame_counters:
                message["camera_frame_counters"] = camera_frame_counters

            logger.debug(
                f"Publishing aggregated joint state for {twin_uuid}: "
                f"{len(joint_positions)} joints (source_type: {effective_source_type})"
            )
        else:
            # Flat format: positions as top-level keys
            message = {
                "source_type": effective_source_type,
                **joint_positions,
            }
            logger.debug(
                f"Publishing joint state for {twin_uuid}: {len(joint_positions)} joints "
                f"(source_type: {effective_source_type})"
            )

        self.publish(topic, message)

    def update_aggregated_joints_state(
        self,
        twin_uuid: str,
        joint_positions: Dict[str, float],
        source_type: Optional[str] = None,
        velocities: Optional[Dict[str, float]] = None,
        efforts: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
        source_subtype: Optional[str] = None,
        workload_uuid: Optional[str] = None,
        session_id: Optional[str] = None,
        camera_frame_counters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Alias for update_joints_state with aggregated format.

        Deprecated: Use update_joints_state() instead. This method is kept for
        backward compatibility but simply delegates to update_joints_state().
        """
        # Force aggregated format by ensuring timestamp is set
        if timestamp is None:
            timestamp = time.time()

        return self.update_joints_state(
            twin_uuid=twin_uuid,
            joint_positions=joint_positions,
            source_type=source_type,
            velocities=velocities,
            efforts=efforts,
            timestamp=timestamp,
            source_subtype=source_subtype,
            workload_uuid=workload_uuid,
            session_id=session_id,
            camera_frame_counters=camera_frame_counters,
        )

    def publish_initial_observation(
        self, twin_uuid: str, observations: Dict[str, Any], fps: float = 30.0
    ):
        """Send initial observation to the leader twin."""
        if twin_uuid not in self.twin_uuids_with_telemetry_start:
            metadata = {
                "fps": fps,
                "observations": observations,
            }
            self._handle_twin_update_with_telemetry(twin_uuid, metadata)
        else:
            topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/telemetry"
            message = {
                "type": "initial_observation",
                "observations": observations,
                "fps": fps,
                "timestamp": time.time(),
            }
            self.publish(topic, message)

    # Sensor stream MQTT methods
    def subscribe_video_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to video stream via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/video"
        self.subscribe(topic, on_frame)

    def subscribe_depth_stream(
        self, twin_uuid: str, on_frame: Optional[Callable] = None
    ):
        """Subscribe to depth stream via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/depth"
        self.subscribe(topic, on_frame)

    def subscribe_pointcloud_stream(
        self, twin_uuid: str, on_pointcloud: Optional[Callable] = None
    ):
        """Subscribe to colored point cloud via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/pointcloud"
        self.subscribe(topic, on_pointcloud)

    def publish_depth_frame(
        self,
        twin_uuid: str,
        depth_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ):
        """Publish depth frame data via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/depth"
        message = {
            "type": "depth_data",
            "data": depth_data,
            "timestamp": timestamp or time.time(),
        }
        self.publish(topic, message)

    def publish_webrtc_message(self, twin_uuid: str, webrtc_data: Dict[str, Any]):
        """Publish WebRTC signaling message via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        msg_type = webrtc_data.get("type")
        if msg_type == "offer":
            topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-offer"
        elif msg_type == "answer":
            topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-answer"
        elif msg_type == "candidate":
            topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-candidate"
        else:
            topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc"
        self.publish(topic, webrtc_data)

    def subscribe_webrtc_messages(
        self, twin_uuid: str, on_message: Optional[Callable] = None
    ):
        """Subscribe to WebRTC signaling messages via MQTT."""
        self._handle_twin_update_with_telemetry(twin_uuid)
        # Subscribe to specialized topics
        self.subscribe(
            f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-offer", on_message
        )
        self.subscribe(
            f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-answer", on_message
        )
        self.subscribe(
            f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/webrtc-candidate",
            on_message,
        )

    def publish_command_message(self, twin_uuid: str, status):
        """Publish command response message via MQTT.

        Args:
            twin_uuid: The twin UUID to publish to
            status: Either a string status (e.g., "ok") or a dict with status and other fields
        """
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/command"
        if isinstance(status, dict):
            message = status  # Use dict directly
        else:
            message = {"status": status}  # Wrap string in dict
        self.publish(topic, message)

    def subscribe_command_message(
        self, twin_uuid: str, on_command: Optional[Callable] = None
    ):
        """Subscribe to Egde command messages via MQTT."""
        topic = f"{self.topic_prefix}cyberwave/twin/{twin_uuid}/command"
        self.subscribe(topic, on_command)

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


# Export the main client class
__all__ = ["CyberwaveMQTTClient"]
