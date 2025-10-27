"""
MQTT client wrapper for real-time communication with Cyberwave platform
"""

import logging
from typing import Callable, Optional, Dict, Any

try:
    from .mqtt.messaging import Messaging  # type: ignore
    from .mqtt.twinPositionPayload import TwinPositionPayload  # type: ignore
    from .mqtt.twinRotationPayload import TwinRotationPayload  # type: ignore
    from .mqtt.twinScalePayload import TwinScalePayload  # type: ignore
    from .mqtt.jointStatePayload import JointStatePayload  # type: ignore
    from .mqtt.pingPayload import PingPayload  # type: ignore
    from .mqtt.pongPayload import PongPayload  # type: ignore
    from .mqtt.webRTCOfferPayload import WebRTCOfferPayload  # type: ignore
except ImportError as e:
    logging.warning(f"MQTT modules not found. Real-time features will be disabled. Error: {e}")
    Messaging = None

from .config import CyberwaveConfig
from .exceptions import CyberwaveMQTTError

logger = logging.getLogger(__name__)


class CyberwaveMQTTClient:
    """
    Wrapper for MQTT communication with the Cyberwave platform.
    
    Provides high-level methods for publishing and subscribing to twin updates,
    joint states, and other real-time events.
    """
    
    def __init__(self, config: CyberwaveConfig):
        """
        Initialize MQTT client
        
        Args:
            config: Cyberwave configuration object
        """
        if Messaging is None:
            raise CyberwaveMQTTError(
                "MQTT client not available. Please ensure the MQTT SDK is generated "
                "in the 'mqtt' directory."
            )
        
        self.config = config
        self._client: Optional[Messaging] = None
        self._subscriptions: Dict[str, Callable] = {}
        self._connected = False
    
    def connect(self):
        """Connect to the MQTT broker"""
        if self._connected:
            return
        
        # Create config dict for Messaging class
        mqtt_config = {
            'host': self.config.mqtt_host,
            'port': str(self.config.mqtt_port),
        }
        
        if self.config.mqtt_username:
            mqtt_config['username'] = self.config.mqtt_username
        
        if self.config.mqtt_password:
            mqtt_config['password'] = self.config.mqtt_password
        
        try:
            # Create a simple config-like object
            class ConfigDict(dict):
                def get(self, key, default=None):
                    return super().get(key, default)
            
            config_obj = ConfigDict(mqtt_config)
            self._client = Messaging(config_obj)
            self._connected = True
            logger.info(f"Connected to MQTT broker at {self.config.mqtt_host}:{self.config.mqtt_port}")
        except Exception as e:
            raise CyberwaveMQTTError(f"Failed to connect to MQTT broker: {e}")
    
    def disconnect(self):
        """Disconnect from the MQTT broker"""
        if self._client and self._connected:
            # paho-mqtt client doesn't have explicit disconnect in loop mode
            self._connected = False
            logger.info("Disconnected from MQTT broker")
    
    def subscribe_twin_position(self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to twin position updates
        
        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when position updates are received
        """
        topic = f"cyberwave.twin.{twin_uuid}.position"
        
        def on_message(client, userdata, msg):
            try:
                payload = TwinPositionPayload.from_json(msg.payload.decode('utf-8'))
                callback(payload.to_dict() if hasattr(payload, 'to_dict') else payload)
            except Exception as e:
                logger.error(f"Error processing position update: {e}")
        
        self._subscribe(topic, on_message)
    
    def subscribe_twin_rotation(self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to twin rotation updates
        
        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when rotation updates are received
        """
        topic = f"cyberwave.twin.{twin_uuid}.rotation"
        
        def on_message(client, userdata, msg):
            try:
                payload = TwinRotationPayload.from_json(msg.payload.decode('utf-8'))
                callback(payload.to_dict() if hasattr(payload, 'to_dict') else payload)
            except Exception as e:
                logger.error(f"Error processing rotation update: {e}")
        
        self._subscribe(topic, on_message)
    
    def subscribe_joint_states(self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to joint state updates
        
        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when joint state updates are received
        """
        topic = f"cyberwave.twin.{twin_uuid}.joint-states"
        
        def on_message(client, userdata, msg):
            try:
                payload = JointStatePayload.from_json(msg.payload.decode('utf-8'))
                callback(payload.to_dict() if hasattr(payload, 'to_dict') else payload)
            except Exception as e:
                logger.error(f"Error processing joint state update: {e}")
        
        self._subscribe(topic, on_message)
    
    def publish_twin_position(self, twin_uuid: str, x: float, y: float, z: float):
        """
        Publish twin position update
        
        Args:
            twin_uuid: UUID of the twin
            x, y, z: Position coordinates
        """
        if not self._connected:
            self.connect()
        
        topic = f"cyberwave.twin.{twin_uuid}.position"
        payload = TwinPositionPayload(x=x, y=y, z=z)
        
        try:
            payload_json = payload.to_json()
            if self._client:
                self._client.publish(topic, payload_json)
            logger.debug(f"Published position to {topic}")
        except Exception as e:
            raise CyberwaveMQTTError(f"Failed to publish position: {e}")
    
    def publish_twin_rotation(self, twin_uuid: str, x: float, y: float, z: float, w: float):
        """
        Publish twin rotation update (quaternion)
        
        Args:
            twin_uuid: UUID of the twin
            x, y, z, w: Quaternion components
        """
        if not self._connected:
            self.connect()
        
        topic = f"cyberwave.twin.{twin_uuid}.rotation"
        payload = TwinRotationPayload(x=x, y=y, z=z, w=w)
        
        try:
            payload_json = payload.to_json()
            if self._client:
                self._client.publish(topic, payload_json)
            logger.debug(f"Published rotation to {topic}")
        except Exception as e:
            raise CyberwaveMQTTError(f"Failed to publish rotation: {e}")
    
    def _subscribe(self, topic: str, callback: Callable):
        """
        Internal method to subscribe to a topic
        
        Args:
            topic: MQTT topic to subscribe to
            callback: Callback function for messages
        """
        if not self._connected:
            self.connect()
        
        if topic not in self._subscriptions:
            messenger = Messaging(
                ConfigDict({'host': self.config.mqtt_host, 'port': str(self.config.mqtt_port)}),
                subscription=topic,
                on_message=callback
            )
            messenger.loop_start()
            self._subscriptions[topic] = messenger
            logger.info(f"Subscribed to {topic}")


class ConfigDict(dict):
    """Helper class to make dict behave like config object"""
    def get(self, key, default=None):
        return super().get(key, default)

