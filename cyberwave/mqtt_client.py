"""
MQTT client wrapper for real-time communication with Cyberwave platform.

This module provides a compatibility layer that adapts the CyberwaveMQTTClient
from the mqtt module to work with the CyberwaveConfig object used by the main client.
"""

import logging
from typing import Callable, Optional, Dict, Any, List, Union

from .config import CyberwaveConfig, DEFAULT_MQTT_PORT, DEFAULT_MQTT_USERNAME
from .mqtt import CyberwaveMQTTClient as BaseMQTTClient
from .mqtt import TELEMETRY_CUT_SENDER, TELEMETRY_CUT_SOURCE_SUBTYPE, _UNSET

logger = logging.getLogger(__name__)


class CyberwaveMQTTClient:
    """
    Wrapper for MQTT communication with the Cyberwave platform.

    This class adapts the BaseMQTTClient to work with CyberwaveConfig objects,
    providing a compatibility layer for the main Cyberwave client.

    Provides high-level methods for publishing and subscribing to twin updates,
    joint states, and other real-time events.

    TODO: We should just use the BaseMQTTClient and call that CyberwaveMQTTClient
    """

    def __init__(self, config: CyberwaveConfig, mqtt_password: Optional[str] = None):
        """
        Initialize MQTT client from a CyberwaveConfig object.

        Args:
            config: Cyberwave configuration object containing MQTT settings
            mqtt_password: Optional explicit MQTT password override
        """
        self.config = config

        # Determine broker, port, username, and API key from config.
        mqtt_broker = config.mqtt_host or "mqtt.cyberwave.com"
        mqtt_port = config.mqtt_port or DEFAULT_MQTT_PORT
        mqtt_username = config.mqtt_username or DEFAULT_MQTT_USERNAME
        api_key = config.api_key
        # Explicit MQTT password (ctor arg or CYBERWAVE_MQTT_PASSWORD) wins over API key
        # so CI can use legacy broker credentials while REST keeps the API token.
        # Forwarded only when genuinely explicit: the base client reads a password
        # here as legacy credentials unless it is an API token, in which case it
        # still derives the username from it.
        explicit_mqtt_password = mqtt_password or config.mqtt_password
        if not (explicit_mqtt_password or api_key):
            raise ValueError(
                "API key or mqtt_password is required. "
                "Set CYBERWAVE_API_KEY or pass mqtt_password explicitly"
            )

        # Determine topic prefix from config (which handles env vars)
        topic_prefix = config.topic_prefix or ""

        self._topic_prefix = topic_prefix

        # Initialize the base MQTT client
        self._client = BaseMQTTClient(
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            mqtt_username=mqtt_username,
            api_key=api_key,
            mqtt_password=explicit_mqtt_password,
            use_tls=config.mqtt_use_tls,
            tls_ca_cert=config.mqtt_tls_ca_cert,
            topic_prefix=topic_prefix,
            runtime_mode=config.runtime_mode,
            source_type=config.source_type,
            protocol=config.mqtt_protocol,
        )

    @property
    def connected(self) -> bool:
        """Check if the client is connected to the MQTT broker."""
        return self._client.connected

    @property
    def disconnect_count(self) -> int:
        """Number of disconnect callbacks, including outages between driver ticks."""
        return self._client.disconnect_count

    @property
    def topic_prefix(self) -> str:
        """Get the topic prefix used by this MQTT client."""
        return self._topic_prefix

    @property
    def client_id(self) -> str:
        """Get the MQTT client ID."""
        return self._client.client_id

    def connect(self):
        """Connect to the MQTT broker."""
        if not self.connected:
            self._client.connect()

    @property
    def connecting_in_background(self) -> bool:
        """Whether paho's network thread is (re)connecting on its own.

        True for :meth:`connect_in_background`, and after a dropped connection
        while paho reconnects; :meth:`connect` would only wait for it then.
        """
        return (
            self._client.connecting_in_background
            or self._client._network_loop_running
        )

    def connect_in_background(self):
        """Keep connecting from the network thread until the broker answers."""
        if not self.connected:
            self._client.connect_in_background()

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
        # Include topic prefix to match backend pattern
        prefix = self.topic_prefix
        topic = f"{prefix}cyberwave/twin/{twin_uuid}/position"
        return self._client.subscribe(topic, callback)

    def subscribe_twin_rotation(
        self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to twin rotation updates.

        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when rotation updates are received
        """
        # Include topic prefix to match backend pattern
        prefix = self.topic_prefix
        topic = f"{prefix}cyberwave/twin/{twin_uuid}/rotation"
        return self._client.subscribe(topic, callback)

    def subscribe_joint_states(
        self, twin_uuid: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to joint state updates.

        Args:
            twin_uuid: UUID of the twin to monitor
            callback: Function to call when joint state updates are received
        """
        # Include topic prefix to match backend pattern
        prefix = self.topic_prefix
        topic = f"{prefix}cyberwave/joint/{twin_uuid}/+"
        return self._client.subscribe(topic, callback)

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

    def publish_metric(
        self,
        twin_uuid: str,
        channel: str,
        value: Any,
        *,
        timestamp: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> bool:
        """
        Publish one float, int or bool sample on ``metrics/{channel}``.

        See ``cyberwave.mqtt.CyberwaveMQTTClient.publish_metric``.

        Args:
            twin_uuid: UUID of the twin
            channel: ``.``-separated name, e.g. ``"arm.gripper"``
            value: A float, an int or a bool
            timestamp: Measurement time in Unix seconds, not ms (default: now)
            source_type: Override the default source type
        """
        return self._client.publish_metric(
            twin_uuid, channel, value, timestamp=timestamp, source_type=source_type
        )

    def publish_metrics(
        self,
        twin_uuid: str,
        metrics: Dict[str, Any],
        *,
        timestamp: Optional[float] = None,
        source_type: Optional[str] = None,
    ) -> bool:
        """
        Publish several float, int or bool samples in one ``metrics`` bag.

        See ``cyberwave.mqtt.CyberwaveMQTTClient.publish_metrics``.

        Args:
            twin_uuid: UUID of the twin
            metrics: Channel name -> sample
            timestamp: Measurement time in Unix seconds, not ms (default: now)
            source_type: Override the default source type
        """
        return self._client.publish_metrics(
            twin_uuid, metrics, timestamp=timestamp, source_type=source_type
        )

    def update_twin_scale(self, twin_uuid: str, scale: Dict[str, float]):
        """
        Update twin scale via MQTT.

        Args:
            twin_uuid: UUID of the twin
            scale: Dictionary with scale values
        """
        return self._client.update_twin_scale(twin_uuid, scale)

    def publish_initial_observation(
        self, twin_uuid: str, observations: Dict[str, Any], fps: float = 30.0
    ):
        """
        Send initial observations to the leader twin.

        Args:
            twin_uuid: UUID of the twin
            observations: Dictionary of observations
        """
        return self._client.publish_initial_observation(twin_uuid, observations, fps)

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
        return self._client.publish_telemetry_start(twin_uuid, metadata)

    def publish_telemetry_start_message(
        self, twin_uuid: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publish telemetry_start message unconditionally (no already_started check).

        Registers the twin and publishes connect + telemetry_start. Caller is
        responsible for ensuring this is only called once.

        Args:
            twin_uuid: UUID of the twin
            metadata: Optional dict (e.g. {"fps": 100, "observations": {...}})
        """
        return self._client.publish_telemetry_start_message(twin_uuid, metadata)

    def publish_telemetry_end(
        self,
        twin_uuid: str,
        sensor: Optional[str] = None,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
    ) -> None:
        """
        Publish telemetry_end. A twin-wide end (no ``sensor``) also clears SDK
        telemetry-start tracking for this twin; an end that names a ``sensor``
        closes only that stream and leaves the twin's session open.

        Match edge teleop cleanup so recordings/sessions end cleanly.
        """
        return self._client.publish_telemetry_end(
            twin_uuid,
            sensor=sensor,
            stream_source=stream_source,
            stream_instance_id=stream_instance_id,
        )

    def publish_telemetry_cut(
        self,
        twin_uuid: str,
        source_type: Optional[str] = None,
        sender: str = TELEMETRY_CUT_SENDER,
        source_subtype: str = TELEMETRY_CUT_SOURCE_SUBTYPE,
    ) -> None:
        """Publish a recording-session hard cut (telemetry_end + telemetry_start).

        For a publisher that does not own the twin's telemetry session (e.g. a
        workflow recorder node bounding a window in a driver's continuous
        stream). Both messages are marked ``recording_boundary: True`` so
        consumers do not mistake the cut for a peer disconnect and tear down a
        live WebRTC stream. See ``MQTTClient.publish_telemetry_cut`` for the
        full rationale.
        """
        return self._client.publish_telemetry_cut(
            twin_uuid,
            source_type=source_type,
            sender=sender,
            source_subtype=source_subtype,
        )

    def publish_recording_boundary(
        self,
        twin_uuid: str,
        state: str,
        *,
        stream_source: str,
        stream_instance_id: str,
        observations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Open (``"start"``) or close (``"end"``) one recording window.

        For a publisher that switches recording on and off while its streams
        keep running, such as a simulator's recording toggle. See
        ``MQTTClient.publish_recording_boundary`` for the full rationale.
        """
        return self._client.publish_recording_boundary(
            twin_uuid,
            state,
            stream_source=stream_source,
            stream_instance_id=stream_instance_id,
            observations=observations,
        )

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
        return self._client.update_joint_state(
            twin_uuid,
            joint_name,
            position,
            velocity,
            effort,
            timestamp,
            source_type=source_type,
        )

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
        as_targets: bool = False,
    ):
        """
        Update multiple joints at once via MQTT.

        Supports two formats based on provided parameters:

        1. **Flat format** (joint_positions only):
           Simple and lightweight, sends positions as top-level keys.

        2. **Aggregated format** (with velocities, efforts, timestamp, or metadata):
           Sends structured payload with nested objects and full telemetry metadata.

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
            camera_frame_counters: Optional dict mapping track_id to frame info used
                for robot-camera synchronization.
            as_targets: When True, publish as a *command* using ``target_*`` field
                names (target_positions/target_velocities/target_efforts) instead
                of the measured names. Used by controllers so the plant and viewers
                never confuse a commanded setpoint with measured robot state.
        """
        return self._client.update_joints_state(
            twin_uuid,
            joint_positions,
            source_type,
            velocities,
            efforts,
            timestamp,
            source_subtype,
            workload_uuid,
            session_id,
            camera_frame_counters,
            as_targets,
        )

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

        Deprecated: Use update_joints_state() instead.
        """
        return self._client.update_aggregated_joints_state(
            twin_uuid,
            joint_positions,
            source_type,
            velocities,
            efforts,
            timestamp,
            source_subtype,
            workload_uuid,
            session_id,
            camera_frame_counters,
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

    def publish_frame_poses(
        self,
        twin_uuid: str,
        frames: List[Dict[str, Any]],
        *,
        substrate: Optional[str] = None,
        pose_provenance: Optional[str] = None,
        frame_id: str = "environment",
        source_subtype: Optional[str] = None,
        producer_instance: Optional[str] = None,
        producer_sequence: Optional[Union[int, str]] = None,
        workload_uuid: Optional[str] = None,
        simulation_time_s: Optional[float] = None,
    ) -> bool:
        """
        Publish environment-frame poses for one or more robot frames.

        Args:
            twin_uuid: UUID of the twin
            frames: ``[{"child_frame_id", "position": {x,y,z},
                "orientation": {x,y,z,w}, "pose_provenance"?,
                "pose_covariance"?}, ...]``, at least one
            substrate: ``"sim"`` or ``"hardware"``; defaults from the client's
                source type.
            pose_provenance: How the poses were obtained (``"odometry"``,
                ``"slam"``...), for entries that do not say. Ground truth by
                default on a simulator; required on hardware.
            frame_id: Must be ``"environment"``.
            source_subtype, producer_instance, producer_sequence,
            workload_uuid, simulation_time_s: Envelope fields, as
                :meth:`cyberwave.mqtt.CyberwaveMQTTClient.publish_frame_poses`
                takes them.

        Raises:
            ValueError: ``twin_uuid`` is not a UUID, ``frames`` is empty, a
                frame or envelope field is malformed, or the frames break a
                contract rule.
        """
        return self._client.publish_frame_poses(
            twin_uuid,
            frames,
            substrate=substrate,
            pose_provenance=pose_provenance,
            frame_id=frame_id,
            source_subtype=source_subtype,
            producer_instance=producer_instance,
            producer_sequence=producer_sequence,
            workload_uuid=workload_uuid,
            simulation_time_s=simulation_time_s,
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

    def publish_depth_frame(
        self,
        twin_uuid: str,
        depth_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Publish depth frame data via MQTT."""
        return self._client.publish_depth_frame(twin_uuid, depth_data, timestamp)

    def publish_pointcloud(
        self,
        twin_uuid: str,
        point_cloud_data: Any,
        timestamp: Optional[float] = None,
        *,
        stride: int = 6,
        sensor_id: str | None = None,
        camera_pose: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a point cloud frame via MQTT."""
        return self._client.publish_pointcloud(
            twin_uuid,
            point_cloud_data,
            timestamp,
            stride=stride,
            **({"sensor_id": sensor_id} if sensor_id is not None else {}),
            **({"camera_pose": camera_pose} if camera_pose is not None else {}),
        )

    def publish_webrtc_message(self, twin_uuid: str, webrtc_data: Dict[str, Any]):
        """Publish WebRTC signaling message via MQTT."""
        return self._client.publish_webrtc_message(twin_uuid, webrtc_data)

    def subscribe_webrtc_messages(
        self, twin_uuid: str, on_message: Optional[Callable] = None
    ):
        """Subscribe to WebRTC signaling messages via MQTT."""
        return self._client.subscribe_webrtc_messages(twin_uuid, on_message)

    def publish_command_message(self, twin_uuid: str, status):
        """Publish Edge command response message via MQTT.

        Args:
            twin_uuid: The twin UUID to publish to
            status: Either a string status (e.g., "ok") or a dict with status and other fields
        """
        return self._client.publish_command_message(twin_uuid, status)

    def subscribe_command_message(
        self, twin_uuid: str, on_command: Optional[Callable] = None
    ):
        """Subscribe to Edge command message via MQTT."""
        return self._client.subscribe_command_message(twin_uuid, on_command)

    def ping(self, resource_uuid: str):
        """Send ping message to test connectivity."""
        return self._client.ping(resource_uuid)

    def subscribe_pong(self, resource_uuid: str, on_pong: Optional[Callable] = None):
        """Subscribe to pong responses."""
        return self._client.subscribe_pong(resource_uuid, on_pong)

    # Low-level MQTT methods for advanced use cases
    def subscribe(
        self,
        topic: str,
        handler: Optional[Callable] = None,
        qos: int = 0,
        *,
        no_local: bool = False,
        subscriber_key: Any = None,
    ):
        """
        Subscribe to any MQTT topic.

        Args:
            topic: MQTT topic pattern
            handler: Callback function for messages
            qos: Quality of service level (0, 1, or 2)
            no_local: MQTT v5 only — broker will not echo this client's publishes
            subscriber_key: Opaque per-subscriber key. Distinct keys let
                independent subscribers coexist on one topic; re-subscribing
                under the same key replaces in place (see BaseMQTTClient).
        """
        return self._client.subscribe(
            topic, handler, qos, no_local=no_local, subscriber_key=subscriber_key
        )

    @property
    def is_mqtt_v5(self) -> bool:
        """True when the underlying client negotiates MQTT v5."""
        return self._client.is_mqtt_v5

    def unsubscribe(self, topic: str, subscriber_key: Any = _UNSET) -> None:
        """
        Unsubscribe from an MQTT topic.

        Args:
            topic: MQTT topic to unsubscribe from
            subscriber_key: When given, removes only that subscriber's handler
                and keeps the broker subscription alive while others remain.
                Omitted (default) removes all handlers and tears down the
                broker subscription.
        """
        return self._client.unsubscribe(topic, subscriber_key)

    def is_subscribed(self, topic: str) -> bool:
        """Whether the broker has acknowledged (SUBACK) the subscription to *topic*.

        Request/response flows such as a WebRTC offer check this before
        publishing the request, so the response cannot arrive first.
        """
        return self._client.is_subscribed(topic)

    def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        qos: int = 0,
        *,
        properties: Any = None,
    ) -> bool:
        """
        Publish a message to any MQTT topic.

        Returns local client acceptance, not broker or robot acknowledgement.

        Args:
            topic: MQTT topic
            message: Message payload as dictionary
            qos: Quality of service level (0, 1, or 2)
            properties: paho MQTT v5 PUBLISH properties (v5 connections only)
        """
        if properties is None:
            return self._client.publish(topic, message, qos)
        return self._client.publish(topic, message, qos, properties=properties)

    def request(
        self,
        topic: str,
        payload: Any,
        *,
        response_topic: str,
        timeout_s: float = 5.0,
        qos: int = 1,
    ) -> Any:
        """Publish a request over MQTT v5 and return the reply to it.

        The generic request/response pattern for any topic pair; see
        :meth:`cyberwave.mqtt.CyberwaveMQTTClient.request`. Raises
        :class:`~cyberwave.exceptions.RequestTimeoutError` when nothing answers.
        """
        return self._client.request(
            topic,
            payload,
            response_topic=response_topic,
            timeout_s=timeout_s,
            qos=qos,
        )

    def request_command(
        self,
        twin_uuid: str,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        timeout_s: float = 5.0,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a twin command over MQTT v5 and return its reply.

        See :meth:`cyberwave.mqtt.CyberwaveMQTTClient.request_command`. Governed
        twins refuse direct publishes; use ``twin.commands.send`` (the command
        gateway) for those.
        """
        return self._client.request_command(
            twin_uuid, name, data, timeout_s=timeout_s, source_type=source_type
        )
