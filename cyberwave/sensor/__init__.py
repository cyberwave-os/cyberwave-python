"""Sensor streaming functionality for Cyberwave SDK.

This module provides abstract base classes and implementations for various sensor types.
Currently supports RGB and Depth sensors via CV2 and RealSense cameras.

Sensor capabilities are defined in the twin's capabilities dictionary:
    {
        "sensors": [
            {"id": "uuid", "type": "rgb", "offset": {...}},
            {"id": "uuid", "type": "depth", "offset": {...}}
        ]
    }
"""

import abc
import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference

logger = logging.getLogger(__name__)

DEFAULT_TURN_SERVERS = [
    {
        "urls": [
            "stun:turn.cyberwave.com:3478",
        ]
    },
    {
        "urls": "turn:turn.cyberwave.com:3478",
        "username": "cyberwave-user",
        "credential": "cyberwave-admin",
    },
]


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseVideoTrack(VideoStreamTrack, abc.ABC):
    """Abstract base class for video stream tracks.

    Subclasses must implement:
        - __init__: Initialize the video track with camera-specific configuration
        - recv: Receive and encode the next video frame
        - close: Release camera resources
    """

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.data_channel: Optional[RTCDataChannel] = None
        self.should_sync: bool = True
        self.frame_count: int = 0
        self.frame_0_timestamp: Optional[float] = None
        self.frame_0_timestamp_monotonic: Optional[float] = None
        self.channel_queue: list[dict[str, Any]] = []

    def set_data_channel(self, data_channel: RTCDataChannel):
        """Set the data channel for sending metadata."""
        self.data_channel = data_channel

    def set_should_sync(self, should_sync: bool):
        """Set whether to sync frames."""
        self.should_sync = should_sync

    def _queue_sync_frame(self, timestamp: float, timestamp_monotonic: float, pts: int):
        """Queue a sync frame to be sent later."""
        self.channel_queue.append(
            {
                "timestamp": timestamp,
                "timestamp_monotonic": timestamp_monotonic,
                "pts": pts,
                "track_id": self.id,
                "type": "sync_frame",
            }
        )

    def _send_sync_frame(self, timestamp: float, timestamp_monotonic: float, pts: int):
        """Send a sync frame to the data channel."""
        if self.data_channel and self.should_sync:
            if len(self.channel_queue) > 0:
                self.data_channel.send(json.dumps(self.channel_queue[0]))
                self.channel_queue = []
            self.data_channel.send(
                json.dumps(
                    {
                        "type": "sync_frame",
                        "read_timestamp": timestamp,
                        "read_timestamp_monotonic": timestamp_monotonic,
                        "pts": pts,
                        "track_id": self.id,
                    }
                )
            )
        else:
            self._queue_sync_frame(timestamp, timestamp_monotonic, pts)

    def get_stream_attributes(self) -> Dict[str, Any]:
        """Get streaming attributes for the offer payload.

        Subclasses should override this to provide camera-specific attributes.

        Returns:
            Dictionary with stream attributes (width, height, fps, camera_type, etc.)
        """
        return {}

    @abc.abstractmethod
    async def recv(self):
        """Receive and encode the next video frame."""
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def close(self):
        """Release camera resources."""
        raise NotImplementedError("Subclasses must implement this method")


class BaseVideoStreamer(abc.ABC):
    """Abstract base class for WebRTC video streaming to Cyberwave platform.

    Manages WebRTC peer connections, signaling, and automatic reconnection.

    Subclasses must implement:
        - initialize_track: Create and return the appropriate video track
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
    ):
        """Initialize the video streamer.

        Args:
            client: Cyberwave MQTT client instance
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
        """
        self.client = client
        self.twin_uuid: Optional[str] = twin_uuid
        self.auto_reconnect = auto_reconnect
        # Use explicit None check so empty list [] disables TURN servers
        self.turn_servers = turn_servers if turn_servers is not None else DEFAULT_TURN_SERVERS
        self.time_reference = time_reference

        # WebRTC state
        self.pc: Optional[RTCPeerConnection] = None
        self.streamer: Optional[BaseVideoTrack] = None
        self.channel: Optional[RTCDataChannel] = None

        # Answer handling state
        self._answer_received = False
        self._answer_data: Optional[Dict[str, Any]] = None

        # Reconnection state
        self._should_reconnect = False
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Recording state
        self._should_record = True

    @abc.abstractmethod
    def initialize_track(self) -> BaseVideoTrack:
        """Initialize and return the video track.

        Subclasses must implement this to create the appropriate track type.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _reset_state(self):
        """Reset internal state for fresh connection."""
        self._answer_received = False
        self._answer_data = None

    def _publish_frame_0_timestamp(self, timestamp: float, timestamp_monotonic: float):
        """Publish the frame_0_timestamp via MQTT."""
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/telemetry"
        payload = {
            "type": "video_start_timestamp",
            "sender": "edge",
            "timestamp": timestamp,
            "timestamp_monotonic": timestamp_monotonic,
            "track_id": self.streamer.id if self.streamer else None,
        }
        self._publish_message(topic, payload)
        logger.info(f"Published frame_0_timestamp: {timestamp}")

    async def _wait_and_publish_frame_0_timestamp(self, timeout: float = 10.0):
        """Wait for the first frame to be sent and publish its timestamp."""
        start_time = time.time()
        while self.streamer and self.streamer.frame_count == 0:
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for first frame")
                return
            await asyncio.sleep(0.1)

        if self.streamer:
            frame_0_timestamp = getattr(self.streamer, "frame_0_timestamp", None)
            frame_0_timestamp_monotonic = getattr(
                self.streamer, "frame_0_timestamp_monotonic", None
            )

            if frame_0_timestamp is not None:
                self._publish_frame_0_timestamp(
                    frame_0_timestamp, frame_0_timestamp_monotonic or 0.0
                )

    # -------------------------------------------------------------------------
    # Public API - Start/Stop
    # -------------------------------------------------------------------------

    async def start(self, twin_uuid: Optional[str] = None):
        """Start streaming camera to Cyberwave.

        Args:
            twin_uuid: UUID of the digital twin (uses instance twin_uuid if not provided)
        """
        self._reset_state()

        if twin_uuid is not None:
            self.twin_uuid = twin_uuid
        elif self.twin_uuid is None:
            raise ValueError(
                "twin_uuid must be provided either during initialization or when calling start()"
            )

        logger.info(f"Starting camera stream for twin {self.twin_uuid}")

        self._subscribe_to_answer()
        await asyncio.sleep(2.5)
        await self._setup_webrtc()
        await self._perform_signaling()

        logger.debug("WebRTC connection established")
        asyncio.create_task(self._wait_and_publish_frame_0_timestamp())

    async def stop(self):
        """Stop streaming and cleanup resources."""
        if self.streamer:
            try:
                self.streamer.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error closing streamer: {e}")
            finally:
                self.streamer = None
        if self.pc:
            try:
                await self.pc.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
            finally:
                self.pc = None
        self._reset_state()
        logger.info("Camera streaming stopped")

    async def run_with_auto_reconnect(
        self,
        stop_event: Optional[asyncio.Event] = None,
        command_callback: Optional[Callable] = None,
    ):
        """Run camera streaming with automatic reconnection and MQTT command handling.

        Args:
            stop_event: Optional asyncio.Event to signal when to stop
            command_callback: Optional callback function(status, message) for command responses
        """
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before running")

        self._is_running = True
        self._event_loop = asyncio.get_running_loop()
        stop = stop_event or asyncio.Event()

        self._subscribe_to_commands(command_callback)

        if self.auto_reconnect:
            self._monitor_task = asyncio.create_task(self._monitor_connection(stop))

        try:
            while not stop.is_set() and self._is_running:
                await asyncio.sleep(0.1)
        finally:
            await self._cleanup_run()

    # -------------------------------------------------------------------------
    # WebRTC Setup
    # -------------------------------------------------------------------------

    async def _setup_webrtc(self):
        """Initialize WebRTC peer connection and video track."""
        self.streamer = self.initialize_track()

        ice_servers = [RTCIceServer(**server) for server in self.turn_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        self._setup_pc_handlers()

        self.channel = self.pc.createDataChannel("track_info", negotiated=True, id=1)
        logger.info(f"Data channel created: {self.channel}, id={self.channel.id}")
        self.pc.addTrack(self.streamer)

        self._setup_channel_handlers()

    def _setup_pc_handlers(self):
        """Set up peer connection event handlers."""

        @self.pc.on("connectionstatechange")
        def on_connectionstatechange():
            state = self.pc.connectionState
            logger.info(f"WebRTC connection state changed: {state}")

        @self.pc.on("iceconnectionstatechange")
        def on_iceconnectionstatechange():
            state = self.pc.iceConnectionState
            logger.info(f"WebRTC ICE connection state changed: {state}")

    def _setup_channel_handlers(self):
        """Set up data channel event handlers."""
        color_track = self.streamer

        @self.channel.on("open")
        def on_open():
            self.streamer.set_data_channel(self.channel)
            msg = {"type": "track_info", "color_track_id": color_track.id}
            logger.info(f"Data channel opened, sending track info: {msg}")
            self.channel.send(json.dumps(msg))

        @self.channel.on("message")
        def on_message(msg):
            logger.info(f"Received message: {msg}")
            parsed = json.loads(msg)

            if self.channel.readyState != "open":
                return

            if parsed["type"] == "ping":
                self.channel.send(
                    json.dumps({"type": "pong", "timestamp": time.time()})
                )
            elif parsed["type"] == "pong":
                self.channel.send(
                    json.dumps({"type": "ping", "timestamp": time.time()})
                )
            elif parsed["type"] == "sync_frame_command":
                self.streamer.set_should_sync(True)

    # -------------------------------------------------------------------------
    # WebRTC Signaling
    # -------------------------------------------------------------------------

    async def _perform_signaling(self):
        """Perform WebRTC offer/answer signaling."""
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        modified_sdp = self._filter_sdp(self.pc.localDescription.sdp)
        self._send_offer(modified_sdp)

        await self._wait_for_answer()

    def _send_offer(self, sdp: str):
        """Send WebRTC offer via MQTT."""
        prefix = self.client.topic_prefix
        offer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-offer"

        # Get stream attributes from the track
        stream_attributes = {}
        if self.streamer:
            stream_attributes = self.streamer.get_stream_attributes()

        offer_payload = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": sdp,
            "timestamp": time.time(),
            "recording": self._should_record,
            "stream_attributes": stream_attributes,
        }

        self._publish_message(offer_topic, offer_payload)
        logger.debug(f"WebRTC offer sent to {offer_topic}")

        self._signal_datachannel_info()

    async def _wait_for_answer(self, timeout: float = 60.0):
        """Wait for WebRTC answer from backend."""
        start_time = time.time()
        while not self._answer_received:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for WebRTC answer")
            await asyncio.sleep(0.1)

        logger.debug("WebRTC answer received")

        if self._answer_data is None:
            raise RuntimeError("Answer received flag set but answer data is None")

        answer = (
            json.loads(self._answer_data)
            if isinstance(self._answer_data, str)
            else self._answer_data
        )

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    def _filter_sdp(self, sdp: str) -> str:
        """Filter SDP to remove VP8 codec lines."""
        VP8_PREFIXES = (
            "a=rtpmap:97",
            "a=rtpmap:98",
            "a=rtcp-fb:97 nack",
            "a=rtcp-fb:97 nack pli",
            "a=rtcp-fb:97 goog-remb",
            "a=rtcp-fb:98 nack",
            "a=rtcp-fb:98 nack pli",
            "a=rtcp-fb:98 goog-remb",
            "a=fmtp:98",
        )

        sdp_lines = sdp.split("\r\n")
        final_sdp_lines = []
        m_video_parts = []

        for line in sdp_lines:
            if line.startswith("m=video"):
                parts = line.split()
                for part in parts:
                    if part not in ["97", "98"]:
                        m_video_parts.append(part)
                final_sdp_lines.append(" ".join(m_video_parts))
            elif line.startswith(VP8_PREFIXES):
                continue
            else:
                final_sdp_lines.append(line)

        return "\r\n".join(final_sdp_lines)

    def _signal_datachannel_info(self):
        """Signal DataChannel SCTP parameters to mediasoup via MQTT."""
        if not self.channel or not self.twin_uuid:
            return

        stream_id = getattr(self.channel, "id", None)
        ordered = getattr(self.channel, "ordered", True)

        if stream_id is None:
            logger.warning("DataChannel stream_id is None, using default of 1")
            stream_id = 1

        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/datachannel-info"

        payload = {
            "type": "datachannel_ready",
            "sender": "edge",
            "stream_id": stream_id,
            "ordered": ordered,
            "label": self.channel.label
            if hasattr(self.channel, "label")
            else "track_info",
            "timestamp": time.time(),
        }

        self._publish_message(topic, payload)
        logger.info(
            f"Signaled DataChannel info to mediasoup: stream_id={stream_id}, ordered={ordered}"
        )

    # -------------------------------------------------------------------------
    # MQTT Communication
    # -------------------------------------------------------------------------

    def _subscribe_to_answer(self):
        """Subscribe to WebRTC answer topic."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before subscribing")

        prefix = self.client.topic_prefix
        answer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-answer"
        logger.info(f"Subscribing to WebRTC answer topic: {answer_topic}")

        def on_answer(data):
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
                logger.info(f"Received message: type={payload.get('type')}")
                logger.debug(f"Full payload: {payload}")

                if payload.get("type") == "offer":
                    logger.debug("Skipping offer message")
                    return
                elif payload.get("type") == "answer":
                    if payload.get("target") == "edge":
                        logger.info("Processing answer targeted at edge")
                        self._answer_data = payload
                        self._answer_received = True
                    else:
                        logger.debug("Skipping answer message not targeted at edge")
                else:
                    raise ValueError(f"Unknown message type: {payload.get('type')}")
            except Exception as e:
                raise e

        self.client.subscribe(answer_topic, on_answer)

    def _subscribe_to_commands(self, command_callback: Optional[Callable] = None):
        """Subscribe to start/stop command messages via MQTT."""
        prefix = self.client.topic_prefix
        command_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/command"
        logger.info(f"Subscribing to command topic: {command_topic}")

        def on_command(data):
            try:
                payload = data if isinstance(data, dict) else json.loads(data)

                if "status" in payload:
                    return

                command_type = payload.get("command")
                if not command_type:
                    logger.warning("Command message missing command field")
                    return

                if command_type == "start_video":
                    data_dict = payload.get("data", {})
                    if isinstance(data_dict, dict):
                        recording = data_dict.get("recording", True)
                    else:
                        recording = payload.get("recording", True)
                    self._should_record = recording
                    logger.info(f"Setting recording state to: {recording}")
                    asyncio.run_coroutine_threadsafe(
                        self._handle_start_command(command_callback), self._event_loop
                    )
                elif command_type == "stop_video":
                    asyncio.run_coroutine_threadsafe(
                        self._handle_stop_command(command_callback), self._event_loop
                    )
                else:
                    logger.warning(f"Unknown command type: {command_type}")

            except Exception as e:
                logger.error(f"Error processing command message: {e}", exc_info=True)

        self.client.subscribe(command_topic, on_command)

    def _publish_message(self, topic: str, payload: Dict[str, Any]):
        """Publish a message via MQTT."""
        self.client.publish(topic, payload, qos=2)
        logger.info(f"Published to {topic}")

    # -------------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------------

    async def _handle_start_command(self, callback: Optional[Callable] = None):
        """Handle start_video command."""
        try:
            if self.pc is not None:
                logger.info("Video stream already running")
                if callback:
                    callback("ok", "Video stream already running")
                return

            logger.info(f"Starting video stream - Recording: {self._should_record}")
            await self.start()
            self._should_reconnect = self.auto_reconnect
            logger.info("Camera streaming started successfully!")

            if callback:
                callback("ok", "Camera streaming started")

        except Exception as e:
            logger.error(f"Error starting video stream: {e}", exc_info=True)
            if callback:
                callback("error", str(e))

    async def _handle_stop_command(self, callback: Optional[Callable] = None):
        """Handle stop_video command."""
        try:
            if self.pc is None:
                logger.info("Video stream not running")
                if callback:
                    callback("ok", "Video stream not running")
                return

            logger.info("Stopping video stream")
            self._should_reconnect = False
            await self.stop()
            logger.info("Camera stream stopped successfully")

            if callback:
                callback("ok", "Camera stream stopped")

        except Exception as e:
            logger.error(f"Error stopping video stream: {e}", exc_info=True)
            if callback:
                callback("error", str(e))

    # -------------------------------------------------------------------------
    # Connection Monitoring & Reconnection
    # -------------------------------------------------------------------------

    async def _monitor_connection(self, stop_event: asyncio.Event):
        """Monitor WebRTC connection and automatically reconnect on disconnection."""
        reconnect_delay = 2.0
        max_reconnect_attempts = 10
        reconnect_attempt = 0

        while not stop_event.is_set() and self._is_running:
            if not self._should_reconnect or self.pc is None:
                await asyncio.sleep(1.0)
                continue

            if self._is_connection_lost():
                reconnect_attempt = await self._attempt_reconnect(
                    stop_event,
                    reconnect_attempt,
                    reconnect_delay,
                    max_reconnect_attempts,
                )
                if reconnect_attempt < 0:
                    break

            await asyncio.sleep(1.0)

    def _is_connection_lost(self) -> bool:
        """Check if WebRTC connection is lost."""
        connection_state = getattr(self.pc, "connectionState", None)
        ice_connection_state = getattr(self.pc, "iceConnectionState", None)

        is_disconnected = connection_state in (
            "disconnected",
            "failed",
            "closed",
        ) or ice_connection_state in ("disconnected", "failed", "closed")

        if is_disconnected:
            logger.warning(
                f"WebRTC connection lost (connectionState={connection_state}, "
                f"iceConnectionState={ice_connection_state})"
            )

        return is_disconnected

    async def _attempt_reconnect(
        self,
        stop_event: asyncio.Event,
        attempt: int,
        base_delay: float,
        max_attempts: int,
    ) -> int:
        """Attempt to reconnect the WebRTC connection.

        Returns:
            New attempt count, or -1 to signal stopping
        """
        try:
            try:
                await self.stop()
            except Exception as e:
                logger.warning(f"Error stopping old streamer during reconnect: {e}")

            await asyncio.sleep(base_delay)

            if not self._should_reconnect or stop_event.is_set():
                logger.info("Reconnect cancelled (stream was stopped)")
                return -1

            logger.info(f"Reconnecting camera stream (attempt {attempt + 1})...")
            await self.start()
            logger.info("Camera stream reconnected successfully!")
            return 0

        except Exception as e:
            attempt += 1
            logger.error(f"Reconnection attempt {attempt} failed: {e}", exc_info=True)

            if attempt >= max_attempts:
                logger.error(
                    f"Max reconnection attempts ({max_attempts}) reached. "
                    "Stopping reconnection attempts."
                )
                self._should_reconnect = False
                return -1

            backoff_delay = min(base_delay * (2**attempt), 30.0)
            await asyncio.sleep(backoff_delay)
            return attempt

    async def _cleanup_run(self):
        """Cleanup after run_with_auto_reconnect exits."""
        self._is_running = False
        self._should_reconnect = False
        self._event_loop = None

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self.pc is not None:
            try:
                await self.stop()
            except Exception as e:
                logger.error(f"Error stopping streamer during cleanup: {e}")


# =============================================================================
# Exports
# =============================================================================

# Import configuration classes
from .config import (  # noqa: E402
    CameraType,
    Resolution,
    CameraConfig,
    EdgeCameraConfig,
    RealSenseConfig,
    StreamProfile,
    SensorOption,
    RealSenseDeviceInfo,
    RealSenseDiscovery,
    PRESET_LOW_BANDWIDTH,
    PRESET_STANDARD,
    PRESET_HD,
    PRESET_FULL_HD,
)

# Import concrete implementations for convenience
from .camera_cv2 import CV2VideoTrack, CV2CameraStreamer  # noqa: E402
from .camera_rs import RealSenseVideoTrack, RealSenseStreamer  # noqa: E402

__all__ = [
    # Base classes
    "BaseVideoTrack",
    "BaseVideoStreamer",
    # Configuration
    "CameraType",
    "Resolution",
    "CameraConfig",
    "EdgeCameraConfig",
    "RealSenseConfig",
    "StreamProfile",
    "SensorOption",
    "RealSenseDeviceInfo",
    "RealSenseDiscovery",
    "PRESET_LOW_BANDWIDTH",
    "PRESET_STANDARD",
    "PRESET_HD",
    "PRESET_FULL_HD",
    # CV2 implementations
    "CV2VideoTrack",
    "CV2CameraStreamer",
    # RealSense implementations
    "RealSenseVideoTrack",
    "RealSenseStreamer",
    # Constants
    "DEFAULT_TURN_SERVERS",
]
