"""Base classes for WebRTC video streaming to the Cyberwave platform.

Provides :class:`BaseVideoTrack` (abstract video frame source) and
:class:`BaseVideoStreamer` (WebRTC peer-connection lifecycle, MQTT signaling,
automatic reconnection, and health-check reporting).  Concrete implementations
live in sibling modules (``camera_cv2``, ``camera_virtual``, ``camera_rs``,
``camera_sim``).
"""

import abc
import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference
else:
    EdgeHealthCheck = None

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

CONNECTION_LOSS_CONFIRMATION_CHECKS = 3
SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS = 60
SDK_EDGE_HEALTH_INTERVAL_SECONDS = 5


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
        self.frame_count: int = 0
        self.frame_0_timestamp: Optional[float] = None
        self.frame_0_timestamp_monotonic: Optional[float] = None
        self.sync_frame_target: int = 30
        self.sync_frame_pts: Optional[int] = None
        self.sync_frame_timestamp: Optional[float] = None
        self.sync_frame_timestamp_monotonic: Optional[float] = None

    def _capture_sync_frame(
        self, timestamp: float, timestamp_monotonic: float, pts: int
    ):
        """Capture sync frame data at the exact moment of frame capture.

        This is used for the MQTT camera_sync_frame message which provides
        the anchor point for video/robot synchronization.
        """
        if pts == self.sync_frame_target and self.sync_frame_pts is None:
            self.sync_frame_pts = pts
            self.sync_frame_timestamp = timestamp
            self.sync_frame_timestamp_monotonic = timestamp_monotonic

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
        enable_health_check: bool = True,
        camera_name: Optional[str] = None,
        stream_source: Optional[str] = None,
        stream_instance_id: Optional[str] = None,
        frontend_type: Optional[str] = None,
    ):
        """Initialize the video streamer.

        Args:
            client: Cyberwave MQTT client instance
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
            enable_health_check: Whether to enable automatic health check reporting (default: True)
            camera_name: Sensor/camera identifier used for WebRTC signaling routing
                (MQTT offer ``sensor`` field). Omit only for non-recording or legacy
                paths; for recording, pass :attr:`cyberwave.twin.CameraTwin.default_camera_name`
                or an explicit per-stream name when a twin has multiple video streams.
            frontend_type: Track type sent in the WebRTC offer (e.g. "rgb", "depth").
                Must match what the browser consumer sends so the SFU can pair them.
                Defaults to None (SFU uses "rgb").
        """
        self.client = client
        self.twin_uuid: Optional[str] = twin_uuid
        self.camera_name: Optional[str] = camera_name
        self.stream_source: Optional[str] = stream_source
        self.stream_instance_id: Optional[str] = stream_instance_id
        self.frontend_type: Optional[str] = frontend_type
        self.auto_reconnect = auto_reconnect
        # Use explicit None check so empty list [] disables TURN servers
        self.turn_servers = (
            turn_servers if turn_servers is not None else DEFAULT_TURN_SERVERS
        )
        self.time_reference = time_reference
        self.enable_health_check = enable_health_check

        # WebRTC state
        self.pc: Optional[RTCPeerConnection] = None
        self.streamer: Optional[BaseVideoTrack] = None

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

        # Health check state
        self._health_check: Optional[Any] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._last_frame_count = 0
        self._bad_connection_checks = 0

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
        self._bad_connection_checks = 0

    def _publish_camera_sync_frame(
        self, pts: int, timestamp: float, timestamp_monotonic: float
    ):
        """Publish a camera sync frame via MQTT.

        This sync frame is sent after ~1 second of streaming when the connection
        has stabilized. It provides an anchor point for video/robot synchronization:
        - pts: The edge frame counter at this sync point
        - timestamp: Wall-clock time when this frame was captured

        During recording processing, the video is trimmed to start at this sync frame,
        and the timestamp becomes the video's start time. No interpolation needed.
        """
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/telemetry"

        payload = {
            "type": "camera_sync_frame",
            "sender": "edge",
            "pts": pts,
            "timestamp": timestamp,
            "timestamp_monotonic": timestamp_monotonic,
            "track_id": self.streamer.id if self.streamer else None,
            "twin_uuid": self.twin_uuid,
            "sensor": self.camera_name,
        }
        self._publish_message(topic, payload)
        logger.info(
            f"Published camera_sync_frame: pts={pts}, timestamp={timestamp:.3f}"
        )

    async def _wait_and_publish_camera_sync_frame(
        self, sync_frame: int = 30, timeout: float = 10.0
    ):
        """Wait for the sync frame to be captured and publish it via MQTT.

        Args:
            sync_frame: Frame number to use as sync point (default: 30 = ~1sec at 30fps)
            timeout: Maximum time to wait for the sync frame

        The sync frame data (pts + timestamp) is captured at the exact moment
        the frame is produced in the camera track's recv() method. This ensures
        the timestamp precisely matches when the frame was captured, providing
        accurate video/robot synchronization.
        """
        if self.streamer:
            self.streamer.sync_frame_target = sync_frame

        start_time = time.time()

        while self.streamer and self.streamer.sync_frame_pts is None:
            if time.time() - start_time > timeout:
                logger.warning(
                    f"Timeout waiting for sync frame {sync_frame}, "
                    f"current frame: {self.streamer.frame_count if self.streamer else 0}"
                )
                return
            await asyncio.sleep(0.05)

        if self.streamer and self.streamer.sync_frame_pts is not None:
            pts = self.streamer.sync_frame_pts
            timestamp = self.streamer.sync_frame_timestamp
            timestamp_monotonic = self.streamer.sync_frame_timestamp_monotonic

            if timestamp is not None:
                self._publish_camera_sync_frame(
                    pts, timestamp, timestamp_monotonic or 0.0
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
        try:
            await self._perform_signaling()
        except Exception:
            try:
                if self.pc is not None:
                    await self.pc.close()
                    await asyncio.sleep(0.5)
            except Exception:
                pass
            self.pc = None
            if self.streamer is not None:
                try:
                    self.streamer.close()
                except Exception:
                    pass
                self.streamer = None
            raise

        logger.debug("WebRTC connection established")
        asyncio.create_task(self._wait_and_publish_camera_sync_frame())

        if self.enable_health_check:
            self._start_health_check()

    async def stop(self):
        """Stop streaming and cleanup resources.

        IMPORTANT: Close peer connection BEFORE stopping tracks. aiortc can segfault
        if tracks are stopped before pc.close() (see aiortc/aiortc#283).
        """
        self._stop_health_check()

        if self.pc:
            try:
                await self.pc.close()
                # Allow aioice STUN retry callbacks to settle before continuing
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
            finally:
                self.pc = None
        if self.streamer:
            try:
                self.streamer.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error closing streamer: {e}")
            finally:
                self.streamer = None
        self._reset_state()
        logger.info("Camera streaming stopped")

    async def run_with_auto_reconnect(
        self,
        stop_event: Optional[asyncio.Event] = None,
        command_callback: Optional[Callable] = None,
    ):
        """Run camera streaming with automatic reconnection and MQTT command handling.

        Auto-starts the stream immediately so the camera is ready when the frontend
        connects. Also responds to start_video/stop_video commands for manual control.

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

        if self.pc is None:
            try:
                if command_callback:
                    try:
                        command_callback("connecting", "Starting camera stream")
                    except TypeError:
                        pass
                await self.start()
                self._should_reconnect = self.auto_reconnect
                if command_callback:
                    command_callback("ok", "Camera streaming started")
            except Exception as e:
                logger.error(f"Auto-start camera stream failed: {e}", exc_info=True)
                if command_callback:
                    try:
                        command_callback("error", str(e))
                    except TypeError:
                        pass

        if self.auto_reconnect:
            self._monitor_task = asyncio.create_task(self._monitor_connection(stop))

        _next_retry_at: float = time.monotonic() + 15.0
        _retry_backoff: float = 30.0
        _initial_stream_connected: bool = False

        try:
            while not stop.is_set() and self._is_running:
                if self.pc is None and self.auto_reconnect and not _initial_stream_connected:
                    if time.monotonic() >= _next_retry_at:
                        try:
                            logger.info(
                                "No active camera stream — retrying offer (pc is None)..."
                            )
                            await self.start()
                            self._should_reconnect = self.auto_reconnect
                            logger.info("Camera stream retry succeeded.")
                            _initial_stream_connected = True
                        except Exception as retry_exc:
                            logger.info(
                                "Camera stream retry failed (%s). "
                                "Will retry in %.0fs.",
                                retry_exc,
                                _retry_backoff,
                            )
                            _next_retry_at = time.monotonic() + _retry_backoff
                            _retry_backoff = min(_retry_backoff * 2, 120.0)
                await asyncio.sleep(0.5)
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
        self.pc.addTrack(self.streamer)

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
            "sensor": self.camera_name,
            "track_id": self.streamer.id if self.streamer else None,
            "session_id": f"{self.client.client_id}_{self.camera_name}",
        }
        if self.stream_source:
            offer_payload["stream_source"] = self.stream_source
        if self.stream_instance_id:
            offer_payload["stream_instance_id"] = self.stream_instance_id
        if self.frontend_type:
            offer_payload["frontend_type"] = self.frontend_type

        self._publish_message(offer_topic, offer_payload)
        logger.debug(f"WebRTC offer sent to {offer_topic}")

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

        for line in sdp_lines:
            if line.startswith("m=video"):
                parts = line.split()
                filtered_parts = [part for part in parts if part not in ["97", "98"]]
                final_sdp_lines.append(" ".join(filtered_parts))
            elif line.startswith(VP8_PREFIXES):
                continue
            else:
                final_sdp_lines.append(line)

        return "\r\n".join(final_sdp_lines)

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
                logger.debug(f"Received message: type={payload.get('type')}")
                logger.debug(f"Full payload: {payload}")

                if payload.get("type") == "offer":
                    logger.debug("Skipping offer message")
                    return
                elif payload.get("type") == "answer":
                    if payload.get("target") == "edge":
                        if "m=video" not in payload.get("sdp", ""):
                            logger.debug("Ignoring answer with no m=video (likely audio stream)")
                            return
                        answer_sensor = payload.get("sensor") or payload.get("camera")
                        expected = self.camera_name
                        answer_stream_source = payload.get("stream_source") or "live"
                        expected_stream_source = self.stream_source or "live"
                        answer_stream_instance_id = (
                            payload.get("stream_instance_id") or "default"
                        )
                        expected_stream_instance_id = (
                            self.stream_instance_id or "default"
                        )
                        if (
                            (answer_sensor is None or answer_sensor == expected)
                            and answer_stream_source == expected_stream_source
                            and answer_stream_instance_id == expected_stream_instance_id
                        ):
                            logger.info(
                                "Processing answer targeted at edge"
                                + (
                                    f" (sensor={expected}, answer_sensor={answer_sensor})"
                                    if expected != "default"
                                    else ""
                                )
                            )
                            self._answer_data = payload
                            self._answer_received = True
                        else:
                            logger.debug(
                                "Ignoring answer with mismatched stream identity: "
                                f"expected_sensor={expected}, got_sensor={answer_sensor}, "
                                f"expected_stream_source={expected_stream_source}, "
                                f"got_stream_source={answer_stream_source}, "
                                f"expected_stream_instance_id={expected_stream_instance_id}, "
                                f"got_stream_instance_id={answer_stream_instance_id}"
                            )
                    else:
                        logger.debug("Skipping answer message not targeted at edge")
                elif payload.get("type") == "candidate":
                    if payload.get("target") == "edge":
                        self._handle_candidate(payload)
                else:
                    logger.debug(f"Ignoring message type: {payload.get('type')}")
            except Exception as e:
                logger.error(f"Error in on_answer: {e}")

        self.client.subscribe(answer_topic, on_answer)
        candidate_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-candidate"
        self.client.subscribe(candidate_topic, on_answer)

    def _handle_candidate(self, payload: Dict[str, Any]):
        """Handle incoming ICE candidate."""
        if not self.pc or not payload.get("candidate"):
            return

        try:
            from aiortc import RTCIceCandidate

            cand_data = payload["candidate"]
            candidate = RTCIceCandidate(
                candidate=cand_data["candidate"],
                sdpMid=cand_data.get("sdpMid"),
                sdpMLineIndex=cand_data.get("sdpMLineIndex"),
            )
            asyncio.create_task(self.pc.addIceCandidate(candidate))
            logger.info("Added remote ICE candidate")
        except Exception as e:
            logger.warning(f"Failed to add remote ICE candidate: {e}")

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

        is_bad_state = connection_state in (
            "disconnected",
            "failed",
            "closed",
        ) or ice_connection_state in ("disconnected", "failed", "closed")

        if is_bad_state:
            self._bad_connection_checks += 1
            if self._bad_connection_checks < CONNECTION_LOSS_CONFIRMATION_CHECKS:
                logger.debug(
                    "WebRTC in temporary bad state "
                    f"(check {self._bad_connection_checks}/{CONNECTION_LOSS_CONFIRMATION_CHECKS}): "
                    f"connectionState={connection_state}, "
                    f"iceConnectionState={ice_connection_state}"
                )
                return False

            logger.warning(
                f"WebRTC connection lost after {self._bad_connection_checks} consecutive checks "
                f"(connectionState={connection_state}, "
                f"iceConnectionState={ice_connection_state})"
            )
            return True

        if self._bad_connection_checks > 0:
            logger.debug(
                "WebRTC connection recovered before reconnect "
                f"(connectionState={connection_state}, iceConnectionState={ice_connection_state})"
            )
            self._bad_connection_checks = 0

        return False

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

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        self._stop_health_check()

        if self.pc is not None:
            try:
                await self.stop()
            except Exception as e:
                logger.error(f"Error stopping streamer during cleanup: {e}")

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def _start_health_check(self):
        """Start health check monitoring."""
        if not self.enable_health_check or not self.twin_uuid:
            return

        try:
            global EdgeHealthCheck
            if EdgeHealthCheck is None:
                from ..edge.health import EdgeHealthCheck

            self._health_check = EdgeHealthCheck(
                mqtt_client=self.client,
                twin_uuids=[self.twin_uuid],
                edge_id=self.twin_uuid,
                stale_timeout=SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
                interval=SDK_EDGE_HEALTH_INTERVAL_SECONDS,
            )
            self._health_check.start()
            self._last_frame_count = 0

            self._health_monitor_task = asyncio.create_task(self._monitor_frame_count())
            logger.debug("Health check started")
        except Exception as e:
            logger.warning(f"Failed to start health check: {e}")

    def _stop_health_check(self):
        """Stop health check monitoring."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            self._health_monitor_task = None

        if self._health_check:
            try:
                self._health_check.stop()
            except Exception as e:
                logger.warning(f"Error stopping health check: {e}")
            self._health_check = None

        self._last_frame_count = 0

    async def _monitor_frame_count(self):
        """Monitor streamer frame count and update health check."""
        while self._is_running or self.pc is not None:
            try:
                if self.streamer and self._health_check:
                    current_frame_count = getattr(self.streamer, "frame_count", 0)
                    if current_frame_count < self._last_frame_count:
                        self._last_frame_count = current_frame_count
                    if current_frame_count > self._last_frame_count:
                        frames_delta = current_frame_count - self._last_frame_count
                        for _ in range(frames_delta):
                            self._health_check.update_frame_count()
                        self._last_frame_count = current_frame_count
            except Exception as e:
                logger.debug(f"Health check monitoring error: {e}")
            await asyncio.sleep(0.1)
