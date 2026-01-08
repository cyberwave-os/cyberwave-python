"""Camera streaming functionality for Cyberwave SDK."""

import abc
import asyncio
import base64
import fractions
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
import numpy as np
from .mqtt_client import CyberwaveMQTTClient
from .utils import TimeReference

# Make pyrealsense2 optional
try:
    import pyrealsense2 as rs
    _has_realsense = True
except ImportError:
    rs = None  
    _has_realsense = False

logger = logging.getLogger(__name__)
# logging.getLogger("aiortc").setLevel(logging.DEBUG)
# logging.getLogger("aioice").setLevel(logging.DEBUG)
DEFAULT_TURN_SERVERS = [
    {
        "urls": [
            "stun:turn.cyberwave.com:3478",
            # "stun:stun.cloudflare.com:3478",
            # "stun:stun.fbsbx.com:3478",
        ]
    },
    {
        "urls": "turn:turn.cyberwave.com:3478",
        "username": "cyberwave-user",
        "credential": "cyberwave-admin",
    },
]


class BaseVideoTrack(VideoStreamTrack, abc.ABC):
    """Video stream track using OpenCV for camera capture."""
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def set_data_channel(self, data_channel):
        """Set the data channel for sending metadata."""
        self.data_channel = data_channel

    def set_should_sync(self, should_sync: bool):
        """Set whether to sync frames."""
        self.should_sync = should_sync

    @abc.abstractmethod
    async def recv(self):
        """Receive and encode the next video frame."""
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def close(self):
        """Release camera resources."""
        raise NotImplementedError("Subclasses must implement this method")


class CV2VideoTrack(BaseVideoTrack):
    """Video stream track using OpenCV for camera capture."""

    def __init__(
        self, camera_id: int = 0, fps: int = 30, time_reference: TimeReference = None
    ):
        """
        Initialize the video stream track.

        Args:
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
        """
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fps = fps
        self.frame_count = 0
        self.data_channel = None
        self.should_sync = True
        self.time_reference = time_reference
        self.frame_0_timestamp = None
        logger.info(f"Initialized camera {camera_id} at {fps} FPS")

    
    def _send_sync_frame(self, timestamp: float, timestamp_monotonic: float, pts: int):
        """Send a sync frame to the data channel."""
        if self.data_channel and self.should_sync:
            # logger.info(f"Sending sync frame at frame {self.frame_count}")
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

    async def recv(self):
        """Receive and encode the next video frame."""
        # logger.debug(f"Sending frame {self.frame_count}")

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        timestamp, timestamp_monotonic = self.time_reference.read()

        # Create video frame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.fps)
        # logger.info(f"Data channel: {self.data_channel}")
        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
        self.frame_count += 1
        return video_frame

    def close(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")


class RealSenseVideoTrack(BaseVideoTrack):
    def __init__(
        self,
        color_stream_fps: int = 30,
        depth_stream_fps: int = 30,
        color_stream_width: int = 640,
        color_stream_height: int = 480,
        depth_stream_width: int = 640,
        depth_stream_height: int = 480,
        enable_depth: bool = True,
        client: "CyberwaveMQTTClient" = None,
        time_reference: TimeReference = None,
        twin_uuid: Optional[str] = None,
    ):
        if not _has_realsense:
            raise ImportError(
                "RealSense camera support requires pyrealsense2. "
            )
        super().__init__()
        self.client = client
        self.time_reference = time_reference
        self.color_stream_width = color_stream_width
        self.color_stream_height = color_stream_height
        self.depth_stream_width = depth_stream_width
        self.depth_stream_height = depth_stream_height
        self.color_stream_fps = color_stream_fps
        self.depth_stream_fps = depth_stream_fps
        self.enable_depth = enable_depth
        self.depth_sample_count = 0
        self.twin_uuid = twin_uuid
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None

        self.start_time = None
        self.frame_count = 0
        self.frame_0_timestamp = None
        self.data_channel = None
        self.should_sync = True

        if not self.client and self.enable_depth or not self.twin_uuid:
            raise ValueError("To enable stream depth, client and twin_uuid must be provided")
        self.camera_initialized = self.initialize_realsense()
        if not self.camera_initialized:
            raise RuntimeError("Failed to initialize RealSense camera")

    def initialize_realsense(self) -> bool:
        """Initialize RealSense camera and pipeline.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create a context to manage devices
            ctx = rs.context()
            devices = ctx.query_devices()

            if len(devices) == 0:
                logger.error("No RealSense devices found!")
                return False

            logger.info(f"Found {len(devices)} RealSense device(s)")

            # Print device information
            for i, device in enumerate(devices):
                logger.debug(
                    f"Device {i}: {device.get_info(rs.camera_info.name)} "
                    f"Serial: {device.get_info(rs.camera_info.serial_number)}"
                )

            # Create pipeline and configuration
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure streams
            self.config.enable_stream(
                rs.stream.color,
                self.color_stream_width,
                self.color_stream_height,
                rs.format.bgr8,
                self.color_stream_fps,
            )
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.depth_stream_width,
                    self.depth_stream_height,
                    rs.format.z16,
                    self.depth_stream_fps,
                )

            # Create alignment object to align depth to color
            self.align = rs.align(rs.stream.color)

            # Start pipeline
            profile = self.pipeline.start(self.config)

            # Get device information
            device = profile.get_device()
            logger.info(
                f"Started pipeline from: {device.get_info(rs.camera_info.name)}"
            )

            # Let camera warm up
            logger.info("Warming up camera...")
            for i in range(30):
                _ = self.pipeline.wait_for_frames()
                logger.debug(f"Warmup frame {i + 1}/30")

            logger.info("RealSense camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing RealSense camera: {e}")
            return False

    def close(self):
        """Release camera resources."""
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Camera released")

    def get_frames(
        self,
    ) -> Optional[Tuple[bool, Optional[Tuple[np.ndarray, np.ndarray]]]]:
        """Get aligned color and depth frames from RealSense camera.

        Returns:
            Tuple of (color_frame, depth_frame) or None if failed
        """
        timeout = 1000 // self.color_stream_fps
        try:
            # Get frames from the camera
            if not self.pipeline:
                logger.error("Pipeline not initialized")
                return False, None
            if not self.align:
                logger.error("Align not initialized")
                return False, None
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                logger.warning("Could not get both color and depth frames")
                return False, None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return True, (color_image, depth_image)
        except RuntimeError as e:
            # Handle specific RealSense runtime errors
            if "Frame didn't arrive" in str(e):
                logger.warning(f"Frame timeout after {timeout}ms")
            else:
                logger.error(f"RealSense runtime error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Error getting frames: {e}")
            return False, None
    
    def _send_sync_frame(self, timestamp: float, timestamp_monotonic: float, pts: int):
        """Send a sync frame to the data channel."""
        if self.data_channel and self.should_sync:
            # logger.info(f"Sending sync frame at frame {self.frame_count}")
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

    async def recv(self):
        """Receive and encode the next video frame."""
        ret, frames = self.get_frames()
        if not ret:
            logger.error("Failed to read frames from realsense camera")
            return None

        timestamp, timestamp_monotonic = self.time_reference.read()
        color_image, depth_image = frames

        # Create video frame
        video_frame = VideoFrame.from_ndarray(color_image, format="bgr24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.color_stream_fps)
        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        # Publish depth frame every 30 frames
        if self.enable_depth and self.frame_count % 30 == 0:
            if depth_image.dtype != np.uint16:
                depth_image = depth_image.astype(np.uint16)
        
            depth_binary = base64.b64encode(depth_image.tobytes()).decode('utf-8')
            self.client.publish_depth_frame(self.twin_uuid, depth_binary, timestamp)
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
        self.frame_count += 1
        return video_frame


class CameraStreamer:
    """
    Manages WebRTC camera streaming to Cyberwave platform.

    Note: It's recommended to use the Cyberwave.video_stream() method instead
    of instantiating this class directly for a better developer experience.

    Example (Recommended):
        >>> from cyberwave import Cyberwave
        >>> import asyncio
        >>>
        >>> client = Cyberwave(token="your_token")
        >>> streamer = client.video_stream(twin_uuid="your_twin_uuid", camera_id=0)
        >>> asyncio.run(streamer.start())

    Example (Direct instantiation):
        >>> from cyberwave import Cyberwave, CameraStreamer
        >>> import asyncio
        >>>
        >>> client = Cyberwave(token="your_token")
        >>> streamer = CameraStreamer(client.mqtt, camera_id=0, twin_uuid="your_twin_uuid")
        >>> asyncio.run(streamer.start())
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        camera_id: int = 0,
        fps: int = 30,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: TimeReference = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize the camera streamer.

        Args:
            client: Cyberwave SDK client instance
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin (can be provided here or in start())
            auto_reconnect: Whether to automatically reconnect on disconnection (default: True)
        """
        # Configuration
        self.client = client
        self.camera_id = camera_id
        self.fps = fps
        self.twin_uuid: Optional[str] = twin_uuid
        self.auto_reconnect = auto_reconnect
        self.turn_servers = turn_servers or DEFAULT_TURN_SERVERS
        self.time_reference = time_reference
        # WebRTC state
        self.pc: Optional[RTCPeerConnection] = None
        self.streamer: Optional[CV2VideoTrack] = None
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

    def initialize_track(self):
        """Initialize the video track."""
        self.streamer = CV2VideoTrack(self.camera_id, self.fps, self.time_reference)

    def _reset_state(self):
        """Reset internal state for fresh connection."""
        self._answer_received = False
        self._answer_data = None
        # Note: _should_record is intentionally NOT reset here to preserve the recording
        # preference set by the command handler before start() is called

    def _publish_frame_0_timestamp(self, timestamp: float):
        """Publish the frame_0_timestamp via MQTT."""
        prefix = self.client.topic_prefix
        topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/telemetry"
        payload = {
            "type": "video_start_timestamp",
            "sender": "edge",
            "timestamp": timestamp,
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
        
        if self.streamer and self.streamer.frame_0_timestamp is not None:
            self._publish_frame_0_timestamp(self.streamer.frame_0_timestamp)

    # -------------------------------------------------------------------------
    # Public API - Start/Stop
    # -------------------------------------------------------------------------

    async def start(self, twin_uuid: Optional[str] = None):
        """
        Start streaming camera to Cyberwave.

        Args:
            twin_uuid: UUID of the digital twin (uses instance twin_uuid if not provided)
        """
        self._reset_state()

        # Validate twin_uuid
        if twin_uuid is not None:
            self.twin_uuid = twin_uuid
        elif self.twin_uuid is None:
            raise ValueError(
                "twin_uuid must be provided either during initialization or when calling start()"
            )

        logger.info(f"Starting camera stream for twin {self.twin_uuid}")

        # Subscribe to WebRTC answer topic BEFORE doing anything else
        self._subscribe_to_answer()

        # Give MQTT time to fully connect and subscribe
        await asyncio.sleep(2.5)

        # Initialize WebRTC
        await self._setup_webrtc()

        # Perform signaling
        await self._perform_signaling()

        logger.debug("WebRTC connection established")

        # Wait for first frame and publish its timestamp
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
        """
        Run camera streaming with automatic reconnection and MQTT command handling.

        This method:
        - Subscribes to start/stop commands via MQTT
        - Monitors WebRTC connection state
        - Automatically reconnects on disconnection
        - Runs until stop_event is set or explicitly stopped

        Args:
            stop_event: Optional asyncio.Event to signal when to stop
            command_callback: Optional callback function(status, message) for command responses
        """
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before running")

        self._is_running = True
        self._event_loop = asyncio.get_running_loop()
        stop = stop_event or asyncio.Event()

        # Subscribe to command messages
        self._subscribe_to_commands(command_callback)

        # Start connection monitoring if auto_reconnect is enabled
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
        # Initialize video stream
        self.initialize_track()

        # Create peer connection with STUN/TURN servers
        ice_servers = [RTCIceServer(**server) for server in self.turn_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        # Set up event handlers
        self._setup_pc_handlers()

        # Create data channel with explicit stream_id (negotiated mode)
        # Using id=1 for the first offerer DataChannel (odd numbers for offerer)
        # This allows us to signal the stream_id before the channel opens
        self.channel = self.pc.createDataChannel("track_info", negotiated=True, id=1)
        logger.info(f"Data channel created: {self.channel}, id={self.channel.id}")
        self.pc.addTrack(self.streamer)

        # Set up data channel handlers
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
        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering to complete
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        # Filter SDP and send offer
        modified_sdp = self._filter_sdp(self.pc.localDescription.sdp)
        self._send_offer(modified_sdp)

        # Wait for and process answer
        await self._wait_for_answer()

    def _send_offer(self, sdp: str):
        """Send WebRTC offer via MQTT."""
        prefix = self.client.topic_prefix
        offer_topic = f"{prefix}cyberwave/twin/{self.twin_uuid}/webrtc-offer"

        offer_payload = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": sdp,
            "color_track_id": self.streamer.id,
            "timestamp": time.time(),
            "recording": getattr(self, "_should_record", False),
        }

        self._publish_message(offer_topic, offer_payload)
        logger.debug(f"WebRTC offer sent to {offer_topic}")

        # Signal DataChannel info immediately after offer
        # We use negotiated mode with explicit id, so stream_id is known
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
        """
        Filter SDP to remove VP8 codec lines.

        Args:
            sdp: Original SDP string

        Returns:
            Modified SDP string
        """
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
        """Signal DataChannel SCTP parameters to mediasoup via MQTT.

        This allows mediasoup to set up the DataProducer with the correct
        SCTP stream parameters to receive DataChannel messages.

        NOTE: This must be called AFTER the DataChannel is open, otherwise
        the stream_id will be None.
        """
        if not self.channel or not self.twin_uuid:
            return

        # Get SCTP stream ID from the DataChannel
        # aiortc assigns stream IDs after the channel is open
        # For the offerer (client), first DataChannel gets stream_id=1 (odd numbers)
        stream_id = getattr(self.channel, "id", None)
        ordered = getattr(self.channel, "ordered", True)

        # If stream_id is still None, use default of 1 (first offerer DataChannel)
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
            """Callback for WebRTC answer messages."""
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
            """Handle incoming command messages."""
            try:
                payload = data if isinstance(data, dict) else json.loads(data)

                # Skip status messages
                if "status" in payload:
                    return

                command_type = payload.get("command")
                if not command_type:
                    logger.warning("Command message missing command field")
                    return

                if command_type == "start_video":
                    recording = payload.get("recording", False)
                    self._should_record = recording
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
        """Handle start_video command.

        Args:
            callback: Optional callback function to report status
            recording: Whether to record the video stream on the backend
        """
        try:
            if self.pc is not None:
                logger.info("Video stream already running")
                if callback:
                    callback("ok", "Video stream already running")
                return

            logger.info(
                f"Starting video stream - Camera ID: {self.camera_id}, FPS: {self.fps}, Recording: {self._should_record}"
            )
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
                if reconnect_attempt < 0:  # Signal to stop
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
        """
        Attempt to reconnect the WebRTC connection.

        Returns:
            New attempt count, or -1 to signal stopping
        """
        try:
            # Clean up old connection
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
            return 0  # Reset attempt counter on success

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

            # Exponential backoff
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


class RealSenseStreamer(CameraStreamer):
    """
    Manages WebRTC realsense camera streaming to Cyberwave platform.

    Example (Direct instantiation):
        >>> from cyberwave import Cyberwave, RealSenseStreamer
        >>> import asyncio
        >>>
        >>> client = Cyberwave(token="your_token")
        >>> streamer = RealSenseStreamer(client.mqtt, twin_uuid="your_twin_uuid")
        >>> asyncio.run(streamer.start())
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        color_stream_fps: int = 30,
        depth_stream_fps: int = 30,
        color_stream_width: int = 640,
        color_stream_height: int = 480,
        depth_stream_width: int = 640,
        depth_stream_height: int = 480,
        enable_depth: bool = True,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: TimeReference = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize the camera streamer.

        Args:
            client: Cyberwave SDK client instance
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin (can be provided here or in start())
            auto_reconnect: Whether to automatically reconnect on disconnection (default: True)
        """
        if not _has_realsense:
            raise ImportError(
                "RealSense camera support requires pyrealsense2. "
            )
        # Configuration
        super().__init__(client, twin_uuid=twin_uuid, time_reference=time_reference, auto_reconnect=auto_reconnect) 
        self.client = client
        self.color_stream_fps = color_stream_fps
        self.depth_stream_fps = depth_stream_fps
        self.color_stream_width = color_stream_width
        self.color_stream_height = color_stream_height
        self.depth_stream_width = depth_stream_width
        self.depth_stream_height = depth_stream_height
        self.enable_depth = enable_depth
        self.twin_uuid: Optional[str] = twin_uuid
        self.auto_reconnect = auto_reconnect
        self.turn_servers = turn_servers or DEFAULT_TURN_SERVERS
        self.time_reference = time_reference
        # WebRTC state
        self.pc: Optional[RTCPeerConnection] = None
        self.streamer: Optional[RealSenseVideoTrack] = None
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

    def initialize_track(self):
        """Initialize the video track."""
        self.streamer = RealSenseVideoTrack(
            self.color_stream_fps,
            self.depth_stream_fps,
            self.color_stream_width,
            self.color_stream_height,
            self.depth_stream_width,
            self.depth_stream_height,
            self.enable_depth,
            self.client,
            self.time_reference,
            self.twin_uuid,
        )

