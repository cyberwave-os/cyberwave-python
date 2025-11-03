"""Camera streaming functionality for Cyberwave SDK."""

import asyncio
import fractions
import json
import logging
import time
from typing import Any, Dict, Optional

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

from .mqtt_client import CyberwaveMQTTClient

logger = logging.getLogger(__name__)


class CV2VideoStreamTrack(VideoStreamTrack):
    """Video stream track using OpenCV for camera capture."""

    def __init__(self, camera_id: int = 0, fps: int = 10):
        """
        Initialize the video stream track.

        Args:
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
        """
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        self.frame_count = 0
        self.data_channel = None
        logger.info(f"Initialized camera {camera_id} at {fps} FPS")

    def set_data_channel(self, data_channel):
        """Set the data channel for sending metadata."""
        self.data_channel = data_channel

    async def recv(self):
        """Receive and encode the next video frame."""
        self.frame_count += 1
        logger.debug(f"Sending frame {self.frame_count}")

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create video frame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.fps)

        return video_frame

    def close(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")


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

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        camera_id: int = 0,
        fps: int = 10,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
    ):
        """
        Initialize the camera streamer.

        Args:
            client: Cyberwave SDK client instance
            camera_id: Camera device ID (default: 0)
            fps: Frames per second (default: 10)
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin (can be provided here or in start())
        """
        self.client = client
        self.camera_id = camera_id
        self.fps = fps
        self.pc: Optional[RTCPeerConnection] = None
        self.streamer: Optional[CV2VideoStreamTrack] = None
        self.channel: Optional[RTCDataChannel] = None
        self.twin_uuid: Optional[str] = twin_uuid
        self._answer_received = False
        self._answer_data: Optional[Dict[str, Any]] = None
        self._answer_messenger = (
            None  # Store messenger reference to prevent garbage collection
        )

        # Default TURN servers
        self.turn_servers = turn_servers or [
            {
                "urls": [
                    "stun:turn.cyberwave.com:3478",
                    "stun:stun.cloudflare.com:3478",
                    "stun:stun.fbsbx.com:3478",
                ]
            },
            {
                "urls": "turn:turn.cyberwave.com:3478",
                "username": "cyberwave-user",
                "credential": "cyberwave-admin",
            },
        ]

    async def start(self, twin_uuid: Optional[str] = None):
        """
        Start streaming camera to Cyberwave.

        Args:
            twin_uuid: UUID of the digital twin (uses instance twin_uuid if not provided)
        """
        # Use provided twin_uuid or fall back to instance twin_uuid
        if twin_uuid is not None:
            self.twin_uuid = twin_uuid
        elif self.twin_uuid is None:
            raise ValueError(
                "twin_uuid must be provided either during initialization or when calling start()"
            )

        logger.info(f"Starting camera stream for twin {self.twin_uuid}")

        # Assume the MQTT client is already connected
        # if self.client:
        #     self.client.connect()
        #     logger.info("MQTT client connected")
        # else:
        #     raise RuntimeError("MQTT client not available")

        # Subscribe to WebRTC answer topic BEFORE doing anything else
        self._subscribe_to_answer()

        # Give MQTT time to fully connect and subscribe
        await asyncio.sleep(2.5)

        # Initialize video stream
        self.streamer = CV2VideoStreamTrack(self.camera_id, self.fps)

        # Create peer connection with STUN/TURN servers
        ice_servers = [RTCIceServer(**server) for server in self.turn_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        # Create data channel for metadata
        self.channel = self.pc.createDataChannel("track_info")

        # Add video track
        color_track = self.streamer
        self.pc.addTrack(color_track)

        @self.channel.on("open")
        def on_open():
            self.streamer.set_data_channel(self.channel)
            msg = {"type": "track_info", "color_track_id": color_track.id}
            logger.info(f"Data channel opened, sending track info: {msg}")
            self.channel.send(json.dumps(msg))

        @self.channel.on("message")
        def on_message(msg):
            logger.info(f"Received message: {msg}")
            msg = json.loads(msg)

            # Check if channel is open before sending
            if self.channel.readyState == "open":
                if msg["type"] == "ping":
                    self.channel.send(
                        json.dumps({"type": "pong", "timestamp": time.time()})
                    )
                elif msg["type"] == "pong":
                    self.channel.send(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )

        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering to complete
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        # Filter SDP to remove VP8 codecs
        modified_sdp = self._filter_sdp(self.pc.localDescription.sdp)

        # Send offer via MQTT using SDK
        # Match the working code's topic format: {twin_uuid}/webrtc-offer
        offer_topic = f"cyberwave/twin/{self.twin_uuid}/webrtc-offer"
        offer_payload = {
            "target": "backend",
            "sender": "edge",
            "type": self.pc.localDescription.type,
            "sdp": modified_sdp,
            "color_track_id": color_track.id,
            "timestamp": time.time(),
        }

        self._publish_message(offer_topic, offer_payload)
        logger.info("WebRTC offer sent")

        # Wait for answer
        timeout = 60  # 60 second timeout
        start_time = time.time()
        while not self._answer_received:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for WebRTC answer")
            await asyncio.sleep(0.1)

        logger.info("WebRTC answer received")

        if self._answer_data is None:
            raise RuntimeError("Answer received flag set but answer data is None")

        if isinstance(self._answer_data, str):
            answer = json.loads(self._answer_data)
        else:
            answer = self._answer_data

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

        logger.info("WebRTC connection established")

    def _filter_sdp(self, sdp: str) -> str:
        """
        Filter SDP to remove VP8 codec lines.

        Args:
            sdp: Original SDP string

        Returns:
            Modified SDP string
        """
        sdp_lines = sdp.split("\r\n")
        final_sdp_lines = []
        m_video_parts = []

        vp8_prefixes = (
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

        for line in sdp_lines:
            if line.startswith("m=video"):
                parts = line.split()
                for part in parts:
                    if part not in ["97", "98"]:
                        m_video_parts.append(part)
                final_sdp_lines.append(" ".join(m_video_parts))
            elif line.startswith(vp8_prefixes):
                continue
            else:
                final_sdp_lines.append(line)

        return "\r\n".join(final_sdp_lines)

    def _subscribe_to_answer(self):
        """Subscribe to WebRTC answer topic."""
        if not self.twin_uuid:
            raise ValueError("twin_uuid must be set before subscribing")

        # Match the working code's topic format: {twin_uuid}/webrtc-answer
        answer_topic = f"cyberwave/twin/{self.twin_uuid}/webrtc-answer"

        def on_answer(data):
            """Callback for WebRTC answer messages."""
            try:
                # Data is already parsed by the MQTT client
                payload = data if isinstance(data, dict) else json.loads(data)
                logger.info(f"Received message: type={payload.get('type')}")
                logger.debug(f"Full payload: {payload}")

                # Skip if this is an offer message
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
                        return
                else:
                    raise ValueError(f"Unknown message type: {payload.get('type')}")
            except Exception as e:
                # logger.error(f"Error processing WebRTC answer: {e}")
                raise e

        # Use SDK's MQTT client to subscribe
        # The SDK creates separate Messaging instances for each subscription
        self.client.subscribe(answer_topic, on_answer)
        logger.info(f"Subscribed to {answer_topic}")

    def _publish_message(self, topic: str, payload: Dict[str, Any]):
        """
        Publish a message via MQTT.

        Args:
            topic: MQTT topic
            payload: Message payload as dictionary
        """
        # The MQTT client's publish method accepts a dict and handles JSON conversion
        # QoS 2 ensures message delivery (matching working implementation)
        self.client.publish(topic, payload, qos=2)
        logger.info(f"Published to {topic}")

    async def stop(self):
        """Stop streaming and cleanup resources."""
        if self.streamer:
            self.streamer.close()
        if self.pc:
            await self.pc.close()
        logger.info("Camera streaming stopped")
