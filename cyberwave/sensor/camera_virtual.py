"""Virtual camera streaming using callback-based frame providers.

Use VirtualCameraStreamer for streaming from custom sources like simulations,
robot feeds, or processing pipelines. Pass a get_frame() callback that returns
RGB numpy arrays (HxWx3 uint8) or None for placeholder frames.

Example:
    >>> def my_frame_source():
    ...     return np.zeros((480, 640, 3), dtype=np.uint8)  # RGB frame
    >>>
    >>> streamer = VirtualCameraStreamer(
    ...     client.mqtt, get_frame=my_frame_source,
    ...     width=640, height=480, fps=30, twin_uuid="camera_id"
    ... )
    >>> await streamer.start()
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
from av import VideoFrame

from . import BaseVideoTrack, BaseVideoStreamer

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference

logger = logging.getLogger(__name__)


class VirtualVideoTrack(BaseVideoTrack):
    """Video track that calls get_frame() to fetch frames at the specified fps.

    Returns placeholder (blue frame or custom image) when get_frame() returns None.

    Args:
        get_frame: Callable returning RGB ndarray (HxWx3 uint8) or None
        width: Frame width in pixels (default: 640)
        height: Frame height in pixels (default: 480)
        fps: Target frames per second (default: 15)
        time_reference: Optional TimeReference for clock sync
        placeholder_image: Optional RGB placeholder image when get_frame returns None
    """

    def __init__(
        self,
        get_frame: Callable[[], Optional[np.ndarray]],
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        time_reference: Optional["TimeReference"] = None,
        placeholder_image: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.get_frame = get_frame
        self.width = width
        self.height = height
        self.fps = fps
        self._last_time = None
        self.time_reference = time_reference

        if placeholder_image is not None:
            self._placeholder = np.ascontiguousarray(placeholder_image, dtype=np.uint8)
            logger.info(
                "VirtualVideoTrack initialized with custom placeholder image (%dx%d @ %dfps)",
                width,
                height,
                fps,
            )
        else:
            # No cached placeholder by default. A solid-color frame will be created on demand.
            self._placeholder = None
            logger.info(
                "VirtualVideoTrack initialized with fallback placeholder (%dx%d @ %dfps)",
                width,
                height,
                fps,
            )

    def get_stream_attributes(self) -> dict:
        """Get streaming attributes for the offer payload."""
        return {
            "camera_type": "virtual",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    async def recv(self):
        """Receive and encode the next video frame."""
        # Maintain target frame rate
        now = time.time()
        if self._last_time is not None:
            elapsed = now - self._last_time
            wait = max(0.0, (1.0 / float(self.fps)) - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_time = time.time()

        # Get timestamps before fetching frame
        if self.time_reference is not None:
            timestamp, timestamp_monotonic = self.time_reference.read()
        else:
            timestamp = time.time()
            timestamp_monotonic = time.monotonic()

        # Store first frame timestamp
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        # Fetch frame from callback
        frame = None
        try:
            frame = self.get_frame()
        except Exception as e:
            logger.warning("Virtual frame provider error: %s", e)

        # Use placeholder if no frame available
        if frame is None:
            if self._placeholder is not None:
                # Modulate placeholder brightness with 2-second period
                t = time.time()
                modulation = 0.5 + 0.5 * np.sin(2 * np.pi * t / 2.0)
                frame = (self._placeholder * modulation).astype(np.uint8)
            else:
                # Solid blue fallback (RGB)
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                frame[...] = (0, 0, 255)

        # Log timestamps periodically
        if self.frame_count % 60 == 0 and self.frame_count > 0:
            logger.debug(
                "%d: ts=%.3f, ts_monotonic=%.3f",
                self.frame_count,
                timestamp,
                timestamp_monotonic,
            )

        # Convert to video frame
        arr = np.ascontiguousarray(frame)
        video_frame = VideoFrame.from_ndarray(arr, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.fps)

        # Send sync frame via inherited method
        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        self.frame_count += 1

        return video_frame

    def close(self):
        """Release resources (no-op for virtual track)."""
        pass


class VirtualCameraStreamer(BaseVideoStreamer):
    """Stream video from custom sources via callback function.

    Your get_frame() callback should return RGB numpy arrays (HxWx3 uint8)
    or None to show a placeholder. Keep callbacks fast (<10ms) for smooth streaming.

    Example:
        >>> def my_frames():
        ...     return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>>
        >>> streamer = VirtualCameraStreamer(
        ...     client.mqtt, 
        ...     get_frame=my_frames,
        ...     width=640, 
        ...     height=480, 
        ...     fps=30, 
        ...     twin_uuid="camera_twin",
        ...     time_reference=time_reference
        ... )
        >>> await streamer.start()
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        get_frame: Callable[[], Optional[np.ndarray]],
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        time_reference: Optional["TimeReference"] = None,
        twin_uuid: Optional[str] = None,
        auto_reconnect: bool = True,
        placeholder_image: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize virtual camera streamer.

        Args:
            client: MQTT client (client.mqtt)
            get_frame: Callback returning RGB frame (HxWx3 uint8) or None
            width: Frame width in pixels (default: 640)
            height: Frame height in pixels (default: 480)
            fps: Target frames per second (default: 15)
            time_reference: Time reference for synchronization
            twin_uuid: UUID of the digital twin
            turn_servers: Optional list of TURN server configurations
            auto_reconnect: Whether to automatically reconnect on disconnection
            placeholder_image: Optional placeholder image when get_frame returns None
        """
        super().__init__(
            client=client,
            turn_servers=[
                {
                    "urls": "turn:turn.cyberwave.com:3478",
                    "username": "cyberwave-user",
                    "credential": "cyberwave-admin",
                    }
            ],
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
        )

        # Store virtual camera-specific parameters
        self.get_frame = get_frame
        self.width = width
        self.height = height
        self.fps = fps
        self.placeholder_image = placeholder_image

        logger.info(
            f"✅ VirtualCameraStreamer initialized for twin {twin_uuid} "
            f"({width}x{height} @ {fps}fps)"
        )

    def initialize_track(self) -> BaseVideoTrack:
        """Create the video track (called internally by BaseVideoStreamer)."""
        return VirtualVideoTrack(
            self.get_frame,
            width=self.width,
            height=self.height,
            fps=self.fps,
            time_reference=self.time_reference,
            placeholder_image=self.placeholder_image,
        )

