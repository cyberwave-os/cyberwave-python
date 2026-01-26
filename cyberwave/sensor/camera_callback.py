"""Callback-based camera streaming for Cyberwave SDK.

Provides a video track and streamer that use an external frame provider.
This is ideal for robot WebRTC feeds or custom pipelines where frames
are produced elsewhere and supplied via a callable.
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


class CallbackVideoTrack(BaseVideoTrack):
    """Video track that pulls frames from a callback.

    Args:
        get_frame: Callable returning an HxWx3 RGB uint8 ndarray or None.
        width: Expected frame width (used for placeholder fallback).
        height: Expected frame height (used for placeholder fallback).
        fps: Target frames per second.
        time_reference: Optional TimeReference for timestamp sync.
        placeholder_image: Optional placeholder image (HxWx3 uint8 RGB).
        placeholder_color: RGB tuple for placeholder if no image provided.
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
        placeholder_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()
        self.get_frame = get_frame
        self.width = width
        self.height = height
        self.fps = fps
        self.time_reference = time_reference
        self._last_time: Optional[float] = None

        if placeholder_image is not None:
            self._placeholder = np.ascontiguousarray(placeholder_image, dtype=np.uint8)
        else:
            self._placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            self._placeholder[..., 0] = placeholder_color[0]
            self._placeholder[..., 1] = placeholder_color[1]
            self._placeholder[..., 2] = placeholder_color[2]

    def get_stream_attributes(self) -> dict:
        return {
            "camera_type": "callback",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    async def recv(self):
        # Maintain target frame rate
        now = time.time()
        if self._last_time is not None:
            elapsed = now - self._last_time
            wait = max(0.0, (1.0 / float(self.fps)) - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_time = time.time()

        # Fetch frame from callback
        frame = None
        try:
            frame = self.get_frame()
        except Exception as e:
            logger.warning("Callback frame provider error: %s", e)

        if frame is None:
            frame = self._placeholder

        arr = np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            logger.warning("Invalid frame format; expected HxWx3 RGB, got %s", arr.shape)
            arr = self._placeholder

        if self.time_reference is not None:
            timestamp, timestamp_monotonic = self.time_reference.read()
        else:
            timestamp = time.time()
            timestamp_monotonic = time.monotonic()

        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        video_frame = VideoFrame.from_ndarray(arr, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, int(self.fps))

        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        self.frame_count += 1

        return video_frame

    def close(self):
        """Release resources (no-op for callback-based track)."""
        return


class CallbackCameraStreamer(BaseVideoStreamer):
    """WebRTC streamer for external frame sources.

    Use this when frames come from a custom source (robot WebRTC, pipeline,
    shared memory, etc.) and are provided via a callback.
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        get_frame: Callable[[], Optional[np.ndarray]],
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
        placeholder_image: Optional[np.ndarray] = None,
        placeholder_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(
            client=client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
        )
        self._get_frame = get_frame
        self._width = width
        self._height = height
        self._fps = fps
        self._placeholder_image = placeholder_image
        self._placeholder_color = placeholder_color

    def initialize_track(self) -> BaseVideoTrack:
        return CallbackVideoTrack(
            self._get_frame,
            width=self._width,
            height=self._height,
            fps=self._fps,
            time_reference=self.time_reference,
            placeholder_image=self._placeholder_image,
            placeholder_color=self._placeholder_color,
        )
