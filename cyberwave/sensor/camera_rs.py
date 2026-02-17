"""RealSense camera implementation for Cyberwave SDK.

Provides video streaming using Intel RealSense cameras with RGB and depth support.
"""

import base64
import fractions
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from av import VideoFrame

from . import BaseVideoTrack, BaseVideoStreamer
from .config import Resolution, RealSenseConfig

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference

# Make pyrealsense2 optional
try:
    import pyrealsense2 as rs

    _has_realsense = True
except ImportError:
    rs = None
    _has_realsense = False

logger = logging.getLogger(__name__)


def require_realsense():
    """Check if RealSense is available, raise ImportError if not."""
    if not _has_realsense:
        raise ImportError(
            "RealSense camera support requires pyrealsense2. "
            "Install with: pip install pyrealsense2"
        )


class RealSenseVideoTrack(BaseVideoTrack):
    """Video stream track using Intel RealSense camera.

    Supports RGB and depth streaming with frame alignment.
    """

    def __init__(
        self,
        color_fps: int = 30,
        depth_fps: int = 30,
        color_resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        depth_resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        enable_depth: bool = True,
        client: Optional["CyberwaveMQTTClient"] = None,
        time_reference: Optional["TimeReference"] = None,
        twin_uuid: Optional[str] = None,
        depth_publish_interval: int = 30,
    ):
        """Initialize the RealSense video stream track.

        Args:
            color_fps: FPS for color stream (default: 30)
            depth_fps: FPS for depth stream (default: 30)
            color_resolution: RGB stream resolution (default: VGA 640x480)
            depth_resolution: Depth stream resolution (default: VGA 640x480)
            enable_depth: Whether to enable depth streaming (default: True)
            client: MQTT client for publishing depth frames
            time_reference: Time reference for synchronization
            twin_uuid: UUID of the digital twin
            depth_publish_interval: Publish depth every N frames (default: 30)
        """
        require_realsense()
        super().__init__()

        self.client = client
        self.time_reference = time_reference
        self.twin_uuid = twin_uuid

        # Parse color resolution
        if isinstance(color_resolution, Resolution):
            self.color_width = color_resolution.width
            self.color_height = color_resolution.height
        else:
            self.color_width, self.color_height = color_resolution

        # Parse depth resolution
        if isinstance(depth_resolution, Resolution):
            self.depth_width = depth_resolution.width
            self.depth_height = depth_resolution.height
        else:
            self.depth_width, self.depth_height = depth_resolution

        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self.enable_depth = enable_depth
        self.depth_publish_interval = depth_publish_interval

        # RealSense objects
        self.pipeline: Optional["rs.pipeline"] = None
        self.config: Optional["rs.config"] = None
        self.align: Optional["rs.align"] = None

        # Cached frame for fallback when reading fails
        self._cached_color_image: Optional[np.ndarray] = None
        self._cached_depth_image: Optional[np.ndarray] = None

        # Validate depth requirements
        if self.enable_depth and (not self.client or not self.twin_uuid):
            raise ValueError(
                "To enable depth streaming, client and twin_uuid must be provided"
            )

        # Initialize camera
        self.camera_initialized = self._initialize_realsense()
        if not self.camera_initialized:
            raise RuntimeError("Failed to initialize RealSense camera")

    def _initialize_realsense(self) -> bool:
        """Initialize RealSense camera and pipeline.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            ctx = rs.context()
            devices = ctx.query_devices()

            if len(devices) == 0:
                logger.error("No RealSense devices found!")
                return False

            logger.info(f"Found {len(devices)} RealSense device(s)")

            for i, device in enumerate(devices):
                logger.debug(
                    f"Device {i}: {device.get_info(rs.camera_info.name)} "
                    f"Serial: {device.get_info(rs.camera_info.serial_number)}"
                )

            # Create pipeline and configuration
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure color stream
            self.config.enable_stream(
                rs.stream.color,
                self.color_width,
                self.color_height,
                rs.format.bgr8,
                self.color_fps,
            )

            # Configure depth stream if enabled
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.depth_width,
                    self.depth_height,
                    rs.format.z16,
                    self.depth_fps,
                )

            # Create alignment object
            self.align = rs.align(rs.stream.color)

            # Start pipeline
            profile = self.pipeline.start(self.config)

            device = profile.get_device()
            logger.info(
                f"Started RealSense pipeline: {device.get_info(rs.camera_info.name)}, "
                f"color={self.color_width}x{self.color_height}@{self.color_fps}fps"
                + (f", depth={self.depth_width}x{self.depth_height}@{self.depth_fps}fps" if self.enable_depth else "")
            )

            # Warm up camera
            logger.info("Warming up camera...")
            for i in range(30):
                _ = self.pipeline.wait_for_frames()
                logger.debug(f"Warmup frame {i + 1}/30")

            logger.info("RealSense camera initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing RealSense camera: {e}")
            return False

    def _get_frames(
        self,
    ) -> Tuple[bool, Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]:
        """Get aligned color and depth frames from RealSense camera.

        Returns:
            Tuple of (success, (color_frame, depth_frame)) or (False, None) if failed
        """
        # Use a more generous timeout to handle occasional delays (at least 100ms, or 2x frame time)
        frame_time_ms = 1000 // self.color_fps
        timeout = max(100, frame_time_ms * 2)  # At least 100ms, or 2x frame time
        
        try:
            if not self.pipeline:
                logger.error("Pipeline not initialized")
                return False, None
            if not self.align:
                logger.error("Align not initialized")
                return False, None

            frames = self.pipeline.wait_for_frames(timeout_ms=timeout)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if self.enable_depth else None

            if not color_frame:
                logger.warning("Could not get color frame")
                return False, None

            # Only require depth frame if depth is enabled
            if self.enable_depth and not depth_frame:
                logger.warning("Could not get depth frame")
                return False, None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = (
                np.asanyarray(depth_frame.get_data()) if depth_frame else None
            )

            # Cache frames for fallback
            self._cached_color_image = color_image.copy()
            if depth_image is not None:
                self._cached_depth_image = depth_image.copy()
            elif not self.enable_depth:
                # Clear cached depth if depth is disabled
                self._cached_depth_image = None

            return True, (color_image, depth_image)

        except RuntimeError as e:
            error_msg = str(e)
            if "Frame didn't arrive" in error_msg:
                logger.warning(f"Frame timeout after {timeout}ms (expected frame time: {frame_time_ms}ms)")
            elif "device is disconnected" in error_msg.lower() or "device not found" in error_msg.lower():
                logger.error(f"RealSense device disconnected: {e}")
                # Don't return None immediately - let retry logic in recv() handle it
                return False, None
            else:
                logger.error(f"RealSense runtime error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Error getting frames: {e}", exc_info=True)
            return False, None

    def _publish_depth_frame(self, depth_image: np.ndarray, timestamp: float):
        """Publish depth frame via MQTT."""
        if self.client is None or self.twin_uuid is None:
            return

        if depth_image.dtype != np.uint16:
            depth_image = depth_image.astype(np.uint16)

        # Use actual image dimensions (depth is aligned to color, so dimensions may differ)
        height, width = depth_image.shape[:2]

        depth_binary = base64.b64encode(depth_image.tobytes()).decode("utf-8")
        depth_data = {
            "depth_binary": depth_binary,
            "width": width,
            "height": height,
            "dtype": "uint16",
        }
        self.client.publish_depth_frame(self.twin_uuid, depth_data, timestamp)

    async def recv(self):
        """Receive and encode the next video frame."""
        ret, frames = self._get_frames()
        
        # If reading failed, use cached frame
        if not ret or frames is None:
            if self._cached_color_image is not None:
                logger.debug("Failed to read frames from RealSense camera, using cached frame")
                color_image = self._cached_color_image
                depth_image = self._cached_depth_image if self.enable_depth else None
            else:
                # No cached frame available - return None
                logger.error("Failed to read frames and no cached frame available")
                return None
        else:
            color_image, depth_image = frames

        # Update time reference to capture current timestamp at frame capture moment.
        # This ensures video frame timestamps reflect actual capture time, not
        # a potentially stale timestamp from the teleop loop.
        timestamp, timestamp_monotonic = self.time_reference.update()

        # Store frame 0 timestamp for publishing
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        # Create video frame
        video_frame = VideoFrame.from_ndarray(color_image, format="bgr24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, int(self.color_fps))

        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)

        # Publish depth frame at configured interval
        if (
            self.enable_depth
            and depth_image is not None
            and self.frame_count % self.depth_publish_interval == 0
        ):
            self._publish_depth_frame(depth_image, timestamp)

        self.frame_count += 1
        return video_frame

    def get_stream_attributes(self) -> dict:
        """Get streaming attributes for the offer payload.

        Returns:
            Dictionary with RealSense camera stream attributes
        """
        return {
            "camera_type": "realsense",
            "color_width": self.color_width,
            "color_height": self.color_height,
            "color_fps": self.color_fps,
            "depth_enabled": self.enable_depth,
            "depth_width": self.depth_width if self.enable_depth else None,
            "depth_height": self.depth_height if self.enable_depth else None,
            "depth_fps": self.depth_fps if self.enable_depth else None,
            "depth_publish_interval": self.depth_publish_interval if self.enable_depth else None,
        }

    def close(self):
        """Release camera resources."""
        if self.pipeline:
            self.pipeline.stop()
            logger.info("RealSense camera released")


class RealSenseStreamer(BaseVideoStreamer):
    """WebRTC camera streamer using Intel RealSense for video capture.

    Supports RGB and depth streaming with automatic depth publishing via MQTT.

    Example:
        >>> from cyberwave import Cyberwave
        >>> from cyberwave.sensor import RealSenseStreamer, Resolution
        >>> import asyncio
        >>>
        >>> client = Cyberwave(token="your_token")
        >>> streamer = RealSenseStreamer(
        ...     client.mqtt,
        ...     twin_uuid="your_twin_uuid",
        ...     color_resolution=Resolution.HD,
        ...     enable_depth=True
        ... )
        >>> asyncio.run(streamer.start())

    Example with RealSenseConfig:
        >>> from cyberwave.sensor import RealSenseStreamer, RealSenseConfig, Resolution
        >>>
        >>> config = RealSenseConfig(
        ...     color_resolution=Resolution.HD,
        ...     depth_resolution=Resolution.VGA,
        ...     color_fps=30,
        ...     depth_fps=15,
        ...     enable_depth=True
        ... )
        >>> streamer = RealSenseStreamer.from_config(client.mqtt, config, twin_uuid="...")
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        color_fps: int = 30,
        depth_fps: int = 30,
        color_resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        depth_resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        enable_depth: bool = True,
        depth_publish_interval: int = 30,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
    ):
        """Initialize the RealSense camera streamer.

        Args:
            client: Cyberwave MQTT client instance
            color_fps: FPS for color stream (default: 30)
            depth_fps: FPS for depth stream (default: 30)
            color_resolution: RGB stream resolution (default: VGA 640x480)
            depth_resolution: Depth stream resolution (default: VGA 640x480)
            enable_depth: Whether to enable depth streaming (default: True)
            depth_publish_interval: Publish depth every N frames (default: 30)
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
        """
        require_realsense()
        super().__init__(
            client=client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
        )

        # RealSense specific configuration
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self.color_resolution = color_resolution
        self.depth_resolution = depth_resolution
        self.enable_depth = enable_depth
        self.depth_publish_interval = depth_publish_interval

    @classmethod
    def from_config(
        cls,
        client: "CyberwaveMQTTClient",
        config: RealSenseConfig,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
        validate: bool = True,
    ) -> "RealSenseStreamer":
        """Create streamer from RealSenseConfig.

        Args:
            client: Cyberwave MQTT client instance
            config: RealSense configuration
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
            validate: Whether to validate config against device capabilities (default: True)

        Returns:
            Configured RealSenseStreamer instance

        Raises:
            ValueError: If validate=True and config is not supported by device
        """
        if validate:
            is_valid, errors = config.validate()
            if not is_valid:
                raise ValueError(f"RealSense config validation failed: {'; '.join(errors)}")

        return cls(
            client=client,
            color_fps=config.color_fps,
            depth_fps=config.depth_fps,
            color_resolution=(config.color_width, config.color_height),
            depth_resolution=(config.depth_width, config.depth_height),
            enable_depth=config.enable_depth,
            depth_publish_interval=config.depth_publish_interval,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
        )

    @classmethod
    def from_device(
        cls,
        client: "CyberwaveMQTTClient",
        serial_number: Optional[str] = None,
        prefer_resolution: Resolution = Resolution.VGA,
        prefer_fps: int = 30,
        enable_depth: bool = True,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
    ) -> "RealSenseStreamer":
        """Create streamer with auto-detected device configuration.

        Automatically discovers device capabilities and creates optimal config.

        Args:
            client: Cyberwave MQTT client instance
            serial_number: Target device serial number (None = first device)
            prefer_resolution: Preferred resolution (will find closest match)
            prefer_fps: Preferred FPS (will find closest match)
            enable_depth: Whether to enable depth streaming
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection

        Returns:
            Configured RealSenseStreamer instance
        """
        config = RealSenseConfig.from_device(
            serial_number=serial_number,
            prefer_resolution=prefer_resolution,
            prefer_fps=prefer_fps,
            enable_depth=enable_depth,
        )
        logger.info(f"Auto-configured RealSense: {config}")

        return cls.from_config(
            client=client,
            config=config,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
            validate=False,  # Already validated during from_device
        )

    def initialize_track(self) -> RealSenseVideoTrack:
        """Initialize and return the RealSense video track."""
        self.streamer = RealSenseVideoTrack(
            color_fps=self.color_fps,
            depth_fps=self.depth_fps,
            color_resolution=self.color_resolution,
            depth_resolution=self.depth_resolution,
            enable_depth=self.enable_depth,
            client=self.client,
            time_reference=self.time_reference,
            twin_uuid=self.twin_uuid,
            depth_publish_interval=self.depth_publish_interval,
        )
        return self.streamer
