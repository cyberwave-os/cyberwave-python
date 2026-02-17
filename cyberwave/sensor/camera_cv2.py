"""CV2 (OpenCV) camera implementation for Cyberwave SDK.

Provides video streaming using standard USB/webcam cameras and IP cameras via OpenCV.

Supports:
- Local cameras: camera_id=0, camera_id=1 (device index)
- IP cameras: camera_id="http://192.168.1.100/snapshot.jpg"
- RTSP streams: camera_id="rtsp://192.168.1.100:554/stream"
"""

import fractions
import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, Union

import cv2
import numpy as np
from av import VideoFrame

from . import BaseVideoTrack, BaseVideoStreamer
from .config import CameraConfig, Resolution

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference

logger = logging.getLogger(__name__)


def _get_default_keyframe_interval() -> Optional[int]:
    """Get default keyframe interval from environment variable.

    Returns:
        Keyframe interval in frames, or None if not configured.
        Recommended values: 30 (1sec at 30fps), 60 (2sec at 30fps)
    """
    env_value = os.environ.get("CYBERWAVE_KEYFRAME_INTERVAL")
    if env_value:
        try:
            interval = int(env_value)
            if interval > 0:
                return interval
        except ValueError:
            pass
    return None


class CV2VideoTrack(BaseVideoTrack):
    """Video stream track using OpenCV for camera capture.

    Supports:
    - Standard USB cameras and webcams (camera_id as int)
    - IP cameras via HTTP (camera_id as URL string)
    - RTSP streams (camera_id as rtsp:// URL)

    Example:
        >>> # Local USB camera
        >>> track = CV2VideoTrack(camera_id=0, fps=30, resolution=Resolution.HD)
        >>>
        >>> # IP camera
        >>> track = CV2VideoTrack(
        ...     camera_id="rtsp://192.168.1.100:554/stream",
        ...     fps=15,
        ...     resolution=Resolution.VGA
        ... )
    """

    def __init__(
        self,
        camera_id: Union[int, str] = 0,
        fps: int = 30,
        resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        time_reference: Optional["TimeReference"] = None,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        """Initialize the CV2 video stream track.

        Args:
            camera_id: Camera device ID (int) or stream URL (str)
                - int: Local camera device index (0, 1, etc.)
                - str: URL for IP camera (http://, rtsp://, https://)
            fps: Frames per second (default: 30)
            resolution: Video resolution as Resolution enum or (width, height) tuple
                       (default: Resolution.VGA = 640x480)
            time_reference: Time reference for synchronization
            keyframe_interval: Force a keyframe every N frames. If None, uses
                CYBERWAVE_KEYFRAME_INTERVAL env var, or disables forced keyframes.
                Recommended: fps * 2 (e.g., 60 for 30fps = keyframe every 2 seconds)
            frame_callback: Optional callback called for each frame.
                Signature: callback(frame: np.ndarray, frame_count: int) -> None
                Called after frame normalization, before encoding.
        """
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.time_reference = time_reference
        self.frame_callback = frame_callback

        # Keyframe interval: use provided value, env var, or None (disabled)
        self.keyframe_interval = (
            keyframe_interval
            if keyframe_interval is not None
            else _get_default_keyframe_interval()
        )
        self._frames_since_keyframe = 0

        # Frame format warning flags (log once)
        self._logged_frame_info = False
        self._warned_frame_format = False
        self._warned_frame_dtype = False

        # Parse resolution
        if isinstance(resolution, Resolution):
            self.requested_width = resolution.width
            self.requested_height = resolution.height
            self.resolution: Optional[Resolution] = resolution
        else:
            self.requested_width, self.requested_height = resolution
            self.resolution = Resolution.from_size(*resolution)

        # Initialize camera with appropriate backend
        self.cap = self._open_capture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        # Configure camera settings
        self._configure_camera()

        # Get actual values after configuration
        self.actual_width, self.actual_height, self.actual_fps = self._get_actual_settings()

        log_msg = (
            f"Initialized CV2 camera {camera_id}: "
            f"requested={self.requested_width}x{self.requested_height}@{fps}fps, "
            f"actual={self.actual_width}x{self.actual_height}@{self.actual_fps}fps"
        )
        if self.keyframe_interval:
            log_msg += f", keyframe_interval={self.keyframe_interval}"
        logger.info(log_msg)

        # Warn if actual differs from requested
        if (
            self.actual_width != self.requested_width
            or self.actual_height != self.requested_height
        ):
            logger.warning(
                f"Camera resolution mismatch: requested {self.requested_width}x{self.requested_height}, "
                f"got {self.actual_width}x{self.actual_height}"
            )

    def _select_capture_backends(self, camera_id: Union[int, str]) -> list[int]:
        """Select appropriate capture backends based on source type.

        Args:
            camera_id: Camera device ID or URL

        Returns:
            List of cv2 backend constants to try
        """
        # Check for explicit backend override
        backend_env = os.environ.get("CYBERWAVE_CV2_BACKEND", "").strip().lower()
        backend_map = {
            "ffmpeg": cv2.CAP_FFMPEG,
            "gstreamer": cv2.CAP_GSTREAMER,
            "any": cv2.CAP_ANY,
        }
        if backend_env:
            backend = backend_map.get(backend_env)
            if backend is None:
                logger.warning(
                    f"Unknown CYBERWAVE_CV2_BACKEND '{backend_env}'; using default"
                )
                return []
            return [backend]

        # For URL sources, prefer FFMPEG then GStreamer
        if isinstance(camera_id, str) and camera_id.startswith(
            ("rtsp://", "http://", "https://")
        ):
            return [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]

        # For local cameras, use default
        return []

    def _open_capture(self, camera_id: Union[int, str]) -> cv2.VideoCapture:
        """Open video capture with appropriate backend.

        Args:
            camera_id: Camera device ID or URL

        Returns:
            Opened VideoCapture object
        """
        backends = self._select_capture_backends(camera_id)
        backend_names = {
            cv2.CAP_FFMPEG: "FFMPEG",
            cv2.CAP_GSTREAMER: "GSTREAMER",
            cv2.CAP_ANY: "AUTO",
        }

        # For RTSP streams, use TCP transport for better reliability with HEVC
        # This avoids "Could not find ref with POC" errors from lost UDP packets
        is_rtsp = isinstance(camera_id, str) and camera_id.startswith("rtsp://")
        if is_rtsp:
            # Set FFmpeg options for TCP transport
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            logger.info("Using TCP transport for RTSP stream")

        # Try each backend in order
        for backend in backends:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                logger.info(
                    f"Opened video capture with {backend_names.get(backend, backend)} backend"
                )
                return cap
            cap.release()

        # Fall back to default
        return cv2.VideoCapture(camera_id)

    def _configure_camera(self):
        """Configure camera capture settings."""
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Enable RGB conversion
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        # For RTSP/HTTP streams, minimize buffer to reduce latency
        if isinstance(self.camera_id, str) and self.camera_id.startswith(
            ("rtsp://", "http://", "https://")
        ):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _get_actual_settings(self) -> tuple[int, int, float]:
        """Get actual camera settings after configuration.

        Returns:
            Tuple of (width, height, fps) as actually set by the camera
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return width, height, fps

    @property
    def width(self) -> int:
        """Get actual frame width."""
        return self.actual_width

    @property
    def height(self) -> int:
        """Get actual frame height."""
        return self.actual_height

    def get_stream_attributes(self) -> dict:
        """Get streaming attributes for the offer payload.

        Returns:
            Dictionary with CV2 camera stream attributes
        """
        # Mask URL credentials if present
        camera_id_display = self.camera_id
        if isinstance(self.camera_id, str) and "@" in self.camera_id:
            # Hide credentials in URL
            parts = self.camera_id.split("://", 1)
            if len(parts) == 2 and "@" in parts[1]:
                protocol = parts[0]
                rest = parts[1].split("@", 1)[-1]  # Get part after @
                camera_id_display = f"{protocol}://***@{rest}"

        return {
            "camera_type": "cv2",
            "camera_id": camera_id_display,
            "is_ip_camera": isinstance(self.camera_id, str),
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps or self.fps,
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "requested_fps": self.fps,
            "resolution": str(self.resolution) if self.resolution else None,
            "keyframe_interval": self.keyframe_interval,
        }

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to BGR24 format for encoding.

        Handles various input formats from different camera sources:
        - Grayscale (2D array)
        - BGRA (4 channels)
        - YUV formats
        - Non-uint8 dtypes

        Args:
            frame: Input frame from camera

        Returns:
            Normalized BGR24 frame as contiguous uint8 array
        """
        if not self._logged_frame_info:
            logger.info(f"Camera frame format: shape={frame.shape} dtype={frame.dtype}")
            self._logged_frame_info = True

        # Handle non-uint8 dtypes
        if frame.dtype != np.uint8:
            if not self._warned_frame_dtype:
                logger.warning(
                    f"Non-uint8 frame dtype detected ({frame.dtype}); converting to uint8"
                )
                self._warned_frame_dtype = True
            if frame.dtype == np.uint16:
                frame = (frame / 256).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Handle different channel configurations
        if frame.ndim == 2:
            # Grayscale
            if not self._warned_frame_format:
                logger.warning("Grayscale frame detected; converting to BGR")
                self._warned_frame_format = True
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 1:
                # Single channel
                if not self._warned_frame_format:
                    logger.warning("Single-channel frame detected; converting to BGR")
                    self._warned_frame_format = True
                frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif channels == 2:
                # YUV 4:2:2
                if not self._warned_frame_format:
                    logger.warning("YUV 4:2:2 frame detected; converting to BGR")
                    self._warned_frame_format = True
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                except cv2.error:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
            elif channels == 4:
                # BGRA
                if not self._warned_frame_format:
                    logger.warning("BGRA frame detected; converting to BGR")
                    self._warned_frame_format = True
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif channels != 3:
                # Unknown format, truncate to 3 channels
                if not self._warned_frame_format:
                    logger.warning(
                        f"Unexpected channel count {channels}; truncating to 3 channels"
                    )
                    self._warned_frame_format = True
                frame = frame[:, :, :3]

        return np.ascontiguousarray(frame, dtype=np.uint8)

    @classmethod
    def get_supported_resolutions(cls, camera_id: int = 0) -> list[Resolution]:
        """Probe camera to find supported resolutions.

        Args:
            camera_id: Camera device ID to probe

        Returns:
            List of supported Resolution values
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id} for probing")
            return []

        supported = []
        for resolution in Resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution.height)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == resolution.width and actual_height == resolution.height:
                supported.append(resolution)
                logger.debug(f"Camera {camera_id} supports {resolution}")
            else:
                logger.debug(
                    f"Camera {camera_id} does not support {resolution} "
                    f"(got {actual_width}x{actual_height})"
                )

        cap.release()
        return supported

    @classmethod
    def get_camera_info(cls, camera_id: int = 0) -> dict:
        """Get detailed camera information.

        Args:
            camera_id: Camera device ID

        Returns:
            Dictionary with camera properties
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {"error": f"Cannot open camera {camera_id}"}

        info = {
            "camera_id": camera_id,
            "backend": cap.getBackendName(),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "autofocus": cap.get(cv2.CAP_PROP_AUTOFOCUS),
        }

        cap.release()
        return info

    async def recv(self):
        """Receive and encode the next video frame."""
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        # Update time reference to capture current timestamp at frame capture moment.
        # This ensures video frame timestamps reflect actual capture time.
        timestamp, timestamp_monotonic = self.time_reference.update()

        # Store frame 0 timestamp for publishing
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        # Normalize frame format
        frame = self._normalize_frame(frame)

        # Call frame callback if set (for ML inference, etc.)
        if self.frame_callback:
            try:
                self.frame_callback(frame, self.frame_count)
            except Exception as e:
                logger.warning(f"Frame callback error: {e}")

        # Create video frame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")

        # Force keyframe periodically for better streaming start
        force_keyframe = False
        if self.keyframe_interval:
            if (
                self._frames_since_keyframe >= self.keyframe_interval
                or self.frame_count == 0
            ):
                force_keyframe = True
                self._frames_since_keyframe = 0
                logger.debug(f"Forcing keyframe at frame {self.frame_count}")
            else:
                self._frames_since_keyframe += 1

        if force_keyframe:
            try:
                from av.video.frame import PictureType

                video_frame.pict_type = PictureType.I
            except (ImportError, AttributeError):
                pass
            try:
                video_frame.key_frame = 1
            except AttributeError:
                pass

        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, int(self.actual_fps or self.fps))

        self._send_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
        self.frame_count += 1

        return video_frame

    def close(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            logger.info("CV2 camera released")


class CV2CameraStreamer(BaseVideoStreamer):
    """WebRTC camera streamer using OpenCV for video capture.

    Supports local cameras, IP cameras, and RTSP streams.

    Example with local camera:
        >>> from cyberwave import Cyberwave
        >>> from cyberwave.sensor import CV2CameraStreamer, Resolution
        >>> import asyncio
        >>>
        >>> client = Cyberwave(token="your_token")
        >>> streamer = CV2CameraStreamer(
        ...     client.mqtt,
        ...     camera_id=0,
        ...     resolution=Resolution.HD,
        ...     twin_uuid="your_twin_uuid"
        ... )
        >>> asyncio.run(streamer.start())

    Example with IP camera:
        >>> streamer = CV2CameraStreamer(
        ...     client.mqtt,
        ...     camera_id="rtsp://192.168.1.100:554/stream",
        ...     fps=15,
        ...     resolution=Resolution.VGA,
        ...     twin_uuid="your_twin_uuid"
        ... )

    Example with CameraConfig:
        >>> from cyberwave.sensor import CV2CameraStreamer, CameraConfig, Resolution
        >>>
        >>> config = CameraConfig(
        ...     resolution=Resolution.VGA,
        ...     fps=30,
        ...     camera_id=0
        ... )
        >>> streamer = CV2CameraStreamer.from_config(client.mqtt, config, twin_uuid="...")
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        camera_id: Union[int, str] = 0,
        fps: int = 30,
        resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        """Initialize the CV2 camera streamer.

        Args:
            client: Cyberwave MQTT client instance
            camera_id: Camera device ID (int) or stream URL (str)
                - int: Local camera device index (0, 1, etc.)
                - str: URL for IP camera (http://, rtsp://, https://)
            fps: Frames per second (default: 30)
            resolution: Video resolution as Resolution enum or (width, height) tuple
                       (default: Resolution.VGA = 640x480)
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
            keyframe_interval: Force a keyframe every N frames for better streaming start.
                If None, uses CYBERWAVE_KEYFRAME_INTERVAL env var, or disables forced keyframes.
                Recommended: fps * 2 (e.g., 60 for 30fps = keyframe every 2 seconds)
            frame_callback: Optional callback for each frame (ML inference, etc.).
                Signature: callback(frame: np.ndarray, frame_count: int) -> None
        """
        super().__init__(
            client=client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
        )
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.keyframe_interval = keyframe_interval
        self.frame_callback = frame_callback

    @classmethod
    def from_config(
        cls,
        client: "CyberwaveMQTTClient",
        config: CameraConfig,
        turn_servers: Optional[list] = None,
        twin_uuid: Optional[str] = None,
        time_reference: Optional["TimeReference"] = None,
        auto_reconnect: bool = True,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> "CV2CameraStreamer":
        """Create streamer from CameraConfig.

        Args:
            client: Cyberwave MQTT client instance
            config: Camera configuration
            turn_servers: Optional list of TURN server configurations
            twin_uuid: Optional UUID of the digital twin
            time_reference: Time reference for synchronization
            auto_reconnect: Whether to automatically reconnect on disconnection
            keyframe_interval: Force a keyframe every N frames
            frame_callback: Optional callback for each frame

        Returns:
            Configured CV2CameraStreamer instance
        """
        return cls(
            client=client,
            camera_id=config.camera_id,
            fps=config.fps,
            resolution=config.resolution,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
            keyframe_interval=keyframe_interval,
            frame_callback=frame_callback,
        )

    def initialize_track(self) -> CV2VideoTrack:
        """Initialize and return the CV2 video track."""
        self.streamer = CV2VideoTrack(
            camera_id=self.camera_id,
            fps=self.fps,
            resolution=self.resolution,
            time_reference=self.time_reference,
            keyframe_interval=self.keyframe_interval,
            frame_callback=self.frame_callback,
        )
        return self.streamer


# Backwards compatibility aliases
CameraStreamer = CV2CameraStreamer
