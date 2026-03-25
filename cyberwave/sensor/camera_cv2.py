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

# Default pixel format for local V4L2/USB when ``fourcc`` is omitted (see ``CV2VideoTrack``).
_DEFAULT_LOCAL_FOURCC = "MJPG"


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

    For **local** USB/V4L2 devices, if ``fourcc`` is omitted, the SDK tries ``MJPG``
    by default and reopens the device without a FOURCC override if negotiation fails.

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

    @staticmethod
    def _is_url_stream(camera_id: Union[int, str]) -> bool:
        return isinstance(camera_id, str) and camera_id.startswith(
            ("rtsp://", "http://", "https://")
        )

    @staticmethod
    def _fourcc_from_cap(cap: cv2.VideoCapture) -> str:
        code = int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF
        if code == 0:
            return ""
        raw = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        return raw.replace("\x00", "").strip()

    @staticmethod
    def _fourcc_tags_match(wanted: str, actual: str) -> bool:
        w = (wanted or "").strip()[:4].upper()
        a = (actual or "").strip()[:4].upper()
        if not w:
            return True
        if not a:
            return False
        return w == a

    def __init__(
        self,
        camera_id: Union[int, str] = 0,
        fps: int = 30,
        resolution: Union[Resolution, tuple[int, int]] = Resolution.VGA,
        time_reference: Optional["TimeReference"] = None,
        keyframe_interval: Optional[int] = None,
        frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        fourcc: Optional[str] = None,
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
            fourcc: Optional FOURCC for local USB/V4L2 cameras (e.g. ``'MJPG'``, ``'YUYV'``).
                If omitted for a **local** device, the SDK tries ``MJPG`` by default for better
                bandwidth/FPS; if that does not stick, it reopens the device without a FOURCC
                override (OpenCV/V4L2 default). URL/RTSP sources ignore FOURCC.
        """
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.time_reference = time_reference
        self.frame_callback = frame_callback
        self.fourcc = fourcc
        self._negotiated_fourcc_ascii: Optional[str] = None
        self._fourcc_attempted: Optional[str] = None
        self._fourcc_auto_mjpg: bool = False
        self._fourcc_fallback_reopen: bool = False

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

        self._negotiate_and_configure_capture()

        # Get actual values after configuration
        self.actual_width, self.actual_height, self.actual_fps = (
            self._get_actual_settings()
        )

        log_msg = (
            f"Initialized CV2 camera {camera_id}: "
            f"requested={self.requested_width}x{self.requested_height}@{fps}fps, "
            f"actual={self.actual_width}x{self.actual_height}@{self.actual_fps}fps"
        )
        if self.keyframe_interval:
            log_msg += f", keyframe_interval={self.keyframe_interval}"
        if self._negotiated_fourcc_ascii:
            log_msg += f", fourcc={self._negotiated_fourcc_ascii}"
        if self._fourcc_auto_mjpg:
            log_msg += ", fourcc_auto_mjpg=True"
        if self._fourcc_fallback_reopen:
            log_msg += ", fourcc_fallback_reopen=True"
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

    def _apply_capture_geometry_and_buffer(
        self,
        cap: cv2.VideoCapture,
        *,
        fourcc_str: Optional[str],
    ) -> None:
        """Set optional FOURCC (local sources only), then resolution, FPS, RGB, buffer."""
        if fourcc_str and not self._is_url_stream(self.camera_id):
            try:
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc_str[:4])
                cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
                logger.debug("Set camera FOURCC to %s", fourcc_str)
            except Exception as e:
                logger.warning("Failed to set FOURCC %s: %s", fourcc_str, e)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _negotiate_and_configure_capture(self) -> None:
        """Try FOURCC negotiation for local cameras; reopen without FOURCC if it fails."""
        is_local = not self._is_url_stream(self.camera_id)
        user_tag = (self.fourcc or "").strip()
        user_fourcc = user_tag[:4] if user_tag else None

        effective_try: Optional[str] = None
        if is_local:
            if user_fourcc:
                effective_try = user_fourcc
            else:
                effective_try = _DEFAULT_LOCAL_FOURCC
                self._fourcc_auto_mjpg = True

        if logger.isEnabledFor(logging.DEBUG):
            c = self.cap
            logger.debug(
                "CV2 after open: backend=%s size=%dx%d fps=%s fourcc=%r",
                c.getBackendName(),
                int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                c.get(cv2.CAP_PROP_FPS),
                self._fourcc_from_cap(c),
            )

        if effective_try:
            self._fourcc_attempted = effective_try
            self._apply_capture_geometry_and_buffer(self.cap, fourcc_str=effective_try)
            got = self._fourcc_from_cap(self.cap)
            if not self._fourcc_tags_match(effective_try, got):
                logger.warning(
                    "Camera FOURCC did not stick: tried %r, got %r; reopening without "
                    "FOURCC override",
                    effective_try,
                    got or "(empty)",
                )
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = self._open_capture(self.camera_id)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to reopen camera {self.camera_id}")
                self._fourcc_fallback_reopen = True
                self._apply_capture_geometry_and_buffer(self.cap, fourcc_str=None)
        else:
            self._apply_capture_geometry_and_buffer(self.cap, fourcc_str=None)

        neg = self._fourcc_from_cap(self.cap)
        self._negotiated_fourcc_ascii = neg or None

        logger.debug(
            "CV2 FOURCC negotiation: user_fourcc=%r attempted=%r negotiated=%r "
            "auto_mjpg=%s fallback_reopen=%s",
            user_fourcc,
            self._fourcc_attempted,
            self._negotiated_fourcc_ascii,
            self._fourcc_auto_mjpg,
            self._fourcc_fallback_reopen,
        )

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

        attrs: dict = {
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
        if self._negotiated_fourcc_ascii:
            attrs["fourcc"] = self._negotiated_fourcc_ascii
        if self.fourcc:
            attrs["fourcc_requested"] = self.fourcc[:4]
        if self._fourcc_auto_mjpg:
            attrs["fourcc_auto_mjpg"] = True
        if self._fourcc_fallback_reopen:
            attrs["fourcc_fallback_open_cv_default"] = True
        return attrs

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

        # Read time reference to capture current timestamp at frame capture moment.
        # This ensures video frame timestamps reflect actual capture time.
        timestamp, timestamp_monotonic = self.time_reference.read()

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

        self._capture_sync_frame(timestamp, timestamp_monotonic, video_frame.pts)
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
        >>> client = Cyberwave(api_key="your_api_key")
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
        camera_name: Optional[str] = None,
        fourcc: Optional[str] = None,
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
            camera_name: Optional sensor identifier for multi-stream twins. Use the sensor
                id from twin capabilities (e.g. "head_camera") when the twin has multiple cameras.
            fourcc: Optional FOURCC for local USB/V4L2 (e.g. ``'MJPG'``). If omitted,
                :class:`CV2VideoTrack` tries ``MJPG`` by default for better bandwidth/FPS.
                URL/RTSP sources ignore this.
        """
        super().__init__(
            client=client,
            turn_servers=turn_servers,
            twin_uuid=twin_uuid,
            time_reference=time_reference,
            auto_reconnect=auto_reconnect,
            camera_name=camera_name,
        )
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.keyframe_interval = keyframe_interval
        self.frame_callback = frame_callback
        self.fourcc = fourcc

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
        camera_name: Optional[str] = None,
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
            camera_name: Optional sensor identifier for multi-stream twins

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
            camera_name=camera_name,
            fourcc=getattr(config, "fourcc", None),
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
            fourcc=self.fourcc,
        )
        return self.streamer


# Backwards compatibility aliases
CameraStreamer = CV2CameraStreamer
