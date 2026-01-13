"""Camera configuration classes for Cyberwave SDK.

Provides resolution presets and camera configuration utilities.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Make pyrealsense2 optional
try:
    import pyrealsense2 as rs

    _has_realsense = True
except ImportError:
    rs = None
    _has_realsense = False


class Resolution(Enum):
    """Standard video resolutions for camera streaming.

    Each resolution is defined as (width, height) tuple.
    """

    QVGA = (320, 240)  # Quarter VGA
    VGA = (640, 480)  # Video Graphics Array
    SVGA = (800, 600)  # Super VGA
    HD = (1280, 720)  # 720p HD
    FULL_HD = (1920, 1080)  # 1080p Full HD

    @property
    def width(self) -> int:
        """Get resolution width in pixels."""
        return self.value[0]

    @property
    def height(self) -> int:
        """Get resolution height in pixels."""
        return self.value[1]

    @property
    def size(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        return self.value

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        return self.value[0] / self.value[1]

    @property
    def pixel_count(self) -> int:
        """Get total pixel count."""
        return self.value[0] * self.value[1]

    def __str__(self) -> str:
        return f"{self.value[0]}x{self.value[1]}"

    @classmethod
    def from_size(cls, width: int, height: int) -> Optional["Resolution"]:
        """Get Resolution enum from width and height.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Resolution enum if found, None otherwise
        """
        for resolution in cls:
            if resolution.value == (width, height):
                return resolution
        return None

    @classmethod
    def closest(cls, width: int, height: int) -> "Resolution":
        """Find the closest standard resolution to given dimensions.

        Args:
            width: Target width in pixels
            height: Target height in pixels

        Returns:
            Closest Resolution enum value
        """
        target_pixels = width * height
        return min(cls, key=lambda r: abs(r.pixel_count - target_pixels))


@dataclass
class CameraConfig:
    """Configuration for camera capture settings.

    Attributes:
        resolution: Video resolution (default: VGA 640x480)
        fps: Frames per second (default: 30)
        camera_id: Camera device ID (default: 0)
    """

    resolution: Resolution = Resolution.VGA
    fps: int = 30
    camera_id: int = 0

    @property
    def width(self) -> int:
        """Get configured width in pixels."""
        return self.resolution.width

    @property
    def height(self) -> int:
        """Get configured height in pixels."""
        return self.resolution.height

    def __str__(self) -> str:
        return f"CameraConfig(camera={self.camera_id}, {self.resolution}, {self.fps}fps)"


# =============================================================================
# RealSense Stream Profile
# =============================================================================


@dataclass
class StreamProfile:
    """A supported stream profile (resolution + format + fps combination).

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        format: Pixel format string (e.g., 'RGB8', 'BGR8', 'Z16', 'Y8')
        stream_type: Stream type string (e.g., 'color', 'depth', 'infrared')
    """

    width: int
    height: int
    fps: int
    format: str
    stream_type: str

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        return (self.width, self.height)

    @property
    def standard_resolution(self) -> Optional[Resolution]:
        """Get matching standard Resolution enum, if any."""
        return Resolution.from_size(self.width, self.height)

    def __str__(self) -> str:
        return f"{self.stream_type}: {self.width}x{self.height}@{self.fps}fps ({self.format})"


@dataclass
class SensorOption:
    """A sensor option/setting with its range and current value.

    Attributes:
        name: Option name (e.g., 'exposure', 'gain', 'laser_power')
        value: Current value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Value increment step
        default: Default value
        description: Human-readable description
    """

    name: str
    value: float
    min_value: float
    max_value: float
    step: float
    default: float
    description: str

    def __str__(self) -> str:
        return f"{self.name}: {self.value} (range: {self.min_value}-{self.max_value}, default: {self.default})"


@dataclass
class RealSenseDeviceInfo:
    """Information about a connected RealSense device.

    Attributes:
        name: Device name (e.g., 'Intel RealSense D435')
        serial_number: Device serial number
        firmware_version: Firmware version string
        usb_type: USB connection type (e.g., '3.2')
        product_line: Product line (e.g., 'D400')
        sensors: List of sensor names on the device
        color_profiles: Supported color stream profiles
        depth_profiles: Supported depth stream profiles
        infrared_profiles: Supported infrared stream profiles
        sensor_options: Dict of sensor name -> list of options
    """

    name: str
    serial_number: str
    firmware_version: str
    usb_type: str
    product_line: str
    sensors: List[str] = field(default_factory=list)
    color_profiles: List[StreamProfile] = field(default_factory=list)
    depth_profiles: List[StreamProfile] = field(default_factory=list)
    infrared_profiles: List[StreamProfile] = field(default_factory=list)
    sensor_options: Dict[str, List[SensorOption]] = field(default_factory=dict)

    def get_color_resolutions(self) -> List[Tuple[int, int]]:
        """Get unique color stream resolutions."""
        return sorted(set((p.width, p.height) for p in self.color_profiles))

    def get_depth_resolutions(self) -> List[Tuple[int, int]]:
        """Get unique depth stream resolutions."""
        return sorted(set((p.width, p.height) for p in self.depth_profiles))

    def get_color_fps_options(
        self, width: int, height: int, format: Optional[str] = None
    ) -> List[int]:
        """Get available FPS options for a color resolution."""
        fps_set = set()
        for p in self.color_profiles:
            if p.width == width and p.height == height:
                if format is None or p.format == format:
                    fps_set.add(p.fps)
        return sorted(fps_set)

    def get_depth_fps_options(
        self, width: int, height: int, format: Optional[str] = None
    ) -> List[int]:
        """Get available FPS options for a depth resolution."""
        fps_set = set()
        for p in self.depth_profiles:
            if p.width == width and p.height == height:
                if format is None or p.format == format:
                    fps_set.add(p.fps)
        return sorted(fps_set)

    def get_color_formats(self, width: int, height: int) -> List[str]:
        """Get available pixel formats for a color resolution."""
        return sorted(
            set(p.format for p in self.color_profiles if p.width == width and p.height == height)
        )

    def get_depth_formats(self, width: int, height: int) -> List[str]:
        """Get available pixel formats for a depth resolution."""
        return sorted(
            set(p.format for p in self.depth_profiles if p.width == width and p.height == height)
        )

    def supports_color_profile(
        self, width: int, height: int, fps: int, format: Optional[str] = None
    ) -> bool:
        """Check if a specific color profile is supported."""
        for p in self.color_profiles:
            if p.width == width and p.height == height and p.fps == fps:
                if format is None or p.format == format:
                    return True
        return False

    def supports_depth_profile(
        self, width: int, height: int, fps: int, format: Optional[str] = None
    ) -> bool:
        """Check if a specific depth profile is supported."""
        for p in self.depth_profiles:
            if p.width == width and p.height == height and p.fps == fps:
                if format is None or p.format == format:
                    return True
        return False

    def __str__(self) -> str:
        return (
            f"RealSenseDevice({self.name}, SN: {self.serial_number}, "
            f"FW: {self.firmware_version}, USB: {self.usb_type})"
        )


# =============================================================================
# RealSense Device Discovery
# =============================================================================


class RealSenseDiscovery:
    """Discover and query RealSense device capabilities.

    Example:
        >>> from cyberwave.sensor.config import RealSenseDiscovery
        >>>
        >>> # List all devices
        >>> devices = RealSenseDiscovery.list_devices()
        >>> for dev in devices:
        ...     print(dev.name, dev.serial_number)
        >>>
        >>> # Get detailed info for first device
        >>> info = RealSenseDiscovery.get_device_info()
        >>> print(info.color_profiles)
        >>> print(info.sensor_options)
    """

    @staticmethod
    def _require_realsense():
        """Check if RealSense is available."""
        if not _has_realsense:
            raise ImportError(
                "RealSense discovery requires pyrealsense2. "
                "Install with: pip install pyrealsense2"
            )

    @staticmethod
    def _format_to_string(fmt) -> str:
        """Convert RealSense format enum to string."""
        if not _has_realsense:
            return str(fmt)
        format_map = {
            rs.format.z16: "Z16",
            rs.format.rgb8: "RGB8",
            rs.format.bgr8: "BGR8",
            rs.format.rgba8: "RGBA8",
            rs.format.bgra8: "BGRA8",
            rs.format.y8: "Y8",
            rs.format.y16: "Y16",
            rs.format.yuyv: "YUYV",
            rs.format.uyvy: "UYVY",
            rs.format.raw16: "RAW16",
            rs.format.raw10: "RAW10",
            rs.format.raw8: "RAW8",
            rs.format.disparity16: "DISPARITY16",
            rs.format.disparity32: "DISPARITY32",
            rs.format.xyz32f: "XYZ32F",
            rs.format.motion_raw: "MOTION_RAW",
            rs.format.motion_xyz32f: "MOTION_XYZ32F",
            rs.format.gpio_raw: "GPIO_RAW",
        }
        return format_map.get(fmt, str(fmt))

    @staticmethod
    def _stream_to_string(stream) -> str:
        """Convert RealSense stream enum to string."""
        if not _has_realsense:
            return str(stream)
        stream_map = {
            rs.stream.color: "color",
            rs.stream.depth: "depth",
            rs.stream.infrared: "infrared",
            rs.stream.fisheye: "fisheye",
            rs.stream.gyro: "gyro",
            rs.stream.accel: "accel",
            rs.stream.pose: "pose",
            rs.stream.confidence: "confidence",
        }
        return stream_map.get(stream, str(stream))

    @staticmethod
    def _option_to_string(option) -> str:
        """Convert RealSense option enum to string."""
        if not _has_realsense:
            return str(option)
        # Use the enum name directly
        return str(option).replace("option.", "")

    @classmethod
    def is_available(cls) -> bool:
        """Check if RealSense SDK is available."""
        return _has_realsense

    @classmethod
    def list_devices(cls) -> List[RealSenseDeviceInfo]:
        """List all connected RealSense devices with basic info.

        Returns:
            List of RealSenseDeviceInfo objects (without detailed profiles)
        """
        cls._require_realsense()

        devices = []
        ctx = rs.context()

        for dev in ctx.query_devices():
            try:
                info = RealSenseDeviceInfo(
                    name=dev.get_info(rs.camera_info.name),
                    serial_number=dev.get_info(rs.camera_info.serial_number),
                    firmware_version=dev.get_info(rs.camera_info.firmware_version),
                    usb_type=dev.get_info(rs.camera_info.usb_type_descriptor)
                    if dev.supports(rs.camera_info.usb_type_descriptor)
                    else "unknown",
                    product_line=dev.get_info(rs.camera_info.product_line)
                    if dev.supports(rs.camera_info.product_line)
                    else "unknown",
                    sensors=[sensor.get_info(rs.camera_info.name) for sensor in dev.query_sensors()],
                )
                devices.append(info)
            except Exception as e:
                logger.warning(f"Error querying device: {e}")

        return devices

    @classmethod
    def get_device_info(cls, serial_number: Optional[str] = None) -> Optional[RealSenseDeviceInfo]:
        """Get detailed information about a RealSense device.

        Args:
            serial_number: Device serial number. If None, uses first available device.

        Returns:
            RealSenseDeviceInfo with full profile and option details, or None if not found
        """
        cls._require_realsense()

        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            logger.warning("No RealSense devices found")
            return None

        # Find the requested device
        target_device = None
        for dev in devices:
            if serial_number is None:
                target_device = dev
                break
            if dev.get_info(rs.camera_info.serial_number) == serial_number:
                target_device = dev
                break

        if target_device is None:
            logger.warning(f"Device with serial {serial_number} not found")
            return None

        # Build device info
        info = RealSenseDeviceInfo(
            name=target_device.get_info(rs.camera_info.name),
            serial_number=target_device.get_info(rs.camera_info.serial_number),
            firmware_version=target_device.get_info(rs.camera_info.firmware_version),
            usb_type=target_device.get_info(rs.camera_info.usb_type_descriptor)
            if target_device.supports(rs.camera_info.usb_type_descriptor)
            else "unknown",
            product_line=target_device.get_info(rs.camera_info.product_line)
            if target_device.supports(rs.camera_info.product_line)
            else "unknown",
        )

        # Query each sensor
        for sensor in target_device.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            info.sensors.append(sensor_name)

            # Get stream profiles
            for profile in sensor.get_stream_profiles():
                if profile.is_video_stream_profile():
                    video_profile = profile.as_video_stream_profile()
                    stream_type = cls._stream_to_string(profile.stream_type())
                    fmt = cls._format_to_string(profile.format())

                    sp = StreamProfile(
                        width=video_profile.width(),
                        height=video_profile.height(),
                        fps=profile.fps(),
                        format=fmt,
                        stream_type=stream_type,
                    )

                    if stream_type == "color":
                        info.color_profiles.append(sp)
                    elif stream_type == "depth":
                        info.depth_profiles.append(sp)
                    elif stream_type == "infrared":
                        info.infrared_profiles.append(sp)

            # Get sensor options
            sensor_options = []
            for option in rs.option:
                try:
                    if sensor.supports(option):
                        option_range = sensor.get_option_range(option)
                        current_value = sensor.get_option(option)
                        description = sensor.get_option_description(option)

                        sensor_options.append(
                            SensorOption(
                                name=cls._option_to_string(option),
                                value=current_value,
                                min_value=option_range.min,
                                max_value=option_range.max,
                                step=option_range.step,
                                default=option_range.default,
                                description=description,
                            )
                        )
                except Exception:
                    # Some options may not be queryable
                    pass

            if sensor_options:
                info.sensor_options[sensor_name] = sensor_options

        return info

    @classmethod
    def get_device_count(cls) -> int:
        """Get number of connected RealSense devices."""
        cls._require_realsense()
        ctx = rs.context()
        return len(ctx.query_devices())


# =============================================================================
# RealSense Config with Validation
# =============================================================================


@dataclass
class RealSenseConfig:
    """Configuration for RealSense camera capture settings.

    Supports validation against actual device capabilities.

    Attributes:
        color_resolution: RGB stream resolution (default: VGA 640x480)
        depth_resolution: Depth stream resolution (default: VGA 640x480)
        color_fps: Color stream FPS (default: 30)
        depth_fps: Depth stream FPS (default: 30)
        color_format: Color pixel format (default: 'BGR8')
        depth_format: Depth pixel format (default: 'Z16')
        enable_depth: Whether to enable depth streaming (default: True)
        depth_publish_interval: Publish depth every N frames (default: 30)
        serial_number: Target device serial number (None = first device)

    Example:
        >>> from cyberwave.sensor.config import RealSenseConfig, Resolution
        >>>
        >>> # Create config
        >>> config = RealSenseConfig(
        ...     color_resolution=Resolution.HD,
        ...     depth_resolution=Resolution.VGA,
        ...     color_fps=30,
        ...     depth_fps=15
        ... )
        >>>
        >>> # Validate against device
        >>> is_valid, errors = config.validate()
        >>> if not is_valid:
        ...     print("Config errors:", errors)
        >>>
        >>> # Auto-create from device capabilities
        >>> config = RealSenseConfig.from_device()
    """

    color_resolution: Union[Resolution, Tuple[int, int]] = Resolution.VGA
    depth_resolution: Union[Resolution, Tuple[int, int]] = Resolution.VGA
    color_fps: int = 30
    depth_fps: int = 30
    color_format: str = "BGR8"
    depth_format: str = "Z16"
    enable_depth: bool = True
    depth_publish_interval: int = 30
    serial_number: Optional[str] = None

    @property
    def color_width(self) -> int:
        """Get color stream width in pixels."""
        if isinstance(self.color_resolution, Resolution):
            return self.color_resolution.width
        return self.color_resolution[0]

    @property
    def color_height(self) -> int:
        """Get color stream height in pixels."""
        if isinstance(self.color_resolution, Resolution):
            return self.color_resolution.height
        return self.color_resolution[1]

    @property
    def depth_width(self) -> int:
        """Get depth stream width in pixels."""
        if isinstance(self.depth_resolution, Resolution):
            return self.depth_resolution.width
        return self.depth_resolution[0]

    @property
    def depth_height(self) -> int:
        """Get depth stream height in pixels."""
        if isinstance(self.depth_resolution, Resolution):
            return self.depth_resolution.height
        return self.depth_resolution[1]

    def validate(self, device_info: Optional[RealSenseDeviceInfo] = None) -> Tuple[bool, List[str]]:
        """Validate configuration against device capabilities.

        Args:
            device_info: Device info to validate against. If None, queries the device.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Get device info if not provided
        if device_info is None:
            if not _has_realsense:
                return True, []  # Can't validate without RealSense SDK
            device_info = RealSenseDiscovery.get_device_info(self.serial_number)
            if device_info is None:
                errors.append("No RealSense device found for validation")
                return False, errors

        # Validate color stream
        if not device_info.supports_color_profile(
            self.color_width, self.color_height, self.color_fps, self.color_format
        ):
            available_fps = device_info.get_color_fps_options(
                self.color_width, self.color_height, self.color_format
            )
            available_formats = device_info.get_color_formats(self.color_width, self.color_height)
            available_resolutions = device_info.get_color_resolutions()

            errors.append(
                f"Color profile not supported: {self.color_width}x{self.color_height}@{self.color_fps}fps ({self.color_format}). "
                f"Available resolutions: {available_resolutions}, "
                f"FPS for this resolution: {available_fps}, "
                f"Formats: {available_formats}"
            )

        # Validate depth stream if enabled
        if self.enable_depth:
            if not device_info.supports_depth_profile(
                self.depth_width, self.depth_height, self.depth_fps, self.depth_format
            ):
                available_fps = device_info.get_depth_fps_options(
                    self.depth_width, self.depth_height, self.depth_format
                )
                available_formats = device_info.get_depth_formats(self.depth_width, self.depth_height)
                available_resolutions = device_info.get_depth_resolutions()

                errors.append(
                    f"Depth profile not supported: {self.depth_width}x{self.depth_height}@{self.depth_fps}fps ({self.depth_format}). "
                    f"Available resolutions: {available_resolutions}, "
                    f"FPS for this resolution: {available_fps}, "
                    f"Formats: {available_formats}"
                )

        return len(errors) == 0, errors

    @classmethod
    def from_device(
        cls,
        serial_number: Optional[str] = None,
        prefer_resolution: Resolution = Resolution.VGA,
        prefer_fps: int = 30,
        enable_depth: bool = True,
    ) -> "RealSenseConfig":
        """Create configuration from device capabilities.

        Automatically selects the best matching profile from available options.

        Args:
            serial_number: Target device serial number (None = first device)
            prefer_resolution: Preferred resolution (will find closest match)
            prefer_fps: Preferred FPS (will find closest match)
            enable_depth: Whether to enable depth streaming

        Returns:
            RealSenseConfig configured for the device

        Raises:
            ImportError: If pyrealsense2 is not available
            RuntimeError: If no device is found
        """
        if not _has_realsense:
            raise ImportError("RealSense SDK required for device discovery")

        device_info = RealSenseDiscovery.get_device_info(serial_number)
        if device_info is None:
            raise RuntimeError("No RealSense device found")

        # Find best color profile
        color_resolutions = device_info.get_color_resolutions()
        if not color_resolutions:
            raise RuntimeError("No color stream profiles found")

        # Find closest resolution
        target_pixels = prefer_resolution.pixel_count
        best_color_res = min(color_resolutions, key=lambda r: abs(r[0] * r[1] - target_pixels))

        # Find closest FPS for that resolution (prefer BGR8 format)
        color_fps_options = device_info.get_color_fps_options(
            best_color_res[0], best_color_res[1], "BGR8"
        )
        if not color_fps_options:
            # Fall back to any format
            color_fps_options = device_info.get_color_fps_options(
                best_color_res[0], best_color_res[1]
            )
        best_color_fps = min(color_fps_options, key=lambda f: abs(f - prefer_fps))

        # Determine color format
        color_formats = device_info.get_color_formats(best_color_res[0], best_color_res[1])
        color_format = "BGR8" if "BGR8" in color_formats else color_formats[0]

        # Find best depth profile if enabled
        depth_resolution = best_color_res
        depth_fps = best_color_fps
        depth_format = "Z16"

        if enable_depth:
            depth_resolutions = device_info.get_depth_resolutions()
            if depth_resolutions:
                best_depth_res = min(
                    depth_resolutions, key=lambda r: abs(r[0] * r[1] - target_pixels)
                )
                depth_resolution = best_depth_res

                depth_fps_options = device_info.get_depth_fps_options(
                    best_depth_res[0], best_depth_res[1], "Z16"
                )
                if depth_fps_options:
                    depth_fps = min(depth_fps_options, key=lambda f: abs(f - prefer_fps))

                depth_formats = device_info.get_depth_formats(best_depth_res[0], best_depth_res[1])
                depth_format = "Z16" if "Z16" in depth_formats else depth_formats[0]

        return cls(
            color_resolution=best_color_res,
            depth_resolution=depth_resolution,
            color_fps=best_color_fps,
            depth_fps=depth_fps,
            color_format=color_format,
            depth_format=depth_format,
            enable_depth=enable_depth,
            serial_number=serial_number,
        )

    def __str__(self) -> str:
        depth_str = (
            f", depth={self.depth_width}x{self.depth_height}@{self.depth_fps}fps ({self.depth_format})"
            if self.enable_depth
            else ""
        )
        return (
            f"RealSenseConfig(color={self.color_width}x{self.color_height}@{self.color_fps}fps "
            f"({self.color_format}){depth_str})"
        )


# Common configuration presets
PRESET_LOW_BANDWIDTH = CameraConfig(resolution=Resolution.QVGA, fps=15)
PRESET_STANDARD = CameraConfig(resolution=Resolution.VGA, fps=30)
PRESET_HD = CameraConfig(resolution=Resolution.HD, fps=30)
PRESET_FULL_HD = CameraConfig(resolution=Resolution.FULL_HD, fps=30)
