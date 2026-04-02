"""Sensor streaming functionality for Cyberwave SDK.

This module provides abstract base classes and implementations for various sensor types
including video (CV2, RealSense, virtual/callback, MuJoCo simulation) and audio
(microphone) streaming over WebRTC.

Sensor capabilities are defined in the twin's capabilities dictionary:
    {
        "sensors": [
            {"id": "uuid", "type": "rgb", "offset": {...}},
            {"id": "uuid", "type": "depth", "offset": {...}}
        ]
    }
"""

# Base video classes and shared constants
from .base_video import (  # noqa: F401
    BaseVideoTrack,
    BaseVideoStreamer,
    DEFAULT_TURN_SERVERS,
    CONNECTION_LOSS_CONFIRMATION_CHECKS,
    SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS,
    SDK_EDGE_HEALTH_INTERVAL_SECONDS,
)

# Configuration classes
from .config import (  # noqa: F401
    CameraType,
    Resolution,
    CameraConfig,
    EdgeCameraConfig,
    SimStreamingConfig,
    cameras_from_schema,
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

# Concrete video implementations
from .camera_cv2 import CV2VideoTrack, CV2CameraStreamer  # noqa: F401
from .camera_rs import RealSenseVideoTrack, RealSenseStreamer  # noqa: F401
from .camera_virtual import VirtualVideoTrack, VirtualCameraStreamer  # noqa: F401

# Simulation (MuJoCo) imports are optional — mujoco is not installed on edge devices
try:
    from .camera_sim import (  # noqa: F401
        ThreadSafeFrameBuffer,
        SimVideoTrack,
        SimCameraStreamer,
        MujocoMultiCameraStreamer,
        CyberwaveSimStreaming,
    )

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
    ThreadSafeFrameBuffer = None  # type: ignore[misc, assignment]
    SimVideoTrack = None  # type: ignore[misc, assignment]
    SimCameraStreamer = None  # type: ignore[misc, assignment]
    MujocoMultiCameraStreamer = None  # type: ignore[misc, assignment]
    CyberwaveSimStreaming = None  # type: ignore[misc, assignment]

# Camera stream manager
from .manager import CameraStreamManager, run_streamer_in_background  # noqa: F401

# Audio classes
from .microphone import (  # noqa: F401
    BaseAudioTrack,
    BaseAudioStreamer,
    MicrophoneAudioTrack,
    MicrophoneAudioStreamer,
    AUDIO_PTIME,
    DEFAULT_SAMPLE_RATE,
)

# Multimedia (combined video + audio) streamer
from .av_streamer import MultimediaStreamer  # noqa: F401

# Backward-compatibility aliases
CallbackAudioTrack = MicrophoneAudioTrack
CallbackAudioStreamer = MicrophoneAudioStreamer

__all__ = [
    # Base classes
    "BaseVideoTrack",
    "BaseVideoStreamer",
    "BaseAudioTrack",
    "BaseAudioStreamer",
    "MicrophoneAudioTrack",
    "MicrophoneAudioStreamer",
    "CallbackAudioTrack",
    "CallbackAudioStreamer",
    # Configuration
    "CameraType",
    "Resolution",
    "CameraConfig",
    "EdgeCameraConfig",
    "SimStreamingConfig",
    "cameras_from_schema",
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
    # Virtual camera implementations
    "VirtualCameraStreamer",
    "VirtualVideoTrack",
    # RealSense implementations
    "RealSenseVideoTrack",
    "RealSenseStreamer",
    # Manager
    "CameraStreamManager",
    "run_streamer_in_background",
    # Multimedia
    "MultimediaStreamer",
    # Constants
    "DEFAULT_TURN_SERVERS",
    "CONNECTION_LOSS_CONFIRMATION_CHECKS",
    "SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS",
    "SDK_EDGE_HEALTH_INTERVAL_SECONDS",
    "AUDIO_PTIME",
    "DEFAULT_SAMPLE_RATE",
]

# Simulation (MuJoCo) exports are only available when mujoco is installed
if _HAS_MUJOCO:
    __all__ += [
        "ThreadSafeFrameBuffer",
        "SimVideoTrack",
        "SimCameraStreamer",
        "MujocoMultiCameraStreamer",
        "CyberwaveSimStreaming",
    ]
