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

Heavy dependencies (OpenCV, aiortc, RealSense, …) are loaded lazily via :pep:`562`
``__getattr__`` so lightweight submodules can be imported in tests and minimal
environments without pulling the full WebRTC stack.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__all__ = [
    # NOTE: keep in sync with _LAZY_ATTRS and optional Mujoco block below.
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

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None

_MUJOCO_EXPORT_NAMES: frozenset[str] = frozenset(
    {
        "ThreadSafeFrameBuffer",
        "SimVideoTrack",
        "SimCameraStreamer",
        "MujocoMultiCameraStreamer",
        "CyberwaveSimStreaming",
    }
)

if _HAS_MUJOCO:
    __all__ += [
        "ThreadSafeFrameBuffer",
        "SimVideoTrack",
        "SimCameraStreamer",
        "MujocoMultiCameraStreamer",
        "CyberwaveSimStreaming",
    ]

# (submodule path relative to this package, attribute name)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BaseVideoTrack": (".base_video", "BaseVideoTrack"),
    "BaseVideoStreamer": (".base_video", "BaseVideoStreamer"),
    "DEFAULT_TURN_SERVERS": (".base_video", "DEFAULT_TURN_SERVERS"),
    "CONNECTION_LOSS_CONFIRMATION_CHECKS": (".base_video", "CONNECTION_LOSS_CONFIRMATION_CHECKS"),
    "SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS": (".base_video", "SDK_EDGE_HEALTH_STALE_TIMEOUT_SECONDS"),
    "SDK_EDGE_HEALTH_INTERVAL_SECONDS": (".base_video", "SDK_EDGE_HEALTH_INTERVAL_SECONDS"),
    "CameraType": (".config", "CameraType"),
    "Resolution": (".config", "Resolution"),
    "CameraConfig": (".config", "CameraConfig"),
    "EdgeCameraConfig": (".config", "EdgeCameraConfig"),
    "SimStreamingConfig": (".config", "SimStreamingConfig"),
    "cameras_from_schema": (".config", "cameras_from_schema"),
    "RealSenseConfig": (".config", "RealSenseConfig"),
    "StreamProfile": (".config", "StreamProfile"),
    "SensorOption": (".config", "SensorOption"),
    "RealSenseDeviceInfo": (".config", "RealSenseDeviceInfo"),
    "RealSenseDiscovery": (".config", "RealSenseDiscovery"),
    "PRESET_LOW_BANDWIDTH": (".config", "PRESET_LOW_BANDWIDTH"),
    "PRESET_STANDARD": (".config", "PRESET_STANDARD"),
    "PRESET_HD": (".config", "PRESET_HD"),
    "PRESET_FULL_HD": (".config", "PRESET_FULL_HD"),
    "CV2VideoTrack": (".camera_cv2", "CV2VideoTrack"),
    "CV2CameraStreamer": (".camera_cv2", "CV2CameraStreamer"),
    "RealSenseVideoTrack": (".camera_rs", "RealSenseVideoTrack"),
    "RealSenseStreamer": (".camera_rs", "RealSenseStreamer"),
    "VirtualVideoTrack": (".camera_virtual", "VirtualVideoTrack"),
    "VirtualCameraStreamer": (".camera_virtual", "VirtualCameraStreamer"),
    "CameraStreamManager": (".manager", "CameraStreamManager"),
    "run_streamer_in_background": (".manager", "run_streamer_in_background"),
    "BaseAudioTrack": (".microphone", "BaseAudioTrack"),
    "BaseAudioStreamer": (".microphone", "BaseAudioStreamer"),
    "MicrophoneAudioTrack": (".microphone", "MicrophoneAudioTrack"),
    "MicrophoneAudioStreamer": (".microphone", "MicrophoneAudioStreamer"),
    "AUDIO_PTIME": (".microphone", "AUDIO_PTIME"),
    "DEFAULT_SAMPLE_RATE": (".microphone", "DEFAULT_SAMPLE_RATE"),
    "MultimediaStreamer": (".av_streamer", "MultimediaStreamer"),
}

_MUJOCO_ATTRS: dict[str, tuple[str, str]] = {
    "ThreadSafeFrameBuffer": (".camera_sim", "ThreadSafeFrameBuffer"),
    "SimVideoTrack": (".camera_sim", "SimVideoTrack"),
    "SimCameraStreamer": (".camera_sim", "SimCameraStreamer"),
    "MujocoMultiCameraStreamer": (".camera_sim", "MujocoMultiCameraStreamer"),
    "CyberwaveSimStreaming": (".camera_sim", "CyberwaveSimStreaming"),
}


def __getattr__(name: str) -> Any:
    if name == "CallbackAudioTrack":
        return __getattr__("MicrophoneAudioTrack")
    if name == "CallbackAudioStreamer":
        return __getattr__("MicrophoneAudioStreamer")

    if name in _MUJOCO_EXPORT_NAMES:
        if not _HAS_MUJOCO:
            # Match previous eager-import behavior when mujoco/camera_sim was unavailable.
            return None
        submod, attr = _MUJOCO_ATTRS[name]
        mod = importlib.import_module(submod, package=__name__)
        return getattr(mod, attr)

    if name in _LAZY_ATTRS:
        submod, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(submod, package=__name__)
        return getattr(mod, attr)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
