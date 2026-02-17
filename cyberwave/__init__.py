"""
Cyberwave SDK - Python client for the Cyberwave Digital Twin Platform

This SDK provides a comprehensive interface for interacting with the Cyberwave platform,
including REST APIs, MQTT messaging, and high-level abstractions for digital twins.

Quick Start:
    >>> from cyberwave import Cyberwave
    >>> cw = Cyberwave(api_key="your_key")
    >>> workspaces = cw.workspaces.list()

    Video Streaming (requires: pip install cyberwave[camera]):
    >>> cw = Cyberwave(token="your_token")
    >>> twin = cw.twin("cyberwave/generic-camera")
    >>> twin.start_streaming()
"""

# Core client
from .client import Cyberwave

# Configuration
from .config import CyberwaveConfig, get_config, set_config

# High-level abstractions
from .twin import (
    Twin,
    JointController,
    TwinControllerHandle,
    CameraTwin,
    DepthCameraTwin,
    FlyingTwin,
    GripperTwin,
    FlyingCameraTwin,
    GripperCameraTwin,
    create_twin,
)

# Alerts
from .alerts import Alert, TwinAlertManager

# Motion and navigation
from .motion import (
    TwinMotionHandle,
    ScopedMotionHandle,
    TwinNavigationHandle,
)
from .navigation import NavigationPlan

# Keyboard teleop
from .keyboard import KeyboardBindings, KeyboardTeleop

# Exceptions
from .exceptions import (
    CyberwaveError,
    CyberwaveAPIError,
    CyberwaveConnectionError,
    CyberwaveTimeoutError,
    CyberwaveValidationError,
)

# Compact API - convenience functions
from .compact import (
    configure,
    twin,
    get_client,
)

# Resource managers (optional, available through client instance)
from .resources import (
    WorkspaceManager,
    ProjectManager,
    EnvironmentManager,
    AssetManager,
    EdgeManager,
    TwinManager,
)

# MQTT client (optional, for direct MQTT access)
from .mqtt import CyberwaveMQTTClient

# Camera streaming (optional, requires additional dependencies)
try:
    from .sensor import (
        CV2VideoTrack,
        CV2CameraStreamer,
        VirtualVideoTrack,
        VirtualCameraStreamer,
        RealSenseVideoTrack,
        RealSenseStreamer,
        BaseVideoTrack,
        BaseVideoStreamer,
    )

    # Legacy alias for backwards compatibility
    CameraStreamer = CV2CameraStreamer

    _has_camera = True
except ImportError:
    _has_camera = False
    CameraStreamer = None  # type: ignore
    CV2VideoTrack = None  # type: ignore
    CV2CameraStreamer = None  # type: ignore
    VirtualVideoTrack = None  # type: ignore
    VirtualCameraStreamer = None  # type: ignore
    CallbackVideoTrack = None  # type: ignore
    CallbackCameraStreamer = None  # type: ignore
    RealSenseVideoTrack = None  # type: ignore
    RealSenseStreamer = None  # type: ignore
    BaseVideoTrack = None  # type: ignore
    BaseVideoStreamer = None  # type: ignore

# Edge controller
from .controller import EdgeController

# Utils
from .utils import TimeReference

# Device fingerprinting (for edge devices)
from .fingerprint import (
    generate_fingerprint,
    get_device_info,
    format_device_info_table,
)

# Constants
from .constants import (
    SOURCE_TYPE_EDGE,
    SOURCE_TYPE_TELE,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
    SOURCE_TYPES,
)

# Version information
__version__ = "0.3.9"

# Define public API
__all__ = [
    # Core client
    "Cyberwave",
    # Configuration
    "CyberwaveConfig",
    "get_config",
    "set_config",
    # High-level abstractions
    "Twin",
    "JointController",
    "TwinControllerHandle",
    "CameraTwin",
    "DepthCameraTwin",
    "FlyingTwin",
    "GripperTwin",
    "FlyingCameraTwin",
    "GripperCameraTwin",
    "create_twin",
    # Alerts
    "Alert",
    "TwinAlertManager",
    # Motion and navigation
    "TwinMotionHandle",
    "ScopedMotionHandle",
    "TwinNavigationHandle",
    "NavigationPlan",
    # Keyboard teleop
    "KeyboardBindings",
    "KeyboardTeleop",
    # Exceptions
    "CyberwaveError",
    "CyberwaveAPIError",
    "CyberwaveConnectionError",
    "CyberwaveTimeoutError",
    "CyberwaveValidationError",
    # Compact API
    "configure",
    "twin",
    "get_client",
    # Resource managers
    "WorkspaceManager",
    "ProjectManager",
    "EnvironmentManager",
    "AssetManager",
    "EdgeManager",
    "TwinManager",
    # MQTT client
    "CyberwaveMQTTClient",
    # Camera streaming (optional)
    "CameraStreamer",  # Legacy alias for CV2CameraStreamer
    "CV2VideoTrack",
    "CV2CameraStreamer",
    "VirtualVideoTrack",
    "VirtualCameraStreamer",
    "RealSenseVideoTrack",
    "RealSenseStreamer",
    "BaseVideoTrack",
    "BaseVideoStreamer",
    # Edge controller
    "EdgeController",
    # Constants
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPES",
    # Utils
    "TimeReference",
    # Device fingerprinting
    "generate_fingerprint",
    "get_device_info",
    "format_device_info_table",
    # Version
    "__version__",
]
