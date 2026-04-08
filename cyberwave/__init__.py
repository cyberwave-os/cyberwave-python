"""
Cyberwave SDK - Python client for the Cyberwave Digital Twin Platform

This SDK provides a comprehensive interface for interacting with the Cyberwave platform,
including REST APIs, MQTT messaging, and high-level abstractions for digital twins.

Quick Start:
    >>> from cyberwave import Cyberwave
    >>> cw = Cyberwave(api_key="your_key")
    >>> workspaces = cw.workspaces.list()

    Video Streaming (requires: pip install cyberwave[camera]):
    >>> cw = Cyberwave(api_key="your_api_key")
    >>> twin = cw.twin("cyberwave/generic-camera")
    >>> twin.start_streaming()  # blocking; use stream_video_background() in async code
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
    TwinCameraHandle,
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

# Resource managers depend on the auto-generated REST client (cyberwave.rest)
# which may not be present in edge-only installs.
try:
    from .resources import (
        WorkspaceManager,
        ProjectManager,
        EnvironmentManager,
        AssetManager,
        EdgeManager,
        TwinManager,
    )
except ImportError:
    WorkspaceManager = None  # type: ignore[assignment,misc]
    ProjectManager = None  # type: ignore[assignment,misc]
    EnvironmentManager = None  # type: ignore[assignment,misc]
    AssetManager = None  # type: ignore[assignment,misc]
    EdgeManager = None  # type: ignore[assignment,misc]
    TwinManager = None  # type: ignore[assignment,misc]

# Workflow management
try:
    from .workflows import (
        Workflow,
        WorkflowRun,
        WorkflowManager,
        WorkflowRunManager,
    )
except ImportError:
    Workflow = None  # type: ignore[assignment,misc]
    WorkflowRun = None  # type: ignore[assignment,misc]
    WorkflowManager = None  # type: ignore[assignment,misc]
    WorkflowRunManager = None  # type: ignore[assignment,misc]

# Worker API
from .workers import HookContext

# Manifest schema
from .manifest import ManifestSchema, detect_dispatch_mode, validate_manifest

# Model API
from .models import ModelManager, LoadedModel, Detection, BoundingBox, PredictionResult

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
        CameraStreamManager,
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
    CameraStreamManager = None  # type: ignore

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
    SOURCE_TYPE_EDGE_FOLLOWER,
    SOURCE_TYPE_EDGE_LEADER,
    SOURCE_TYPE_TELE,
    SOURCE_TYPE_EDIT,
    SOURCE_TYPE_SIM,
    SOURCE_TYPE_SIM_TELE,
    SOURCE_TYPES,
)

# Worker API
from .workers import HookContext, HookRegistration, HookRegistry, SynchronizedGroup

# Model output types
from .models import BoundingBox, Detection, PredictionResult

# Scene Composition
from .scene import Scene

# Version information
from ._version import get_version

__version__ = get_version()

# Define public API
__all__ = [
    # Core client
    "Cyberwave",
    # Scene
    "Scene",
    # Configuration
    "CyberwaveConfig",
    "get_config",
    "set_config",
    # High-level abstractions
    "Twin",
    "JointController",
    "TwinControllerHandle",
    "TwinCameraHandle",
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
    # Workflow management
    "Workflow",
    "WorkflowRun",
    "WorkflowManager",
    "WorkflowRunManager",
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
    "CameraStreamManager",
    # Edge controller
    "EdgeController",
    # Worker API
    "HookContext",
    # Model API
    "ModelManager",
    "LoadedModel",
    "Detection",
    "BoundingBox",
    "PredictionResult",
    # Constants
    "SOURCE_TYPE_EDGE",
    "SOURCE_TYPE_EDGE_FOLLOWER",
    "SOURCE_TYPE_EDGE_LEADER",
    "SOURCE_TYPE_TELE",
    "SOURCE_TYPE_EDIT",
    "SOURCE_TYPE_SIM",
    "SOURCE_TYPE_SIM_TELE",
    "SOURCE_TYPES",
    # Utils
    "TimeReference",
    # Device fingerprinting
    "generate_fingerprint",
    "get_device_info",
    "format_device_info_table",
    # Worker API
    "HookContext",
    "HookRegistration",
    "HookRegistry",
    "SynchronizedGroup",
    # Model output types
    "BoundingBox",
    "Detection",
    "PredictionResult",
    # Manifest
    "ManifestSchema",
    "detect_dispatch_mode",
    "validate_manifest",
    # Version
    "__version__",
]
