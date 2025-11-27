"""
Cyberwave SDK - Python client for the Cyberwave Digital Twin Platform

This SDK provides a comprehensive interface for interacting with the Cyberwave platform,
including REST APIs, MQTT messaging, and high-level abstractions for digital twins.

Quick Start:
    >>> from cyberwave import Cyberwave
    >>> client = Cyberwave(base_url="http://localhost:8000", api_key="your_key")
    >>> workspaces = client.workspaces.list()

    Or use the compact API:
    >>> import cyberwave as cw
    >>> cw.configure(api_key="your_key", base_url="http://localhost:8000")
    >>> robot = cw.twin("the-robot-studio/so101")
    >>> robot.move(x=1, y=0, z=0.5)

    Video Streaming (requires: pip install cyberwave[camera]):
    >>> client = Cyberwave(token="your_token")
    >>> streamer = client.video_stream(twin_uuid="your_twin_uuid")
    >>> await streamer.start()
"""

# Core client
from .client import Cyberwave

# Configuration
from .config import CyberwaveConfig, get_config, set_config

# High-level abstractions
from .twin import Twin, JointController

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
    TwinManager,
)

# MQTT client (optional, for direct MQTT access)
from .mqtt import CyberwaveMQTTClient

# Camera streaming (optional, requires additional dependencies)
try:
    from .camera import CameraStreamer, CV2VideoStreamTrack

    _has_camera = True
except ImportError:
    _has_camera = False
    CameraStreamer = None  # type: ignore
    CV2VideoStreamTrack = None  # type: ignore

# Edge controller
from .controller import EdgeController

# Version information
__version__ = "0.2.4"

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
    # Exceptions
    "CyberwaveError",
    "CyberwaveAPIError",
    "CyberwaveConnectionError",
    "CyberwaveTimeoutError",
    "CyberwaveValidationError",
    # Compact API
    "configure",
    "twin",
    "simulation",
    "get_client",
    # Resource managers
    "WorkspaceManager",
    "ProjectManager",
    "EnvironmentManager",
    "AssetManager",
    "TwinManager",
    # MQTT client
    "CyberwaveMQTTClient",
    # Camera streaming (optional)
    "CameraStreamer",
    "CV2VideoStreamTrack",
    # Edge controller
    "EdgeController",
    # Utils
    "TimeReference",
    # Version
    "__version__",
]
