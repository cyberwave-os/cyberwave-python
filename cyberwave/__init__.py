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

from __future__ import annotations

import importlib
from typing import Any

# ---- Eagerly loaded (lightweight, no transitive heavy deps) -----------------

import yaml as _yaml  # noqa: F401  — imported to anchor sys.modules["yaml"]
# before test stubs (e.g. test_camera_virtual_streamer.py) can replace it
# with a SimpleNamespace via sys.modules.setdefault().  Only costs ~15ms.

from .config import CyberwaveConfig, get_config, set_config
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
from .exceptions import (
    CyberwaveError,
    CyberwaveAPIError,
    CyberwaveConnectionError,
    CyberwaveInsufficientCreditsError,
    CyberwaveTimeoutError,
    CyberwaveValidationError,
)
from ._version import get_version

__version__ = get_version()

# ---- Eagerly loaded compact API & image module ------------------------------
# ``twin`` and ``image`` collide with submodule filenames (twin.py, image.py).
# Python's import machinery sets package attributes when submodules are loaded,
# overwriting any lazy wrapper we might install.  Eagerly importing these
# ensures the compact-API *function* (not the module) is the canonical binding.
# The import chain (compact -> client -> data) is lightweight (~130ms) and
# does NOT trigger the heavy REST layer.
from .compact import configure, twin, get_client  # noqa: E402
from . import image  # noqa: E402

# ---- Lazily loaded (deferred until first access) ----------------------------
#
# The auto-generated REST client (cyberwave.rest) contains a ~98k-line
# default_api.py that takes >1 second to parse on fast hardware and 5-10
# seconds on edge devices.  Deferring it (and everything that depends on
# it) behind __getattr__ turns ``import cyberwave`` from a multi-second
# penalty into a ~50ms baseline.  The full cost is only paid when a
# caller first accesses a symbol that needs the REST layer (e.g.
# ``Cyberwave``, ``WorkspaceManager``, etc.).

# Maps public symbol names to (module_path, attribute_name).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core client
    "Cyberwave": (".client", "Cyberwave"),
    # High-level abstractions
    "Twin": (".twin", "Twin"),
    "JointTwin": (".twin", "JointTwin"),
    "TwinCameraHandle": (".twin", "TwinCameraHandle"),
    "CameraTwin": (".twin", "CameraTwin"),
    "DepthCameraTwin": (".twin", "DepthCameraTwin"),
    "FlyingTwin": (".twin", "FlyingTwin"),
    "GripperTwin": (".twin", "GripperTwin"),
    "FlyingCameraTwin": (".twin", "FlyingCameraTwin"),
    "GripperCameraTwin": (".twin", "GripperCameraTwin"),
    "create_twin": (".twin", "create_twin"),
    # Alerts
    "Alert": (".alerts", "Alert"),
    "TwinAlertManager": (".alerts", "TwinAlertManager"),
    # Agent/action API helpers
    "ActionsClient": (".actions", "ActionsClient"),
    "AgentManager": (".agents", "AgentManager"),
    "ControlAgentClient": (".agents", "ControlAgentClient"),
    "EmbodimentAgentClient": (".agents", "EmbodimentAgentClient"),
    "EnvironmentAgentClient": (".agents", "EnvironmentAgentClient"),
    "WorkflowAgentClient": (".agents", "WorkflowAgentClient"),
    # Motion and navigation
    "TwinMotionHandle": (".motion", "TwinMotionHandle"),
    "ScopedMotionHandle": (".motion", "ScopedMotionHandle"),
    "TwinNavigationHandle": (".motion", "TwinNavigationHandle"),
    "NavigationPlan": (".navigation", "NavigationPlan"),
    "LOCOMOTION_VELOCITY_COMMAND_CONTRACT": (
        ".locomotion_contracts",
        "LOCOMOTION_VELOCITY_COMMAND_CONTRACT",
    ),
    "AERIAL_VELOCITY_COMMAND_CONTRACT": (
        ".locomotion_contracts",
        "AERIAL_VELOCITY_COMMAND_CONTRACT",
    ),
    "LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS": (
        ".locomotion_contracts",
        "LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS",
    ),
    "AERIAL_VELOCITY_COMMAND_REQUIRED_FIELDS": (
        ".locomotion_contracts",
        "AERIAL_VELOCITY_COMMAND_REQUIRED_FIELDS",
    ),
    "LocomotionVelocityCommand": (
        ".locomotion_contracts",
        "LocomotionVelocityCommand",
    ),
    "BodyVelocityCommand": (".locomotion_contracts", "BodyVelocityCommand"),
    "LocomotionVelocityCommandError": (
        ".locomotion_contracts",
        "LocomotionVelocityCommandError",
    ),
    "build_locomotion_velocity_command": (
        ".locomotion_contracts",
        "build_locomotion_velocity_command",
    ),
    "normalize_locomotion_velocity_command": (
        ".locomotion_contracts",
        "normalize_locomotion_velocity_command",
    ),
    "normalize_body_velocity_command": (
        ".locomotion_contracts",
        "normalize_body_velocity_command",
    ),
    "stop_locomotion_velocity_command": (
        ".locomotion_contracts",
        "stop_locomotion_velocity_command",
    ),
    "stop_aerial_velocity_command": (
        ".locomotion_contracts",
        "stop_aerial_velocity_command",
    ),
    "hold_seconds_for_velocity_command": (
        ".locomotion_contracts",
        "hold_seconds_for_velocity_command",
    ),
    # Keyboard teleop
    "KeyboardBindings": (".keyboard", "KeyboardBindings"),
    "KeyboardTeleop": (".keyboard", "KeyboardTeleop"),
    # Compact API (twin, configure, get_client are eagerly imported above
    # to avoid submodule name collision with cyberwave/twin/ package)
    # Resource managers (these pull in the heavy REST layer)
    "WorkspaceManager": (".resources", "WorkspaceManager"),
    "ProjectManager": (".resources", "ProjectManager"),
    "EnvironmentManager": (".resources", "EnvironmentManager"),
    "AssetManager": (".resources", "AssetManager"),
    "AssetControllerSetupRecommendation": (
        ".resources",
        "AssetControllerSetupRecommendation",
    ),
    "AssetControllerSetupRuntimeOption": (
        ".resources",
        "AssetControllerSetupRuntimeOption",
    ),
    "AssetControllerSetupRuntimePolicy": (
        ".resources",
        "AssetControllerSetupRuntimePolicy",
    ),
    "AssetControllerSetupView": (".resources", "AssetControllerSetupView"),
    "ControlRuntimeTargetPayload": (".resources", "ControlRuntimeTargetPayload"),
    "PolicyRefPayload": (".resources", "PolicyRefPayload"),
    "EdgeManager": (".resources", "EdgeManager"),
    "TwinManager": (".resources", "TwinManager"),
    # Workflow management (also pulls in REST layer)
    "Workflow": (".workflows", "Workflow"),
    "WorkflowRun": (".workflows", "WorkflowRun"),
    "WorkflowManager": (".workflows", "WorkflowManager"),
    "WorkflowRunManager": (".workflows", "WorkflowRunManager"),
    # Worker API
    "HookContext": (".workers", "HookContext"),
    "HookRegistration": (".workers", "HookRegistration"),
    "HookRegistry": (".workers", "HookRegistry"),
    "ScheduleRegistration": (".workers", "ScheduleRegistration"),
    "SynchronizedGroup": (".workers", "SynchronizedGroup"),
    # Manifest
    "ManifestSchema": (".manifest", "ManifestSchema"),
    "detect_dispatch_mode": (".manifest", "detect_dispatch_mode"),
    "validate_manifest": (".manifest", "validate_manifest"),
    # Model API (edge runtime)
    "ModelManager": (".models", "ModelManager"),
    "LoadedModel": (".models", "LoadedModel"),
    "Detection": (".models", "Detection"),
    "BoundingBox": (".models", "BoundingBox"),
    "PredictionResult": (".models", "PredictionResult"),
    # Playground API (cloud inference)
    "StructuredAction": (".models.playground", "StructuredAction"),
    "STRUCTURED_ACTIONS": (".models.playground", "STRUCTURED_ACTIONS"),
    # Image helpers (image module is eagerly imported above to avoid
    # submodule collision with cyberwave/image.py)
    "decode_image_base64": (".image", "decode_image_base64"),
    "encode_image_base64": (".image", "encode_image_base64"),
    "read_annotated_metadata": (".image", "read_annotated_metadata"),
    "save_annotated_image": (".image", "save_annotated_image"),
    # MQTT client
    "CyberwaveMQTTClient": (".mqtt", "CyberwaveMQTTClient"),
    # Edge controller
    "EdgeController": (".controller", "EdgeController"),
    # Utils
    "TimeReference": (".utils", "TimeReference"),
    # Device fingerprinting
    "generate_fingerprint": (".fingerprint", "generate_fingerprint"),
    "get_device_info": (".fingerprint", "get_device_info"),
    "format_device_info_table": (".fingerprint", "format_device_info_table"),
    # Scene
    "Scene": (".scene", "Scene"),
    # Centered placement helpers
    "CenteredPlacement": (".placement", "CenteredPlacement"),
    "GENERIC_CUBE_BOUNDS": (".placement", "GENERIC_CUBE_BOUNDS"),
    "compute_centered_placement": (".placement", "compute_centered_placement"),
    "compute_center_from_origin": (".placement", "compute_center_from_origin"),
    # RL task scene-entity + task-spec helpers
    "RLTaskClient": (".rl_tasks", "RLTaskClient"),
    "TaskSpecExport": (".rl_tasks", "TaskSpecExport"),
}

# Optional camera streaming symbols — only available with extra deps.
_OPTIONAL_CAMERA_IMPORTS: dict[str, tuple[str, str]] = {
    "CV2VideoTrack": (".sensor", "CV2VideoTrack"),
    "CV2CameraStreamer": (".sensor", "CV2CameraStreamer"),
    "VirtualVideoTrack": (".sensor", "VirtualVideoTrack"),
    "VirtualCameraStreamer": (".sensor", "VirtualCameraStreamer"),
    "RealSenseVideoTrack": (".sensor", "RealSenseVideoTrack"),
    "RealSenseStreamer": (".sensor", "RealSenseStreamer"),
    "BaseVideoTrack": (".sensor", "BaseVideoTrack"),
    "BaseVideoStreamer": (".sensor", "BaseVideoStreamer"),
    "CameraStreamManager": (".sensor", "CameraStreamManager"),
    "CameraStreamer": (".sensor", "CV2CameraStreamer"),
}


def __getattr__(name: str) -> Any:
    # Standard lazy imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, package=__name__)
        if attr_name is None:
            value = mod
        else:
            value = getattr(mod, attr_name)
        globals()[name] = value
        return value

    # Optional camera imports
    if name in _OPTIONAL_CAMERA_IMPORTS:
        module_path, attr_name = _OPTIONAL_CAMERA_IMPORTS[name]
        try:
            mod = importlib.import_module(module_path, package=__name__)
            value = getattr(mod, attr_name)
        except ImportError:
            value = None
        globals()[name] = value
        return value

    # Backward compat: _has_camera flag
    if name == "_has_camera":
        try:
            importlib.import_module(".sensor", package=__name__)
            globals()["_has_camera"] = True
            return True
        except ImportError:
            globals()["_has_camera"] = False
            return False

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define public API
__all__ = [
    # Core client
    "Cyberwave",
    # Scene
    "Scene",
    # Centered placement helpers
    "CenteredPlacement",
    "GENERIC_CUBE_BOUNDS",
    "compute_centered_placement",
    "compute_center_from_origin",
    # RL task helpers
    "RLTaskClient",
    "TaskSpecExport",
    # Configuration
    "CyberwaveConfig",
    "get_config",
    "set_config",
    # High-level abstractions
    "Twin",
    "JointTwin",
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
    # Agent/action API helpers
    "ActionsClient",
    "AgentManager",
    "ControlAgentClient",
    "EmbodimentAgentClient",
    "EnvironmentAgentClient",
    "WorkflowAgentClient",
    # Motion and navigation
    "TwinMotionHandle",
    "ScopedMotionHandle",
    "TwinNavigationHandle",
    "NavigationPlan",
    "LOCOMOTION_VELOCITY_COMMAND_CONTRACT",
    "AERIAL_VELOCITY_COMMAND_CONTRACT",
    "LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS",
    "AERIAL_VELOCITY_COMMAND_REQUIRED_FIELDS",
    "LocomotionVelocityCommand",
    "BodyVelocityCommand",
    "LocomotionVelocityCommandError",
    "build_locomotion_velocity_command",
    "normalize_locomotion_velocity_command",
    "normalize_body_velocity_command",
    "stop_locomotion_velocity_command",
    "stop_aerial_velocity_command",
    "hold_seconds_for_velocity_command",
    # Keyboard teleop
    "KeyboardBindings",
    "KeyboardTeleop",
    # Exceptions
    "CyberwaveError",
    "CyberwaveAPIError",
    "CyberwaveConnectionError",
    "CyberwaveInsufficientCreditsError",
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
    "AssetControllerSetupRecommendation",
    "AssetControllerSetupRuntimeOption",
    "AssetControllerSetupRuntimePolicy",
    "AssetControllerSetupView",
    "ControlRuntimeTargetPayload",
    "PolicyRefPayload",
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
    "CameraStreamer",
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
    # Model API (edge runtime)
    "ModelManager",
    "LoadedModel",
    "Detection",
    "BoundingBox",
    "PredictionResult",
    # Playground API (cloud inference)
    "StructuredAction",
    "STRUCTURED_ACTIONS",
    # Image helpers
    "image",
    "encode_image_base64",
    "decode_image_base64",
    "save_annotated_image",
    "read_annotated_metadata",
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
    "ScheduleRegistration",
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
