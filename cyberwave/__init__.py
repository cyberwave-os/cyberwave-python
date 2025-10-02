"""
CyberWave - A Python project for robot control and automation
"""

from .robot import Robot
from .digital_twin import (
    AbstractAsset,
    StaticAsset,
    RobotAsset,
    PhysicalDevice,
    DigitalTwin,
)
# Skeleton data structures for teleoperation and pose tracking
from .skeleton import (
    Joint3D,
    HandSkeleton,
    BodySkeleton, 
    RobotPose,
    BasicSkeletonMapper,
)
from .trainer import VideoTrainer, perform_welding
from .client import Client, CyberWaveError, APIError, AuthenticationError
from .sdk import Cyberwave, Mission
from .runtime import CyberwaveTask
from .geometry import Mesh
from .assets_api import AssetsAPI
from .constants import (
    DEFAULT_BACKEND_URL,
    BACKEND_URL_ENV_VAR,
    USERNAME_ENV_VAR,
    PASSWORD_ENV_VAR,
)

# geometry primitives (Mesh already imported above)

# Import centralized schema system components
from .centralized_schema import (
    convert_sdk_to_centralized,
    generate_centralized_level_yaml,
    validate_centralized_level,
    CYBERWAVE_API_VERSION,
    CYBERWAVE_LEVEL_API_VERSION,
    CentralizedSchemaError,
)

# Import compact API
from .compact_api import twin, configure, simulation, AuthTrigger, pose, alert, dispatch
from .twins import TwinsAPI  # expose for type hints and access to .get handle
from .mission import agent, CompactAgent, MissionPlanResult, MissionTaskSummary

# Import environment constants
from .constants import CyberWaveEnvironment, ENVIRONMENT_URLS

# Import utilities for advanced use cases
from .utils import EnvironmentUtils, TwinUtils, URLUtils, CompactAPIUtils

__version__ = "0.1.4" 

__all__ = [
    "Client",
    "TwinsAPI",
    "Cyberwave",
    "Mission",
    "CyberwaveTask",
    "AssetsAPI",
    # Skeleton data structures
    "Joint3D",
    "HandSkeleton", 
    "BodySkeleton",
    "RobotPose",
    "BasicSkeletonMapper",
    "CyberWaveError",
    "APIError",
    "AuthenticationError",
    "Mesh",
    "RobotDriver",
    "Robot",
    # Compact API
    "twin",
    "configure",
    "simulation",
    "pose",
    "alert",
    "dispatch",
    "agent",
    "CompactAgent",
    "MissionPlanResult",
    "MissionTaskSummary",
    "AuthTrigger",
    # Environment Configuration
    "CyberWaveEnvironment",
    "ENVIRONMENT_URLS",
    # Utilities
    "EnvironmentUtils",
    "TwinUtils", 
    "URLUtils",
    "CompactAPIUtils",
    # Centralized schema system
    "convert_sdk_to_centralized",
    "generate_centralized_level_yaml", 
    "validate_centralized_level",
    "CYBERWAVE_API_VERSION",
    "CYBERWAVE_LEVEL_API_VERSION",
    "CentralizedSchemaError",
    # Constants
    "DEFAULT_BACKEND_URL",
    "BACKEND_URL_ENV_VAR",
    "USERNAME_ENV_VAR",
    "PASSWORD_ENV_VAR",
    "Mesh",
    "AbstractAsset",
    "StaticAsset",
    "RobotAsset",
    "PhysicalDevice",
    "DigitalTwin",
    # Compact API
    "twin",
    "configure",
    "simulation",
]
