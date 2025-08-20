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
try:
    from cyberwave_robotics_integrations.factory import Robot as RobotDriver
except ModuleNotFoundError:
    RobotDriver = None
from .trainer import VideoTrainer, perform_welding
from .client import Client, CyberWaveError, APIError, AuthenticationError
from .sdk import Cyberwave, Mission
from .geometry import Mesh
from .arm import Arm
from .constants import (
    DEFAULT_BACKEND_URL,
    BACKEND_URL_ENV_VAR,
    USERNAME_ENV_VAR,
    PASSWORD_ENV_VAR,
)

# geometry primitives
from .geometry import Mesh

# Import level module components
from .level import LevelDefinition, load_level, save_level

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
from .compact_api import twin, configure, simulation

__version__ = "0.1.3" 

__all__ = [
    "Client",
    "Cyberwave",
    "Mission",
    "CyberWaveError",
    "APIError",
    "AuthenticationError",
    "Mesh",
    "RobotDriver",
    "Robot",
    "LevelDefinition",
    "load_level",
    "save_level",
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
    "Arm",
    # Compact API
    "twin",
    "configure",
    "simulation",
]
