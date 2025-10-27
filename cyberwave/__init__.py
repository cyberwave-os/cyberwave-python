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
    >>> robot = cw.twin("cyberwave/so101")
    >>> robot.move(x=1, y=0, z=0.5)
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
    simulation,
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

# Version information
__version__ = "0.1.0"

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
    
    # Version
    "__version__",
]