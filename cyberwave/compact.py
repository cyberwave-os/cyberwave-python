"""
Compact API for quick and easy interaction with Cyberwave

This module provides a simplified, module-level API for common operations.
It manages a global client instance for convenience.

Example:
    >>> import cyberwave as cw
    >>> cw.configure(api_key="your_key", base_url="http://localhost:8000")
    >>> robot = cw.twin("cyberwave/so101")
    >>> robot.move(x=1, y=0, z=0.5)
"""

from typing import Optional
from .client import Cyberwave
from .twin import Twin
from .config import CyberwaveConfig, get_config, set_config
from .exceptions import CyberwaveError


# Global client instance
_global_client: Optional[Cyberwave] = None


def _get_client() -> Cyberwave:
    """Get or create the global client instance"""
    global _global_client
    if _global_client is None:
        config = get_config()
        _global_client = Cyberwave(
            base_url=config.base_url,
            token=config.token,
            api_key=config.api_key,
            mqtt_host=config.mqtt_host,
            mqtt_port=config.mqtt_port,
            environment_id=config.environment_id,
            workspace_id=config.workspace_id,
        )
    return _global_client


def configure(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    environment: Optional[str] = None,
    workspace: Optional[str] = None,
    mqtt_host: Optional[str] = None,
    mqtt_port: Optional[int] = None,
    **kwargs
):
    """
    Configure the global Cyberwave client
    
    Args:
        base_url: Base URL of the Cyberwave backend
        api_key: API key for authentication
        token: Bearer token for authentication
        environment: Default environment ID
        workspace: Default workspace ID
        mqtt_host: MQTT broker host
        mqtt_port: MQTT broker port
        **kwargs: Additional configuration options
    
    Example:
        >>> import cyberwave as cw
        >>> cw.configure(
        ...     base_url="http://localhost:8000",
        ...     api_key="your_api_key",
        ...     environment="env_uuid"
        ... )
    """
    global _global_client
    
    # Update global config
    config = get_config()
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    if token:
        config.token = token
    if environment:
        config.environment_id = environment
    if workspace:
        config.workspace_id = workspace
    if mqtt_host:
        config.mqtt_host = mqtt_host
    if mqtt_port:
        config.mqtt_port = mqtt_port
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    set_config(config)
    
    # Reset global client to pick up new config
    if _global_client:
        _global_client.disconnect()
        _global_client = None


def twin(asset_key: str, environment: Optional[str] = None, **kwargs) -> Twin:
    """
    Create or get a digital twin (compact API)
    
    Args:
        asset_key: Asset identifier (e.g., "cyberwave/so101")
        environment: Environment ID (uses default from config if not provided)
        **kwargs: Additional twin creation parameters
    
    Returns:
        Twin instance
    
    Example:
        >>> import cyberwave as cw
        >>> robot = cw.twin("cyberwave/so101")
        >>> robot.move(x=1, y=0, z=0.5)
        >>> robot.joints.arm_joint = 45
    """
    client = _get_client()
    return client.twin(asset_key, environment_id=environment, **kwargs)


class SimulationControl:
    """
    Control interface for simulation
    
    Provides methods to play, pause, step, and reset simulations.
    """
    
    def __init__(self):
        self._is_playing = False
    
    def play(self):
        """Start the simulation"""
        client = _get_client()
        # TODO: Implement simulation control via API
        # This would call an endpoint like: client.api.simulation_play()
        self._is_playing = True
        print("Simulation play - API endpoint not yet implemented")
    
    def pause(self):
        """Pause the simulation"""
        client = _get_client()
        # TODO: Implement simulation control via API
        self._is_playing = False
        print("Simulation pause - API endpoint not yet implemented")
    
    def step(self, steps: int = 1):
        """
        Step the simulation forward
        
        Args:
            steps: Number of simulation steps to advance
        """
        client = _get_client()
        # TODO: Implement simulation control via API
        print(f"Simulation step ({steps}) - API endpoint not yet implemented")
    
    def reset(self):
        """Reset the simulation to initial state"""
        client = _get_client()
        # TODO: Implement simulation control via API
        print("Simulation reset - API endpoint not yet implemented")
    
    @property
    def is_playing(self) -> bool:
        """Check if simulation is currently playing"""
        return self._is_playing
    
    def set_speed(self, speed: float):
        """
        Set simulation speed multiplier
        
        Args:
            speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
        """
        client = _get_client()
        # TODO: Implement simulation control via API
        print(f"Simulation speed set to {speed}x - API endpoint not yet implemented")


# Global simulation control instance
simulation = SimulationControl()


# Convenience function to get the global client
def get_client() -> Cyberwave:
    """
    Get the global Cyberwave client instance
    
    Returns:
        Global Cyberwave client
    
    Example:
        >>> import cyberwave as cw
        >>> cw.configure(api_key="your_key")
        >>> client = cw.get_client()
        >>> workspaces = client.workspaces.list()
    """
    return _get_client()

