"""
Compact API for quick and easy interaction with Cyberwave

This module provides a simplified, module-level API for common operations.
It manages a global client instance for convenience.

Example:
    >>> import cyberwave as cw
    >>> cw.configure(api_key="your_key", base_url="http://localhost:8000")
    >>> robot = cw.twin("the-robot-studio/so101")
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
        config_kwargs = {}
        if config.environment_id:
            config_kwargs["environment_id"] = config.environment_id
        if config.workspace_id:
            config_kwargs["workspace_id"] = config.workspace_id

        _global_client = Cyberwave(
            base_url=config.base_url,
            token=config.token,
            api_key=config.api_key,
            mqtt_host=config.mqtt_host,
            mqtt_port=config.mqtt_port,
            **config_kwargs,
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
    **kwargs,
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
        asset_key: Asset identifier (e.g., "the-robot-studio/so101")
        environment: Environment ID (uses default from config if not provided)
        **kwargs: Additional twin creation parameters

    Returns:
        Twin instance

    Example:
        >>> import cyberwave as cw
        >>> robot = cw.twin("the-robot-studio/so101")
        >>> robot.move(x=1, y=0, z=0.5)
        >>> robot.joints.arm_joint = 45
    """
    client = _get_client()
    return client.twin(asset_key, environment_id=environment, **kwargs)


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
