"""
Configuration management for the Cyberwave SDK
"""

import os
from typing import Optional
from dataclasses import dataclass

from cyberwave.constants import SOURCE_TYPE_EDGE

# Production defaults values
DEFAULT_BASE_URL = "https://api.cyberwave.com"
DEFAULT_MQTT_HOST = "mqtt.cyberwave.com"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_USERNAME = "mqttcyb"
DEFAULT_MQTT_PASSWORD = "mqttcyb231"
DEFAULT_TIMEOUT = 30


@dataclass
class CyberwaveConfig:
    """
    Configuration for the Cyberwave SDK

    Args:
        base_url: Base URL of the Cyberwave backend (e.g., "https://api.cyberwave.com")
        api_key: API key for authentication (sent as Bearer token)
        token: Bearer token for authentication (takes precedence over api_key)
        mqtt_host: MQTT broker host (defaults to base_url host)
        mqtt_port: MQTT broker port (defaults to 1883)
        mqtt_username: MQTT username (optional)
        mqtt_password: MQTT password (optional)
        environment_id: Default environment ID to use
        workspace_id: Default workspace ID to use
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    token: Optional[str] = None
    mqtt_host: Optional[str] = None
    mqtt_port: int = DEFAULT_MQTT_PORT
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    environment_id: Optional[str] = None
    workspace_id: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    verify_ssl: bool = True
    source_type: str = SOURCE_TYPE_EDGE

    def __post_init__(self):
        """Load configuration from environment variables if not provided"""
        if not self.api_key and not self.token:
            self.api_key = os.getenv("CYBERWAVE_API_KEY")
            self.token = os.getenv("CYBERWAVE_TOKEN")

        if self.base_url == DEFAULT_BASE_URL:
            self.base_url = os.getenv("CYBERWAVE_BASE_URL", DEFAULT_BASE_URL)

        if not self.mqtt_host:
            self.mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST", DEFAULT_MQTT_HOST)

        if self.mqtt_port == DEFAULT_MQTT_PORT:
            port_str = os.getenv("CYBERWAVE_MQTT_PORT")
            if port_str:
                self.mqtt_port = int(port_str)

        if not self.mqtt_username:
            self.mqtt_username = os.getenv(
                "CYBERWAVE_MQTT_USERNAME", DEFAULT_MQTT_USERNAME
            )

        if not self.mqtt_password:
            self.mqtt_password = os.getenv(
                "CYBERWAVE_MQTT_PASSWORD", DEFAULT_MQTT_PASSWORD
            )

        if not self.environment_id:
            self.environment_id = os.getenv("CYBERWAVE_ENVIRONMENT_ID")

        if not self.workspace_id:
            self.workspace_id = os.getenv("CYBERWAVE_WORKSPACE_ID")

        if not self.source_type:
            self.source_type = os.getenv("CYBERWAVE_SOURCE_TYPE", SOURCE_TYPE_EDGE)

    @property
    def auth_header(self) -> dict:
        """Get the authorization header for API requests"""
        # Use token first, fallback to api_key - both use Bearer
        auth_value = self.token or self.api_key
        if auth_value:
            return {"Authorization": f"Bearer {auth_value}"}
        return {}


# Global configuration instance
_global_config: Optional[CyberwaveConfig] = None


def get_config() -> CyberwaveConfig:
    """Get the global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = CyberwaveConfig()
    return _global_config


def set_config(config: CyberwaveConfig):
    """Set the global configuration instance"""
    global _global_config
    _global_config = config
