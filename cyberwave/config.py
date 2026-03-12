"""
Configuration management for the Cyberwave SDK
"""

import os
from dataclasses import dataclass
from typing import Optional

from cyberwave.constants import SOURCE_TYPE_EDGE

# Production defaults values
DEFAULT_BASE_URL = "https://api.cyberwave.com"
DEFAULT_MQTT_HOST = "mqtt.cyberwave.com"
DEFAULT_MQTT_PORT = 8883
DEFAULT_MQTT_USERNAME = "mqttcyb"
DEFAULT_TIMEOUT = 30


def _parse_bool_env(value: Optional[str], default: bool = False) -> bool:
    """Parse common boolean env var representations."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CyberwaveConfig:
    """
    Configuration for the Cyberwave SDK

    Args:
        base_url: Base URL of the Cyberwave backend (e.g., "https://api.cyberwave.com")
        api_key: API key for authentication (sent as Bearer auth value)
        token: Deprecated alias for api_key (kept for backwards compatibility)
        mqtt_host: MQTT broker host (defaults to mqtt.cyberwave.com)
        mqtt_port: MQTT broker port (defaults to 8883)
        mqtt_username: MQTT username (optional)
        mqtt_use_tls: Whether to enable MQTT TLS transport
        mqtt_tls_ca_cert: Path to CA certificate bundle for MQTT TLS
        environment_id: Default environment ID to use
        workspace_id: Default workspace ID to use
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    token: Optional[str] = None
    mqtt_host: Optional[str] = None
    mqtt_port: int | None = None
    mqtt_username: Optional[str] = None
    mqtt_use_tls: bool = False
    mqtt_tls_ca_cert: Optional[str] = None
    environment_id: Optional[str] = None
    workspace_id: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    verify_ssl: bool = True
    source_type: str = SOURCE_TYPE_EDGE
    topic_prefix: Optional[str] = None

    def __post_init__(self):
        """Load configuration from environment variables if not provided"""
        if not self.api_key and self.token:
            self.api_key = self.token

        if not self.api_key:
            self.api_key = os.getenv("CYBERWAVE_API_KEY")

        if self.base_url == DEFAULT_BASE_URL:
            self.base_url = os.getenv("CYBERWAVE_BASE_URL", DEFAULT_BASE_URL)

        if not self.mqtt_host:
            self.mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST", DEFAULT_MQTT_HOST)

        if self.mqtt_port is None:
            port_str = os.getenv("CYBERWAVE_MQTT_PORT")
            self.mqtt_port = int(port_str) if port_str else DEFAULT_MQTT_PORT

        if not self.mqtt_username:
            self.mqtt_username = os.getenv(
                "CYBERWAVE_MQTT_USERNAME", DEFAULT_MQTT_USERNAME
            )

        self.mqtt_use_tls = _parse_bool_env(
            os.getenv("CYBERWAVE_MQTT_USE_TLS"), default=self.mqtt_use_tls
        )
        if self.mqtt_port == 8883 and not self.mqtt_use_tls:
            # Port 8883 is the conventional MQTT-over-TLS port.
            self.mqtt_use_tls = True
        if not self.mqtt_tls_ca_cert:
            self.mqtt_tls_ca_cert = os.getenv("CYBERWAVE_MQTT_TLS_CA_CERT")

        if not self.environment_id:
            self.environment_id = os.getenv("CYBERWAVE_ENVIRONMENT_ID")

        if not self.workspace_id:
            self.workspace_id = os.getenv("CYBERWAVE_WORKSPACE_ID")

        if not self.source_type:
            self.source_type = os.getenv("CYBERWAVE_SOURCE_TYPE", SOURCE_TYPE_EDGE)

        if not self.topic_prefix:
            # Check for explicit prefix first
            self.topic_prefix = os.getenv("CYBERWAVE_MQTT_TOPIC_PREFIX")

            # If not set, derive from environment (legacy behavior)
            if not self.topic_prefix:
                env_value = os.getenv("CYBERWAVE_ENVIRONMENT", "").strip()
                if env_value and env_value.lower() != "production":
                    self.topic_prefix = env_value
                else:
                    self.topic_prefix = ""

    @property
    def auth_header(self) -> dict:
        """Get the authorization header for API requests"""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
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
