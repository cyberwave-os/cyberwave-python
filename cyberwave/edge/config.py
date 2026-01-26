"""
Configuration for edge nodes.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeNodeConfig:
    """
    Base configuration for all edge nodes.

    This configuration can be loaded from environment variables or passed directly.

    Environment Variables:
        CYBERWAVE_TOKEN: API token for authentication
        CYBERWAVE_BASE_URL: Base URL of the Cyberwave backend
        MQTT_HOST: MQTT broker hostname
        MQTT_PORT: MQTT broker port
        MQTT_USERNAME: MQTT username
        MQTT_PASSWORD: MQTT password
        EDGE_UUID: UUID of this edge device
        TWIN_UUID: UUID of the default twin
        TOPIC_PREFIX: MQTT topic prefix (environment-specific)
        HEALTH_INTERVAL: Health publish interval in seconds
        LOG_LEVEL: Logging level
    """

    # Cyberwave connection
    cyberwave_token: Optional[str] = field(
        default_factory=lambda: os.getenv("CYBERWAVE_TOKEN")
    )
    cyberwave_base_url: str = field(
        default_factory=lambda: os.getenv(
            "CYBERWAVE_BASE_URL", "https://api.cyberwave.com"
        )
    )

    # MQTT
    mqtt_host: Optional[str] = field(
        default_factory=lambda: os.getenv("MQTT_HOST")
    )
    mqtt_port: int = field(
        default_factory=lambda: int(os.getenv("MQTT_PORT", "1883"))
    )
    mqtt_username: Optional[str] = field(
        default_factory=lambda: os.getenv("MQTT_USERNAME")
    )
    mqtt_password: Optional[str] = field(
        default_factory=lambda: os.getenv("MQTT_PASSWORD")
    )

    # Topic prefix for MQTT (environment-specific)
    topic_prefix: str = field(
        default_factory=lambda: os.getenv("TOPIC_PREFIX", "")
    )

    # Device identity
    edge_uuid: str = field(
        default_factory=lambda: os.getenv("EDGE_UUID", "")
    )
    twin_uuid: Optional[str] = field(
        default_factory=lambda: os.getenv("TWIN_UUID")
    )

    # Health & resilience
    health_publish_interval: int = field(
        default_factory=lambda: int(os.getenv("HEALTH_INTERVAL", "5"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # Source type for telemetry
    source_type: str = field(
        default_factory=lambda: os.getenv("SOURCE_TYPE", "edge")
    )

    @classmethod
    def from_env(cls) -> "EdgeNodeConfig":
        """Create configuration from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not self.cyberwave_token:
            raise ValueError(
                "CYBERWAVE_TOKEN is required. "
                "Get yours at https://cyberwave.com/profile"
            )
        if not self.edge_uuid:
            raise ValueError(
                "EDGE_UUID is required. Set it via environment variable or config."
            )
