"""Cyberwave Python SDK — edge driver framework (:mod:`cyberwave.driver`).

Public surface for authors building drivers with :class:`~cyberwave.driver.BaseDriver`:

- Lifecycle shell, cloud MQTT, twin binding, alerts, telemetry
- Declarative interface registry (``cw-driver.yml`` / ``metadata["mqtt"]`` and optional ``metadata["zenoh"]``)
- Optional mixins (video, audio, imperative Zenoh publish/subscribe)

Install the ``cyberwave`` package from PyPI; import from ``cyberwave.driver`` or ``from cyberwave import ...``.
"""

# Public API is re-exported from the focused subpackages (interface/, cloud/,
# transports/, sensors/, support/). Deep imports like ``cyberwave.driver.cloud.alerts``
# also work, but ``from cyberwave.driver import X`` is the supported surface.
from .base import BaseDriver
from .status import DriverLifecycleState
from .cloud import (
    AlertCode,
    AlertManager,
    AlertSeverity,
    AlertState,
    DriverAlert,
    create_battery_critical_alert,
    create_battery_warning_alert,
    create_camera_failure_alert,
    create_command_rejected_alert,
    create_command_timeout_alert,
    create_config_error_alert,
    create_connection_alert,
    create_disconnection_alert,
    create_invalid_command_alert,
    create_lidar_failure_alert,
    create_overtemp_alert,
    create_safety_violation_alert,
    create_sensor_failure_alert,
    create_stale_telemetry_alert,
)
from .interface import (
    COMMAND_SOURCE_TYPES,
    DEFAULT_PUBLISH_SOURCE_TYPE,
    DEFAULT_SIM_PUBLISH_SOURCE_TYPE,
    CallbackGroup,
    CommandArg,
    CommandArgs,
    CommandInbox,
    DriverInterfaceRegistry,
    DriverOperationMode,
    InterfaceRegistryMixin,
    ProtocolArgs,
    PublisherArgs,
    TopicSpec,
    accepts_inbound,
    default_management_commands,
    dump_cw_driver_yml,
    filtered_listener,
    resolve_topic_path,
)
from .sensors import AudioStreamMixin, VideoStreamMixin
from .support import (
    ColoredFormatter,
    check_device_reachable_async,
    get_colored_formatter,
    get_sdk_version,
    load_driver_manifest,
    setup_colored_logging,
)
from .transports import CommandContext, ZenohPublisherMixin, ZenohSubscriberMixin

__all__ = [
    # Driver base
    "BaseDriver",
    "DriverLifecycleState",
    "DriverOperationMode",
    "InterfaceRegistryMixin",
    "DriverInterfaceRegistry",
    "CallbackGroup",
    "TopicSpec",
    "ProtocolArgs",
    "CommandArg",
    "CommandArgs",
    "PublisherArgs",
    "default_management_commands",
    "resolve_topic_path",
    # Command hand-off + source-type policy
    "CommandInbox",
    "COMMAND_SOURCE_TYPES",
    "DEFAULT_PUBLISH_SOURCE_TYPE",
    "DEFAULT_SIM_PUBLISH_SOURCE_TYPE",
    "accepts_inbound",
    "filtered_listener",
    "AudioStreamMixin",
    "VideoStreamMixin",
    "ZenohPublisherMixin",
    "ZenohSubscriberMixin",
    "CommandContext",
    # Shared utilities
    "check_device_reachable_async",
    "load_driver_manifest",
    "dump_cw_driver_yml",
    "get_sdk_version",
    # Alert handling
    "DriverAlert",
    "AlertSeverity",
    "AlertState",
    "AlertCode",
    "AlertManager",
    "create_connection_alert",
    "create_overtemp_alert",
    "create_stale_telemetry_alert",
    "create_command_timeout_alert",
    "create_disconnection_alert",
    "create_battery_critical_alert",
    "create_battery_warning_alert",
    "create_sensor_failure_alert",
    "create_lidar_failure_alert",
    "create_camera_failure_alert",
    "create_command_rejected_alert",
    "create_invalid_command_alert",
    "create_safety_violation_alert",
    "create_config_error_alert",
    # Colored logging
    "ColoredFormatter",
    "setup_colored_logging",
    "get_colored_formatter",
]
