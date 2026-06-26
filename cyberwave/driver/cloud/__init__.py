"""Cyberwave cloud connectivity and the cloud-facing reactions to driver state.

Everything the driver does to talk to the platform: MQTT/twin connect and the
reconnect watchdog (:class:`~cyberwave.driver.cloud.connection.CloudConnectionMixin`),
the lifecycle **alerts** that react to transitions
(:class:`~cyberwave.driver.cloud.lifecycle_alerts.LifecycleAlertsMixin` — the state
machine itself lives in :mod:`cyberwave.driver.status`), telemetry-session markers,
the alert manager + driver-facing alert API, and twin introspection.
"""

from .alert_api import DriverAlertsMixin
from .alerts import (
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
from .connection import CloudConnectionMixin
from .lifecycle_alerts import LifecycleAlertsMixin
from .telemetry_session import TelemetrySessionMixin
from .twin_binding import (
    refresh_driver_twin_from_api,
    resolve_twin_attached_controller,
)

__all__ = [
    "DriverAlertsMixin",
    "CloudConnectionMixin",
    "LifecycleAlertsMixin",
    "TelemetrySessionMixin",
    "refresh_driver_twin_from_api",
    "resolve_twin_attached_controller",
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
]
