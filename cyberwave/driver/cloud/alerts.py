"""Structured alert handling for :mod:`cyberwave.driver` (Python SDK).

Lifecycle tracking, deduplication, optional auto-recovery, and REST/MQTT
integration with the Cyberwave API via the bound ``cyberwave`` client on
:class:`~cyberwave.driver.BaseDriver`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_node_ip() -> str:
    """Return the primary outbound IP of this host.

    Tries two strategies before giving up:
    1. UDP connect trick — asks the OS which interface would route to 8.8.8.8
       (no traffic sent).  Gives the correct external-facing IP on multi-homed hosts.
    2. Hostname resolution — covers hosts with no default route (e.g. isolated LANs).

    Returns a descriptive tag on failure rather than a generic string so that
    alert consumers can distinguish "not resolved" from "resolved as loopback".
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        pass

    return "<ip-unresolvable>"


class AlertSeverity(Enum):
    """Alert severity levels (maps 1:1 to backend Alert.severity)."""

    CRITICAL = "critical"  # Driver cannot continue, requires immediate action
    ERROR = "error"  # Feature degraded, needs attention
    WARNING = "warning"  # Potential issue, monitor
    INFO = "info"  # Informational only


class AlertState(Enum):
    """Alert lifecycle states."""

    ACTIVE = "active"  # Currently occurring
    RESOLVED = "resolved"  # Fixed automatically or manually
    ACKNOWLEDGED = "acknowledged"  # Operator aware, working on it
    SUPPRESSED = "suppressed"  # Intentionally ignored


class AlertCode(Enum):
    """Standardized alert codes organized by severity.

    Ranges:
        1xxx: CRITICAL - Driver cannot continue, immediate action required
        2xxx: ERROR - Feature degraded, needs attention
        3xxx: WARNING - Potential issue, monitor closely
    """

    # CRITICAL alerts (1xxx) - System cannot continue safely
    SAFETY_VIOLATION = 1001  # Safety limits exceeded
    BATTERY_CRITICAL = 1002  # Battery below critical threshold (<5%)
    MOTOR_OVERHEAT = 1003  # Motor temperature critically high
    WATCHDOG_EXPIRED = 1004  # System watchdog timer expired
    RESOURCE_EXHAUSTION = 1005  # Out of memory or CPU resources
    CONFIGURATION_ERROR = 1006  # Invalid configuration preventing startup

    # ERROR level (2xxx) - Feature degraded, needs attention
    WEBRTC_CONNECTION_FAILED = 2001  # WebRTC connection failed
    WEBRTC_DISCONNECTED = 2002  # WebRTC disconnected unexpectedly
    MQTT_CONNECTION_FAILED = 2003  # MQTT connection failed
    MQTT_DISCONNECTED = 2004  # MQTT disconnected unexpectedly
    ROBOT_UNREACHABLE = 2005  # Robot hardware unreachable
    SENSOR_FAILURE = 2006  # Sensor malfunction or failure
    LIDAR_FAILURE = 2007  # LIDAR system failure
    CAMERA_FAILURE = 2008  # Camera system failure
    COMMAND_TIMEOUT = 2009  # Command failed to execute in time
    TELEMETRY_VALIDATION_FAILED = 2010  # Telemetry data validation failed
    MISSING_SENSOR_DATA = 2011  # Expected sensor data not received
    STATE_MACHINE_ERROR = 2012  # Driver state machine error

    # WARNING level (3xxx) - Potential issues, monitor
    STALE_TELEMETRY = 3001  # Telemetry data is outdated
    COMMAND_REJECTED = 3002  # Command rejected by robot
    INVALID_COMMAND = 3003  # Command validation failed
    BATTERY_WARNING = 3004  # Battery below warning threshold (<10%)

    # INFO (4xxx) - Operational notices (non-actionable)
    DRIVER_LIFECYCLE = 4001  # Lifecycle state transition
    DRIVER_ACTIVE = 4002  # Driver entered active / publishing


@dataclass
class DriverAlert:
    """Structured alert representation with lifecycle tracking."""

    # Identity
    alert_code: AlertCode
    severity: AlertSeverity
    component: str  # e.g., "webrtc", "mqtt", "lidar", "motion_controller"

    # Description
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    # Context
    timestamp: float = field(default_factory=time.time)
    state: AlertState = AlertState.ACTIVE
    occurrence_count: int = 1
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(init=False)

    # Recovery
    recovery_action: str | None = None
    auto_recoverable: bool = False
    recovery_attempted: bool = False
    recovery_success: bool = False

    # Metadata
    stack_trace: str | None = None
    related_alerts: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate unique alert ID, stamp node identity, and align last_occurrence."""
        self.alert_id = f"{self.component}_{self.alert_code.name}_{int(self.timestamp)}"
        self.last_occurrence = self.first_occurrence
        # Node identity: prefer explicit env-var overrides, fall back to system values
        self.node_hostname: str = os.getenv(
            "CYBERWAVE_NODE_HOSTNAME", socket.gethostname()
        )
        self.node_ip: str = os.getenv("CYBERWAVE_NODE_IP", _resolve_node_ip())

    def increment_occurrence(self):
        """Record another occurrence of this alert."""
        self.occurrence_count += 1
        self.last_occurrence = time.time()

    def resolve(self, resolution_note: str | None = None):
        """Mark alert as resolved.

        Args:
            resolution_note: Optional note about how it was resolved
        """
        self.state = AlertState.RESOLVED
        if resolution_note:
            self.details["resolution_note"] = resolution_note
        self.details["resolved_at"] = time.time()

    def acknowledge(self):
        """Mark alert as acknowledged by operator."""
        self.state = AlertState.ACKNOWLEDGED
        self.details["acknowledged_at"] = time.time()

    def suppress(self, reason: str):
        """Suppress this alert (intentionally ignore).

        Args:
            reason: Reason for suppression
        """
        self.state = AlertState.SUPPRESSED
        self.details["suppression_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for MQTT publishing
        """
        return {
            "alert_id": self.alert_id,
            "alert_code": self.alert_code.name,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "occurrence_count": self.occurrence_count,
            "first_occurrence": self.first_occurrence,
            "last_occurrence": self.last_occurrence,
            "recovery_action": self.recovery_action,
            "auto_recoverable": self.auto_recoverable,
            "recovery_attempted": self.recovery_attempted,
            "recovery_success": self.recovery_success,
            "source_type": "edge",  # Required by Cyberwave
            "node_hostname": self.node_hostname,
            "node_ip": self.node_ip,
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())


# ========================================
# BACKEND INTEGRATION MAPPINGS
# ========================================

# Map AlertCode to backend alert_type strings
ALERT_CODE_TO_ALERT_TYPE = {
    AlertCode.SAFETY_VIOLATION: "safety_violation",
    AlertCode.BATTERY_CRITICAL: "battery_critical",
    AlertCode.MOTOR_OVERHEAT: "motor_overheat",
    AlertCode.WATCHDOG_EXPIRED: "watchdog_expired",
    AlertCode.RESOURCE_EXHAUSTION: "resource_exhaustion",
    AlertCode.CONFIGURATION_ERROR: "configuration_error",
    AlertCode.WEBRTC_CONNECTION_FAILED: "webrtc_connection_failed",
    AlertCode.WEBRTC_DISCONNECTED: "webrtc_disconnected",
    AlertCode.MQTT_CONNECTION_FAILED: "mqtt_connection_failed",
    AlertCode.MQTT_DISCONNECTED: "mqtt_disconnected",
    AlertCode.ROBOT_UNREACHABLE: "robot_unreachable",
    AlertCode.SENSOR_FAILURE: "sensor_failure",
    AlertCode.LIDAR_FAILURE: "lidar_failure",
    AlertCode.CAMERA_FAILURE: "camera_failure",
    AlertCode.COMMAND_TIMEOUT: "command_timeout",
    AlertCode.TELEMETRY_VALIDATION_FAILED: "telemetry_validation_failed",
    AlertCode.MISSING_SENSOR_DATA: "missing_sensor_data",
    AlertCode.STATE_MACHINE_ERROR: "state_machine_error",
    AlertCode.STALE_TELEMETRY: "stale_telemetry",
    AlertCode.COMMAND_REJECTED: "command_rejected",
    AlertCode.INVALID_COMMAND: "invalid_command",
    AlertCode.BATTERY_WARNING: "battery_warning",
    AlertCode.DRIVER_LIFECYCLE: "driver_lifecycle",
    AlertCode.DRIVER_ACTIVE: "driver_active",
}

# Map AlertSeverity to backend severity strings
SEVERITY_TO_BACKEND_SEVERITY = {
    AlertSeverity.CRITICAL: "critical",
    AlertSeverity.ERROR: "error",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.INFO: "info",
}

# Driver lifecycle/active alerts are informational state mirrors that are already
# surfaced as curated, auto-resolving twin notices (see cloud/lifecycle_alerts.py).
# Don't *also* create backend alert rows for them — those stay "active" until
# shutdown and read as duplicate, verbose clutter in the twin UI. They remain in
# the in-memory AlertManager for get_active_alerts()/summaries.
_BACKEND_PUSH_SUPPRESSED_CODES = frozenset(
    {AlertCode.DRIVER_LIFECYCLE, AlertCode.DRIVER_ACTIVE}
)


def _format_alert_description(alert: "DriverAlert") -> str:
    """Readable twin-alert description: a Component line + one ``key: value`` per
    detail, instead of a raw ``json.dumps`` blob."""
    lines: list[str] = []
    if alert.component:
        lines.append(f"Component: {alert.component}")
    for key, value in (alert.details or {}).items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


class AlertManager:
    """Manages alert lifecycle, deduplication, and auto-recovery.

    Example:
        >>> manager = AlertManager()
        >>> manager.register_handler(AlertCode.MQTT_DISCONNECTED, reconnect_mqtt)
        >>> alert = DriverAlert(
        ...     alert_code=AlertCode.MQTT_DISCONNECTED,
        ...     severity=AlertSeverity.ERROR,
        ...     component="mqtt",
        ...     message="Connection lost",
        ...     auto_recoverable=True
        ... )
        >>> manager.raise_alert(alert)
    """

    def __init__(
        self,
        max_history: int = 100,
        sdk_client: Any | None = None,
        twin_uuid: str | None = None,
        environment_uuid: str | None = None,
        enable_backend_push: bool = False,
    ):
        """Initialize alert manager.

        Args:
            max_history: Maximum number of alerts to keep in history
            sdk_client: Cyberwave SDK client for backend push (optional)
            twin_uuid: Twin UUID for attaching alerts (required if backend enabled)
            environment_uuid: Environment UUID for alert context (optional)
            enable_backend_push: Enable pushing alerts to backend
        """
        self._active_alerts: dict[str, DriverAlert] = {}
        self._alert_history: list[DriverAlert] = []
        self._max_history = max_history
        self._alert_handlers: dict[AlertCode, Callable] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Backend integration
        self._sdk_client = sdk_client
        self._twin_uuid = twin_uuid
        self._environment_uuid = environment_uuid
        self._enable_backend_push = enable_backend_push
        self._backend_alert_ids: dict[str, str] = {}  # alert_key -> backend alert UUID
        self._twin_handle: Any | None = None  # cached twin handle

        logger.info("AlertManager initialized")

    def enable_backend_integration(
        self,
        sdk_client: Any,
        twin_uuid: str,
        environment_uuid: str | None = None,
        mqtt_client: Any | None = None,
    ) -> None:
        """Upgrade this instance to push alerts to the backend.

        Migrates any active alerts that were raised before the backend was
        available. Safe to call on an already-configured instance (no-op if
        the parameters are unchanged). Pass ``mqtt_client`` to also start the
        alert listener so operator acknowledgements sync back to the driver.

        Args:
            sdk_client: Cyberwave SDK client
            twin_uuid: Twin UUID for attaching alerts
            environment_uuid: Optional environment UUID
            mqtt_client: Optional MQTT client to start the alert listener
        """
        with self._lock:
            self._sdk_client = sdk_client
            self._twin_uuid = twin_uuid
            self._environment_uuid = environment_uuid
            self._enable_backend_push = True
            self._twin_handle = None  # Reset cached handle for new client

            # Push any active alerts that were raised before backend was available
            for alert_key, alert in self._active_alerts.items():
                if alert_key in self._backend_alert_ids:
                    continue
                if alert.severity in (
                    AlertSeverity.CRITICAL,
                    AlertSeverity.ERROR,
                    AlertSeverity.WARNING,
                ) or alert.alert_code in (
                    AlertCode.DRIVER_LIFECYCLE,
                    AlertCode.DRIVER_ACTIVE,
                ):
                    self._push_alert_to_backend(alert, alert_key)

        # Start alert listener outside the lock (subscribes to MQTT)
        if mqtt_client is not None:
            self.start_alert_listener(mqtt_client)

    def sync_lifecycle_alerts_to_backend(self, component: str) -> None:
        """Push active driver lifecycle/active alerts for *component* to the twin."""
        with self._lock:
            if not self._enable_backend_push:
                return
            for code in (AlertCode.DRIVER_LIFECYCLE, AlertCode.DRIVER_ACTIVE):
                alert_key = f"{component}_{code.name}"
                alert = self._active_alerts.get(alert_key)
                if alert is not None and alert_key not in self._backend_alert_ids:
                    self._push_alert_to_backend(alert, alert_key)

    def shutdown(self) -> None:
        """Flush remaining active alerts to backend and release resources.

        Should be called once during driver shutdown to ensure all in-flight
        alerts are visible in the backend before the process exits.
        """
        with self._lock:
            if not self._enable_backend_push:
                return

            for alert_key, alert in list(self._active_alerts.items()):
                if alert_key not in self._backend_alert_ids:
                    if alert.severity in (
                        AlertSeverity.CRITICAL,
                        AlertSeverity.ERROR,
                        AlertSeverity.WARNING,
                    ) or alert.alert_code in (
                        AlertCode.DRIVER_LIFECYCLE,
                        AlertCode.DRIVER_ACTIVE,
                    ):
                        self._push_alert_to_backend(alert, alert_key)
                else:
                    # Resolve already-pushed alerts so operators don't see stale open alerts
                    self._resolve_backend_alert(alert_key, "Driver shut down cleanly")

            # Release cached handle so the SDK connection can be torn down
            self._twin_handle = None

    def start_alert_listener(self, mqtt_client: Any) -> None:
        """Subscribe to backend alert updates for this twin.

        When an operator acknowledges, resolves, or silences an alert from
        the frontend, the backend publishes the updated alert object on the
        MQTT topic ``{prefix}cyberwave/twin/{twin_uuid}/alert``. This method
        subscribes to that topic and updates local ``DriverAlert`` state so the
        in-process view stays in sync with the backend.

        Must be called after ``enable_backend_integration`` so that
        ``_twin_uuid`` and ``_backend_alert_ids`` are populated.

        Args:
            mqtt_client: Connected ``CyberwaveMQTTClient`` instance.
        """
        if not self._twin_uuid:
            logger.warning(
                "AlertManager: cannot start alert listener — twin_uuid not set"
            )
            return

        prefix = getattr(mqtt_client, "topic_prefix", "")
        topic = f"{prefix}cyberwave/twin/{self._twin_uuid}/alert"

        def _on_alert_update(data: Any) -> None:
            try:
                payload = data if isinstance(data, dict) else json.loads(data)
            except Exception as exc:
                logger.warning(f"AlertManager: failed to parse alert update: {exc}")
                return

            backend_uuid = payload.get("uuid")
            status = payload.get("status")
            if not backend_uuid or not status:
                return

            with self._lock:
                # Reverse-lookup the local alert key for this backend UUID
                alert_key = next(
                    (
                        k
                        for k, v in self._backend_alert_ids.items()
                        if v == backend_uuid
                    ),
                    None,
                )
                if alert_key is None:
                    return  # Not an alert we own

                alert = self._active_alerts.get(alert_key)
                if alert is None:
                    return

                if status == "acknowledged":
                    alert.acknowledge()
                    logger.info(f"Alert '{alert_key}' acknowledged by operator")

                elif status in ("resolved", "silenced"):
                    note = (
                        "Silenced by operator via frontend"
                        if status == "silenced"
                        else "Resolved by operator via frontend"
                    )
                    alert.resolve(note)

                    # Move to history
                    self._alert_history.append(alert)
                    if len(self._alert_history) > self._max_history:
                        self._alert_history.pop(0)

                    del self._active_alerts[alert_key]
                    del self._backend_alert_ids[alert_key]
                    logger.info(f"Alert '{alert_key}' {status} by operator")

        mqtt_client.subscribe(topic, _on_alert_update)
        logger.info(f"AlertManager subscribed to alert updates on {topic}")

    async def raise_alert_async(self, alert: DriverAlert) -> DriverAlert:
        """Async-safe version of raise_alert.

        Runs the synchronous raise_alert (which may make blocking HTTP calls to
        the backend) in a thread-pool executor so it does not block the asyncio
        event loop.  Use this instead of raise_alert when calling from inside a
        coroutine (e.g. recv_camera_stream, setup_data_channels).

        Args:
            alert: Alert to raise

        Returns:
            The raised or updated alert
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.raise_alert, alert)

    def register_handler(self, alert_code: AlertCode, handler: Callable):
        """Register automatic handler for specific alert code.

        Args:
            alert_code: Alert code to handle
            handler: Function to call when alert occurs (receives DriverAlert)

        Example:
            >>> def reconnect_mqtt(alert: DriverAlert):
            ...     mqtt_client.reconnect()
            >>> manager.register_handler(AlertCode.MQTT_DISCONNECTED, reconnect_mqtt)
        """
        with self._lock:
            self._alert_handlers[alert_code] = handler

    def raise_alert(self, alert: DriverAlert) -> DriverAlert:
        """Raise a new alert or update existing one.

        If an alert with the same component and alert_code already exists,
        it will be deduplicated (occurrence count incremented).

        Args:
            alert: Alert to raise

        Returns:
            The raised or updated alert
        """
        with self._lock:
            # Check if this alert already exists (by component + code)
            alert_key = f"{alert.component}_{alert.alert_code.name}"

            if alert_key in self._active_alerts:
                # Update existing alert (deduplication)
                existing = self._active_alerts[alert_key]
                existing.increment_occurrence()
                logger.warning(
                    f"Repeated alert: {alert.alert_code.name} "
                    f"(count: {existing.occurrence_count})"
                )
                return existing
            else:
                # New alert
                self._active_alerts[alert_key] = alert
                _log = {
                    AlertSeverity.CRITICAL: logger.critical,
                    AlertSeverity.ERROR: logger.error,
                    AlertSeverity.WARNING: logger.warning,
                    AlertSeverity.INFO: logger.info,
                }.get(alert.severity, logger.error)
                _log(
                    f"New {alert.severity.value} alert: {alert.alert_code.name} "
                    f"in {alert.component}: {alert.message}"
                )

                # Persist on the twin via ``twin.alerts.create`` when backend integration is on
                if self._enable_backend_push:
                    self._push_alert_to_backend(alert, alert_key)

                # Try automatic recovery if available
                if alert.auto_recoverable and alert.alert_code in self._alert_handlers:
                    try:
                        alert.recovery_attempted = True
                        self._alert_handlers[alert.alert_code](alert)
                        alert.recovery_success = True
                        logger.info(
                            f"Auto-recovery successful for {alert.alert_code.name}"
                        )
                    except Exception as e:
                        logger.error(f"Auto-recovery failed: {e}")
                        alert.recovery_success = False

                return alert

    def resolve_alert(
        self, component: str, alert_code: AlertCode, resolution_note: str | None = None
    ):
        """Resolve an active alert.

        Args:
            component: Component that had the alert
            alert_code: Alert code to resolve
            resolution_note: Optional note about how it was resolved
        """
        with self._lock:
            alert_key = f"{component}_{alert_code.name}"

            if alert_key in self._active_alerts:
                alert = self._active_alerts[alert_key]
                alert.resolve(resolution_note)

                # Resolve backend alert if pushed
                if self._enable_backend_push and alert_key in self._backend_alert_ids:
                    self._resolve_backend_alert(alert_key, resolution_note)

                # Move to history
                self._alert_history.append(alert)
                if len(self._alert_history) > self._max_history:
                    self._alert_history.pop(0)

                # Remove from active
                del self._active_alerts[alert_key]

                logger.info(f"Resolved alert: {alert_code.name} in {component}")

    def get_active_alerts(
        self, severity: AlertSeverity | None = None
    ) -> list[DriverAlert]:
        """Get all active alerts, optionally filtered by severity.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts

    def has_critical_alerts(self) -> bool:
        """Check if there are any critical alerts.

        Returns:
            True if any critical alerts exist
        """
        with self._lock:
            return any(
                a.severity == AlertSeverity.CRITICAL
                for a in self._active_alerts.values()
            )

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of alert state.

        Returns:
            Dictionary with alert counts by severity and component
        """
        with self._lock:
            summary = {
                "total_active": len(self._active_alerts),
                "by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
                "by_component": {},
            }

            for alert in self._active_alerts.values():
                summary["by_severity"][alert.severity.value] += 1
                if alert.component not in summary["by_component"]:
                    summary["by_component"][alert.component] = 0
                summary["by_component"][alert.component] += 1

            return summary

    def clear_resolved_alerts(self):
        """Clear all resolved alerts from history."""
        with self._lock:
            self._alert_history = [
                a for a in self._alert_history if a.state != AlertState.RESOLVED
            ]

    def _get_twin(self) -> Any | None:
        """Return a cached twin handle, fetching it lazily if needed."""
        if not self._sdk_client or not self._twin_uuid:
            return None
        if self._twin_handle is None:
            self._twin_handle = self._sdk_client.twin(twin_id=self._twin_uuid)
        return self._twin_handle

    def _push_alert_to_backend(self, alert: DriverAlert, alert_key: str):
        """Push alert to backend as Alert (non-blocking, graceful degradation).

        Args:
            alert: Alert to push
            alert_key: Internal alert key for tracking
        """
        if not self._sdk_client or not self._twin_uuid:
            return
        if alert.alert_code in _BACKEND_PUSH_SUPPRESSED_CODES:
            # Surfaced as a curated, auto-resolving twin notice instead.
            return

        try:
            twin = self._get_twin()
            if twin is None:
                return

            # Map local alert code to backend alert_type
            alert_type = ALERT_CODE_TO_ALERT_TYPE.get(
                alert.alert_code, alert.alert_code.name.lower()
            )
            severity = SEVERITY_TO_BACKEND_SEVERITY.get(alert.severity, "warning")

            # Create alert on backend
            backend_alert = twin.alerts.create(
                name=alert.message,
                description=_format_alert_description(alert),
                severity=severity,
                alert_type=alert_type,
                source_type="edge",
            )

            # Store mapping for updates and resolution
            self._backend_alert_ids[alert_key] = backend_alert.uuid

            logger.info(
                f"[SUCCESS] Pushed alert to backend: {alert_type} "
                f"(UUID: {backend_alert.uuid})"
            )

        except Exception as e:
            # Graceful degradation - don't crash driver if backend fails
            logger.warning(f"Failed to push alert to backend: {e}")

    def _resolve_backend_alert(self, alert_key: str, resolution_note: str | None):
        """Resolve backend alert.

        Args:
            alert_key: Internal alert key
            resolution_note: Optional resolution note
        """
        if not self._sdk_client:
            return

        try:
            twin = self._get_twin()
            if twin is None:
                return
            backend_alert_uuid = self._backend_alert_ids.get(alert_key)

            if not backend_alert_uuid:
                return

            alert_obj = twin.alerts.get(backend_alert_uuid)

            # Add resolution note if provided
            if resolution_note:
                current_desc = getattr(alert_obj, "description", "")
                alert_obj.update(
                    description=f"{current_desc}\n\nResolution: {resolution_note}"
                )

            # Mark as resolved
            alert_obj.resolve()

            # Remove mapping (resolved alerts not tracked)
            del self._backend_alert_ids[alert_key]

            logger.info(f"[RESOLVED] Backend alert: {alert_key}")

        except Exception as e:
            logger.warning(f"Failed to resolve backend alert: {e}")


# Helper functions for creating common alerts
# Organized by severity: CRITICAL (1xxx) → ERROR (2xxx) → WARNING (3xxx)

# ========================================
# CRITICAL ALERT HELPERS (1xxx)
# ========================================


def create_safety_violation_alert(
    component: str, violation_type: str, details_dict: dict | None = None
) -> DriverAlert:
    """Create a safety violation alert (CRITICAL).

    Args:
        component: Component that detected safety violation
        violation_type: Type of safety violation
        details_dict: Optional additional details

    Returns:
        DriverAlert for safety violation
    """
    return DriverAlert(
        alert_code=AlertCode.SAFETY_VIOLATION,
        severity=AlertSeverity.CRITICAL,
        component=component,
        message=f"Safety violation detected: {violation_type}",
        details=details_dict or {"violation_type": violation_type},
        recovery_action="Stop all motion and assess safety conditions",
        auto_recoverable=False,
    )


def create_battery_warning_alert(
    component: str, battery_level: float, warning_threshold: float = 10.0
) -> DriverAlert:
    """Create a battery warning alert (WARNING).

    Args:
        component: Component reporting battery level
        battery_level: Current battery level (percentage)
        warning_threshold: Warning threshold (default: 10%)

    Returns:
        DriverAlert for low battery warning
    """
    return DriverAlert(
        alert_code=AlertCode.BATTERY_WARNING,
        severity=AlertSeverity.WARNING,
        component=component,
        message=f"Battery low: {battery_level:.1f}% (warning threshold: {warning_threshold}%)",
        details={
            "battery_level": battery_level,
            "warning_threshold": warning_threshold,
        },
        recovery_action="Return to charging station soon",
        auto_recoverable=False,
    )


def create_battery_critical_alert(
    component: str, battery_level: float, critical_threshold: float = 5.0
) -> DriverAlert:
    """Create a critical battery alert (CRITICAL).

    Args:
        component: Component reporting battery level
        battery_level: Current battery level (percentage)
        critical_threshold: Critical threshold (default: 5%)

    Returns:
        DriverAlert for critical battery
    """
    return DriverAlert(
        alert_code=AlertCode.BATTERY_CRITICAL,
        severity=AlertSeverity.CRITICAL,
        component=component,
        message=f"Battery critically low: {battery_level:.1f}% (critical: {critical_threshold}%)",
        details={
            "battery_level": battery_level,
            "critical_threshold": critical_threshold,
        },
        recovery_action="Return to charging station or stop operations immediately",
        auto_recoverable=False,
    )


def create_overtemp_alert(
    component: str, temperature: float, limit: float
) -> DriverAlert:
    """Create an overtemperature alert (CRITICAL).

    Args:
        component: Component that is overheating
        temperature: Current temperature
        limit: Temperature limit

    Returns:
        DriverAlert for overtemperature condition
    """
    return DriverAlert(
        alert_code=AlertCode.MOTOR_OVERHEAT,
        severity=AlertSeverity.CRITICAL,
        component=component,
        message=f"Temperature {temperature}°C exceeds limit {limit}°C",
        details={"temperature": temperature, "limit": limit},
        recovery_action="Reduce load or allow cooling period",
        auto_recoverable=False,
    )


# ========================================
# ERROR LEVEL ALERT HELPERS (2xxx)
# ========================================


def create_connection_alert(
    component: str, target: str, details: dict | None = None
) -> DriverAlert:
    """Create a connection alert (ERROR).

    Args:
        component: Component that failed (e.g., "webrtc", "mqtt")
        target: Target that couldn't be reached (e.g., IP address, hostname)
        details: Optional additional details

    Returns:
        DriverAlert for connection failure
    """
    alert_code_map = {
        "webrtc": AlertCode.WEBRTC_CONNECTION_FAILED,
        "mqtt": AlertCode.MQTT_CONNECTION_FAILED,
    }

    return DriverAlert(
        alert_code=alert_code_map.get(component, AlertCode.ROBOT_UNREACHABLE),
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"Failed to connect to {target}",
        details=details or {},
        recovery_action=f"Check network connectivity to {target}",
        auto_recoverable=True,
    )


def create_disconnection_alert(
    component: str, target: str, duration_seconds: float | None = None
) -> DriverAlert:
    """Create a disconnection alert (ERROR).

    Args:
        component: Component that disconnected (e.g., "webrtc", "mqtt")
        target: Target that was disconnected from
        duration_seconds: How long the connection was active before disconnect

    Returns:
        DriverAlert for disconnection
    """
    alert_code_map = {
        "webrtc": AlertCode.WEBRTC_DISCONNECTED,
        "mqtt": AlertCode.MQTT_DISCONNECTED,
    }

    details = {"target": target}
    if duration_seconds is not None:
        details["connection_duration_seconds"] = duration_seconds

    return DriverAlert(
        alert_code=alert_code_map.get(component, AlertCode.ROBOT_UNREACHABLE),
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"Disconnected from {target}",
        details=details,
        recovery_action=f"Reconnect to {target}",
        auto_recoverable=True,
    )


def create_sensor_failure_alert(
    component: str, sensor_name: str, failure_description: str
) -> DriverAlert:
    """Create a sensor failure alert (ERROR).

    Args:
        component: Component with failed sensor
        sensor_name: Name of the failed sensor
        failure_description: Description of the failure

    Returns:
        DriverAlert for sensor failure
    """
    return DriverAlert(
        alert_code=AlertCode.SENSOR_FAILURE,
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"Sensor '{sensor_name}' failure: {failure_description}",
        details={
            "sensor_name": sensor_name,
            "failure_description": failure_description,
        },
        recovery_action="Check sensor connection and calibration",
        auto_recoverable=False,
    )


def create_lidar_failure_alert(component: str, reason: str) -> DriverAlert:
    """Create a LIDAR failure alert (ERROR).

    Args:
        component: Component with LIDAR failure
        reason: Reason for failure

    Returns:
        DriverAlert for LIDAR failure
    """
    return DriverAlert(
        alert_code=AlertCode.LIDAR_FAILURE,
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"LIDAR failure: {reason}",
        details={"reason": reason},
        recovery_action="Check LIDAR power and connection",
        auto_recoverable=False,
    )


def create_camera_failure_alert(
    component: str, camera_id: str, reason: str
) -> DriverAlert:
    """Create a camera failure alert (ERROR).

    Args:
        component: Component with camera failure
        camera_id: Identifier of the failed camera
        reason: Reason for failure

    Returns:
        DriverAlert for camera failure
    """
    return DriverAlert(
        alert_code=AlertCode.CAMERA_FAILURE,
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"Camera '{camera_id}' failure: {reason}",
        details={"camera_id": camera_id, "reason": reason},
        recovery_action="Check camera connection and drivers",
        auto_recoverable=False,
    )


def create_command_timeout_alert(
    component: str, command_type: str, timeout_ms: float
) -> DriverAlert:
    """Create a command timeout alert (ERROR).

    Args:
        component: Component that timed out
        command_type: Type of command that timed out
        timeout_ms: Timeout duration in milliseconds

    Returns:
        DriverAlert for command timeout
    """
    return DriverAlert(
        alert_code=AlertCode.COMMAND_TIMEOUT,
        severity=AlertSeverity.ERROR,
        component=component,
        message=f"Command '{command_type}' timed out after {timeout_ms}ms",
        details={"command_type": command_type, "timeout_ms": timeout_ms},
        recovery_action="Retry command or check robot responsiveness",
        auto_recoverable=True,
    )


# ========================================
# WARNING LEVEL ALERT HELPERS (3xxx)
# ========================================


def create_stale_telemetry_alert(
    component: str, age_seconds: float, max_age: float
) -> DriverAlert:
    """Create a stale telemetry alert (WARNING).

    Args:
        component: Component with stale data
        age_seconds: How old the data is
        max_age: Maximum acceptable age

    Returns:
        DriverAlert for stale telemetry
    """
    return DriverAlert(
        alert_code=AlertCode.STALE_TELEMETRY,
        severity=AlertSeverity.WARNING,
        component=component,
        message=f"Telemetry is {age_seconds:.1f}s old (max: {max_age:.1f}s)",
        details={"age_seconds": age_seconds, "max_age": max_age},
        recovery_action="Check data source connection",
        auto_recoverable=True,
    )


def create_command_rejected_alert(
    component: str, command_type: str, rejection_reason: str
) -> DriverAlert:
    """Create a command rejected alert (WARNING).

    Args:
        component: Component that rejected the command
        command_type: Type of command that was rejected
        rejection_reason: Reason for rejection

    Returns:
        DriverAlert for command rejection
    """
    return DriverAlert(
        alert_code=AlertCode.COMMAND_REJECTED,
        severity=AlertSeverity.WARNING,
        component=component,
        message=f"Command '{command_type}' rejected: {rejection_reason}",
        details={"command_type": command_type, "rejection_reason": rejection_reason},
        recovery_action="Check command validity and robot state",
        auto_recoverable=False,
    )


def create_invalid_command_alert(
    component: str, command_type: str, validation_error: str
) -> DriverAlert:
    """Create an invalid command alert (WARNING).

    Args:
        component: Component that received invalid command
        command_type: Type of invalid command
        validation_error: Validation error message

    Returns:
        DriverAlert for invalid command
    """
    return DriverAlert(
        alert_code=AlertCode.INVALID_COMMAND,
        severity=AlertSeverity.WARNING,
        component=component,
        message=f"Invalid command '{command_type}': {validation_error}",
        details={"command_type": command_type, "validation_error": validation_error},
        recovery_action="Validate command parameters",
        auto_recoverable=False,
    )


def create_config_error_alert(
    component: str, resource_type: str, resource_id: str, details: dict | None = None
) -> DriverAlert:
    """Create a configuration error alert (CRITICAL).

    Used for invalid configuration that prevents driver startup, such as
    missing or invalid digital twin UUIDs, environment IDs, or API keys.

    Args:
        component: Component with configuration error (e.g., "twin_config", "api_config")
        resource_type: Type of resource not found (e.g., "Digital Twin", "Environment")
        resource_id: ID/UUID of the missing resource
        details: Additional context (environment UUID, API URL, error message, etc.)

    Returns:
        DriverAlert for configuration error

    Example:
        >>> alert = create_config_error_alert(
        ...     component="twin_config",
        ...     resource_type="Digital Twin",
        ...     resource_id="abc-123-def",
        ...     details={"environment_uuid": "env-456", "error": "HTTP 404"}
        ... )
    """
    return DriverAlert(
        alert_code=AlertCode.CONFIGURATION_ERROR,
        severity=AlertSeverity.CRITICAL,
        component=component,
        message=f"{resource_type} not found: {resource_id}",
        details=details or {},
        recovery_action="Verify configuration (environment variables, .env file) and ensure resource exists in backend",
        auto_recoverable=False,
    )


def create_driver_lifecycle_alert(
    component: str,
    *,
    from_state: str,
    to_state: str,
) -> DriverAlert:
    """INFO alert when a driver lifecycle state changes."""
    return DriverAlert(
        alert_code=AlertCode.DRIVER_LIFECYCLE,
        severity=AlertSeverity.INFO,
        component=component,
        message=f"Driver lifecycle: {from_state} → {to_state}",
        details={"from_state": from_state, "to_state": to_state},
        auto_recoverable=True,
    )


def create_driver_active_alert(component: str, *, details: dict[str, Any] | None = None) -> DriverAlert:
    """INFO alert when the driver enters ACTIVE and begins publishing."""
    return DriverAlert(
        alert_code=AlertCode.DRIVER_ACTIVE,
        severity=AlertSeverity.INFO,
        component=component,
        message="Driver active",
        details=details or {},
        auto_recoverable=True,
    )
