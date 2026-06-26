"""Cloud reaction to driver lifecycle transitions: alerts + twin-UI notices.

The lifecycle **state machine** lives in :mod:`cyberwave.driver.status`; this is
one *consumer* of it. :class:`LifecycleAlertsMixin` overrides
:meth:`~cyberwave.driver.status.LifecycleStateMixin._on_lifecycle_transition` to
turn each state change into ``AlertManager`` alerts and a twin-UI notice. Mixed
into :class:`~cyberwave.driver.base.BaseDriver` **before** ``LifecycleStateMixin``
so this override wins over the default no-op.

**Host contract** — expects on ``self``: ``_lifecycle_state``,
``_lifecycle_twin_pending_notice``, ``_alert_manager``, ``_twin``,
``registry_id``, ``twin_uuid``, ``create_twin_alert(...)``.
"""

from __future__ import annotations

import logging

from ..status import DriverLifecycleState
from .alerts import (
    AlertCode,
    create_driver_active_alert,
    create_driver_lifecycle_alert,
)

logger = logging.getLogger(__name__)


# Twin-visible notices for :meth:`LifecycleAlertsMixin._notify_lifecycle_twin_alert`.
#
# Only the states an operator actually cares about emit a twin-UI alert. The
# transient startup/teardown states (CONFIGURING, CONNECTING, INACTIVE,
# DEACTIVATING) are still tracked via the internal AlertManager and logs, but no
# longer spam the twin with ~6 notices per startup. Non-critical notices below
# auto-resolve after a few seconds so they self-clear; ERROR persists.
_LIFECYCLE_TWIN_ALERTS: dict[DriverLifecycleState, tuple[str, str, str]] = {
    DriverLifecycleState.ACTIVE: (
        "Driver active",
        "Publishing telemetry and sensor streams.",
        "info",
    ),
    DriverLifecycleState.RECONNECTING: (
        "Driver reconnecting",
        "Reconnecting to the Cyberwave MQTT broker.",
        "warning",
    ),
    DriverLifecycleState.FINALIZED: (
        "Driver stopped",
        "Edge driver process has shut down.",
        "info",
    ),
    DriverLifecycleState.ERROR: (
        "Driver error",
        "The driver entered an error state; check logs for details.",
        "error",
    ),
}

# Seconds after which a non-critical lifecycle twin notice self-resolves.
_LIFECYCLE_TWIN_AUTO_RESOLVE_S = 5.0


class LifecycleAlertsMixin:
    """Publishes lifecycle alerts + twin notices on each state transition.

    Implements the ``_on_lifecycle_transition`` hook from
    :class:`~cyberwave.driver.status.LifecycleStateMixin`.
    """

    def _on_lifecycle_transition(
        self,
        from_state: DriverLifecycleState,
        to_state: DriverLifecycleState,
    ) -> None:
        """Raise lifecycle alerts via :class:`AlertManager` and twin API when bound."""
        component = self._lifecycle_alert_component()
        if from_state != DriverLifecycleState.UNCONFIGURED:
            self._alert_manager.resolve_alert(component, AlertCode.DRIVER_LIFECYCLE)

        self._alert_manager.raise_alert(
            create_driver_lifecycle_alert(
                component,
                from_state=from_state.value,
                to_state=to_state.value,
            )
        )

        if to_state == DriverLifecycleState.ACTIVE:
            self._alert_manager.raise_alert(
                create_driver_active_alert(
                    component,
                    details={
                        "registry_id": self.registry_id,
                        "twin_uuid": self.twin_uuid,
                    },
                )
            )
        elif to_state in {
            DriverLifecycleState.DEACTIVATING,
            DriverLifecycleState.FINALIZED,
            DriverLifecycleState.ERROR,
        }:
            self._alert_manager.resolve_alert(component, AlertCode.DRIVER_ACTIVE)

        self._notify_lifecycle_twin_alert(from_state, to_state)

    def _lifecycle_alert_component(self) -> str:
        """Stable alert component id (class name)."""
        return type(self).__name__

    def _notify_lifecycle_twin_alert(
        self,
        from_state: DriverLifecycleState,
        to_state: DriverLifecycleState,
    ) -> None:
        """Create a twin alert for the platform UI when the twin handle is available."""
        copy = _LIFECYCLE_TWIN_ALERTS.get(to_state)
        if copy is None:
            return
        if self._twin is None:
            self._lifecycle_twin_pending_notice = True
            return
        name, description, severity = copy
        # Non-critical notices self-clear; ERROR (and any future critical) persists.
        auto_resolve_after = (
            None if severity in {"error", "critical"} else _LIFECYCLE_TWIN_AUTO_RESOLVE_S
        )
        self.create_twin_alert(
            name,
            description=description,
            alert_type="driver_lifecycle",
            severity=severity,
            metadata={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "registry_id": self.registry_id,
                "driver_class": self._lifecycle_alert_component(),
            },
            auto_resolve_after=auto_resolve_after,
            _async_dispatch=False,
        )
        self._lifecycle_twin_pending_notice = False

    def _sync_lifecycle_alerts_after_connect(self) -> None:
        """Push lifecycle alerts raised before MQTT/twin were ready."""
        self._alert_manager.sync_lifecycle_alerts_to_backend(
            self._lifecycle_alert_component()
        )
        if self._lifecycle_twin_pending_notice:
            self._notify_lifecycle_twin_alert(
                DriverLifecycleState.UNCONFIGURED,
                self._lifecycle_state,
            )
