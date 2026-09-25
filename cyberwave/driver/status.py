"""Driver lifecycle **status**: the state enum and the state machine.

This is pure status — no cloud, no alerts. :class:`LifecycleStateMixin` owns the
current state and ``_transition_to``; on every transition it logs, emits a
``driver_info`` telemetry field, and fires the :meth:`_on_lifecycle_transition`
hook. Reactions to transitions (e.g. cloud alerts in
:class:`~cyberwave.driver.cloud.lifecycle_alerts.LifecycleAlertsMixin`, or a
driver's own per-status behavior) subscribe by overriding that hook — the state
machine itself stays decoupled from them.

**Host contract** — expects on ``self``: ``_lifecycle_state`` (set in
``BaseDriver.__init__``) and ``_emit_driver_info(**fields)``.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DriverLifecycleState(str, Enum):
    """Lifecycle states for a Cyberwave edge driver.

    Currently tracked with a plain enum and the ``_transition_to`` helper on
    :class:`LifecycleStateMixin`. If remote lifecycle management (activate/
    deactivate commands from the backend) or reconnect/retry loops that need to
    re-enter states are ever required, this can be migrated to the
    ``transitions`` library (``AsyncMachine``) without changing the state names
    or the public ``lifecycle_state`` property.
    """

    UNCONFIGURED = "unconfigured"  # Initial state before run() is called
    CONFIGURING = "configuring"  # Cloud connection + resource initialisation
    CONNECTING = "connecting"  # Transport connection (WebRTC / serial / …)
    INACTIVE = "inactive"  # Connected and ready; control loop not yet started
    ACTIVE = "active"  # Control loop running
    RECONNECTING = "reconnecting"  # Transport reconnect in progress
    DEACTIVATING = "deactivating"  # Teardown in progress
    FINALIZED = "finalized"  # Teardown complete; process will exit
    ERROR = "error"  # Unrecoverable failure; teardown will still run


class LifecycleStateMixin:
    """The driver lifecycle state machine (status only)."""

    @property
    def lifecycle_state(self) -> DriverLifecycleState:
        """Current lifecycle state of the driver."""
        return self._lifecycle_state

    def _transition_to(self, state: DriverLifecycleState) -> None:
        """Move to *state*: log, emit telemetry, then fire the transition hook."""
        from_state = self._lifecycle_state
        if from_state == state:
            return
        logger.info("[STATE] %s → %s", from_state.value, state.value)
        self._lifecycle_state = state
        self._emit_driver_info(lifecycle_state=state.value)
        self._on_lifecycle_transition(from_state, state)

    def _on_lifecycle_transition(
        self,
        from_state: DriverLifecycleState,
        to_state: DriverLifecycleState,
    ) -> None:
        """Hook fired on every lifecycle transition (default: no-op).

        Override (or mix in a class that overrides) this to run status-specific
        behavior on *any* state — including ``RECONNECTING`` / ``ERROR`` /
        ``FINALIZED`` that have no dedicated author hook. The cloud alert layer
        overrides it to publish lifecycle alerts.
        """
