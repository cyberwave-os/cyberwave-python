"""JointCommandBufferMixin — thread-safe MQTT→executor joint-command hand-off.

MQTT tele commands arrive on the driver asyncio loop; the ROS executor thread
publishes them in ``tick()``. This mixin owns the ``CommandInbox`` (lock, merge,
stale-timestamp guard) and the ``_home_pending`` flag so each driver does not
hand-roll them. Compose with any driver::

    class MyDriver(JointCommandBufferMixin, BaseROS2Driver):
        def __init__(self, ...):
            super().__init__(...)
            self._init_joint_command_buffer()
        def _publish_joint_command(self, positions): ...   # write seam
        def _return_to_home(self): ...                      # home seam
        def tick(self): self.pump_joint_commands()

The mixin also owns teleop observability (``tele_mqtt_rx`` / ``tele_ros_publish``
counters, ``tele_log_due`` 5 s throttle) and an opt-in home-on-teardown helper:
set ``HOME_ON_TEARDOWN = True`` and call ``home_if_teardown_enabled()`` from
``deactivate``/``shutdown``/``request_shutdown`` instead of hand-rolling a
``try/except _return_to_home()`` in each teardown path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..interface.command_inbox import CommandInbox

logger = logging.getLogger(__name__)


def coerce_joint_timestamp(raw: Any) -> float | None:
    """Best-effort float timestamp; ``None`` when absent or unparseable."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid joint command timestamp %r; ignoring", raw)
        return None


class HomePositionNotDefinedError(NotImplementedError):
    """A driver requested home but declared no ``_home_position`` and did not override
    ``_return_to_home``."""


class JointCommandBufferMixin:
    """Buffered joint-command hand-off with a home-request short-circuit."""

    HOME_ON_TEARDOWN: bool = False
    """When True, ``home_if_teardown_enabled()`` returns the arm home during teardown."""

    def _init_joint_command_buffer(self) -> None:
        """Initialize the inbox + home flag + teleop counters. Call once from ``__init__``."""
        self._tele_inbox = CommandInbox()
        self._home_pending = False
        self.tele_mqtt_rx = 0
        self.tele_ros_publish = 0
        self._last_tele_status_log_at = 0.0

    def submit_joint_targets(
        self, positions: dict[str, float], *, timestamp: Any = None
    ) -> bool:
        """Merge *positions* into the pending set. Returns ``False`` if dropped as stale."""
        accepted = self._tele_inbox.submit(
            positions, timestamp=coerce_joint_timestamp(timestamp)
        )
        if accepted:
            self.tele_mqtt_rx += 1
        return accepted

    def request_home(self) -> None:
        """Request a home return on the next ``pump_joint_commands()`` tick."""
        self._home_pending = True

    def cancel_pending(self) -> None:
        """Discard queued commands and clear any pending home request, keeping the staleness
        watermark (unlike ``_tele_inbox.reset()``, which also clears the watermark)."""
        self._tele_inbox.clear()
        self._home_pending = False

    def pump_joint_commands(self) -> None:
        """Drain pending commands and publish (call from the driver ``tick()``)."""
        if self._home_pending:
            self._home_pending = False
            self._return_to_home()
            self._tele_inbox.drain()  # discard commands queued during home return
            return
        pending = self._tele_inbox.drain()
        if not pending:
            return
        # Increment before publishing so a driver's _publish_joint_command hook
        # (which typically logs) observes the count including this event —
        # mirrors submit_joint_targets, which increments before returning.
        self.tele_ros_publish += 1
        self._publish_joint_command(pending)

    def tele_log_due(self, count: int, *, now: float | None = None) -> bool:
        """True (and stamps the throttle) on the first event or ≥5 s since the last log."""
        t = time.monotonic() if now is None else now
        if count == 1 or t - self._last_tele_status_log_at >= 5.0:
            self._last_tele_status_log_at = t
            return True
        return False

    def home_if_teardown_enabled(self) -> None:
        """Return to home during teardown when ``HOME_ON_TEARDOWN`` is set (errors swallowed)."""
        if not type(self).HOME_ON_TEARDOWN:
            return
        try:
            self._return_to_home()
        except Exception:
            logger.warning("home-on-teardown: _return_to_home() failed", exc_info=True)

    # ── standard teleop observability ─────────────────────────────────────────

    def teleop_log_extra(self) -> str:
        """Vendor hook: appended verbatim to publish log lines (default '')."""
        return ""

    def teleop_status_fields(self) -> dict[str, Any]:
        """Standard teleop fields for driver_info/edge_health snapshots.

        Reads composed surfaces defensively: ``operation_mode`` (registry
        mixin) and ``controller_policy_snapshot()`` when present.
        """
        mode = getattr(self, "operation_mode", None)
        mode_value = getattr(mode, "value", None)
        fields: dict[str, Any] = {
            "teleop_robot_status": (
                "teleop" if mode_value in ("teleop_local", "teleop_remote") else "idle"
            ),
            "teleop_operation_mode": mode_value,
            "teleop_mqtt_rx": self.tele_mqtt_rx,
            "teleop_ros_publish": self.tele_ros_publish,
            "teleop_pending_joints": self._tele_inbox.pending_count,
        }
        snapshot = getattr(self, "controller_policy_snapshot", None)
        if callable(snapshot):
            fields.update(snapshot())
        return fields

    def log_tele_rx(self, payload: dict[str, Any], update: dict[str, float]) -> None:
        """Throttled RX log (first event, then >=5s apart via tele_log_due)."""
        if not self.tele_log_due(self.tele_mqtt_rx):
            return
        mode = getattr(getattr(self, "operation_mode", None), "value", None)
        logger.info(
            "MQTT joint/update tele RX #%d (source_type=%r, operation_mode=%s, "
            "joints=%s) — queued for publish",
            self.tele_mqtt_rx,
            payload.get("source_type"),
            mode,
            sorted(update.keys()),
        )

    def log_tele_publish(self, update: dict[str, float]) -> None:
        """Throttled publish log; appends teleop_log_extra()."""
        if not self.tele_log_due(self.tele_ros_publish):
            return
        logger.info(
            "tele publish #%d (%d joint(s): %s)%s",
            self.tele_ros_publish,
            len(update),
            sorted(update.keys()),
            self.teleop_log_extra(),
        )

    def log_tele_ignored(self, payload: dict[str, Any]) -> None:
        """Warn on unparseable payloads, only for the first few RX events."""
        if self.tele_mqtt_rx > 3:
            return
        keys = sorted(
            k for k in payload if k not in {"source_type", "timestamp", "session_id"}
        )
        logger.warning(
            "MQTT joint/update ignored (no controllable joint positions parsed): "
            "source_type=%r, keys=%s",
            payload.get("source_type"),
            keys,
        )

    # ── seams the driver implements ───────────────────────────────────────────

    def _publish_joint_command(self, positions: dict[str, float]) -> None:
        raise NotImplementedError

    def _return_to_home(self) -> None:
        """Publish ``self._home_position`` (a ``{joint: position}`` dict). Override for
        vendor-specific delivery (e.g. repeated publishes). Raises if neither is provided."""
        home = getattr(self, "_home_position", None)
        if home is None:
            raise HomePositionNotDefinedError(
                f"{type(self).__name__} requested home but defined no _home_position; "
                "set self._home_position = {joint: pos, ...} or override _return_to_home()."
            )
        self._publish_joint_command(dict(home))
