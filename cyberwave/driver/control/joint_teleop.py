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

Two opt-in safety seams, inert unless a driver uses them:

* **MQTT-outage motion stop** -- set ``STOP_MOTION_ON_MQTT_OUTAGE = True``.
  The first time a broker disconnect is observed (even one that began and
  ended between two ticks, via the client's ``disconnect_count``) the driver's
  ``_stop_motion()`` runs once, and listener callbacks are refused while the
  link is down. The attached controller and operation mode are kept, so the
  first fresh command after reconnecting works without re-attaching.
* **Joint-target refusal** -- override ``_joint_target_refusal(merged)`` to veto
  a target (e.g. a collision check) on the MERGED posture before
  ``_publish_joint_command``. ``pump_joint_commands`` routes through
  ``publish_joint_targets``, the single gated publish path.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from ..interface.command_inbox import CommandInbox
from ..support.throttle import Throttle
from ..support.utils import driver_logger

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

    STOP_MOTION_ON_MQTT_OUTAGE: bool = False
    """When True, an MQTT outage stops motion once (``_stop_motion``) and gates input."""

    JOINT_TARGET_REFUSAL_LOG_INTERVAL_S: float = 5.0
    """Throttle for the refused-target warning (teleop restreams the same target)."""

    def _init_joint_command_buffer(self) -> None:
        """Initialize the inbox + home flag + teleop counters. Call once from ``__init__``."""
        self._tele_inbox = CommandInbox()
        self._home_pending = False
        self.tele_mqtt_rx = 0
        self.tele_ros_publish = 0
        self._last_tele_status_log_at = 0.0
        self._mqtt_motion_lock = threading.RLock()
        self._mqtt_disconnect_count = 0
        self._mqtt_motion_paused = False
        self.joint_target_refusals = 0
        self._joint_target_refusal_throttle = Throttle(
            type(self).JOINT_TARGET_REFUSAL_LOG_INTERVAL_S
        )

    def submit_joint_targets(
        self, positions: dict[str, float], *, timestamp: Any = None
    ) -> bool:
        """Merge *positions* into the pending set. Returns ``False`` if dropped as stale
        (or, with ``STOP_MOTION_ON_MQTT_OUTAGE``, while the MQTT link is down)."""
        if not self._mqtt_motion_ready():
            return False
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

    def pump_joint_commands(self) -> dict[str, float] | None:
        """Drain and publish pending commands; return the update that was drained.

        The drained update goes through :meth:`publish_joint_targets`, so a
        refused or outage-interrupted target is consumed, never replayed.
        """
        if not self._mqtt_motion_ready():
            return None
        if self._home_pending:
            # A pending home wins its tick: targets queued before it are older
            # than the request and are discarded. Drain BEFORE the hook, so
            # targets submitted while it publishes survive for the next tick.
            self._home_pending = False
            self._tele_inbox.drain()
            self._return_to_home()
            return None
        pending = self._tele_inbox.drain()
        if not pending:
            return None
        # Increment before publishing so a driver's _publish_joint_command hook
        # (which typically logs) observes the count including this event —
        # mirrors submit_joint_targets, which increments before returning.
        self.tele_ros_publish += 1
        self.publish_joint_targets(pending)
        return pending

    # ── gated publish path (joint-target refusal + MQTT outage) ──────────────

    def publish_joint_targets(self, positions: dict[str, float]) -> bool:
        """Publish *positions* through ``_publish_joint_command`` unless refused.

        The one seam every commanded joint target should pass (teleop, queued
        waypoints, gripper-only updates): the refusal check runs on the MERGED
        posture (:meth:`_joint_target_baseline` plus *positions*), because a
        partial teleop update is delivered downstream as a complete posture and
        checking the delta alone answers for a configuration nobody commands.

        A refused target is dropped BEFORE ``_publish_joint_command`` so it
        leaves no residue in the commanded state -- merging first would let the
        next partial update inherit the offending joint and walk the arm into
        the same configuration one axis at a time.

        With ``STOP_MOTION_ON_MQTT_OUTAGE`` the MQTT link is checked before and
        after the refusal check: a target drained before an outage (or an
        outage+reconnect that happened during the check) is not published.

        Home and parking postures deliberately do NOT use this path: refusing
        to park the arm is not a safe outcome.
        """
        epoch = self.mqtt_motion_epoch()
        if not self.mqtt_motion_still_ready(epoch):
            return False
        if self.check_joint_targets(positions) is not None:
            return False
        if not self.mqtt_motion_still_ready(epoch):
            return False
        self._publish_joint_command(positions)
        return True

    def check_joint_targets(self, positions: Mapping[str, float]) -> str | None:
        """Why *positions* must not be commanded (counted + throttled log), or None.

        A seam that raises fails OPEN (logged as an error): "we could not
        check" is not a refusal, and a solver fault must not stop teleop.
        """
        if (
            type(self)._joint_target_refusal
            is JointCommandBufferMixin._joint_target_refusal
        ):
            return None  # not overridden: nothing to check, nothing to merge
        merged = {**self._joint_target_baseline(), **positions}
        try:
            refusal = self._joint_target_refusal(merged)
        except Exception as exc:  # noqa: BLE001 - a checker fault must not stop teleop
            self._teleop_logger().error(
                f"Joint-target check failed to evaluate a command; publishing it "
                f"unchecked: {type(exc).__name__}: {exc}"
            )
            return None
        if refusal is None:
            return None
        self.joint_target_refusals = getattr(self, "joint_target_refusals", 0) + 1
        self._log_joint_target_refusal(positions, refusal)
        return refusal

    def _joint_target_baseline(self) -> dict[str, float]:
        """The posture a partial update is merged onto (default: ``_joint_states``)."""
        return dict(getattr(self, "_joint_states", None) or {})

    def _joint_target_refusal(self, merged_targets: dict[str, float]) -> str | None:
        """Vendor seam: why the MERGED posture must not be commanded, or None."""
        del merged_targets
        return None

    def _log_joint_target_refusal(
        self, positions: Mapping[str, float], reason: str
    ) -> None:
        throttle = getattr(self, "_joint_target_refusal_throttle", None)
        if throttle is not None and not throttle.ready():
            return
        requested = ", ".join(
            f"{name}={float(positions[name]):.3f}" for name in sorted(positions)
        )
        self._teleop_logger().warning(
            f"Joint command refused: {reason}. Requested change: {requested}. "
            "Nothing was published and the commanded state is unchanged "
            f"(refusals={self.joint_target_refusals})"
        )

    # ── MQTT-outage motion stop (opt-in) ──────────────────────────────────────

    def _mqtt_motion_ready(self) -> bool:
        """True when motion may proceed; stops motion once per MQTT outage.

        Always True unless ``STOP_MOTION_ON_MQTT_OUTAGE`` is set, or when no
        MQTT client is bound (local ROS harness, manifest export). An outage is
        detected from the client's ``disconnect_count`` as well as its live
        ``connected`` flag, so a disconnect that began and ended between two
        ticks still stops the arm: the commands queued before it describe a
        world the operator could no longer see.
        """
        if not type(self).STOP_MOTION_ON_MQTT_OUTAGE:
            return True
        client = getattr(self, "_cw", None)
        if client is None:
            return True
        lock = getattr(self, "_mqtt_motion_lock", None)
        if lock is None:
            lock = self._mqtt_motion_lock = threading.RLock()
        with lock:
            mqtt = client.mqtt
            connected = bool(mqtt.connected)
            count = getattr(mqtt, "disconnect_count", 0)
            if count != self.mqtt_motion_epoch() or (
                not connected and not getattr(self, "_mqtt_motion_paused", False)
            ):
                self._mqtt_disconnect_count = count
                self._mqtt_motion_paused = True
                self._stop_motion()
                self._teleop_logger().warning(
                    "MQTT interrupted: motion stopped; controller attachment retained"
                )
            if not connected:
                return False
            if getattr(self, "_mqtt_motion_paused", False):
                self._mqtt_motion_paused = False
                self._teleop_logger().info(
                    "MQTT restored: attached controller ready for fresh commands"
                )
            return True

    def mqtt_motion_epoch(self) -> int:
        """The disconnect count motion was last validated against.

        Capture it before a multi-step publish (e.g. a repeated home burst) and
        pass it to :meth:`mqtt_motion_still_ready` between steps.
        """
        return getattr(self, "_mqtt_disconnect_count", 0)

    def mqtt_motion_still_ready(self, epoch: int) -> bool:
        """``_mqtt_motion_ready()`` and no outage since *epoch* was captured."""
        return self._mqtt_motion_ready() and epoch == self.mqtt_motion_epoch()

    def sync_mqtt_disconnect_count(self) -> None:
        """Adopt the client's current count without stopping (call once wired).

        Disconnects that happened before the driver was wired to MQTT are not
        outages of a session that had motion to stop.
        """
        client = getattr(self, "_cw", None)
        if client is not None:
            self._mqtt_disconnect_count = getattr(client.mqtt, "disconnect_count", 0)

    async def _invoke_listener(self, callback: Any, envelope: dict[str, Any]) -> None:
        """Refuse listener input while the MQTT link is down (when opted in).

        Readiness is evaluated before the handler runs, so a reconnect is
        observed (and motion stopped) before its first input is accepted, even
        if the outage and recovery both fell between two ticks.
        ``controller_changed`` always passes: detaching a controller must work
        during an outage.
        """
        if not (
            envelope.get("command") == "controller_changed" or self._mqtt_motion_ready()
        ):
            return
        parent = getattr(super(), "_invoke_listener", None)
        if parent is not None:
            await parent(callback, envelope)
            return
        result = callback(envelope)
        if inspect.isawaitable(result):
            await result

    def _stop_motion(self) -> None:
        """Stop commanded motion, keeping controller attachment and mode.

        Default: drop queued commands and any pending home. Drivers with
        trajectories, planners or a hold command override this (and usually
        call ``super()._stop_motion()`` or ``cancel_pending()``).
        """
        self.cancel_pending()

    def _teleop_logger(self) -> Any:
        """The driver's ROS logger when it has one, else this module's logger."""
        return driver_logger(self, logger)

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
            "joints=%s) — latest targets received (older targets are replaced)",
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
