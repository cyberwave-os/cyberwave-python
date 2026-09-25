"""Driver-facing alert API: raise/resolve via :class:`AlertManager` and create
twin-UI alerts.

:class:`DriverAlertsMixin` is mixed into
:class:`~cyberwave.driver.base.BaseDriver`. Lifecycle alerts are emitted
automatically (see :mod:`cyberwave.driver.lifecycle`); these methods are for a
driver's own operational notices.

**Host contract** — expects on ``self``: ``_alert_manager`` (``AlertManager``)
and ``_twin`` (twin handle or None).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Severities eligible for timed auto-resolution. Never auto-resolve error/critical
# notices — those must persist until an operator (or the driver) clears them.
_AUTO_RESOLVE_SEVERITIES = frozenset({"info", "warning"})


class DriverAlertsMixin:
    """Raise/resolve alerts and create twin-visible notices."""

    def raise_alert(self, alert: Any) -> None:
        """Raise an alert via the shared :class:`AlertManager`."""
        self._alert_manager.raise_alert(alert)

    async def raise_alert_async(self, alert: Any) -> Any:
        """Async-safe version of :meth:`raise_alert`. Use inside coroutines."""
        return await self._alert_manager.raise_alert_async(alert)

    def resolve_alert(self, component: str, alert_code: Any) -> None:
        """Resolve an alert via the shared :class:`AlertManager`."""
        self._alert_manager.resolve_alert(component, alert_code)

    def create_twin_alert(
        self,
        name: str,
        *,
        description: str = "",
        alert_type: str = "driver_notice",
        severity: str = "info",
        source_type: str = "edge",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
        auto_resolve_after: float | None = None,
        _async_dispatch: bool = True,
    ) -> Any | None:
        """Create an alert on this driver's twin (``twin.alerts.create`` API).

        Prefer this for lifecycle and operational notices that must appear on the
        twin in the platform UI. When called from the asyncio lifecycle loop, the
        HTTP request runs in a background thread so the tick loop is not blocked.

        ``auto_resolve_after`` (seconds): when set and the severity is ``info`` or
        ``warning``, the alert is resolved automatically after that delay via a
        daemon timer. Transient state-change notices (controller assigned, driver
        active, …) use this so they self-clear instead of piling up in the UI.
        ``error``/``critical`` alerts ignore it and stay until explicitly cleared.
        """
        if _async_dispatch:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                threading.Thread(
                    target=self.create_twin_alert,
                    kwargs={
                        "name": name,
                        "description": description,
                        "alert_type": alert_type,
                        "severity": severity,
                        "source_type": source_type,
                        "metadata": metadata,
                        "force": force,
                        "auto_resolve_after": auto_resolve_after,
                        "_async_dispatch": False,
                    },
                    name=f"twin-alert-{alert_type}",
                    daemon=True,
                ).start()
                return None

        twin = self._twin
        if twin is None:
            logger.debug("create_twin_alert skipped: no twin bound yet")
            return None
        alerts_api = getattr(twin, "alerts", None)
        if alerts_api is None or not hasattr(alerts_api, "create"):
            logger.debug("create_twin_alert skipped: twin.alerts unavailable")
            return None
        try:
            created = alerts_api.create(
                name=name,
                description=description,
                alert_type=alert_type,
                severity=severity,
                source_type=source_type,
                metadata=metadata,
                force=force,
            )
            logger.info(
                "Twin alert created: type=%s severity=%s uuid=%s",
                alert_type,
                severity,
                getattr(created, "uuid", created),
            )
            self._schedule_alert_auto_resolve(created, severity, auto_resolve_after)
            return created
        except Exception:
            logger.warning(
                "create_twin_alert failed (type=%s)", alert_type, exc_info=True
            )
            return None

    def _schedule_alert_auto_resolve(
        self, created: Any, severity: str, auto_resolve_after: float | None
    ) -> None:
        """Resolve *created* after ``auto_resolve_after`` s (info/warning only)."""
        if not auto_resolve_after or auto_resolve_after <= 0:
            return
        if (severity or "").strip().lower() not in _AUTO_RESOLVE_SEVERITIES:
            return  # error/critical must persist
        resolve = getattr(created, "resolve", None)
        if not callable(resolve) or not getattr(created, "uuid", None):
            return

        def _auto_resolve() -> None:
            try:
                resolve()
                logger.debug(
                    "Auto-resolved twin alert uuid=%s after %ss",
                    getattr(created, "uuid", None),
                    auto_resolve_after,
                )
            except Exception:
                logger.debug("auto-resolve of twin alert failed", exc_info=True)

        timer = threading.Timer(auto_resolve_after, _auto_resolve)
        timer.daemon = True
        timer.start()
