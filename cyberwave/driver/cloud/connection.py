"""Cyberwave cloud connectivity for a driver: MQTT connect, twin binding,
backend-alert enablement, and the reconnect watchdog.

:class:`CloudConnectionMixin` is mixed into
:class:`~cyberwave.driver.base.BaseDriver`. It is the single place that talks to
the :class:`~cyberwave.Cyberwave` SDK client; the driver template only decides
*when* to call :meth:`_connect_cloud_async` / :meth:`_reconnect_loop_async`.

**Host contract** — expects on ``self``: ``_twin_prebound``, ``_cw``, ``_twin``,
``_alert_manager``, ``_shutdown``/``_connection_lost`` (asyncio events),
``RECONNECT_MAX_ATTEMPTS``/``RECONNECT_BACKOFF_BASE``/``RECONNECT_BACKOFF_MAX``,
``registry_id``, ``twin_uuid``, ``_emit_driver_info``,
``_sync_lifecycle_alerts_after_connect``, ``_transition_to``, ``on_reconnect``,
``_wire_interface_from_registry``/``_unwire_interface_from_registry``/
``_activate_registry_zenoh``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from ..status import DriverLifecycleState
from .alerts import (
    AlertCode,
    create_config_error_alert,
    create_connection_alert,
)

logger = logging.getLogger(__name__)


class CloudConnectionMixin:
    """MQTT/twin connect + reconnect for the driver lifecycle."""

    async def _connect_cloud_async(self) -> None:
        """Connect MQTT, bind the digital twin, and enable backend alerts.

        When ``twin=`` was passed to ``__init__``, reuses ``twin.client`` when present
        instead of fetching the twin again.
        """
        if self._twin_prebound:
            if self._cw is None:
                self._cw = await self._connect_mqtt_async()
            else:
                await self._ensure_mqtt_connected_async()
        else:
            self._cw = await self._connect_mqtt_async()
            self._twin = self._fetch_twin()
        self._enable_backend_alerts()

    async def _ensure_mqtt_connected_async(self) -> None:
        """Wait until ``self._cw.mqtt`` is connected (existing SDK client)."""
        assert self._cw is not None
        mqtt = self._cw.mqtt
        if getattr(mqtt, "connected", False):
            return
        if hasattr(mqtt, "connect"):
            mqtt.connect()
        deadline = time.monotonic() + 10.0
        while not getattr(mqtt, "connected", False) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if not getattr(mqtt, "connected", False):
            raise TimeoutError("MQTT connection timeout on existing Cyberwave client")

    async def _connect_mqtt_async(self) -> Any:
        """Initialize MQTT via :class:`~cyberwave.Cyberwave` (env-driven config).

        Raises :exc:`SystemExit` with code 1 on failure.
        """
        from cyberwave import Cyberwave  # lazy: avoids heavy imports at module load

        from ..support.utils import get_sdk_version

        sdk_version = get_sdk_version()
        logger.info("Cyberwave Python SDK version: %s", sdk_version or "unknown")

        from cyberwave.constants import SOURCE_TYPE_EDGE

        max_attempts = 3
        timeout = 10.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "Attempt %d/%d: Cyberwave MQTT connect (source_type=edge)...",
                    attempt,
                    max_attempts,
                )
                # Same defaults as notebooks/CLI: CyberwaveConfig reads CYBERWAVE_* env.
                cyberwave = Cyberwave(source_type=SOURCE_TYPE_EDGE)
                cyberwave.mqtt.connect()
                deadline = time.monotonic() + timeout
                while not cyberwave.mqtt.connected and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                if cyberwave.mqtt.connected:
                    mqtt_proto = (
                        "MQTT v5"
                        if getattr(cyberwave.mqtt, "is_mqtt_v5", False)
                        else "MQTT v3.1.1"
                    )
                    logger.info(
                        "[SUCCESS] MQTT connection established via Cyberwave SDK (%s)",
                        mqtt_proto,
                    )
                    if not getattr(cyberwave.mqtt, "is_mqtt_v5", False):
                        logger.info(
                            "Tip: set CYBERWAVE_MQTT_PROTOCOL=5 so joint/update "
                            "listeners use no_local and skip echoing edge publishes"
                        )
                    self._alert_manager.resolve_alert(
                        "mqtt", AlertCode.MQTT_CONNECTION_FAILED
                    )
                    return cyberwave
                last_error = TimeoutError(f"MQTT connection timeout after {timeout}s")
                logger.warning("[ATTEMPT %d] %s", attempt, last_error)
            except SystemExit:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("[ATTEMPT %d] SDK connection failed: %s", attempt, exc)
            if attempt < max_attempts:
                await asyncio.sleep(2)

        logger.error("MQTT connection failed after %d attempts", max_attempts)
        if last_error:
            logger.error("Last error: %s", last_error)
        from cyberwave.config import get_config

        alert = create_connection_alert(
            component="mqtt",
            target=get_config().mqtt_host or "mqtt",
            details={"error": str(last_error) if last_error else "connect failed"},
        )
        self._alert_manager.raise_alert(alert)
        raise SystemExit(1) from last_error

    def _fetch_twin(self) -> Any:
        """Fetch the robot twin handle from the API.

        Raises :exc:`SystemExit` with code 1 if the twin cannot be found.
        Skips the API when a twin was passed to ``__init__``.
        """
        if self._twin_prebound and self._twin is not None:
            return self._twin
        assert self._cw is not None, "_fetch_twin called before _connect_mqtt"
        twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID", "").strip()
        try:
            logger.info(
                f"Fetching robot twin from API ({self._cw.config.base_url})..."
            )
            t0 = time.monotonic()
            twin = self._cw.twins.get(twin_uuid)
            self._emit_driver_info()
            logger.info(
                f"Found robot twin: {twin.name} with {twin.uuid}/{twin.environment_id} (took {time.monotonic() - t0:.2f}s)"
            )
            return twin

        except Exception as e:
            logger.error(f"[FAILED] Digital twin not found: {twin_uuid}")
            logger.error(f"   API URL: {self._cw.config.base_url}")
            logger.error(f"   Error: {e}")
            logger.error(
                "   Verify that the twin UUID and environment UUID are correct"
            )
            alert = create_config_error_alert(
                component="twin_config",
                resource_type="Digital Twin",
                resource_id=twin_uuid,
                details={
                    "environment_uuid": None,
                    "registry_id": self.registry_id,
                    "api_url": self._cw.config.base_url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            self._alert_manager.raise_alert(alert)
            logger.debug("Full traceback:", exc_info=True)
            raise SystemExit(1) from e

    def _enable_backend_alerts(self) -> None:
        """Enable backend alert integration now that MQTT is up and twin is confirmed.

        Alerts raised before this point (e.g. MQTT timeout) are local-only.
        That is expected — there is no connection to push them to yet.
        """
        assert self._cw is not None, (
            "_enable_backend_alerts called before _connect_mqtt"
        )
        self._alert_manager.enable_backend_integration(
            sdk_client=self._cw,
            twin_uuid=self.twin_uuid,
            environment_uuid=self._twin.environment_id,
        )
        self._alert_manager.start_alert_listener(self._cw.mqtt)
        self._sync_lifecycle_alerts_after_connect()
        logger.info(
            "[SUCCESS] AlertManager listening for alert updates from backend"
        )

    def _disconnect_cloud_client(self) -> None:
        """Release MQTT and Zenoh resources held by the driver's SDK client."""
        if self._cw is None:
            return
        disconnect = getattr(self._cw, "disconnect", None)
        if not callable(disconnect):
            return
        try:
            disconnect()
        except Exception:
            logger.debug(
                "Cyberwave client disconnect after driver stop",
                exc_info=True,
            )

    async def _reconnect_loop_async(self) -> None:
        """Watch for connection-loss events and drive :meth:`on_reconnect`.

        Sealed — subclasses should override :meth:`on_reconnect`, not this.
        Exits immediately (without error) if reconnect is disabled
        (:attr:`RECONNECT_MAX_ATTEMPTS` == 0 or :meth:`on_reconnect` is not
        overridden) and raises :exc:`RuntimeError` if all attempts are exhausted.
        """
        if self.RECONNECT_MAX_ATTEMPTS == 0:
            # Reconnect disabled — just wait for shutdown and return cleanly.
            await self._shutdown.wait()
            return

        while not self._shutdown.is_set():
            # Block until either a disconnect is signalled or shutdown is requested.
            _done, _ = await asyncio.wait(
                [
                    asyncio.create_task(self._connection_lost.wait()),
                    asyncio.create_task(self._shutdown.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._shutdown.is_set():
                return

            self._transition_to(DriverLifecycleState.RECONNECTING)
            logger.warning(
                "[RECONNECT] Cyberwave MQTT connection lost — starting reconnect sequence"
            )

            delay = self.RECONNECT_BACKOFF_BASE
            for attempt in range(1, self.RECONNECT_MAX_ATTEMPTS + 1):
                logger.info(
                    "[RECONNECT] Attempt %d/%d...", attempt, self.RECONNECT_MAX_ATTEMPTS
                )
                try:
                    success = await self.on_reconnect()
                except Exception:
                    logger.exception("[RECONNECT] on_reconnect() raised an exception")
                    success = False

                if success:
                    self._connection_lost.clear()
                    await self._unwire_interface_from_registry()
                    await self._wire_interface_from_registry()
                    await self._activate_registry_zenoh()
                    self._transition_to(DriverLifecycleState.ACTIVE)
                    logger.info(
                        "[RECONNECT] Cyberwave MQTT connection restored on attempt %d",
                        attempt,
                    )
                    break

                if attempt < self.RECONNECT_MAX_ATTEMPTS:
                    logger.warning(
                        "[RECONNECT] Attempt %d failed, retrying in %.1fs...",
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.RECONNECT_BACKOFF_MAX)
            else:
                logger.error(
                    "[RECONNECT] All %d attempts failed — driver entering ERROR state",
                    self.RECONNECT_MAX_ATTEMPTS,
                )
                raise RuntimeError(
                    f"Transport reconnection failed after {self.RECONNECT_MAX_ATTEMPTS} attempts"
                )
