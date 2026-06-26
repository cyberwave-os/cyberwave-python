"""``cyberwave.driver.base`` — lifecycle shell for Python edge drivers.

:class:`BaseDriver` is the shared runtime for Cyberwave Python drivers. It uses the
**template method** pattern: :meth:`run` / :meth:`run_async` own the lifecycle and call
your hooks in a fixed order. Cloud connectivity (MQTT, twin binding, session
telemetry, backend alerts) and the **interface registry** (manifest export, command
subscribe, rate-limited publishers) are handled here so subclasses focus on hardware.

**Typical construction**::

    twin = cw.twin("intel/realsensed455", twin_id="...")
    MyDriver(twin).run()

**Lifecycle (do not override ``run_async``)**::

    run()  →  asyncio.run(run_async())
    run_async()
        CONFIGURING   →  _connect_cloud_async()     # MQTT, twin, AlertManager
                        →  register_interface_on_twin()  # optional; see auto_register_interface
                        →  on_configure()
        CONNECTING    →  on_connect_to_device()
        INACTIVE      →  on_register_callbacks()    # device-local only
                        →  _wire_interface_from_registry()  # MQTT (+ Zenoh if declared)
                        →  on_activate()
                        →  _activate_registry_zenoh()  # DataBus when dual/zenoh specs
        ACTIVE        →  gather(_tick_loop_async, on_start_monitoring, _reconnect_loop_async)
                        # tick loop: _run_registry_publishers() (MQTT + Zenoh) then on_tick()
        finally       →  on_shutdown(), alert flush, FINALIZED

See :class:`BaseDriver` docstring for what you implement vs what is automatic.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from typing_extensions import Self

from .cloud.alert_api import DriverAlertsMixin
from .cloud.alerts import AlertManager
from .cloud.connection import CloudConnectionMixin
from .cloud.lifecycle_alerts import LifecycleAlertsMixin
from .cloud.telemetry_session import TelemetrySessionMixin
from .interface.registry_mixin import InterfaceRegistryMixin
from .interface.stream_publish_rate import StreamPublishRateLimiter
from .status import DriverLifecycleState, LifecycleStateMixin

# Re-exported for backwards compatibility — historically defined here, now live in
# dedicated modules. Imports such as ``from cyberwave.driver.base import
# DriverLifecycleState`` / ``resolve_twin_attached_controller`` keep working.
from .cloud.twin_binding import (  # noqa: F401
    refresh_driver_twin_from_api,
    resolve_twin_attached_controller,
)

logger = logging.getLogger(__name__)


class BaseDriver(
    InterfaceRegistryMixin,
    # LifecycleAlertsMixin must precede LifecycleStateMixin so its
    # _on_lifecycle_transition override wins over the state machine's no-op default.
    LifecycleAlertsMixin,
    LifecycleStateMixin,
    CloudConnectionMixin,
    TelemetrySessionMixin,
    DriverAlertsMixin,
    ABC,
):
    """Abstract base class for Cyberwave Python edge drivers.

    **Intent** — Give every driver the same cloud shell (MQTT, twin, alerts,
    telemetry session markers, manifest wiring) while keeping device I/O in a small
    set of async hooks. Declare topics and commands once in
    :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.define_interface`;
    the registry publishes your manifest and subscribes MQTT handlers during
    :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin._wire_interface_from_registry`.

  **Preferred construction**::

        twin = cw.twin("intel/realsensed455", twin_id="...")
        twin.driver.set_schema(driver.manifest)  # before run(), recommended
        MyDriver(twin).run()

    **Operator entrypoints** (usually called from ``__main__`` or a demo script):

    - :meth:`run` — blocking; runs :meth:`run_async` on a new event loop (main thread)
    - :meth:`run_async` — full lifecycle; do **not** override
    - :meth:`from_env` — ``cls(params_from_env)`` when ``Params.from_env`` exists
    - :meth:`for_twin` — ``cls(params, twin=twin)``
    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.cw_driver` /
      :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.get_manifest` —
      classmethod: build catalog from :meth:`define_interface` without twin/MQTT; optional
      ``path=`` writes ``cw-driver.yml``
    - :attr:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.manifest` —
      cw-driver.yml root dict for :meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema`
    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.register_interface_on_twin` —
      persist manifest on the bound twin (also called automatically when
      :attr:`auto_register_interface` is true and :attr:`REGISTRY_ID` is set)

    **You must implement (lifecycle hooks)**:

    - :meth:`on_configure` — load config, open files, build non-MQTT objects
    - :meth:`on_connect_to_device` — open serial, ROS, camera SDK, WebRTC, etc.
    - :meth:`on_register_callbacks` — device-native callbacks (not the MQTT registry)
    - :meth:`on_activate` — start hardware streams; driver becomes operational
    - :meth:`on_shutdown` — idempotent cleanup
    - :meth:`create` — only when the entrypoint is :meth:`create_and_run_async`

    **You usually implement (interface registry)**:

    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.define_interface` —
      custom commands, publishers, extra listeners
    - Command listener callbacks (e.g. ``_on_rotate``) — sync or async; registry dispatches on the driver loop
    - Publisher callbacks — return a ``dict`` payload each tick; base publishes at ``PublisherArgs.rate_hz``

    **Provided automatically (no override)**:

    - MQTT connect via :class:`~cyberwave.Cyberwave` using :class:`~cyberwave.config.CyberwaveConfig` (env vars, CLI profile)
    - Twin bind: reuse ``twin=`` + ``twin.client``, or fetch with :attr:`registry_id` + :attr:`twin_uuid`
    - :class:`~cyberwave.driver.alerts.AlertManager` — local + backend alerts after connect
    - Telemetry session: ``telemetry_start``, ``connected``, ``telemetry_end``, ``disconnected``
    - :class:`~cyberwave.telemetry.base.BaseTelemetry` — debounced ``driver_info`` on ``twin/telemetry``
    - Default management commands in :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.define_interface_defaults`
      (``stop``, ``teleoperate``, ``remoteoperate``, ``controller-changed``)

    **Transports (interface registry)**:

    - **MQTT** — remote broker path (SDK, UI, cross-LAN). Default for all topics.
    - **Zenoh** — edge-colocated ``DataBus`` on the same host as workers. Opt in with
      ``TopicSpec(enable_zenoh=True)`` on publishers
      (see :mod:`examples.fake_imu_driver`). ``twin/command`` stays MQTT-only.
    - Registry publisher on ``twin/telemetry`` at :attr:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.TELEMETRY_PUBLISH_RATE_HZ`
    - Tick loop at :attr:`TICK_RATE_HZ` — calls :meth:`_run_registry_publishers` then :meth:`on_tick`
    - SIGINT/SIGTERM → :meth:`request_shutdown` when on the main thread
    - Lifecycle logging via :meth:`_transition_to` and :attr:`lifecycle_state`

    **Built-in components on every instance**:

    - ``_alert_manager`` — :class:`~cyberwave.driver.alerts.AlertManager`
    - ``_interface`` — :class:`~cyberwave.driver.interface.registry.DriverInterfaceRegistry`
    - ``_telemetry`` — :class:`~cyberwave.telemetry.base.BaseTelemetry`
    - ``_operation_mode`` — :class:`~cyberwave.driver.interface.args.DriverOperationMode` (gates listeners)
    - ``_shutdown`` / ``_connection_lost`` — asyncio events for exit and reconnect

    **Operation mode extension point** (``DriverOperationMode``):

    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.on_enter_no_op`
    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.on_enter_teleop_local`
    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.on_enter_teleop_remote`
    - :meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.on_exit_operation`

    ROS 2 drivers (:class:`~cyberwave.driver.ros2.BaseROS2Driver`) map these hooks to
    rclpy lifecycle deactivate/activate. Other drivers may add hardware-specific behavior.

    **Optional overrides**:

    - :meth:`on_tick` — extra periodic work (default no-op)
    - :meth:`on_start_monitoring` — side task during ACTIVE (default waits for shutdown)
    - :meth:`on_reconnect` — return ``True`` when transport restored (default disables reconnect)
    - :meth:`driver_info_extra` — fields merged into telemetry snapshots
    - :attr:`TICK_RATE_HZ`, :attr:`RECONNECT_MAX_ATTEMPTS`, class attr ``REGISTRY_ID``

    **Class attributes to set on your driver**:

    - ``REGISTRY_ID`` — catalog registry ID (e.g. ``"intel/realsensed455"``); used for twin lookup and manifests
    - ``driver_family`` — ``"python"`` (inherited from :class:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin`)
    - ``auto_register_interface`` — if ``True`` (default), :meth:`run_async` calls
      :meth:`register_interface_on_twin` after cloud connect; set ``False`` when you only
      want explicit ``set_schema`` before :meth:`run`
    """

    def __init__(
        self,
        params: Any = None,
        *,
        twin: Any | None = None,
        client: Any | None = None,
        auto_register_interface: bool | None = None,
    ) -> None:
        self.params = params
        self._alert_manager: AlertManager = AlertManager()
        self._cw: Any | None = client or (
            getattr(twin, "client", None) if twin is not None else None
        )
        self._twin: Any | None = twin
        self._twin_prebound: bool = twin is not None
        self._lifecycle_state: DriverLifecycleState = DriverLifecycleState.UNCONFIGURED
        self._shutdown: asyncio.Event = asyncio.Event()
        # Set by the subclass (e.g. from _recv_camera_stream) to trigger reconnect.
        self._connection_lost: asyncio.Event = asyncio.Event()
        # Tick-rate tracking — rolling window of the last 10 tick start times.
        # deque(maxlen) drops old entries atomically; individual appends are
        # thread-safe in CPython (GIL) and avoid the two-step append+pop(0)
        # that a plain list would require.
        self._tick_times: collections.deque[float] = collections.deque(maxlen=10)
        self._last_tick_duration_ms: float = 0.0
        self._lifecycle_twin_pending_notice: bool = False
        self._stream_publish_rate = StreamPublishRateLimiter()
        self._init_interface_registry(auto_register_interface=auto_register_interface)

    # Lifecycle state + transition alerts → LifecycleAlertsMixin (lifecycle.py)

    @property
    def client(self) -> Any:
        """Connected :class:`~cyberwave.Cyberwave` client (after cloud connect)."""
        if self._cw is None:
            raise RuntimeError("Cyberwave client accessed before MQTT/API connection")
        return self._cw

    @property
    def twin(self) -> Any:
        """Resolved digital twin handle (after cloud connect)."""
        if self._twin is None:
            raise RuntimeError("digital twin accessed before Cyberwave connection")
        return self._twin

    @property
    def _mqtt_prefix(self) -> str:
        """MQTT topic prefix from the SDK client (empty string in production)."""
        return (
            getattr(self._cw.mqtt, "topic_prefix", "") if self._cw is not None else ""
        )

    def subscribe_command_topic(
        self, twin_uuid: str, handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Subscribe *handler* to the twin command topic (legacy; prefer registry).

        Prefer :meth:`define_interface` with :class:`~cyberwave.driver.CommandArgs`
        on the ``twin/command`` topic instead of calling this directly.
        """
        if self._cw is None:
            logger.warning(
                "subscribe_command_topic called before Cyberwave MQTT connection"
            )
            return
        topic = f"{self._mqtt_prefix}cyberwave/twin/{twin_uuid}/command"
        self._cw.mqtt.subscribe(topic, handler)
        logger.info("Subscribed to command topic: %s", topic)

    # Telemetry session markers + emit_driver_info → TelemetrySessionMixin
    # (telemetry_session.py)

    def run(self) -> None:
        """Run the full driver lifecycle (blocking)."""
        asyncio.run(self.run_async())

    @classmethod
    def from_env(cls) -> Self:
        """Construct driver from ``Params.from_env()`` when defined, else default ctor.

        Legacy: when ``CYBERWAVE_TWIN_UUID`` is set and no twin is passed, the driver
        fetches the twin during cloud connect. Prefer ``Driver(twin).run()`` instead.
        """
        params_cls = getattr(cls, "Params", None)
        params = params_cls.from_env() if params_cls is not None and hasattr(params_cls, "from_env") else None
        return cls(params)

    @classmethod
    def for_twin(cls, twin: Any, params: Any = None, **kwargs: Any) -> Self:
        """Construct a driver bound to an existing :class:`~cyberwave.twin.base.Twin`."""
        return cls(params, twin=twin, **kwargs)

    # ── Twin / asset identity ────────────────────────────────────────────────

    @property
    def twin_uuid(self) -> str:
        """UUID of the bound digital twin (shortcut for ``self._twin.uuid``)."""
        return self._twin.uuid  # type: ignore[union-attr]

    def _resolve_twin_uuid(self) -> str:
        """Twin UUID usable before the twin is bound (falls back to env var)."""
        if self._twin is not None:
            return self._twin.uuid
        return os.getenv("CYBERWAVE_TWIN_UUID", "?").strip()

    @property
    def registry_id(self) -> str:
        """Catalog registry ID from the subclass ``REGISTRY_ID`` class attribute."""
        key = type(self)._class_registry_id()
        if not key:
            raise RuntimeError(
                f"{type(self).__name__} must set REGISTRY_ID to the catalog registry ID "
                "(e.g. REGISTRY_ID = 'universal_robots/UR7')"
            )
        return key

    @property
    def env_uuid(self) -> str:
        """UUID of the environment the robot belongs to.

        Derived from the fetched twin after :meth:`_connect_cloud` completes.
        Raises :exc:`RuntimeError` if accessed before the twin is available.
        """
        if self._twin is None:
            raise RuntimeError("env_uuid accessed before twin was fetched")
        return self._twin.environment_id

    # ── Abstract hooks ───────────────────────────────────────────────────────

    @abstractmethod
    async def on_configure(self) -> None:
        """Initialize device-specific resources before connecting to the device.

        Called while the driver is in the ``CONFIGURING`` state, after cloud
        connectivity is established but before the transport connection.
        Typical uses: HDF5 recorder, visualization server, callback handler
        construction (no live connection required yet).
        """
        ...

    @abstractmethod
    async def on_connect_to_device(self) -> None:
        """Establish connection to the physical robot.

        e.g. WebRTC for Unitree Go2, serial for custom robots, ROS2 bridge.
        After this returns, the live connection is available for callbacks.
        """
        ...

    @abstractmethod
    async def on_register_callbacks(self) -> None:
        """Subscribe handler methods to data channels/topics.

        Called after :meth:`on_connect_to_device` returns (driver is now
        ``INACTIVE``), so the live connection is guaranteed to be available.
        """
        ...

    @abstractmethod
    async def on_activate(self) -> None:
        """Start streams and set up controllers — make the driver fully operational.

        Called after :meth:`on_register_callbacks`, while the driver is still
        in the ``INACTIVE`` state.  The ACTIVE phase (tick loop + monitoring)
        begins immediately after this returns.  Typical uses: start the camera
        streamer, initialize motion controllers, register telemetry callbacks.
        """
        ...

    @abstractmethod
    async def on_shutdown(self) -> None:
        """Release all device-specific resources.

        Always called in the ``finally`` block of :meth:`run_async`, even if an
        exception occurred.  Should be idempotent.
        """
        ...

    # ── Tick rate ────────────────────────────────────────────────────────────

    TICK_RATE_HZ: float = 10.0
    """Frequency at which :meth:`on_tick` is called during the ACTIVE phase.

    Override at the class level to change the rate for a specific driver::

        class MyDriver(BaseDriver):
            TICK_RATE_HZ = 50.0  # 50 Hz control loop
    """

    STREAM_PUBLISH_MAX_HZ: float = 50.0
    """Default cap for high-frequency streams forwarded to Cyber MQTT/Zenoh.

    ROS sensor topics (e.g. ``joint_states``) often arrive at 100–200 Hz; edge
    drivers should throttle outbound Cyber publishes with
    :meth:`acquire_stream_publish_slot` so broker load stays in the ~50–60 Hz range.
    """

    RECONNECT_MAX_ATTEMPTS: int = 5
    """Maximum number of reconnection attempts before the driver enters ERROR state.

    Set to ``0`` to disable automatic reconnection entirely.
    """

    RECONNECT_BACKOFF_BASE: float = 2.0
    """Initial back-off delay in seconds between reconnect attempts.

    Doubles on each consecutive failure, capped at :attr:`RECONNECT_BACKOFF_MAX`.
    """

    RECONNECT_BACKOFF_MAX: float = 60.0
    """Upper bound for the exponential back-off delay in seconds."""

    def stream_publish_max_hz(self, stream_key: str) -> float:
        """Return the max publish rate for *stream_key* (override per stream)."""
        return type(self).STREAM_PUBLISH_MAX_HZ

    def acquire_stream_publish_slot(
        self, stream_key: str, *, max_hz: float | None = None
    ) -> bool:
        """Return ``True`` when a Cyber publish on *stream_key* is due under the rate cap."""
        hz = self.stream_publish_max_hz(stream_key) if max_hz is None else max_hz
        return self._stream_publish_rate.acquire(stream_key, max_hz=hz)

    # ── Optional hooks ───────────────────────────────────────────────────────

    async def on_tick(self) -> None:  # noqa: B027
        """Called once per tick during the ACTIVE phase.

        Override to step a controller, publish periodic telemetry, or do any
        other work that must run at a fixed rate.  The tick rate is controlled
        by :attr:`TICK_RATE_HZ`.  Default is a no-op.
        """
        pass  # intentional no-op; subclasses may override

    async def on_start_monitoring(self) -> None:
        """Run concurrently with the tick loop during the ACTIVE phase.

        Override for health-monitoring, periodic logging, or any other
        long-running side task.  The default implementation simply waits
        until the driver shuts down, which is the correct behaviour for
        drivers that only use :meth:`on_tick`.
        """
        await self._shutdown.wait()

    async def on_reconnect(self) -> bool:
        """Re-establish the transport connection after an unexpected disconnect.

        Called by :meth:`_reconnect_loop_async` up to :attr:`RECONNECT_MAX_ATTEMPTS`
        times with exponential back-off whenever :attr:`_connection_lost` is set.

        Return ``True`` if the connection was fully restored (live data flowing
        again), ``False`` to trigger another attempt.  Any unhandled exception
        is caught by the loop and treated as ``False``.

        The default implementation returns ``False`` on every call, which opts
        the driver out of automatic reconnection — the process will enter the
        ERROR state after the attempt ceiling is reached.  Override to enable
        reconnect for your transport.

        Typically the implementation should:

        1. Clear ``self._state.first_frame_ready`` (or equivalent)
        2. Close the stale connection (best-effort, ignore errors)
        3. Re-run the connect / register-callbacks / wait-for-first-data sequence
        4. Return ``True``
        """
        return False

    # ── Lifecycle hook: configuration + entrypoint ───────────────────────────

    @classmethod
    @abstractmethod
    def create(cls) -> BaseDriver:
        """Parse CLI arguments, load environment, and return a configured driver instance.

        Subclasses own all driver-specific construction logic here (argument
        parsing, environment loading, config dataclass construction).  The base
        class only calls this once, from :meth:`create_and_run_async`.
        """
        ...

    @classmethod
    async def create_and_run_async(cls) -> None:
        """Canonical entrypoint: construct the driver then run its full lifecycle.

        This is the method that ``__main__.py`` (and therefore the
        docker-entrypoint) should call.  It delegates construction entirely to
        the subclass via :meth:`create` and then hands control to the
        base-class lifecycle via :meth:`run_async`.
        """
        await cls.create().run_async()

    # raise_alert / create_twin_alert / raise_alert_async / resolve_alert
    # → DriverAlertsMixin (driver_alerts.py)

    def request_shutdown(self) -> None:
        """Signal the driver to exit the ACTIVE phase and begin teardown.

        Sets the internal shutdown event, which unblocks :meth:`_tick_loop_async`
        and the default :meth:`on_start_monitoring` implementation.  Returns
        immediately — the caller must still await :meth:`run_async` (or the
        :meth:`create_and_run_async` coroutine) to know that teardown has finished.

        Typical uses::

            # From a SIGTERM handler
            signal.signal(signal.SIGTERM, lambda *_: driver.request_shutdown())

            # From an orchestrator managing multiple drivers
            driver.request_shutdown()
            await run_task  # the original asyncio.create_task(driver.run_async())
        """
        self._shutdown.set()

    # ── Lifecycle entrypoint (not intended to be overridden) ─────────────────

    async def run_async(self) -> None:
        """Run the driver lifecycle.

        This method owns the shared lifecycle and is not intended to be
        overridden.  Override the hook methods instead.
        """
        logger.info(
            "Running Cyberwave driver for asset %s (twin=%s)",
            self.registry_id,
            self._resolve_twin_uuid(),
        )
        try:
            self._transition_to(DriverLifecycleState.CONFIGURING)
            logger.info(
                "Connecting to Cyberwave API and MQTT broker (twin=%s) …",
                self._resolve_twin_uuid(),
            )
            await self._connect_cloud_async()
            logger.info("Connected to Cyberwave servers")
            self._start_driver_telemetry_session()
            if self._auto_register_interface:
                try:
                    self.register_interface_on_twin()
                except Exception:
                    logger.exception("register_interface_on_twin failed; continuing")
            await self.on_configure()

            self._transition_to(DriverLifecycleState.CONNECTING)
            await self.on_connect_to_device()

            self._transition_to(DriverLifecycleState.INACTIVE)
            await self.on_register_callbacks()
            await self._wire_interface_from_registry()
            await self.on_activate()
            await self._activate_registry_zenoh()

            # Register OS signal handlers (main thread only — demo runs driver in a thread).
            if threading.current_thread() is threading.main_thread():
                _loop = asyncio.get_running_loop()
                try:
                    _loop.add_signal_handler(signal.SIGTERM, self.request_shutdown)
                    _loop.add_signal_handler(signal.SIGINT, self.request_shutdown)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

            self._transition_to(DriverLifecycleState.ACTIVE)
            logger.info(
                "Cyberwave driver ACTIVE (operation_mode=%s); entering tick/monitor loops",
                self._operation_mode,
            )
            await self._derive_initial_operation_mode()
            await asyncio.gather(
                self._tick_loop_async(),
                self.on_start_monitoring(),
                self._reconnect_loop_async(),
            )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except BaseException:
            self._transition_to(DriverLifecycleState.ERROR)
            raise
        finally:
            self._shutdown.set()  # unblocks _tick_loop_async and on_start_monitoring
            if self._lifecycle_state != DriverLifecycleState.ERROR:
                self._transition_to(DriverLifecycleState.DEACTIVATING)
            await self._unwire_interface_from_registry()
            self._end_driver_telemetry_session()
            await self.on_shutdown()
            try:
                logger.info("Flushing alerts to backend...")
                self._alert_manager.shutdown()
                logger.info("[SUCCESS] AlertManager shutdown complete")
            except Exception:
                logger.exception("Error shutting down AlertManager")
            self._transition_to(DriverLifecycleState.FINALIZED)
            self._disconnect_cloud_client()

    # ── Base internals ───────────────────────────────────────────────────────

    @property
    def tick_rate_hz(self) -> float:
        """Measured tick rate in Hz, averaged over the last 10 ticks.

        Returns 0.0 until at least 2 tick timestamps have been recorded.
        Safe to read from any thread.
        """
        times = list(self._tick_times)  # snapshot — avoids mutation during iteration
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    async def _tick_loop_async(self) -> None:
        """Call :meth:`on_tick` at :attr:`TICK_RATE_HZ` until shutdown.

        Sealed — subclasses should override :meth:`on_tick`, not this method.
        """
        interval = 1.0 / self.TICK_RATE_HZ
        while not self._shutdown.is_set():
            t0 = time.monotonic()
            try:
                await self._run_registry_publishers()
                await self.on_tick()
            except Exception:
                logger.exception("Unexpected error in on_tick(); continuing")
            elapsed = time.monotonic() - t0
            self._last_tick_duration_ms = elapsed * 1000.0
            # Rolling window: deque(maxlen=10) drops the oldest entry automatically.
            self._tick_times.append(t0)
            sleep_for = interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
