"""base_ros2_driver.py — Python ROS 2 driver base for Cyberwave drivers.

Provides :class:`BaseROS2Driver`, combining :class:`~cyberwave.driver.BaseDriver`
with ``rclpy.lifecycle.LifecycleNode``. Inherits from
:class:`rclpy.lifecycle.LifecycleNode` and routes lifecycle transitions to a
consistent set of driver hooks.

## Hook naming

``LifecycleNode`` already uses on_configure(state), on_activate(state),
on_deactivate(state), and on_shutdown(state) as its lifecycle callbacks.
Python does not support method overloading, so a no-arg override of any of
those names in a subclass would silently break the lifecycle machinery.

To preserve a clean, consistent API while avoiding the conflict, lifecycle
hooks drop the 'on_' prefix:

    Lifecycle stage      Python hook
    ───────────────────  ─────────────────────
    configure            configure()
    connect to device    connect_to_device()
    register callbacks   register_callbacks()
    activate             activate()
    deactivate           deactivate()           (default no-op)
    tick                 tick()                 (default no-op)
    shutdown             shutdown()

Event callbacks keep 'on_' since they have no rclpy naming conflict:

    on_topic_name_changed(topic_entry, new_name)

Service-name parameters are read at startup only; no runtime remap or callback.

## Lifecycle mapping

    ROS 2 transition   Driver hooks called (base first on setup, virtual first on teardown)
    ─────────────────  ────────────────────────────────────────────────────────────────
    configure          connect_cloud() → base configure() → configure()
    activate           base then virtual: connect_to_device, register_callbacks, activate
                       → start tick timer
    deactivate         stop tick timer → deactivate() → base deactivate()
    shutdown           stop tick timer → shutdown() → base shutdown()

## CW_ROS2_AUTO_ACTIVATE

Set CW_ROS2_AUTO_ACTIVATE=true to auto-drive the node to ACTIVE on the
first executor cycle. Placeholder until remote, on-demand lifecycle
activation is available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import String

from .env_params import (
    collect_env_param_overrides,
    node_name_from_env,
    node_namespace_from_env,
    resolve_managed_launch_args,
)
from ..base import BaseDriver, DriverLifecycleState
from .manifest import (
    DRIVER_MANIFEST_FILE_NAME,
    ManifestTopic,
    NodeManifest,
    default_node_manifest,
    dump_combined_driver_manifest,
    merge_combined_driver_manifest,
    resolve_node_manifest,
)
from .ros_publishers import unwire_ros_publishers, wire_ros_publishers
from .ros_setup_env import apply_ros_setup_environment, collect_ros_setup_scripts
from .topic_discovery import resolve_ros_message_class

MsgT = TypeVar("MsgT")

logger = logging.getLogger(__name__)


def _normalize_qos(qos: Union[int, QoSProfile, None]) -> QoSProfile:
    """Convert int (depth) or None to QoSProfile; pass through QoSProfile."""
    if qos is None:
        return QoSProfile(depth=10)
    if isinstance(qos, int):
        return QoSProfile(depth=qos)
    return qos


def _ensure_rclpy_initialized() -> None:
    """Initialize rclpy before constructing a ``LifecycleNode``."""
    if not rclpy.ok():
        rclpy.init()


def _auto_activate_enabled() -> bool:
    return os.environ.get("CW_ROS2_AUTO_ACTIVATE", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def _ros_lifecycle_state_label(node: Any) -> str:
    """Return ROS lifecycle state label (Humble/Jazzy API differences)."""
    getter = getattr(node, "get_current_state", None)
    if callable(getter):
        try:
            return str(getter().label)
        except Exception:
            pass
    state_machine = getattr(node, "_state_machine", None)
    if state_machine is not None:
        try:
            _state_id, label = state_machine.current_state
            return str(label)
        except Exception:
            pass
    return "unknown"


class _BaseDriverAsyncHooks:
    """Implements :class:`BaseDriver` abstract ``on_*`` async methods.

    ``BaseROS2Driver`` must also expose rclpy lifecycle ``on_configure(state)``,
    ``on_activate(state)``, etc. (sync, different signature). Those live on the
    class body; this mixin keeps the asyncio ``BaseDriver`` contract on a
    separate MRO entry. :meth:`BaseROS2Driver.run_async` calls these via
    ``await _BaseDriverAsyncHooks.on_configure(self)``, not ``await self.on_configure()``.
    """

    async def on_configure(self) -> None:
        await self._run_after_cloud_connect()

    async def on_connect_to_device(self) -> None:
        await self._run_connect_to_device()

    async def on_register_callbacks(self) -> None:
        await self._run_register_callbacks()

    async def on_activate(self) -> None:
        await self._run_after_wire_activate()

    async def on_shutdown(self) -> None:
        await self._run_shutdown()


class BaseROS2Driver(_BaseDriverAsyncHooks, BaseDriver, LifecycleNode):
    """Base class for Python Cyberwave ROS 2 edge drivers.

    Combines Cyberwave API/MQTT connectivity (:class:`~cyberwave.driver.BaseDriver`)
    with a ROS 2 :class:`~rclpy.lifecycle.LifecycleNode`. Use :meth:`run` as the
    process entrypoint.

    Example::

        class MyDriver(BaseROS2Driver):
            def configure(self):
                self.get_logger().info("Configuring …")

            def activate(self):
                self._pub = self.create_publisher(String, "/my/topic", 10)

            def tick(self):
                self._pub.publish(String(data="hello"))

            def shutdown(self):
                pass
    """

    def __init__(
        self,
        node_name: str,
        manifest_path: str | None = None,
        *,
        params: Any = None,
        twin: Any | None = None,
        client: Any | None = None,
        auto_register_interface: bool | None = None,
        **kwargs: Any,
    ) -> None:
        actual_name = node_name_from_env(node_name)
        namespace = node_namespace_from_env()
        pre_manifest: NodeManifest = resolve_node_manifest(
            type(self),
            manifest_path,
            node_name=actual_name,
        )
        if pre_manifest.managed_launch is not None:
            apply_ros_setup_environment(
                collect_ros_setup_scripts(pre_manifest.managed_launch)
            )
        _ensure_rclpy_initialized()
        LifecycleNode.__init__(self, actual_name, namespace=namespace, **kwargs)
        BaseDriver.__init__(
            self,
            params,
            twin=twin,
            client=client,
            auto_register_interface=auto_register_interface,
        )

        self._manifest: NodeManifest = pre_manifest
        self._tick_timer: Optional[rclpy.timer.Timer] = None
        self._interface_state_pub = None
        self._param_cb_handle = None
        self._managed_publisher_remaps: dict[str, Callable[[str], None]] = {}
        self._managed_subscription_remaps: dict[str, Callable[[str], None]] = {}
        self._ros_forward_handles: list[Any] = []
        self._ros_out_publishers: dict[tuple[str, type], Any] = {}
        self._driver_loop: asyncio.AbstractEventLoop | None = None
        self._driver_loop_thread: threading.Thread | None = None
        self._ros_active = asyncio.Event()
        self._cloud_connected = False
        self._managed_launch: Any | None = None
        self._change_state_cli: Any | None = None

        if _auto_activate_enabled():
            self.get_logger().info(
                "CW_ROS2_AUTO_ACTIVATE enabled — will activate ROS lifecycle after "
                "connection to Cyberwave servers"
            )
        else:
            self.get_logger().warning(
                "CW_ROS2_AUTO_ACTIVATE disabled — ROS lifecycle will stay "
                "%s until something activates this node externally",
                _ros_lifecycle_state_label(self),
            )

    @classmethod
    def create(cls) -> BaseROS2Driver:
        """Construct from ``CW_DRIVER_MANIFEST`` and ``CW_ROS2_NODE_NAME`` env vars."""
        manifest = os.environ.get("CW_DRIVER_MANIFEST") or None
        node_name = os.environ.get("CW_ROS2_NODE_NAME", "cyberwave_ros2_driver")
        return cls(node_name, manifest)

    def run(self) -> None:
        """Run Cyberwave driver lifecycle and spin this ROS 2 node."""
        _ensure_rclpy_initialized()
        logger.info(
            "Running Cyberwave driver for asset %s (node=%s, ros_lifecycle=%s, "
            "CW_ROS2_AUTO_ACTIVATE=%s)",
            self.registry_id,
            self.get_fully_qualified_name(),
            _ros_lifecycle_state_label(self),
            os.environ.get("CW_ROS2_AUTO_ACTIVATE", "<unset>"),
        )
        self._start_driver_loop()
        run_future = asyncio.run_coroutine_threadsafe(
            self.run_async(), self._driver_loop
        )
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        try:
            while rclpy.ok() and not run_future.done():
                executor.spin_once(timeout_sec=0.1)
        except KeyboardInterrupt:
            self.request_shutdown()
        finally:
            if not run_future.done():
                self.request_shutdown()
            try:
                run_future.result(timeout=120)
            except Exception:
                logger.exception("BaseROS2Driver.run_async failed")

    def _start_driver_loop(self) -> asyncio.AbstractEventLoop:
        if self._driver_loop is not None:
            return self._driver_loop
        loop = asyncio.new_event_loop()
        self._driver_loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._driver_loop_thread = threading.Thread(
            target=_run, name="base-ros2-driver-async", daemon=True
        )
        self._driver_loop_thread.start()
        return loop

    def _require_driver_loop(self) -> asyncio.AbstractEventLoop:
        if self._driver_loop is None:
            raise RuntimeError("driver asyncio loop not started; call run() first")
        return self._driver_loop

    def _signal_ros_active(self) -> None:
        logger.info(
            "ROS lifecycle ACTIVE — wiring MQTT publishers and interface registry "
            "(node=%s)",
            self.get_fully_qualified_name(),
        )
        loop = self._driver_loop
        if loop is None:
            self._ros_active.set()
            return
        loop.call_soon_threadsafe(self._ros_active.set)

    async def run_async(self) -> None:
        """Cyberwave driver lifecycle; waits for ROS ACTIVE before ``from_ros`` wire."""
        try:
            self._transition_to(DriverLifecycleState.CONFIGURING)
            if not self._cloud_connected:
                logger.info(
                    "Connecting to Cyberwave API and MQTT broker (twin=%s) …",
                    self._resolve_twin_uuid(),
                )
                await self._connect_cloud_async()
                self._cloud_connected = True
                self._start_driver_telemetry_session()
                if self._auto_register_interface:
                    try:
                        self.register_interface_on_twin()
                    except Exception:
                        logger.exception("register_interface_on_twin failed; continuing")
            logger.info("Connected to Cyberwave servers")
            await _BaseDriverAsyncHooks.on_configure(self)

            if _auto_activate_enabled():
                self._begin_auto_activate()

            await self._wait_for_ros_active()
            # Edge ROS→MQTT forwarders (e.g. joint_states) are independent of
            # operation_mode and hardware ctrl_mode — wire as soon as ROS is ACTIVE.
            wire_ros_publishers(self)

            logger.info(
                "Wiring interface registry and ROS-to-MQTT forward publishers …"
            )
            self._transition_to(DriverLifecycleState.CONNECTING)
            await _BaseDriverAsyncHooks.on_connect_to_device(self)

            self._transition_to(DriverLifecycleState.INACTIVE)
            await _BaseDriverAsyncHooks.on_register_callbacks(self)
            await self._wire_interface_from_registry()
            await _BaseDriverAsyncHooks.on_activate(self)
            await self._activate_registry_zenoh()

            if threading.current_thread() is threading.main_thread():
                _loop = asyncio.get_running_loop()
                try:
                    import signal

                    _loop.add_signal_handler(signal.SIGTERM, self.request_shutdown)
                    _loop.add_signal_handler(signal.SIGINT, self.request_shutdown)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass

            self._transition_to(DriverLifecycleState.ACTIVE)
            logger.info(
                "Cyberwave driver ACTIVE (operation_mode=%s); entering tick/monitor loops",
                self.operation_mode,
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
            self._shutdown.set()
            if self._lifecycle_state != DriverLifecycleState.ERROR:
                self._transition_to(DriverLifecycleState.DEACTIVATING)
            unwire_ros_publishers(self)
            await self._unwire_interface_from_registry()
            self._end_driver_telemetry_session()
            await _BaseDriverAsyncHooks.on_shutdown(self)
            try:
                self._alert_manager.shutdown()
            except Exception:
                logger.exception("AlertManager shutdown failed")
            self._transition_to(DriverLifecycleState.FINALIZED)
            self._disconnect_cloud_client()

    def ros_publish(
        self,
        topic: str,
        msg: Any,
        *,
        qos_depth: int = 10,
    ) -> None:
        """Publish *msg* on a ROS topic (use from MQTT listener handlers)."""
        msg_type = type(msg)
        key = (topic, msg_type)
        pub = self._ros_out_publishers.get(key)
        if pub is None:
            pub = self.create_publisher(msg_type, topic, qos_depth)
            self._ros_out_publishers[key] = pub
        pub.publish(msg)

    # -------------------------------------------------------------------------
    # rclpy lifecycle callbacks — final, do not override in concrete drivers
    # -------------------------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_configure: declaring parameters from manifest")
        try:
            self._declare_manifest_params()
            self._setup_param_callback()
            self._interface_state_pub = self.create_publisher(
                String, "~/interface_state", 10
            )
            BaseROS2Driver.configure(self)
            self.configure()
        except Exception as e:
            self.get_logger().error(f"on_configure failed: {e}")
            return TransitionCallbackReturn.FAILURE

        self._publish_interface_state()
        self.get_logger().info("on_configure: complete")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate: connecting to device and starting tick")
        try:
            self.connect_to_device()
            self.register_callbacks()
            self.activate()
            self._start_tick_timer()
            self._signal_ros_active()
        except Exception as e:
            self.get_logger().error(f"on_activate failed: {e}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("on_activate: complete")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: stopping tick timer")
        self._stop_tick_timer()
        try:
            self.deactivate()
            BaseROS2Driver.deactivate(self)
        except Exception as e:
            self.get_logger().error(f"on_deactivate failed: {e}")
            return TransitionCallbackReturn.FAILURE

        self._publish_interface_state()
        self.get_logger().info("on_deactivate: complete")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown: running teardown")
        self._stop_tick_timer()
        try:
            self.shutdown()
            BaseROS2Driver.shutdown(self)
        except Exception as e:
            self.get_logger().error(f"on_shutdown failed: {e}")
            return TransitionCallbackReturn.ERROR

        self.get_logger().info("on_shutdown: complete")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(
            "on_cleanup: BaseROS2Driver does not support re-initialization. "
            "Restart the container to reconfigure."
        )
        return TransitionCallbackReturn.FAILURE

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(
            f"on_error: entering error state from '{state.label}', attempting teardown"
        )
        self._stop_tick_timer()
        try:
            self.shutdown()
        except Exception:
            pass
        return TransitionCallbackReturn.ERROR

    # -------------------------------------------------------------------------
    # Driver hooks — override in concrete drivers
    # -------------------------------------------------------------------------

    def configure(self) -> None:
        """Called during ROS 2 configure transition.

        Initialize non-hardware state here (config parsing, buffer allocation).
        Do NOT open device connections — that belongs in connect_to_device().
        """
        if self._manifest.managed_launch is not None:
            from .managed_launch import ManagedRosLaunch

            launch_args = resolve_managed_launch_args(
                self._manifest, self._manifest.managed_launch
            )
            self._managed_launch = ManagedRosLaunch(
                self._manifest.managed_launch,
                node=self,
                launch_args=launch_args,
            )

    def connect_to_device(self) -> None:
        """Called during ROS 2 activate transition.

        Open the physical device / network transport here.
        """
        if self._managed_launch is not None:
            self._managed_launch.start()
            self._managed_launch.wait_ready()

    def register_callbacks(self) -> None:
        """Called during ROS 2 activate transition, after connect_to_device().

        Subscribe to device events and data streams here.
        """

    def activate(self) -> None:
        """Called during ROS 2 activate transition, after register_callbacks().

        Create publishers and start control logic here.
        """

    def deactivate(self) -> None:
        """Called during ROS 2 deactivate transition.

        Stop streams and release resources so the node can be re-activated.
        Mirror what activate() created.
        """

    def tick(self) -> None:
        """Called periodically at tick_rate_hz while the node is ACTIVE.

        Keep this fast: publish data, run a control step. Do not block.
        """

    def shutdown(self) -> None:
        """Called during ROS 2 shutdown transition (full teardown).

        Release all hardware handles, stop streams, destroy publishers.
        """
        if self._managed_launch is not None:
            self._managed_launch.stop()
            self._managed_launch = None

    async def on_enter_no_op(self) -> None:
        await self._ros_request_transition(Transition.TRANSITION_DEACTIVATE)

    async def on_enter_teleop_local(self) -> None:
        await self._ros_request_transition(Transition.TRANSITION_ACTIVATE)

    async def on_enter_teleop_remote(self) -> None:
        await self._ros_request_transition(Transition.TRANSITION_ACTIVATE)

    async def _ros_request_transition(self, transition_id: int) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._ros_request_transition_sync, transition_id
        )

    def _ensure_change_state_client(self) -> None:
        if self._change_state_cli is None:
            self._change_state_cli = self.create_client(
                ChangeState,
                f"{self.get_fully_qualified_name()}/change_state",
            )

    def _trigger_transition_sync(self, transition_id: int) -> None:
        self._ensure_change_state_client()
        deadline = time.monotonic() + 30.0
        while not self._change_state_cli.service_is_ready():
            if time.monotonic() > deadline:
                raise TimeoutError("change_state service not ready")
            time.sleep(0.05)
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = self._change_state_cli.call_async(req)
        done_deadline = time.monotonic() + 30.0
        while not future.done():
            if time.monotonic() > done_deadline:
                raise TimeoutError(
                    f"change_state transition {transition_id} timed out"
                )
            time.sleep(0.05)
        resp = future.result()
        if not resp or not resp.success:
            raise RuntimeError(f"change_state transition {transition_id} failed")

    def _ros_request_transition_sync(self, transition_id: int) -> None:
        label = _ros_lifecycle_state_label(self).lower()
        if transition_id == Transition.TRANSITION_DEACTIVATE:
            if label != "active":
                return
        elif transition_id == Transition.TRANSITION_ACTIVATE:
            if label == "active":
                return
            if label == "unconfigured":
                self._trigger_transition_sync(Transition.TRANSITION_CONFIGURE)
        self._trigger_transition_sync(transition_id)

    # -------------------------------------------------------------------------
    # Event hooks — override to react to runtime name changes
    # -------------------------------------------------------------------------

    def on_topic_name_changed(self, topic_entry: ManifestTopic, new_name: str) -> None:
        """Called when a topic-name parameter changes at runtime.

        Managed remaps run first; override to handle driver-specific topics
        or use add_managed_publisher / add_managed_subscription so the base
        handles remap. Do not call the base from derived.
        """

    # -------------------------------------------------------------------------
    # Managed publisher/subscription helpers
    #
    # Topic names from params: use relative names (no leading /) in the manifest
    # default so the node's namespace is applied. Absolute paths bypass it.
    # Subscriber remap is destroy+recreate; a brief message gap is possible.
    # -------------------------------------------------------------------------

    def remap_publisher(
        self,
        msg_type: type[MsgT],
        attr_name: str,
        topic: str,
        qos: Union[int, QoSProfile] = 10,
    ) -> None:
        """Replace the publisher stored at attr_name with one on topic.

        Destroys the existing publisher if present, creates a new one, and
        sets the attribute. Use from on_topic_name_changed or on_activate.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        profile = _normalize_qos(qos)
        current = getattr(self, attr_name, None)
        if current is not None:
            self.destroy_publisher(current)
        setattr(self, attr_name, self.create_publisher(msg_type, topic, profile))

    def create_publisher_from_param(
        self,
        param_name: str,
        msg_type: type[MsgT],
        qos: Union[int, QoSProfile] = 10,
    ) -> Any:
        """Create a publisher using the current value of the topic-name param.

        No managed remap; use remap_publisher in on_topic_name_changed or
        add_managed_publisher for auto-remap.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        topic = self.get_parameter(param_name).get_parameter_value().string_value
        return self.create_publisher(msg_type, topic, _normalize_qos(qos))

    def add_managed_publisher(
        self,
        param_name: str,
        msg_type: type[MsgT],
        attr_name: str,
        qos: Union[int, QoSProfile] = 10,
    ) -> None:
        """Create publisher from param and register for auto-remap when the
        topic-name parameter changes. The attribute attr_name is updated on
        creation and on every remap; no need to override on_topic_name_changed.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        profile = _normalize_qos(qos)
        topic = self.get_parameter(param_name).get_parameter_value().string_value
        self.remap_publisher(msg_type, attr_name, topic, profile)

        def remap(t: str) -> None:
            self.remap_publisher(msg_type, attr_name, t, profile)

        self._managed_publisher_remaps[param_name] = remap

    def create_subscription_from_param(
        self,
        param_name: str,
        msg_type: type[MsgT],
        callback: Callable[[Any], None],
        qos: Union[int, QoSProfile] = 10,
    ) -> Any:
        """Create a subscription using the current value of the topic-name param.

        No managed remap; use remap_subscription in on_topic_name_changed or
        add_managed_subscription for auto-remap.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        topic = self.get_parameter(param_name).get_parameter_value().string_value
        return self.create_subscription(msg_type, topic, callback, _normalize_qos(qos))

    def remap_subscription(
        self,
        msg_type: type[MsgT],
        attr_name: str,
        topic: str,
        callback: Callable[[Any], None],
        qos: Union[int, QoSProfile] = 10,
    ) -> None:
        """Replace the subscription stored at attr_name with one on topic.

        Destroy + create; brief message gap possible. Logs a warning.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        profile = _normalize_qos(qos)
        self.get_logger().warn(
            "Subscriber remap to '%s' (destroy+recreate; brief message gap possible)"
            % topic
        )
        current = getattr(self, attr_name, None)
        if current is not None:
            self.destroy_subscription(current)
        setattr(self, attr_name, self.create_subscription(msg_type, topic, callback, profile))

    def add_managed_subscription(
        self,
        param_name: str,
        msg_type: type[MsgT],
        qos: Union[int, QoSProfile],
        callback: Callable[[Any], None],
    ) -> None:
        """Create subscription from param and register for auto-remap.

        Callback-only; no handle returned. Use remove_managed_subscription
        (param_name) to stop. Remap is destroy+recreate; warning is logged.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        profile = _normalize_qos(qos)
        topic = self.get_parameter(param_name).get_parameter_value().string_value
        cell: list = [self.create_subscription(msg_type, topic, callback, profile)]

        def remap(t: str) -> None:
            self.get_logger().warn(
                "Subscriber '%s' remapped to '%s' (destroy+recreate; brief message gap possible)"
                % (param_name, t)
            )
            if cell[0] is not None:
                self.destroy_subscription(cell[0])
            cell[0] = self.create_subscription(msg_type, t, callback, profile)

        self._managed_subscription_remaps[param_name] = remap

    def add_managed_subscription_with_ref(
        self,
        param_name: str,
        msg_type: type[MsgT],
        attr_name: str,
        qos: Union[int, QoSProfile],
        callback: Callable[[Any], None],
    ) -> None:
        """Create subscription from param, assign to attr_name, and register
        for auto-remap. The attribute is updated on creation and on every remap.
        qos: depth (int) or rclpy.qos.QoSProfile.
        """
        profile = _normalize_qos(qos)
        topic = self.get_parameter(param_name).get_parameter_value().string_value
        sub = self.create_subscription(msg_type, topic, callback, profile)
        setattr(self, attr_name, sub)

        def remap(t: str) -> None:
            self.get_logger().warn(
                "Subscriber '%s' remapped to '%s' (destroy+recreate; brief message gap possible)"
                % (param_name, t)
            )
            current = getattr(self, attr_name, None)
            if current is not None:
                self.destroy_subscription(current)
            setattr(self, attr_name, self.create_subscription(msg_type, t, callback, profile))

        self._managed_subscription_remaps[param_name] = remap

    def remove_managed_subscription(self, param_name: str) -> None:
        """Stop auto-remap and remove the managed subscription for param_name."""
        self._managed_subscription_remaps.pop(param_name, None)

    # -------------------------------------------------------------------------
    # ROS stream → Cyber publish rate limiting
    # -------------------------------------------------------------------------

    ROS_STREAM_PUBLISH_MAX_HZ: float = 50.0
    """Cap for ROS subscription streams forwarded to Cyber MQTT/Zenoh.

    Sensor topics commonly exceed 100 Hz; use :meth:`acquire_ros_stream_publish_slot`
    in ROS callbacks (or rely on the registry ``from_ros`` forwarder) to stay near
    50–60 Hz on the wire.
    """

    @staticmethod
    def ros_stream_key(ros_topic: str) -> str:
        """Stable throttle key for a ROS topic name."""
        return f"ros:{ros_topic.lstrip('/')}"

    def ros_stream_publish_max_hz(self, ros_topic: str) -> float:
        """Max Cyber publish rate for *ros_topic* (override per topic in subclasses)."""
        return type(self).ROS_STREAM_PUBLISH_MAX_HZ

    def acquire_ros_stream_publish_slot(
        self, ros_topic: str, *, max_hz: float | None = None
    ) -> bool:
        """Return ``True`` when a ROS-sourced Cyber publish on *ros_topic* is due."""
        hz = self.ros_stream_publish_max_hz(ros_topic) if max_hz is None else max_hz
        return self.acquire_stream_publish_slot(self.ros_stream_key(ros_topic), max_hz=hz)

    # -------------------------------------------------------------------------
    # Helpers available to concrete drivers
    # -------------------------------------------------------------------------

    @property
    def manifest(self) -> NodeManifest:
        """The loaded node manifest (available after construction)."""
        return self._manifest

    @classmethod
    def define_node_manifest(cls, node_name: str) -> NodeManifest:
        """ROS node manifest declared in Python (override in concrete drivers)."""
        return default_node_manifest(node_name)

    @classmethod
    def _interface_registry_probe(cls) -> BaseROS2Driver:
        """Minimal instance for MQTT manifest export (no rclpy lifecycle)."""
        inst = cls.__new__(cls)
        inst._init_interface_registry(auto_register_interface=False)
        return inst

    @classmethod
    def get_combined_manifest(cls, *, compiled: bool = False) -> dict[str, Any]:
        """Build unified manifest: ROS node fields + uncompiled cw-driver MQTT catalog."""
        if compiled:
            raise ValueError(
                "compiled driver catalogs are produced server-side; "
                "use compiled=False and twin.driver.set_schema()"
            )
        from .env_params import node_name_from_env

        node_name = node_name_from_env(getattr(cls, "DEFAULT_NODE_NAME", "cyberwave_ros2_driver"))
        node = cls.define_node_manifest(node_name)
        cw = cls._interface_registry_probe().get_driver_manifest(compiled=False)
        return merge_combined_driver_manifest(node, cw)

    @classmethod
    def write_manifest(
        cls,
        path: str | Path | None = None,
        *,
        header_comment: str | None = None,
    ) -> Path:
        """Export :meth:`get_combined_manifest` to ``manifest.yaml`` (ROS + MQTT)."""
        from pathlib import Path as PathCls

        target = PathCls(path or DRIVER_MANIFEST_FILE_NAME)
        comment = header_comment or (
            f"Generated from {cls.__module__}.{cls.__qualname__} "
            "(define_node_manifest + define_interface) — edit Python and re-export."
        )
        dump_combined_driver_manifest(
            cls.get_combined_manifest(compiled=False),
            target,
            header_comment=comment,
        )
        logger.info("Wrote combined driver manifest %s", target.resolve())
        return target.resolve()

    def publish_interface_state(self) -> None:
        """Publish a fresh ~/interface_state snapshot."""
        self._publish_interface_state()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    async def _run_after_cloud_connect(self) -> None:
        """``run_async`` hook after MQTT/twin connect (default no-op).

        ROS-side setup belongs in sync :meth:`configure`, invoked from rclpy
        ``on_configure(state)``.
        """

    async def _run_connect_to_device(self) -> None:
        """``run_async`` hook in ``CONNECTING`` (default no-op).

        Device connection for ROS drivers is usually sync :meth:`connect_to_device`
        from rclpy ``on_activate(state)``.
        """

    async def _run_register_callbacks(self) -> None:
        """``run_async`` hook before registry/MQTT wire (default no-op)."""

    async def _run_after_wire_activate(self) -> None:
        """``run_async`` hook after registry + ``from_ros`` wire (default no-op)."""

    async def _run_shutdown(self) -> None:
        """``run_async`` teardown hook (default no-op; prefer sync :meth:`shutdown`)."""

    async def _wait_for_ros_active(self, *, timeout_s: float = 300.0) -> None:
        """Block until rclpy ``on_activate`` signals ROS lifecycle ACTIVE."""
        auto = "true" if _auto_activate_enabled() else "false"
        ros_state = _ros_lifecycle_state_label(self)
        logger.info(
            "Waiting for ROS lifecycle ACTIVE before wiring MQTT publishers "
            "(current=%s, timeout=%.0fs, CW_ROS2_AUTO_ACTIVATE=%s). "
            "Without auto-activate, call configure→activate on this node.",
            ros_state,
            timeout_s,
            auto,
        )
        if auto != "true":
            logger.warning(
                "ROS lifecycle is still %s — set CW_ROS2_AUTO_ACTIVATE=true (or "
                "activate the node externally) or the driver will idle here.",
                ros_state,
            )

        deadline = asyncio.get_running_loop().time() + timeout_s
        while not self._ros_active.is_set():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"ROS lifecycle did not reach ACTIVE within {timeout_s:.0f}s "
                    f"(still {_ros_lifecycle_state_label(self)}); "
                    "set CW_ROS2_AUTO_ACTIVATE=true"
                )
            try:
                await asyncio.wait_for(
                    self._ros_active.wait(),
                    timeout=min(10.0, remaining),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Still waiting for ROS ACTIVE (lifecycle=%s, auto_activate=%s)",
                    _ros_lifecycle_state_label(self),
                    auto,
                )

    def _declare_manifest_params(self) -> None:
        env_overrides = collect_env_param_overrides(self._manifest)

        def _declare(name: str, type_str: str, default_str: str, description: str, read_only: bool) -> None:
            desc = ParameterDescriptor(description=description, read_only=read_only)
            if name in env_overrides:
                value = env_overrides[name]
            elif type_str == "bool":
                value = default_str.lower() in ("true", "1", "yes")
            elif type_str == "int":
                value = int(default_str) if default_str else 0
            elif type_str == "double":
                value = float(default_str) if default_str else 0.0
            else:
                value = default_str
            self.declare_parameter(name, value, desc)

        for p in self._manifest.params:
            _declare(p.name, p.type, p.default_value, p.description, p.read_only)
        for t in self._manifest.topics:
            _declare(t.name, "string", t.default_value, t.description, False)
        for s in self._manifest.services:
            _declare(s.name, "string", s.default_value, s.description, False)

    def _setup_param_callback(self) -> None:
        def _on_params_set(params):
            from rcl_interfaces.msg import SetParametersResult

            for p in params:
                self._on_param_changed(p.name, p)
            self._publish_interface_state()
            return SetParametersResult(successful=True)

        self._param_cb_handle = self.add_on_set_parameters_callback(_on_params_set)

    def _on_param_changed(self, name: str, param: Parameter) -> None:
        for t in self._manifest.topics:
            if t.name == name:
                new_name = param.get_parameter_value().string_value
                if name in self._managed_publisher_remaps:
                    self._managed_publisher_remaps[name](new_name)
                if name in self._managed_subscription_remaps:
                    self._managed_subscription_remaps[name](new_name)
                self.on_topic_name_changed(t, new_name)
                return
        for s in self._manifest.services:
            if s.name == name:
                return  # service-name params read at startup only; no runtime remap

    def _start_tick_timer(self) -> None:
        tick_hz = 10.0
        try:
            tick_hz = float(
                self.get_parameter("tick_rate_hz").get_parameter_value().integer_value
            )
        except Exception:
            pass
        if tick_hz <= 0:
            tick_hz = 1.0
        self._tick_timer = self.create_timer(1.0 / tick_hz, self._tick_callback)

    def _stop_tick_timer(self) -> None:
        if self._tick_timer is not None:
            self._tick_timer.cancel()
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None

    def _tick_callback(self) -> None:
        try:
            self.tick()
        except Exception as e:
            self.get_logger().error(f"tick() raised an exception: {e}")

    def _publish_interface_state(self) -> None:
        if self._interface_state_pub is None:
            return
        try:
            msg = String(data=self._build_interface_state_json())
            self._interface_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"publish_interface_state failed: {e}")

    def _build_interface_state_json(self) -> str:
        state_label = _ros_lifecycle_state_label(self)
        params: dict = {}
        try:
            names = self.list_parameters([], 0).names
            for name in names:
                p = self.get_parameter(name)
                pv = p.get_parameter_value()
                pt = pv.type
                if pt == ParameterType.PARAMETER_BOOL:
                    params[name] = pv.bool_value
                elif pt == ParameterType.PARAMETER_INTEGER:
                    params[name] = pv.integer_value
                elif pt == ParameterType.PARAMETER_DOUBLE:
                    params[name] = pv.double_value
                elif pt == ParameterType.PARAMETER_STRING:
                    params[name] = pv.string_value
                else:
                    params[name] = str(pv)
        except Exception:
            pass
        return json.dumps({"lifecycle_state": state_label, "parameters": params})

    # -------------------------------------------------------------------------
    # Auto-activate
    # -------------------------------------------------------------------------

    def _begin_auto_activate(self) -> None:
        """Start lifecycle configure→activate after Cyberwave connect (executor spinning)."""
        self.get_logger().info(
            f"CW_ROS2_AUTO_ACTIVATE: starting lifecycle transitions "
            f"(current={_ros_lifecycle_state_label(self)})"
        )
        self._change_state_cli = self.create_client(
            ChangeState,
            f"{self.get_fully_qualified_name()}/change_state",
        )
        # Poll until the service is ready, then trigger transitions.
        self._auto_activate_timer = self.create_timer(
            0.1, self._auto_activate_poll
        )

    def _auto_activate_poll(self) -> None:
        if not self._change_state_cli.service_is_ready():
            now = time.monotonic()
            last = getattr(self, "_auto_activate_service_wait_log_at", 0.0)
            if now - last >= 3.0:
                self._auto_activate_service_wait_log_at = now
                self.get_logger().info(
                    "CW_ROS2_AUTO_ACTIVATE: waiting for change_state service …"
                )
            return
        self._auto_activate_timer.cancel()
        self.destroy_timer(self._auto_activate_timer)
        self._auto_activate_timer = None

        self.get_logger().info("CW_ROS2_AUTO_ACTIVATE: triggering configure")
        req = ChangeState.Request()
        req.transition.id = Transition.TRANSITION_CONFIGURE
        future = self._change_state_cli.call_async(req)
        future.add_done_callback(self._auto_activate_on_configure)

    def _auto_activate_on_configure(self, future) -> None:
        resp = future.result()
        if not resp or not resp.success:
            self.get_logger().error(
                "CW_ROS2_AUTO_ACTIVATE: configure failed, aborting auto-activate"
            )
            return
        self.get_logger().info("CW_ROS2_AUTO_ACTIVATE: triggering activate")
        req = ChangeState.Request()
        req.transition.id = Transition.TRANSITION_ACTIVATE
        future = self._change_state_cli.call_async(req)
        future.add_done_callback(self._auto_activate_on_activate)

    def _auto_activate_on_activate(self, future) -> None:
        resp = future.result()
        if not resp or not resp.success:
            self.get_logger().error("CW_ROS2_AUTO_ACTIVATE: activate failed")
        else:
            self.get_logger().info("CW_ROS2_AUTO_ACTIVATE: node is now ACTIVE")
