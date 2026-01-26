"""
AMR Edge Node base class for Cyberwave.

Extends BaseEdgeNode with AMR/AGV-specific functionality:
- Protocol adapter management (Agilox REST, VDA5050 MQTT, etc.)
- Position streaming from robot to digital twin
- Navigation command handling
- Battery and telemetry reporting
- Map synchronization

Example:
    from cyberwave.edge import AMREdgeNode, EdgeNodeConfig

    class MyAMRNode(AMREdgeNode):
        def _create_adapter(self):
            return MyVendorAdapter(self.adapter_config, self)

    if __name__ == "__main__":
        import asyncio
        node = MyAMRNode(EdgeNodeConfig.from_env())
        asyncio.run(node.run())
"""

import asyncio
import logging
import os
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from cyberwave.edge.base import BaseEdgeNode
from cyberwave.edge.config import EdgeNodeConfig

logger = logging.getLogger(__name__)


class RobotState(str, Enum):
    """Robot operational states."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    CHARGING = "charging"
    TELEOP = "teleop"


class NavigationStatus(str, Enum):
    """Navigation action statuses."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AdapterConfig:
    """Configuration for AMR protocol adapters."""

    # Adapter type (agilox, vda5050, etc.)
    adapter_type: str = field(
        default_factory=lambda: os.getenv("ADAPTER_TYPE", "")
    )

    # Connection settings
    host: str = field(default_factory=lambda: os.getenv("ADAPTER_HOST", ""))
    port: int = field(
        default_factory=lambda: int(os.getenv("ADAPTER_PORT", "0"))
    )

    # Authentication
    username: str = field(
        default_factory=lambda: os.getenv("ADAPTER_USERNAME", "")
    )
    password: str = field(
        default_factory=lambda: os.getenv("ADAPTER_PASSWORD", "")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("ADAPTER_API_KEY", "")
    )

    # Robot identity
    robot_id: str = field(
        default_factory=lambda: os.getenv("ADAPTER_ROBOT_ID", "")
    )

    # Polling/streaming settings
    position_poll_rate_hz: float = field(
        default_factory=lambda: float(os.getenv("POSITION_POLL_RATE_HZ", "10"))
    )
    telemetry_poll_rate_hz: float = field(
        default_factory=lambda: float(os.getenv("TELEMETRY_POLL_RATE_HZ", "1"))
    )

    # VDA5050-specific
    vda_manufacturer: str = field(
        default_factory=lambda: os.getenv("VDA_MANUFACTURER", "")
    )
    vda_serial_number: str = field(
        default_factory=lambda: os.getenv("VDA_SERIAL_NUMBER", "")
    )

    # Extra vendor-specific config (JSON string from env)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AdapterConfig":
        """Create adapter config from environment variables."""
        import json

        config = cls()
        extra_json = os.getenv("ADAPTER_EXTRA", "{}")
        try:
            config.extra = json.loads(extra_json)
        except json.JSONDecodeError:
            logger.warning(f"Invalid ADAPTER_EXTRA JSON: {extra_json}")
        return config


@dataclass
class RobotTelemetry:
    """Robot telemetry data structure."""

    # Position (meters, world frame)
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    rotation: Optional[Dict[str, float]] = None  # quaternion {w, x, y, z}
    velocity: Optional[Dict[str, float]] = None  # {linear, angular}

    # Battery
    battery_level: Optional[float] = None  # 0-100%
    battery_charging: bool = False

    # State
    state: RobotState = RobotState.IDLE
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Active action
    current_action_id: Optional[str] = None
    action_progress: Optional[float] = None  # 0-100%

    # Vendor-specific
    vendor_data: Dict[str, Any] = field(default_factory=dict)


class AMRAdapterProtocol(Protocol):
    """Protocol that all AMR adapters must implement."""

    async def connect(self) -> None:
        """Connect to the vendor robot system."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the vendor robot system."""
        ...

    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        ...

    async def poll_telemetry(self) -> Optional[RobotTelemetry]:
        """Poll current robot telemetry."""
        ...

    async def send_navigation_command(
        self,
        action_id: str,
        command: str,
        position: Optional[Dict[str, float]] = None,
        rotation: Optional[Dict[str, float]] = None,
        waypoints: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> bool:
        """Send navigation command to robot."""
        ...

    async def cancel_navigation(self, action_id: str) -> bool:
        """Cancel an in-progress navigation."""
        ...

    async def pause_navigation(self) -> bool:
        """Pause current navigation."""
        ...

    async def resume_navigation(self) -> bool:
        """Resume paused navigation."""
        ...

    def set_status_callback(
        self, callback: Callable[[str, str, Optional[str], Optional[float]], None]
    ) -> None:
        """Set callback for status updates: (action_id, status, message, progress)."""
        ...


class AMREdgeNode(BaseEdgeNode):
    """
    Base class for AMR/AGV edge nodes.

    Provides:
    - Protocol adapter management
    - Position streaming to Cyberwave
    - Navigation command handling
    - Telemetry reporting (battery, errors)
    - Graceful error handling and reconnection

    Subclasses must implement:
    - _create_adapter(): Create and return the protocol-specific adapter
    """

    def __init__(self, config: EdgeNodeConfig, adapter_config: Optional[AdapterConfig] = None):
        """
        Initialize the AMR edge node.

        Args:
            config: Base edge node configuration
            adapter_config: Optional adapter configuration (defaults from env)
        """
        super().__init__(config)

        self.adapter_config = adapter_config or AdapterConfig.from_env()
        self.adapter: Optional[AMRAdapterProtocol] = None

        # Background tasks
        self._position_task: Optional[asyncio.Task] = None
        self._telemetry_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # State tracking
        self._current_telemetry: Optional[RobotTelemetry] = None
        self._active_actions: Dict[str, Dict[str, Any]] = {}
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5.0

    # =========================================================================
    # Lifecycle (BaseEdgeNode implementation)
    # =========================================================================

    async def _setup(self) -> None:
        """Initialize AMR adapter and start streaming."""
        await self._discover_twins()

        # Create and connect adapter
        self.adapter = self._create_adapter()
        if self.adapter:
            self.adapter.set_status_callback(self._on_adapter_status)
            await self._connect_adapter()

            # Start background tasks
            self._position_task = asyncio.create_task(self._position_stream_loop())
            self._telemetry_task = asyncio.create_task(self._telemetry_poll_loop())

    async def _subscribe_to_commands(self) -> None:
        """Subscribe to navigation and mission command topics."""
        for twin_uuid in self._get_twin_uuids():
            self.subscribe_navigate_command(
                twin_uuid,
                lambda data, tu=twin_uuid: asyncio.create_task(
                    self._handle_navigate_command(tu, data)
                ),
            )
            self.subscribe_mission_command(
                twin_uuid,
                lambda data, tu=twin_uuid: asyncio.create_task(
                    self._handle_mission_command(tu, data)
                ),
            )
            logger.info(f"Subscribed to commands for twin {twin_uuid}")

    async def _cleanup(self) -> None:
        """Stop adapter and background tasks."""
        # Cancel background tasks
        for task in [self._position_task, self._telemetry_task, self._reconnect_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect adapter
        if self.adapter:
            await self.adapter.disconnect()

    def _build_health_status(self) -> Dict[str, Any]:
        """Build AMR-specific health status."""
        telemetry = self._current_telemetry

        return {
            "adapter_type": self.adapter_config.adapter_type,
            "adapter_connected": self.adapter.is_connected() if self.adapter else False,
            "robot_id": self.adapter_config.robot_id,
            "robot_state": telemetry.state.value if telemetry else "unknown",
            "battery_level": telemetry.battery_level if telemetry else None,
            "battery_charging": telemetry.battery_charging if telemetry else False,
            "active_actions": len(self._active_actions),
            "errors": telemetry.errors if telemetry else [],
            "position_rate_hz": self.adapter_config.position_poll_rate_hz,
        }

    # =========================================================================
    # Abstract method - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _create_adapter(self) -> Optional[AMRAdapterProtocol]:
        """
        Create and return the protocol-specific adapter.

        Override this to instantiate the appropriate adapter for your vendor.

        Returns:
            Adapter instance implementing AMRAdapterProtocol, or None.
        """
        pass

    # =========================================================================
    # Adapter Management
    # =========================================================================

    async def _connect_adapter(self) -> None:
        """Connect to adapter with retry logic."""
        if not self.adapter:
            return

        try:
            await self.adapter.connect()
            self._reconnect_attempts = 0
            logger.info(f"Adapter connected: {self.adapter_config.adapter_type}")
        except Exception as e:
            logger.error(f"Adapter connection failed: {e}")
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule adapter reconnection."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self._reconnect_attempts += 1
        delay = self._reconnect_delay * (2 ** min(self._reconnect_attempts - 1, 5))
        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")

        if self._reconnect_task:
            self._reconnect_task.cancel()
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))

    async def _reconnect_after_delay(self, delay: float) -> None:
        """Reconnect after delay."""
        await asyncio.sleep(delay)
        await self._connect_adapter()

    def _on_adapter_status(
        self,
        action_id: str,
        status: str,
        message: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        """
        Callback from adapter for navigation status updates.

        Args:
            action_id: Action ID being updated
            status: New status
            message: Optional message
            progress: Optional progress percentage
        """
        # Update active actions
        if action_id in self._active_actions:
            self._active_actions[action_id]["status"] = status
            if status in ("completed", "failed", "cancelled"):
                twin_uuid = self._active_actions[action_id].get("twin_uuid")
                del self._active_actions[action_id]
            else:
                twin_uuid = self._active_actions[action_id].get("twin_uuid")
        else:
            # Unknown action, use first twin
            twin_uuids = self._get_twin_uuids()
            twin_uuid = twin_uuids[0] if twin_uuids else None

        # Publish status to Cyberwave
        if twin_uuid:
            self.publish_nav_status(twin_uuid, action_id, status, message, progress)

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _position_stream_loop(self) -> None:
        """Stream robot position to Cyberwave at configured rate."""
        interval = 1.0 / self.adapter_config.position_poll_rate_hz

        while self.running:
            try:
                if self.adapter and self.adapter.is_connected():
                    telemetry = await self.adapter.poll_telemetry()
                    if telemetry and telemetry.position:
                        self._current_telemetry = telemetry

                        # Publish position to all twins
                        for twin_uuid in self._get_twin_uuids():
                            self.publish_position(
                                twin_uuid,
                                telemetry.position,
                                telemetry.rotation,
                            )

            except Exception as e:
                logger.warning(f"Position stream error: {e}")

            await asyncio.sleep(interval)

    async def _telemetry_poll_loop(self) -> None:
        """Poll and publish robot telemetry (battery, errors) at slower rate."""
        interval = 1.0 / self.adapter_config.telemetry_poll_rate_hz

        while self.running:
            try:
                if self.adapter and self.adapter.is_connected():
                    telemetry = await self.adapter.poll_telemetry()
                    if telemetry:
                        self._current_telemetry = telemetry

                        # Publish telemetry to all twins
                        for twin_uuid in self._get_twin_uuids():
                            # Battery
                            if telemetry.battery_level is not None:
                                self.publish_telemetry(
                                    twin_uuid,
                                    "battery",
                                    {
                                        "level": telemetry.battery_level,
                                        "charging": telemetry.battery_charging,
                                    },
                                )

                            # Errors
                            if telemetry.errors:
                                self.publish_telemetry(
                                    twin_uuid, "errors", {"errors": telemetry.errors}
                                )

                            # State change events
                            self.publish_event(
                                twin_uuid,
                                "robot_state",
                                {
                                    "state": telemetry.state.value,
                                    "vendor_data": telemetry.vendor_data,
                                },
                            )

            except Exception as e:
                logger.warning(f"Telemetry poll error: {e}")

            await asyncio.sleep(interval)

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _handle_navigate_command(
        self, twin_uuid: str, data: Dict[str, Any]
    ) -> None:
        """
        Handle navigation command from Cyberwave.

        Args:
            twin_uuid: Twin UUID
            data: Command data including:
                - action_id: Tracking ID
                - command: goto, path, stop, pause, resume
                - position: Target position for goto
                - waypoints: List of waypoints for path
                - rotation/yaw: Optional orientation
        """
        if not self.adapter:
            logger.error("No adapter configured, cannot handle navigation command")
            return

        action_id = data.get("action_id") or str(uuid.uuid4())
        command = data.get("command", "goto")

        logger.info(f"Navigation command: {command} (action_id={action_id})")

        # Track the action
        self._active_actions[action_id] = {
            "twin_uuid": twin_uuid,
            "command": command,
            "status": "queued",
            "data": data,
        }

        # Acknowledge receipt
        self.publish_nav_status(twin_uuid, action_id, "queued", "Command received")

        try:
            if command == "stop":
                await self.adapter.cancel_navigation(action_id)
            elif command == "pause":
                await self.adapter.pause_navigation()
            elif command == "resume":
                await self.adapter.resume_navigation()
            else:
                # goto, path, or other navigation commands
                success = await self.adapter.send_navigation_command(
                    action_id=action_id,
                    command=command,
                    position=data.get("position"),
                    rotation=data.get("rotation"),
                    waypoints=data.get("waypoints"),
                    yaw=data.get("yaw"),
                    constraints=data.get("constraints"),
                    metadata=data.get("metadata"),
                )
                if not success:
                    self.publish_nav_status(
                        twin_uuid, action_id, "failed", "Adapter rejected command"
                    )
                    del self._active_actions[action_id]

        except Exception as e:
            logger.error(f"Navigation command error: {e}")
            self.publish_nav_status(twin_uuid, action_id, "failed", str(e))
            if action_id in self._active_actions:
                del self._active_actions[action_id]

    async def _handle_mission_command(
        self, twin_uuid: str, data: Dict[str, Any]
    ) -> None:
        """
        Handle mission command from Cyberwave.

        Missions are orchestrated by the backend. The edge node receives
        individual navigation/action commands as part of mission execution.

        Args:
            twin_uuid: Twin UUID
            data: Command data including:
                - command: start, cancel, pause, resume
                - mission_execution_uuid: Mission execution ID
        """
        command = data.get("command")
        mission_uuid = data.get("mission_execution_uuid")

        logger.info(f"Mission command: {command} (mission={mission_uuid})")

        if command == "cancel":
            # Cancel all active actions for this mission
            for action_id, action_data in list(self._active_actions.items()):
                if action_data.get("mission_uuid") == mission_uuid:
                    await self.adapter.cancel_navigation(action_id)

        elif command == "pause":
            await self.adapter.pause_navigation()

        elif command == "resume":
            await self.adapter.resume_navigation()

    # =========================================================================
    # Map Synchronization (optional, override for vendor-specific)
    # =========================================================================

    async def sync_map(self, twin_uuid: str) -> bool:
        """
        Sync map from vendor robot to Cyberwave.

        Override this for vendor-specific map retrieval and conversion.

        Args:
            twin_uuid: Twin to publish map for

        Returns:
            True if map sync succeeded
        """
        logger.info("Map sync not implemented for this adapter")
        return False
