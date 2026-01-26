"""
Base class for Cyberwave edge nodes.

Provides common infrastructure for connecting to and communicating with the
Cyberwave platform via REST and MQTT. Extend this class to build specialized
edge nodes (AMR, video, sensor, etc.).

Example:
    from cyberwave.edge import BaseEdgeNode, EdgeNodeConfig

    class MyRobotNode(BaseEdgeNode):
        async def _setup(self) -> None:
            # Initialize your hardware
            pass

        async def _subscribe_to_commands(self) -> None:
            # Subscribe to command topics
            for twin_uuid in self._get_twin_uuids():
                self.subscribe_navigate_command(twin_uuid, self.handle_nav)

        async def _cleanup(self) -> None:
            # Release resources
            pass

        def _build_health_status(self) -> dict:
            return {"robot_status": "operational"}

    if __name__ == "__main__":
        import asyncio
        node = MyRobotNode(EdgeNodeConfig.from_env())
        asyncio.run(node.run())
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from cyberwave.edge.config import EdgeNodeConfig

logger = logging.getLogger(__name__)


class BaseEdgeNode(ABC):
    """
    Abstract base class for Cyberwave edge nodes.

    Provides:
    - Cyberwave client (REST + MQTT) connection
    - Twin discovery from edge device registration
    - Periodic health status publishing
    - MQTT publishing helpers (position, status, events)
    - Graceful shutdown handling

    Subclasses must implement:
    - _setup(): Node-specific initialization (hardware, sensors, etc.)
    - _subscribe_to_commands(): Subscribe to MQTT command topics
    - _cleanup(): Node-specific cleanup on shutdown
    - _build_health_status(): Return node-specific health data
    """

    def __init__(self, config: EdgeNodeConfig):
        """
        Initialize the edge node.

        Args:
            config: Configuration for the edge node (can load from env vars)
        """
        self.config = config
        self.running = False
        self.client: Optional[Any] = None  # Cyberwave client instance
        self._health_task: Optional[asyncio.Task] = None
        self._discovered_twins: List[Dict[str, Any]] = []
        self._start_time: float = 0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def run(self) -> None:
        """
        Run the edge node (main entry point).

        Connects to Cyberwave, initializes the node, subscribes to commands,
        and runs until shutdown is requested.
        """
        self.config.validate()
        self.running = True
        self._start_time = time.time()

        try:
            await self._connect()
            await self._setup()
            await self._subscribe_to_commands()

            # Start health publishing
            self._health_task = asyncio.create_task(self._health_loop())

            logger.info(f"Edge node {self.config.edge_uuid} running")
            await self._main_loop()

        except Exception as e:
            logger.error(f"Edge node error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown the edge node."""
        logger.info(f"Shutting down edge node {self.config.edge_uuid}")
        self.running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()

        if self.client:
            self.client.disconnect()

    async def _connect(self) -> None:
        """Connect to the Cyberwave backend (REST + MQTT)."""
        from cyberwave import Cyberwave

        self.client = Cyberwave(
            token=self.config.cyberwave_token,
            base_url=self.config.cyberwave_base_url,
            mqtt_host=self.config.mqtt_host,
            mqtt_port=self.config.mqtt_port,
            mqtt_username=self.config.mqtt_username,
            mqtt_password=self.config.mqtt_password,
            topic_prefix=self.config.topic_prefix,
            source_type=self.config.source_type,
        )
        self.client.mqtt.connect()
        logger.info(f"Connected to Cyberwave at {self.config.cyberwave_base_url}")

    async def _main_loop(self) -> None:
        """
        Main loop. Override for custom behavior.

        Default implementation runs until self.running is False.
        """
        while self.running:
            await asyncio.sleep(1)

    async def _health_loop(self) -> None:
        """Publish health status at configured interval."""
        while self.running:
            await asyncio.sleep(self.config.health_publish_interval)
            try:
                health = self._build_health_status()
                for twin_uuid in self._get_twin_uuids():
                    self.publish_health(twin_uuid, health)
            except Exception as e:
                logger.warning(f"Failed to publish health: {e}")

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def _setup(self) -> None:
        """
        Node-specific setup.

        Called after connecting to Cyberwave. Initialize hardware, load
        adapters, discover twins, etc.
        """
        pass

    @abstractmethod
    async def _subscribe_to_commands(self) -> None:
        """
        Subscribe to MQTT command topics.

        Use the helper methods like subscribe_navigate_command() or
        subscribe directly via self.client.mqtt.subscribe().
        """
        pass

    @abstractmethod
    async def _cleanup(self) -> None:
        """
        Node-specific cleanup on shutdown.

        Stop streams, release hardware, close connections.
        """
        pass

    @abstractmethod
    def _build_health_status(self) -> Dict[str, Any]:
        """
        Build node-specific health status.

        Returns:
            Dictionary with health information specific to this node type.
        """
        pass

    # =========================================================================
    # MQTT Publishing Helpers
    # =========================================================================

    def publish_position(
        self,
        twin_uuid: str,
        position: Dict[str, float],
        rotation: Optional[Dict[str, float]] = None,
        source_type: Optional[str] = None,
    ) -> None:
        """
        Publish position update for a twin.

        Args:
            twin_uuid: UUID of the twin
            position: Position with x, y, z keys
            rotation: Optional quaternion with w, x, y, z keys
            source_type: Override source type (defaults to config)
        """
        if not self.client:
            return

        source = source_type or self.config.source_type
        prefix = self.client.mqtt.topic_prefix

        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/position",
            {"position": position, "source_type": source, "timestamp": time.time()},
        )

        if rotation:
            self.client.mqtt.publish(
                f"{prefix}cyberwave/twin/{twin_uuid}/rotation",
                {"rotation": rotation, "source_type": source, "timestamp": time.time()},
            )

    def publish_joint_states(
        self,
        twin_uuid: str,
        joint_states: Dict[str, float],
        source_type: Optional[str] = None,
    ) -> None:
        """
        Publish joint states for a twin.

        Args:
            twin_uuid: UUID of the twin
            joint_states: Dict mapping joint names to positions (radians)
            source_type: Override source type
        """
        if not self.client:
            return

        source = source_type or self.config.source_type
        prefix = self.client.mqtt.topic_prefix

        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/joint_states",
            {
                "joint_states": joint_states,
                "source_type": source,
                "timestamp": time.time(),
            },
        )

    def publish_nav_status(
        self,
        twin_uuid: str,
        action_id: str,
        status: str,
        message: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        """
        Publish navigation status.

        Args:
            twin_uuid: UUID of the twin
            action_id: Action ID for tracking
            status: Status (queued, running, completed, failed, cancelled)
            message: Optional status message
            progress: Optional progress percentage (0-100)
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        payload: Dict[str, Any] = {
            "action_id": action_id,
            "status": status,
            "timestamp": time.time(),
        }
        if message:
            payload["message"] = message
        if progress is not None:
            payload["progress"] = progress

        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/navigate/status",
            payload,
        )

    def publish_health(self, twin_uuid: str, health_data: Dict[str, Any]) -> None:
        """
        Publish health status for a twin.

        Args:
            twin_uuid: UUID of the twin
            health_data: Health status data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        payload = {
            **health_data,
            "edge_uuid": self.config.edge_uuid,
            "timestamp": time.time(),
            "uptime": time.time() - self._start_time,
        }

        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/edge_health",
            payload,
        )

    def publish_event(
        self, twin_uuid: str, event_type: str, data: Dict[str, Any]
    ) -> None:
        """
        Publish a business event.

        Args:
            twin_uuid: UUID of the twin
            event_type: Type of event (e.g., "object_detected", "task_completed")
            data: Event data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/event",
            {
                "event_type": event_type,
                "source": "edge_node",
                "data": data,
                "timestamp": time.time(),
            },
        )

    def publish_map_update(
        self,
        twin_uuid: str,
        pointcloud: List[List[float]],
        map_type: str = "point_cloud",
        resolution: float = 0.05,
    ) -> None:
        """
        Publish map update for a twin.

        Args:
            twin_uuid: UUID of the twin
            pointcloud: List of [x, y, z, r, g, b] points
            map_type: Type of map (point_cloud or occupancy_grid)
            resolution: Map resolution in meters
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/map_update",
            {
                "pointcloud": pointcloud,
                "map_type": map_type,
                "resolution": resolution,
                "timestamp": time.time(),
            },
        )

    def publish_telemetry(
        self,
        twin_uuid: str,
        telemetry_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Publish telemetry data (battery, errors, diagnostics, etc.).

        Args:
            twin_uuid: UUID of the twin
            telemetry_type: Type of telemetry (battery, error, diagnostic, etc.)
            data: Telemetry payload
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/telemetry",
            {
                "telemetry_type": telemetry_type,
                "data": data,
                "source_type": self.config.source_type,
                "timestamp": time.time(),
            },
        )

    def publish_mission_status(
        self,
        twin_uuid: str,
        mission_execution_uuid: str,
        status: str,
        current_step: int,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish mission execution status update.

        Args:
            twin_uuid: UUID of the twin
            mission_execution_uuid: UUID of the mission execution
            status: Status (queued, running, completed, failed, cancelled)
            current_step: Current step index (0-indexed)
            message: Optional status message
            result: Optional result data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        payload: Dict[str, Any] = {
            "mission_execution_uuid": mission_execution_uuid,
            "status": status,
            "current_step": current_step,
            "timestamp": time.time(),
        }
        if message:
            payload["message"] = message
        if result:
            payload["result"] = result

        self.client.mqtt.publish(
            f"{prefix}cyberwave/twin/{twin_uuid}/mission/status",
            payload,
        )

    # =========================================================================
    # MQTT Subscription Helpers
    # =========================================================================

    def subscribe_navigate_command(
        self, twin_uuid: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to navigation commands for a twin.

        Args:
            twin_uuid: UUID of the twin
            handler: Callback function receiving command data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        topic = f"{prefix}cyberwave/twin/{twin_uuid}/navigate/command"
        self.client.mqtt.subscribe(topic, handler)
        logger.info(f"Subscribed to navigate/command for twin {twin_uuid}")

    def subscribe_motion_command(
        self, twin_uuid: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to motion commands for a twin.

        Args:
            twin_uuid: UUID of the twin
            handler: Callback function receiving command data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        topic = f"{prefix}cyberwave/twin/{twin_uuid}/motion/command"
        self.client.mqtt.subscribe(topic, handler)
        logger.info(f"Subscribed to motion/command for twin {twin_uuid}")

    def subscribe_mission_command(
        self, twin_uuid: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to mission commands for a twin.

        Args:
            twin_uuid: UUID of the twin
            handler: Callback function receiving mission command data
        """
        if not self.client:
            return

        prefix = self.client.mqtt.topic_prefix
        topic = f"{prefix}cyberwave/twin/{twin_uuid}/mission/command"
        self.client.mqtt.subscribe(topic, handler)
        logger.info(f"Subscribed to mission/command for twin {twin_uuid}")

    # =========================================================================
    # Twin Discovery
    # =========================================================================

    def _get_twin_uuids(self) -> List[str]:
        """
        Get list of twin UUIDs this node serves.

        Returns:
            List of twin UUIDs (from discovery or config)
        """
        uuids = [
            t.get("twin_uuid")
            for t in self._discovered_twins
            if t.get("twin_uuid")
        ]
        if not uuids and self.config.twin_uuid:
            uuids = [self.config.twin_uuid]
        return uuids

    async def _discover_twins(self) -> List[Dict[str, Any]]:
        """
        Discover twins paired to this edge device.

        Uses the REST API to find twins associated with this edge UUID.

        Returns:
            List of twin info dictionaries
        """
        if not self.client or not self.config.edge_uuid:
            return []

        try:
            # Query edge device to get paired twins
            edge = self.client.get_edge(self.config.edge_uuid)
            if edge and edge.get("twins"):
                self._discovered_twins = [
                    {"twin_uuid": t.get("uuid"), "name": t.get("name")}
                    for t in edge.get("twins", [])
                ]
            logger.info(f"Discovered {len(self._discovered_twins)} twins")
            return self._discovered_twins
        except Exception as e:
            logger.warning(f"Failed to discover twins: {e}")
            return []
