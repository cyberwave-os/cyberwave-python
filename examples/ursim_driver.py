from __future__ import annotations

import logging
import os
import sys

from cyberwave.driver import CallbackGroup, DriverOperationMode, TopicSpec
from cyberwave.driver.ros2 import BaseROS2Driver, Ros2TopicSpec

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_ID = "universal_robots/UR7"
JOINT_STATES_SLUG = "cyberwave/joint/{twin_uuid}/update"


class UrSimDriver(BaseROS2Driver):
    """Forward UR Sim ROS topics to Cyberwave MQTT."""

    REGISTRY_ID = DEFAULT_REGISTRY_ID
    driver_family = "python"

    def define_interface(self, iface) -> None:
        all_modes = frozenset(DriverOperationMode)
        logger.info("Defining UR Sim interface (joint_states forward)")
        iface.add_publisher(
            TopicSpec(
                topic_slug=JOINT_STATES_SLUG,
                payload_schema_ref="JointStatesPayload",
                description="UR Sim /joint_states forward",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/joint_states"),
            operation_modes=all_modes,
        )

    def configure(self) -> None:
        self.get_logger().info("UrSimDriver.configure (ROS lifecycle configure)")

    def connect_to_device(self) -> None:
        self.get_logger().info("UrSimDriver.connect_to_device (UR Sim / ROS graph)")

    def activate(self) -> None:
        self.get_logger().info("UrSimDriver.activate — tick timer will start")


def main() -> None:
    os.environ.setdefault("CW_ROS2_AUTO_ACTIVATE", "true")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    manifest = os.environ.get("CW_DRIVER_MANIFEST") or None
    node_name = os.environ.get("CW_ROS2_NODE_NAME", "ursim_driver")
    logger.info(
        "Running Cyberwave driver for asset %s (node=%s, manifest=%s, "
        "CW_ROS2_AUTO_ACTIVATE=%s)",
        DEFAULT_REGISTRY_ID,
        node_name,
        manifest or "<defaults>",
        os.environ.get("CW_ROS2_AUTO_ACTIVATE"),
    )
    driver = UrSimDriver(node_name, manifest)
    try:
        driver.run()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    except TimeoutError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
