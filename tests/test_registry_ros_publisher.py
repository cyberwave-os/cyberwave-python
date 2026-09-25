"""Registry tests for add_publisher(..., from_ros=...)."""

from __future__ import annotations

import pytest

from cyberwave.driver import CallbackGroup, DriverInterfaceRegistry, TopicSpec
from cyberwave.driver.ros2 import Ros2TopicSpec


class _ProbeDriver:
    REGISTRY_ID = "test/asset"
    driver_family = "python"

    def __init__(self) -> None:
        self._interface = DriverInterfaceRegistry()
        self.define_interface(self._interface)

    def define_interface(self, iface: DriverInterfaceRegistry) -> None:
        iface.add_publisher(
            TopicSpec(
                topic_slug="cyberwave/test/{twin_uuid}/joint_states",
                payload_schema_ref="JointStatesPayload",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/joint_states"),
        )


def test_add_publisher_from_ros_manifest_export() -> None:
    probe = _ProbeDriver()
    raw = probe._interface.to_cw_driver_dict(registry_id="test/asset")
    assert "ros2" in raw
    forward = raw["ros2"]["forward_publishers"]
    assert len(forward) == 1
    assert forward[0]["ros_topic"] == "/joint_states"
    assert "mqtt_topic_slug" in forward[0]


def test_add_publisher_from_ros_rejects_rate_hz() -> None:
    from cyberwave.driver import PublisherArgs

    iface = DriverInterfaceRegistry()
    with pytest.raises(ValueError, match="rate_hz"):
        iface.add_publisher(
            TopicSpec(
                topic_slug="cyberwave/test/{twin_uuid}/x",
                payload_schema_ref="X",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/x"),
            publisher=PublisherArgs(rate_hz=10.0),
        )


def test_add_listener_rejects_ros2_topic_spec() -> None:
    iface = DriverInterfaceRegistry()
    with pytest.raises(TypeError, match="add_listener expects"):
        iface.add_listener(
            Ros2TopicSpec(topic="/bad"),  # type: ignore[arg-type]
            CallbackGroup(),
        )
