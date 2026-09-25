"""Combined ROS + MQTT manifest helpers."""

from __future__ import annotations

from cyberwave.driver.ros2.manifest import (
    default_node_manifest,
    merge_combined_driver_manifest,
    node_manifest_to_dict,
)


def test_merge_combined_driver_manifest_keeps_ros_and_mqtt() -> None:
    node = default_node_manifest("test_node")
    cw = {
        "registry_id": "test/registry",
        "driver_family": "ros_python",
        "mqtt": {"joint": {"update": {"direction": "both"}}},
    }
    combined = merge_combined_driver_manifest(node, cw)
    assert combined["node_name"] == "test_node"
    assert combined["registry_id"] == "test/registry"
    assert combined["mqtt"]["joint"]["update"]["direction"] == "both"
    assert "parameters" in combined


def test_node_manifest_to_dict_round_trip_keys() -> None:
    node = default_node_manifest("foo")
    data = node_manifest_to_dict(node)
    assert data["node_name"] == "foo"
    assert any(p["name"] == "tick_rate_hz" for p in data["parameters"])
