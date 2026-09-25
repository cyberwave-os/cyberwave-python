"""ROS node manifest defaults when YAML is missing."""

from __future__ import annotations

from cyberwave.driver.ros2.manifest import default_node_manifest, load_manifest


def test_load_manifest_missing_file_uses_defaults() -> None:
    m = load_manifest("/nonexistent/manifest.yaml", node_name="ursim_driver")
    assert m.node_name == "ursim_driver"
    assert any(p.name == "tick_rate_hz" for p in m.params)


def test_load_manifest_none_uses_defaults() -> None:
    m = load_manifest(None, node_name="test_node")
    assert m.node_name == "test_node"
    assert m.description


def test_default_node_manifest_tick_rate() -> None:
    m = default_node_manifest("foo")
    tick = next(p for p in m.params if p.name == "tick_rate_hz")
    assert tick.default_value == "10"
    assert tick.read_only is True
