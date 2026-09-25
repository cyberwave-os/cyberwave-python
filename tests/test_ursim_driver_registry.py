"""UrSim driver declares three ROS-forward publishers without ROS msg imports."""

from __future__ import annotations

import ast
from pathlib import Path


def test_ursim_driver_source_has_no_ros_message_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "ursim_driver.py"
    source = path.read_text()
    tree = ast.parse(source)
    imports = {
        (n.module if isinstance(n, ast.ImportFrom) else alias.name)
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for alias in n.names
    }
    imports |= {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    }
    forbidden = (
        "sensor_msgs",
        "geometry_msgs",
        "std_msgs",
    )
    for name in imports:
        mod = name or ""
        for prefix in forbidden:
            assert not mod.startswith(prefix), f"unexpected ROS msg import: {mod}"


def test_ursim_driver_manifest_ros_forward_count() -> None:
    from cyberwave.driver import CallbackGroup, DriverInterfaceRegistry, TopicSpec
    from cyberwave.driver.ros2 import Ros2TopicSpec

    JOINT_STATES_SLUG = "cyberwave/ursim/{twin_uuid}/joint_states"
    TCP_POSE_SLUG = "cyberwave/ursim/{twin_uuid}/tcp_pose"
    URSCRIPT_SLUG = "cyberwave/ursim/{twin_uuid}/urscript_command"

    iface = DriverInterfaceRegistry()

    def _define(iface: DriverInterfaceRegistry) -> None:
        iface.add_publisher(
            TopicSpec(
                topic_slug=JOINT_STATES_SLUG,
                payload_schema_ref="JointStatesPayload",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/joint_states"),
        )
        iface.add_publisher(
            TopicSpec(
                topic_slug=TCP_POSE_SLUG,
                payload_schema_ref="PoseStampedPayload",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/tcp_pose_broadcaster/pose"),
        )
        iface.add_publisher(
            TopicSpec(
                topic_slug=URSCRIPT_SLUG,
                payload_schema_ref="StringPayload",
            ),
            CallbackGroup(),
            from_ros=Ros2TopicSpec(topic="/urscript_interface/script_command"),
        )

    _define(iface)
    raw = iface.to_cw_driver_dict(registry_id="universalrobots/ur-sim")
    forward = raw.get("ros2", {}).get("forward_publishers", [])
    assert len(forward) == 3
    topics = {item["ros_topic"] for item in forward}
    assert "/joint_states" in topics
    assert "/tcp_pose_broadcaster/pose" in topics
    assert "/urscript_interface/script_command" in topics
    slugs = {item.get("mqtt_topic_slug") for item in forward}
    assert JOINT_STATES_SLUG in slugs
    assert TCP_POSE_SLUG in slugs
    assert URSCRIPT_SLUG in slugs
