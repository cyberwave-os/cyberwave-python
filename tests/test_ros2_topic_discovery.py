"""Tests for ROS topic type discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cyberwave.driver.ros2.topic_discovery import (
    RosTopicDiscoveryError,
    resolve_ros_message_class,
)
from cyberwave.driver.ros2.topic_spec import Ros2TopicSpec


class _FakeMsg:
    pass


def test_resolve_ros_message_class_override() -> None:
    node = MagicMock()
    msg_type, type_string = resolve_ros_message_class(
        node,
        "/joint_states",
        timeout_s=0.1,
        poll_interval_s=0.01,
        msg_type=_FakeMsg,
    )
    assert msg_type is _FakeMsg
    assert _FakeMsg.__name__ in type_string


def test_resolve_ros_message_class_from_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    node = MagicMock()
    node.get_topic_names_and_types.return_value = [
        ("/joint_states", ["sensor_msgs/msg/JointState"]),
    ]

    class _JointState:
        pass

    utilities = ModuleType("rosidl_runtime_py.utilities")
    utilities.get_message = lambda name: _JointState  # type: ignore[attr-defined]
    rosidl_pkg = ModuleType("rosidl_runtime_py")
    rosidl_pkg.utilities = utilities  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", rosidl_pkg)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", utilities)

    msg_type, type_string = resolve_ros_message_class(
        node,
        "/joint_states",
        timeout_s=1.0,
        poll_interval_s=0.01,
    )
    assert msg_type is _JointState
    assert type_string == "sensor_msgs/msg/JointState"


def test_resolve_ros_message_class_timeout() -> None:
    node = MagicMock()
    node.get_topic_names_and_types.return_value = []
    with pytest.raises(RosTopicDiscoveryError):
        resolve_ros_message_class(
            node,
            "/missing",
            timeout_s=0.05,
            poll_interval_s=0.01,
        )


def test_ros2_topic_spec_defaults() -> None:
    spec = Ros2TopicSpec(topic="/foo")
    assert spec.topic == "/foo"
    assert spec.msg_type is None
    assert spec.discovery_timeout_s == 5.0


def test_resolve_ros_message_class_relative_topic_under_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative Ros2TopicSpec topic must resolve under the node's namespace
    before matching get_topic_names_and_types() (which is always fully-qualified) —
    otherwise a namespaced driver (e.g. /twin_<uuid>/joint_states_single) never
    discovers its own relative "joint_states_single" topic."""
    import sys
    from types import ModuleType

    node = MagicMock()
    node.resolve_topic_name.side_effect = lambda t: f"/twin_abc123/{t}"
    node.get_topic_names_and_types.return_value = [
        ("/twin_abc123/joint_states_single", ["sensor_msgs/msg/JointState"]),
    ]

    class _JointState:
        pass

    utilities = ModuleType("rosidl_runtime_py.utilities")
    utilities.get_message = lambda name: _JointState  # type: ignore[attr-defined]
    rosidl_pkg = ModuleType("rosidl_runtime_py")
    rosidl_pkg.utilities = utilities  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", rosidl_pkg)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", utilities)

    msg_type, type_string = resolve_ros_message_class(
        node,
        "joint_states_single",
        timeout_s=1.0,
        poll_interval_s=0.01,
    )
    assert msg_type is _JointState
    assert type_string == "sensor_msgs/msg/JointState"


def test_resolve_ros_message_class_falls_back_without_resolver() -> None:
    """A node without resolve_topic_name (or one that misbehaves) keeps matching
    the raw topic literally — preserves behavior for already-absolute topics."""
    node = MagicMock(spec=["get_topic_names_and_types"])
    node.get_topic_names_and_types.return_value = []
    with pytest.raises(RosTopicDiscoveryError):
        resolve_ros_message_class(
            node,
            "/missing",
            timeout_s=0.05,
            poll_interval_s=0.01,
        )
