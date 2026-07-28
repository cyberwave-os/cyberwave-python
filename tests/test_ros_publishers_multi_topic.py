"""Regression: wire_ros_publishers must not leak per-topic state across
handlers when multiple ``from_ros`` publishers are wired at once.

Each handler closed over ``ros_topic``/``topic_rate_hz`` as free variables
instead of default-arg-capturing them, so with 2+ publishers every handler
resolved both to the *last* loop iteration's values.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cyberwave.driver import CallbackGroup, DriverInterfaceRegistry, TopicSpec
from cyberwave.driver.ros2 import Ros2TopicSpec
from cyberwave.driver.ros2.ros_publishers import wire_ros_publishers


class _FakeMsg:
    pass


def _driver_with_two_ros_publishers() -> MagicMock:
    iface = DriverInterfaceRegistry()
    iface.add_publisher(
        TopicSpec(
            topic_slug="cyberwave/test/{twin_uuid}/slow",
            payload_schema_ref="Slow",
            rate_hz=5.0,
        ),
        CallbackGroup(),
        from_ros=Ros2TopicSpec(topic="/slow_topic", msg_type=_FakeMsg),
    )
    iface.add_publisher(
        TopicSpec(
            topic_slug="cyberwave/test/{twin_uuid}/fast",
            payload_schema_ref="Fast",
            rate_hz=50.0,
        ),
        CallbackGroup(),
        from_ros=Ros2TopicSpec(topic="/fast_topic", msg_type=_FakeMsg),
    )

    driver = MagicMock()
    driver._ros_forward_handles = []
    driver.operation_mode = "teleop_local"
    driver._interface = iface
    driver._require_driver_loop.return_value = MagicMock()
    driver._twin_uuid_for_wire.return_value = "twin-1"
    driver._mqtt_prefix_for_wire.return_value = "cyberwave"
    driver.convert_joints_to_payload = MagicMock(return_value={})
    driver.acquire_ros_stream_publish_slot = MagicMock(return_value=True)

    captured_handlers: dict[str, object] = {}

    def _create_subscription(msg_class, topic, handler, qos):
        captured_handlers[topic] = handler
        return MagicMock()

    driver.create_subscription.side_effect = _create_subscription
    driver._captured_handlers = captured_handlers
    return driver


def test_each_ros_publisher_handler_uses_its_own_topic_and_rate() -> None:
    driver = _driver_with_two_ros_publishers()

    with (
        patch(
            "cyberwave.driver.ros2.ros_publishers.resolve_ros_message_class",
            return_value=(_FakeMsg, "std_msgs/msg/String"),
        ),
        patch(
            "cyberwave.driver.ros2.ros_publishers.ros_message_to_transport_payload",
            return_value={"value": 1},
        ),
        patch(
            "cyberwave.driver.ros2.ros_publishers._publish_forward_payload",
            new=lambda *a, **k: None,
        ),
        patch("asyncio.run_coroutine_threadsafe"),
    ):
        wire_ros_publishers(driver)

        handlers = driver._captured_handlers
        assert set(handlers) == {"/slow_topic", "/fast_topic"}

        handlers["/slow_topic"](_FakeMsg())
        handlers["/fast_topic"](_FakeMsg())

    calls = driver.acquire_ros_stream_publish_slot.call_args_list
    assert len(calls) == 2
    seen = {call.args[0]: call.kwargs["max_hz"] for call in calls}
    assert seen == {"/slow_topic": 5.0, "/fast_topic": 50.0}
