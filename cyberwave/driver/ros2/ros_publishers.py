"""Wire registry ``from_ros`` publishers on a :class:`BaseROS2Driver`."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..interface.registry import resolve_topic_path, _PublisherEntry
from ..interface.args import mqtt_spec, zenoh_spec
from .message_payload import (
    joint_positions_from_transport_payload,
    ros_joint_state_to_transport_payload,
    ros_message_to_transport_payload,
)
from .topic_discovery import RosTopicDiscoveryError, resolve_ros_message_class

if TYPE_CHECKING:
    from .base_ros2_driver import BaseROS2Driver

logger = logging.getLogger(__name__)


@dataclass
class _RosForwardHandle:
    entry: _PublisherEntry
    subscription: Any
    msg_type_string: str


def wire_ros_publishers(driver: BaseROS2Driver) -> None:
    """Subscribe on ROS topics and forward messages to Cyberwave MQTT/Zenoh."""
    if driver._ros_forward_handles:
        unwire_ros_publishers(driver)
    mode = driver.operation_mode
    entries = driver._interface.ros_forward_publishers_for_mode(mode)
    if not entries:
        driver.get_logger().info(
            f"wire_ros_publishers: no from_ros publishers for mode={mode}"
        )
        return

    driver.get_logger().info(
        f"wire_ros_publishers: wiring {len(entries)} ROS subscription(s) (mode={mode})"
    )
    loop = driver._require_driver_loop()
    twin_uuid = driver._twin_uuid_for_wire()
    prefix = driver._mqtt_prefix_for_wire()

    for entry in entries:
        ros_spec = entry.from_ros
        assert ros_spec is not None
        driver.get_logger().info(
            f"wire_ros_publishers: discovering type for {ros_spec.topic} "
            f"(timeout={ros_spec.discovery_timeout_s:.1f}s)"
        )
        try:
            msg_class, type_string = resolve_ros_message_class(
                driver,
                ros_spec.topic,
                timeout_s=ros_spec.discovery_timeout_s,
                poll_interval_s=ros_spec.discovery_poll_interval_s,
                msg_type=ros_spec.msg_type,
            )
        except RosTopicDiscoveryError as exc:
            driver.get_logger().error(f"{exc}")
            continue

        user_cb = entry.callbacks.callback
        if user_cb is None and _is_joint_state_type(type_string):
            transform = ros_joint_state_to_transport_payload
        else:
            transform = user_cb
        rx_count = 0
        last_rx_log_at = 0.0
        ros_topic = ros_spec.topic

        def make_handler(
            pub_entry: _PublisherEntry = entry,
            transform: Any = transform,
        ) -> None:
            def _on_ros_message(msg: Any) -> None:
                nonlocal rx_count, last_rx_log_at
                try:
                    if transform is not None:
                        payload = transform(msg)
                        if payload is None:
                            return
                        if not isinstance(payload, dict):
                            driver.get_logger().error(
                                f"ROS forward callback for {ros_spec.topic} must return "
                                f"dict, got {type(payload).__name__}"
                            )
                            return
                    else:
                        payload = ros_message_to_transport_payload(msg)
                except Exception:
                    logger.exception(
                        "ROS forward callback failed for %s", ros_topic
                    )
                    return

                acquire_slot = getattr(driver, "acquire_ros_stream_publish_slot", None)
                if callable(acquire_slot) and not acquire_slot(ros_topic):
                    return

                rx_count += 1
                now = time.monotonic()
                if rx_count == 1 or now - last_rx_log_at >= 5.0:
                    last_rx_log_at = now
                    joints_note = ""
                    positions = joint_positions_from_transport_payload(payload)
                    if positions:
                        joints_note = f" ({len(positions)} joints)"
                    driver.get_logger().info(
                        f"ROS RX {ros_topic}: {rx_count} message(s) — "
                        f"publishing to Cyberwave MQTT{joints_note}"
                    )

                asyncio.run_coroutine_threadsafe(
                    _publish_forward_payload(
                        driver,
                        pub_entry,
                        payload,
                        twin_uuid=twin_uuid,
                        prefix=prefix,
                    ),
                    loop,
                )

            return _on_ros_message

        qos = (
            ros_spec.qos_profile
            if ros_spec.qos_profile is not None
            else ros_spec.qos_depth
        )
        sub = driver.create_subscription(
            msg_class,
            ros_spec.topic,
            make_handler(),
            qos,
        )
        driver._ros_forward_handles.append(
            _RosForwardHandle(
                entry=entry,
                subscription=sub,
                msg_type_string=type_string,
            )
        )
        driver.get_logger().info(
            f"ROS forward: {ros_spec.topic} ({type_string}) -> Cyberwave MQTT"
        )


def unwire_ros_publishers(driver: BaseROS2Driver) -> None:
    """Destroy ROS subscriptions created by :func:`wire_ros_publishers`."""
    for handle in driver._ros_forward_handles:
        try:
            driver.destroy_subscription(handle.subscription)
        except Exception:
            logger.debug(
                "destroy_subscription failed for %s",
                handle.entry.from_ros.topic if handle.entry.from_ros else "?",
                exc_info=True,
            )
    driver._ros_forward_handles.clear()


async def _publish_forward_payload(
    driver: BaseROS2Driver,
    entry: _PublisherEntry,
    payload: dict[str, Any],
    *,
    twin_uuid: str,
    prefix: str,
) -> None:
    cw = driver._require_client()
    m = mqtt_spec(entry.topic)
    if m is not None and entry.publish_mode in {"mqtt", "dual"}:
        path = resolve_topic_path(m, twin_uuid, prefix=prefix)
        cw.mqtt.publish(path, payload)
    z = zenoh_spec(entry.topic)
    if z is not None and entry.publish_mode in {"zenoh", "dual"}:
        if z.wire_format == "ndarray":
            logger.debug(
                "Skipping Zenoh forward on %s (ndarray wire_format)", z.channel
            )
        else:
            driver._registry_zenoh_publish(z.channel, payload)


def _is_joint_state_type(type_string: str) -> bool:
    normalized = type_string.strip().lower()
    return normalized.endswith("/jointstate") or normalized.endswith("/joint_state")
