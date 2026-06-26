from .manifest import (
    DRIVER_MANIFEST_FILE_NAME,
    ManifestParam,
    ManifestService,
    ManifestTopic,
    NodeManifest,
    ServiceRole,
    TopicRole,
    default_node_manifest,
    dump_combined_driver_manifest,
    load_manifest,
    merge_combined_driver_manifest,
    node_manifest_to_dict,
    resolve_node_manifest,
)
from .message_payload import (
    joint_positions_from_transport_payload,
    ros_joint_state_to_transport_payload,
    ros_message_to_transport_payload,
)
from .topic_discovery import RosTopicDiscoveryError, resolve_ros_message_class
from .topic_spec import Ros2TopicSpec

__all__ = [
    "BaseROS2Driver",
    "Ros2TopicSpec",
    "RosTopicDiscoveryError",
    "resolve_ros_message_class",
    "joint_positions_from_transport_payload",
    "ros_joint_state_to_transport_payload",
    "ros_message_to_transport_payload",
    "NodeManifest",
    "ManifestParam",
    "ManifestTopic",
    "ManifestService",
    "TopicRole",
    "ServiceRole",
    "DRIVER_MANIFEST_FILE_NAME",
    "default_node_manifest",
    "dump_combined_driver_manifest",
    "load_manifest",
    "merge_combined_driver_manifest",
    "node_manifest_to_dict",
    "resolve_node_manifest",
]


def __getattr__(name: str):  # noqa: ANN001
    if name == "BaseROS2Driver":
        from .base_ros2_driver import BaseROS2Driver

        return BaseROS2Driver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
