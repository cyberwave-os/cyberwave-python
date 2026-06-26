"""Declarative driver interface: topic/command specs, the registry that compiles
them to a manifest and wires MQTT/Zenoh, source-type routing, and the thread-safe
command inbox.

A driver author touches this subpackage mainly through
:class:`~cyberwave.driver.interface.args.TopicSpec` /
:class:`~cyberwave.driver.interface.args.CommandArgs` in ``define_interface``.
"""

from .args import (
    CallbackGroup,
    CommandArg,
    CommandArgs,
    DriverOperationMode,
    ProtocolArgs,
    PublisherArgs,
    TopicSpec,
)
from .command_inbox import CommandInbox
from .cw_driver import (
    CW_DRIVER_FILE_NAME,
    dump_cw_driver_yml,
    load_cw_driver_yml,
    resolve_driver_config_dict,
)
from .registry import (
    DriverInterfaceRegistry,
    default_management_commands,
    resolve_topic_path,
)
from .registry_mixin import InterfaceRegistryMixin
from .source_type_policy import (
    COMMAND_SOURCE_TYPES,
    DEFAULT_PUBLISH_SOURCE_TYPE,
    DEFAULT_SIM_PUBLISH_SOURCE_TYPE,
    accepts_inbound,
    filtered_listener,
)
from .stream_publish_rate import StreamPublishRateLimiter

__all__ = [
    "CallbackGroup",
    "CommandArg",
    "CommandArgs",
    "DriverOperationMode",
    "ProtocolArgs",
    "PublisherArgs",
    "TopicSpec",
    "CommandInbox",
    "DriverInterfaceRegistry",
    "default_management_commands",
    "resolve_topic_path",
    "InterfaceRegistryMixin",
    "COMMAND_SOURCE_TYPES",
    "DEFAULT_PUBLISH_SOURCE_TYPE",
    "DEFAULT_SIM_PUBLISH_SOURCE_TYPE",
    "accepts_inbound",
    "filtered_listener",
    "StreamPublishRateLimiter",
    "CW_DRIVER_FILE_NAME",
    "dump_cw_driver_yml",
    "load_cw_driver_yml",
    "resolve_driver_config_dict",
]
