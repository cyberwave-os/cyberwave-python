"""
Edge node infrastructure for Cyberwave.

Provides base classes for building edge nodes that connect to the Cyberwave platform.

Base Classes:
    - BaseEdgeNode: Abstract base for all edge nodes
    - EdgeNodeConfig: Configuration dataclass

AMR/AGV Support:
    - AMREdgeNode: Base class for AMR/AGV nodes
    - AdapterConfig: Configuration for protocol adapters
    - RobotTelemetry: Telemetry data structure
    - RobotState, NavigationStatus: Enums

Example:
    from cyberwave.edge import AMREdgeNode, EdgeNodeConfig, AdapterConfig

    class MyAMRNode(AMREdgeNode):
        def _create_adapter(self):
            return MyVendorAdapter(self.adapter_config, self)

    node = MyAMRNode(EdgeNodeConfig.from_env())
    asyncio.run(node.run())
"""

from cyberwave.edge.base import BaseEdgeNode
from cyberwave.edge.config import EdgeNodeConfig
from cyberwave.edge.amr import (
    AMREdgeNode,
    AdapterConfig,
    RobotTelemetry,
    RobotState,
    NavigationStatus,
    AMRAdapterProtocol,
)
from cyberwave.edge.host_metrics import (
    CPU_THERMAL_ZONE_TYPES,
    HARDWARE_WATCHDOG_DEVICE,
    HostCpuTemperature,
    HostFacts,
    HostMemoryInfo,
    discover_cpu_thermal_zones,
    read_host_cpu_temperature,
    read_host_facts,
    read_host_memory,
    read_thermal_zone_celsius,
)
from cyberwave.edge.platform import is_port_listening, is_usbip_server_running

__all__ = [
    # Base
    "BaseEdgeNode",
    "EdgeNodeConfig",
    # AMR
    "AMREdgeNode",
    "AdapterConfig",
    "RobotTelemetry",
    "RobotState",
    "NavigationStatus",
    "AMRAdapterProtocol",
    # Platform detection
    "is_port_listening",
    "is_usbip_server_running",
    # Host metrics
    "CPU_THERMAL_ZONE_TYPES",
    "HARDWARE_WATCHDOG_DEVICE",
    "HostCpuTemperature",
    "HostFacts",
    "HostMemoryInfo",
    "discover_cpu_thermal_zones",
    "read_host_cpu_temperature",
    "read_host_facts",
    "read_host_memory",
    "read_thermal_zone_celsius",
]
