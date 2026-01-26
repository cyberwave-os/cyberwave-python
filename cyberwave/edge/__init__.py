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
]
