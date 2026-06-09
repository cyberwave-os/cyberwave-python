from __future__ import annotations

from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from ._helpers import _get_asset_capabilities
from .base import Twin
from .classes import (
    CameraTwin,
    DepthCameraTwin,
    FlyingCameraTwin,
    FlyingDepthCameraTwin,
    FlyingGripperCameraTwin,
    FlyingGripperDepthCameraTwin,
    FlyingTwin,
    GripperCameraTwin,
    GripperDepthCameraTwin,
    GripperJointTwin,
    GripperTwin,
    JointTwin,
    LocomoteCameraTwin,
    LocomoteDepthCameraTwin,
    LocomoteGripperCameraTwin,
    LocomoteGripperDepthCameraTwin,
    LocomoteGripperTwin,
    LocomoteTwin,
)

if TYPE_CHECKING:
    from ..client import Cyberwave


def _is_joint_manipulator(capabilities: Dict[str, Any]) -> bool:
    """Stationary manipulators (SO-101), not legged locomotion platforms (Go2)."""
    return bool(capabilities.get("has_joints")) and not bool(
        capabilities.get("can_locomote")
    )


def _select_twin_class(capabilities: Dict[str, Any]) -> Type[Twin]:
    """
    Select the appropriate Twin subclass based on capabilities.

    Args:
        capabilities: Asset capabilities dictionary

    Returns:
        The most appropriate Twin subclass
    """
    has_sensors = bool(capabilities.get("sensors", []))
    has_depth = any(s.get("type") == "depth" for s in capabilities.get("sensors", []))
    can_fly = capabilities.get("can_fly", False)
    can_locomote = capabilities.get("can_locomote", False)
    can_grip = capabilities.get("can_grip", False)

    # Select class based on combination of capabilities
    if can_fly:
        if can_grip and has_depth:
            return FlyingGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return FlyingGripperCameraTwin
        elif has_sensors:
            return FlyingCameraTwin
        elif has_depth:
            return FlyingDepthCameraTwin
        elif can_grip:
            return FlyingGripperCameraTwin
        else:
            return FlyingTwin
    elif can_locomote:
        if can_grip and has_depth:
            return LocomoteGripperDepthCameraTwin
        elif can_grip and has_sensors:
            return LocomoteGripperCameraTwin
        elif can_grip:
            return LocomoteGripperTwin
        elif has_depth:
            return LocomoteDepthCameraTwin
        elif has_sensors:
            return LocomoteCameraTwin
        else:
            return LocomoteTwin
    elif can_grip and has_sensors:
        return GripperCameraTwin
    elif can_grip and has_depth:
        return GripperDepthCameraTwin
    elif can_fly:
        return FlyingTwin
    elif can_locomote:
        return LocomoteTwin
    elif can_grip:
        return GripperJointTwin if _is_joint_manipulator(capabilities) else GripperTwin
    elif has_depth:
        return DepthCameraTwin
    elif has_sensors:
        return CameraTwin
    elif _is_joint_manipulator(capabilities):
        return JointTwin
    else:
        return Twin


def create_twin(
    client: "Cyberwave",
    twin_data: Any,
    registry_id: Optional[str] = None,
) -> Twin:
    """
    Factory function to create the appropriate Twin subclass.

    This function examines the twin's capabilities and returns an instance
    of the most appropriate Twin subclass, providing IDE autocomplete
    for capability-specific methods.

    Args:
        client: Cyberwave client instance
        twin_data: Twin schema data from API
        registry_id: Optional asset registry ID for capability lookup

    Returns:
        Appropriate Twin subclass instance (CameraTwin, FlyingTwin, etc.)

    Example:
        >>> twin = create_twin(client, twin_data, "unitree/go2")
        >>> # twin is CameraTwin with start_streaming() available
    """
    # Get capabilities - prefer cached JSON which has complete capability data
    capabilities = {}

    if registry_id:
        # Use cached capabilities from JSON (most complete source)
        capabilities = _get_asset_capabilities(registry_id)

    # Fall back to twin_data capabilities if no cached data
    if not capabilities:
        if hasattr(twin_data, "capabilities") and twin_data.capabilities:
            caps = twin_data.capabilities
            # Convert to dict if it's an object
            capabilities = (
                caps if isinstance(caps, dict) else getattr(caps, "__dict__", {})
            )
        elif isinstance(twin_data, dict) and twin_data.get("capabilities"):
            capabilities = twin_data["capabilities"]

    # Select and instantiate the appropriate class
    twin_class = _select_twin_class(capabilities)
    return twin_class(client, twin_data)
