"""
Cyberwave Python SDK - Main Entry Point

This SDK integrates two main APIs:
1. REST API - Auto-generated from OpenAPI specification (in /rest folder)
2. MQTT API - Auto-generated from AsyncAPI specification (in /mqtt folder)

The SDK provides a hand-crafted layer on top of the auto-generated code
to deliver a delightful developer experience.

Architecture:
- cyberwave/client.py: Main Cyberwave client integrating REST and MQTT
- cyberwave/twin.py: High-level Twin abstraction for controlling digital twins
- cyberwave/resources.py: Resource managers (Workspaces, Projects, etc.)
- cyberwave/mqtt_client.py: MQTT client wrapper for real-time updates
- cyberwave/compact.py: Simplified module-level API
- cyberwave/config.py: Configuration management
- cyberwave/exceptions.py: Custom exceptions

"""

from cyberwave import (
    Cyberwave,
    Twin,
    configure,
    twin,
    simulation,
    CyberwaveError,
    CyberwaveAPIError,
    CyberwaveConnectionError,
)

__all__ = [
    "Cyberwave",
    "Twin",
    "configure",
    "twin",
    "simulation",
    "CyberwaveError",
    "CyberwaveAPIError",
    "CyberwaveConnectionError",
]


if __name__ == "__main__":
    # Example usage
    print("Cyberwave Python SDK")
    print("=" * 50)
    print()
    print("Quick Start:")
    print()
    print("  from cyberwave import Cyberwave")
    print()
    print("  # Configure the SDK")
    print("  cw = Cyberwave(")
    print('      api_key="your_api_key",')
    print("  )")
    print()
    print("  # Create and control a twin")
    print('  robot = cw.twin("the-robot-studio/so101")')
    print("  robot.edit_position(x=1, y=0, z=0.5)")
    print("  robot.edit_rotation(yaw=90)")
    print("  robot.joints.arm_joint = 45")
    print()
    print("For more examples, see the module docstring or README.md")
