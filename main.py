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

Usage Examples:

1. Compact API (Recommended for quick prototyping):
   ```python
   import cyberwave as cw

   # Configure
   cw.configure(
       api_key="your_api_key",
       environment="env_uuid"
   )

   # Create and control a twin
   robot = cw.twin("cyberwave/so101")
   robot.move(x=1, y=0, z=0.5)
   robot.rotate(yaw=90)
   robot.joints.arm_joint = 45

   # Simulation control
   cw.simulation.play()
   cw.simulation.step()
   cw.simulation.pause()
   ```

2. Advanced API (Full control):
   ```python
   from cyberwave import Cyberwave

   # Create client
   client = Cyberwave(
       token="your_bearer_token",
       environment_id="env_uuid"
   )

   # List workspaces
   workspaces = client.workspaces.list()

   # Create project
   project = client.projects.create(
       name="My Project",
       workspace_id=workspaces[0].uuid
   )

   # Create environment
   env = client.environments.create(
       name="Production",
       project_id=project.uuid
   )

   # Search for assets
   assets = client.assets.search("robot")

   # Create twin
   twin_data = client.twins.create(
       asset_id=assets[0].uuid,
       environment_id=env.uuid,
       name="Robot 1"
   )

   # Work with Twin abstraction
   from cyberwave import Twin
   twin = Twin(client, twin_data)
   twin.move(x=1, y=0, z=0.5)
   twin.joints.set("shoulder_joint", 45)

   # Real-time updates via MQTT
   def on_position_update(data):
       print(f"Position updated: {data}")

   twin.subscribe_updates(on_position=on_position_update)

   # Context manager for automatic cleanup
   with Cyberwave(base_url="http://localhost:8000", token="token") as client:
       twins = client.twins.list()
       # ... do work ...
   # Automatically disconnects
   ```

3. Direct API Access:
   ```python
   from cyberwave import Cyberwave

   client = Cyberwave(base_url="http://localhost:8000", token="token")

   # Access REST API directly
   response = client.api.api_v1_assets_list()

   # Access MQTT client directly
   client.mqtt.connect()
   client.mqtt.subscribe_twin_position("twin_uuid", callback)
   client.mqtt.publish_twin_position("twin_uuid", x=1, y=0, z=0.5)
   ```

Environment Variables:
- CYBERWAVE_BASE_URL: Default base URL
- CYBERWAVE_API_KEY: API key for authentication
- CYBERWAVE_TOKEN: Bearer token for authentication
- CYBERWAVE_ENVIRONMENT_ID: Default environment ID
- CYBERWAVE_WORKSPACE_ID: Default workspace ID
- CYBERWAVE_MQTT_HOST: MQTT broker host
- CYBERWAVE_MQTT_USERNAME: MQTT username
- CYBERWAVE_MQTT_PASSWORD: MQTT password

Installation:
```bash
pip install cyberwave
```

For more information, see:
- README.md for getting started guide
- /rest/README.md for REST API documentation
- /mqtt/README.md for MQTT API documentation
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
    print("  import cyberwave as cw")
    print()
    print("  # Configure the SDK")
    print("  cw.configure(")
    print('      api_key="your_api_key",')
    print('      environment="env_uuid"')
    print("  )")
    print()
    print("  # Create and control a twin")
    print('  robot = cw.twin("cyberwave/so101")')
    print("  robot.move(x=1, y=0, z=0.5)")
    print("  robot.rotate(yaw=90)")
    print("  robot.joints.arm_joint = 45")
    print()
    print("For more examples, see the module docstring or README.md")
