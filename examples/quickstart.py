"""
Cyberwave SDK Quick Start Example

This example demonstrates the basic usage of the Cyberwave SDK
using both the compact API and the advanced API.
"""

import os
from cyberwave import Cyberwave


# ============================================================================
# Example 1: Compact API (Recommended for quick prototyping)
# ============================================================================


def compact_api_example():
    """Example using the compact, module-level API"""

    # Configure the SDK
    cw = Cyberwave()

    # Create a digital twin from an asset
    robot = cw.twin("the-robot-studio/so101")

    # Move the twin
    robot.edit_position(x=1, y=0, z=0.5)

    # Rotate the twin (using euler angles in degrees)
    robot.edit_rotation(yaw=90, pitch=0, roll=0)

    # Control robot joints (for URDF assets)
    robot.joints.set("1", 30)

    # Get joint value
    current_position = robot.joints.get("1")
    print(f"Arm joint position: {current_position}")


# ============================================================================
# Example 2: Real-time Updates via MQTT
# ============================================================================


def mqtt_example():
    """Example using MQTT for real-time updates"""

    client = Cyberwave()

    # Connect to MQTT explicitly before checking connection
    client.mqtt.connect()

    client_obj = getattr(client, "mqtt", None) or getattr(client, "mqtt_client", None)
    try:
        for attr in ("connected", "is_connected", "connection", "state"):
            if hasattr(client_obj, attr):
                try:
                    print(f"client.{attr} = {getattr(client_obj, attr)}")
                except Exception:
                    print(f"client has attribute {attr}")
    except Exception:
        pass

    # Create or get a twin
    robot = client.twin("the-robot-studio/so101")

    def on_update(data):
        print(f"Update received: {data}")

    # Define callbacks for real-time updates
    def on_position_update(data):
        print(f"Position updated: {data}")

    def on_rotation_update(data):
        print(f"Rotation updated: {data}")

    def on_joints_update(data):
        print(f"Joints updated: {data}")

    # Subscribe to movement updates
    robot.subscribe_position(on_position_update)

    # Subscribe to rotation updates
    robot.subscribe_rotation(on_rotation_update)

    # Subscribe to joint updates
    robot.subscribe_joints(on_joints_update)

    # Subscribe to all real-time updates
    robot.subscribe(
        on_update=on_update,
    )

    # Move the robot - this will trigger MQTT updates
    robot.edit_position(x=2, y=1, z=0.5)

    print("Connected to MQTT and moved stuff around successfully")

    # Keep the script running to receive updates
    import time

    time.sleep(2)

    # Cleanup
    client.disconnect()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nCyberwave SDK Examples")
    print("=" * 60)
    print()
    print("Uncomment the example you want to run:")
    print()

    # Uncomment to run examples:
    # compact_api_example()
    mqtt_example()

    print("See the function definitions above for usage examples")
