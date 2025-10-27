"""
Cyberwave SDK Quick Start Example

This example demonstrates the basic usage of the Cyberwave SDK
using both the compact API and the advanced API.
"""

import cyberwave as cw

# ============================================================================
# Example 1: Compact API (Recommended for quick prototyping)
# ============================================================================

def compact_api_example():
    """Example using the compact, module-level API"""
    
    # Configure the SDK
    cw.configure(
        base_url="http://localhost:8000",
        api_key="your_api_key",
        environment="your_environment_uuid"
    )
    
    # Create a digital twin from an asset
    robot = cw.twin("cyberwave/so101")
    
    # Move the twin
    robot.move(x=1, y=0, z=0.5)
    
    # Rotate the twin (using euler angles in degrees)
    robot.rotate(yaw=90, pitch=0, roll=0)
    
    # Control robot joints (for URDF assets)
    robot.joints.arm_joint = 45  # Set to 45 degrees
    
    # Or use explicit method
    robot.joints.set("shoulder_joint", 30)
    
    # Get joint value
    current_position = robot.joints.get("arm_joint")
    print(f"Arm joint position: {current_position}")
    
    # Move to a specific position
    robot.move_to([1.5, 0.5, 0.3])
    
    # Scale the twin
    robot.scale(x=2.0, y=2.0, z=2.0)
    
    # Simulation control
    cw.simulation.play()
    cw.simulation.pause()
    cw.simulation.step(10)  # Step forward 10 frames
    cw.simulation.reset()


# ============================================================================
# Example 2: Advanced API (Full control with context manager)
# ============================================================================

def advanced_api_example():
    """Example using the advanced API with full control"""
    
    # Create client with context manager for automatic cleanup
    with cw.Cyberwave(
        base_url="http://localhost:8000",
        token="your_bearer_token",
        environment_id="your_environment_uuid"
    ) as client:
        
        # List workspaces
        workspaces = client.workspaces.list()
        print(f"Found {len(workspaces)} workspaces")
        
        # Get first workspace
        if workspaces:
            workspace = workspaces[0]
            print(f"Workspace: {workspace.name}")
            
            # List projects in workspace
            projects = client.projects.list()
            
            # Create a new project
            project = client.projects.create(
                name="My Robot Project",
                workspace_id=workspace.uuid,
                description="A project for robot simulations"
            )
            
            # Create an environment
            environment = client.environments.create(
                name="Test Environment",
                project_id=project.uuid
            )
            
            # Search for assets
            assets = client.assets.search("robot")
            
            if assets:
                # Create a twin
                twin_data = client.twins.create(
                    asset_id=assets[0].uuid,
                    environment_id=environment.uuid,
                    name="Robot Instance 1"
                )
                
                # Work with the twin using the Twin abstraction
                twin = cw.Twin(client, twin_data)
                twin.move(x=1, y=0, z=0.5)
                twin.rotate(yaw=45)
                
                # Get all joint names
                joint_names = twin.joints.list()
                print(f"Available joints: {joint_names}")
                
                # Delete the twin when done
                # twin.delete()


# ============================================================================
# Example 3: Real-time Updates via MQTT
# ============================================================================

def mqtt_example():
    """Example using MQTT for real-time updates"""
    
    client = cw.Cyberwave(
        base_url="http://localhost:8000",
        token="your_token",
        environment_id="env_uuid",
        mqtt_host="mqtt.cyberwave.com",
        mqtt_port=1883
    )
    
    # Create or get a twin
    robot = client.twin("cyberwave/so101")
    
    # Define callbacks for real-time updates
    def on_position_update(data):
        print(f"Position updated: {data}")
    
    def on_rotation_update(data):
        print(f"Rotation updated: {data}")
    
    def on_joints_update(data):
        print(f"Joints updated: {data}")
    
    # Subscribe to real-time updates
    robot.subscribe_updates(
        on_position=on_position_update,
        on_rotation=on_rotation_update,
        on_joints=on_joints_update
    )
    
    # Move the robot - this will trigger MQTT updates
    robot.move(x=2, y=1, z=0.5)
    
    # Keep the script running to receive updates
    import time
    time.sleep(10)
    
    # Cleanup
    client.disconnect()


# ============================================================================
# Example 4: Direct API Access
# ============================================================================

def direct_api_example():
    """Example accessing REST and MQTT APIs directly"""
    
    client = cw.Cyberwave(
        base_url="http://localhost:8000",
        token="your_token"
    )
    
    # Access REST API directly (auto-generated OpenAPI client)
    assets_response = client.api.api_v1_assets_list()
    print(f"Assets: {len(assets_response)}")
    
    # Access MQTT client directly
    client.mqtt.connect()
    
    # Subscribe to a topic
    def on_message(data):
        print(f"Received: {data}")
    
    client.mqtt.subscribe_twin_position("twin_uuid", on_message)
    
    # Publish a message
    client.mqtt.publish_twin_position(
        "twin_uuid",
        x=1.0,
        y=0.5,
        z=0.3
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Cyberwave SDK Examples")
    print("=" * 60)
    print()
    print("Uncomment the example you want to run:")
    print()
    
    # Uncomment to run examples:
    # compact_api_example()
    # advanced_api_example()
    # mqtt_example()
    # direct_api_example()
    
    print("See the function definitions above for usage examples")

