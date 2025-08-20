#!/usr/bin/env python3
"""
Cyberwave SDK - Compact API Quickstart
Demonstrates the simple, intuitive interface for digital twin control
"""

import cyberwave as cw

def main():
    """Demonstrate the compact API"""
    print("🚀 Cyberwave SDK - Compact API Demo")
    print("=" * 50)
    
    # Optional: Configure the SDK
    # cw.configure(
    #     api_key="your_api_key",
    #     base_url="http://localhost:8000",
    #     environment="your_environment_id"
    # )
    
    print("\n1. Creating Digital Twins")
    print("-" * 30)
    
    # Create digital twins from registry
    robot_arm = cw.twin("cyberwave/so101")
    print(f"✅ Created robot arm: {robot_arm.name}")
    
    humanoid = cw.twin("berkeley/berkeley_humanoid") 
    print(f"✅ Created humanoid: {humanoid.name}")
    
    drone = cw.twin("dji/tello")
    print(f"✅ Created drone: {drone.name}")
    
    print("\n2. Basic Movement Control")
    print("-" * 30)
    
    # Move robots to different positions
    robot_arm.move(x=0.5, y=0.0, z=0.3)
    print(f"Robot arm moved to: {robot_arm.position}")
    
    humanoid.move(x=1.0, y=0.5, z=0.0)
    print(f"Humanoid moved to: {humanoid.position}")
    
    drone.move(x=0.0, y=0.0, z=2.0)  # Takeoff
    print(f"Drone moved to: {drone.position}")
    
    print("\n3. Rotation Control")
    print("-" * 30)
    
    # Rotate robots
    robot_arm.rotate(yaw=45)
    print(f"Robot arm rotation: {robot_arm.rotation}")
    
    humanoid.rotate(roll=0, pitch=0, yaw=90)
    print(f"Humanoid rotation: {humanoid.rotation}")
    
    print("\n4. Joint Control (URDF Robots)")
    print("-" * 30)
    
    # Control individual joints
    robot_arm.joints.shoulder = 30  # degrees
    robot_arm.joints.elbow = -45
    robot_arm.joints.wrist = 90
    print("✅ Robot arm joints configured")
    
    humanoid.joints.left_arm = 20
    humanoid.joints.right_arm = -20
    print("✅ Humanoid arms positioned")
    
    print("\n5. Advanced Movement")
    print("-" * 30)
    
    # Move to specific poses
    robot_arm.move_to([0.4, 0.2, 0.5], [0, 0, 45])
    print("✅ Robot arm moved to target pose")
    
    # Chain movements
    drone.move(x=1, y=1, z=2)
    drone.rotate(yaw=180)
    drone.move(x=0, y=0, z=2)
    print("✅ Drone completed flight pattern")
    
    print("\n6. Simulation Control")
    print("-" * 30)
    
    # Control simulation
    cw.simulation.play()
    print("✅ Simulation started")
    
    # Run a few steps
    for i in range(5):
        cw.simulation.step()
        print(f"  Step {i+1} completed")
    
    cw.simulation.pause()
    print("✅ Simulation paused")
    
    print("\n7. Property Access")
    print("-" * 30)
    
    # Access twin properties
    print(f"Robot arm position: {robot_arm.position}")
    print(f"Robot arm rotation: {robot_arm.rotation}")
    print(f"Has sensors: {robot_arm.has_sensors}")
    
    # Joint states
    joint_states = robot_arm.joints.all()
    print(f"All joint states: {joint_states}")
    
    print("\n" + "=" * 50)
    print("🎉 Compact API Demo Complete!")
    print("\nKey Benefits:")
    print("- One-liner twin creation: cw.twin('registry_id')")
    print("- Intuitive movement: twin.move(x, y, z)")
    print("- Simple rotation: twin.rotate(yaw=90)")
    print("- Direct joint access: twin.joints.joint_name = value")
    print("- Global simulation: cw.simulation.play()")

if __name__ == "__main__":
    main()
