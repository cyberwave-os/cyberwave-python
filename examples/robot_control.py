#!/usr/bin/env python3
"""
Advanced Robot Control Examples
Demonstrates comprehensive robot control using the Cyberwave SDK
"""

import cyberwave as cw
import time

def demo_robot_arm():
    """Demonstrate robotic arm control"""
    print("\n🦾 Robot Arm Control Demo")
    print("-" * 40)
    
    # Create robot arm twin
    arm = cw.twin("cyberwave/so101", name="Demo Robot Arm")
    print(f"Created: {arm.name}")
    
    # Home position
    arm.move_to([0, 0, 0.3])
    arm.joints.shoulder = 0
    arm.joints.elbow = 0
    arm.joints.wrist = 0
    print("✅ Moved to home position")
    
    # Pick sequence
    print("Executing pick sequence...")
    arm.move_to([0.3, 0.1, 0.2])  # Move above object
    arm.joints.gripper = 0  # Open gripper
    arm.move(z=0.1)  # Move down
    arm.joints.gripper = 50  # Close gripper
    arm.move(z=0.3)  # Lift object
    print("✅ Pick completed")
    
    # Place sequence  
    print("Executing place sequence...")
    arm.move_to([0.0, 0.3, 0.3])  # Move to drop zone
    arm.move(z=0.15)  # Lower
    arm.joints.gripper = 0  # Open gripper
    arm.move(z=0.3)  # Retract
    print("✅ Place completed")
    
    return arm

def demo_quadruped():
    """Demonstrate quadruped robot control"""
    print("\n🐕 Quadruped Control Demo")
    print("-" * 40)
    
    # Create Spot twin
    spot = cw.twin("spot/spot_mini", name="Demo Spot")
    print(f"Created: {spot.name}")
    
    # Basic movements
    spot.move(x=0, y=0, z=0)  # Stand position
    print("✅ Standing")
    
    # Walk pattern
    waypoints = [
        [1, 0, 0],    # Forward
        [1, 1, 0],    # Right
        [0, 1, 0],    # Back
        [0, 0, 0],    # Left (home)
    ]
    
    print("Executing walk pattern...")
    for i, point in enumerate(waypoints):
        spot.move_to(point)
        print(f"  Waypoint {i+1}: {point}")
    
    print("✅ Walk pattern completed")
    
    return spot

def demo_drone():
    """Demonstrate drone control"""
    print("\n🚁 Drone Control Demo")
    print("-" * 40)
    
    # Create drone twin
    drone = cw.twin("dji/tello", name="Demo Tello")
    print(f"Created: {drone.name}")
    
    # Takeoff sequence
    drone.move(x=0, y=0, z=0)  # Ground
    drone.move(z=1.5)  # Takeoff to 1.5m
    print("✅ Takeoff completed")
    
    # Flight pattern - square
    flight_pattern = [
        [2, 0, 1.5],   # Forward
        [2, 2, 1.5],   # Right  
        [0, 2, 1.5],   # Back
        [0, 0, 1.5],   # Left (start)
    ]
    
    print("Executing flight pattern...")
    for i, point in enumerate(flight_pattern):
        drone.move_to(point)
        drone.rotate(yaw=i * 90)  # Rotate at each corner
        print(f"  Flight point {i+1}: {point}")
    
    # Landing
    drone.move(z=0)
    print("✅ Landing completed")
    
    return drone

def demo_humanoid():
    """Demonstrate humanoid robot control"""
    print("\n🤖 Humanoid Control Demo")
    print("-" * 40)
    
    # Create humanoid twin
    humanoid = cw.twin("berkeley/berkeley_humanoid", name="Demo Humanoid")
    print(f"Created: {humanoid.name}")
    
    # Basic pose
    humanoid.move_to([0, 0, 0])
    print("✅ Standing position")
    
    # Arm gestures
    print("Executing arm gestures...")
    
    # Wave gesture
    humanoid.joints.left_shoulder = 90
    humanoid.joints.left_elbow = 45
    print("  Left arm raised")
    
    humanoid.joints.right_shoulder = 90  
    humanoid.joints.right_elbow = 45
    print("  Both arms raised")
    
    # Return to neutral
    humanoid.joints.left_shoulder = 0
    humanoid.joints.left_elbow = 0
    humanoid.joints.right_shoulder = 0
    humanoid.joints.right_elbow = 0
    print("✅ Returned to neutral pose")
    
    # Walking simulation
    print("Simulating walk...")
    walk_positions = [
        [0.5, 0, 0],
        [1.0, 0, 0], 
        [1.5, 0, 0],
    ]
    
    for pos in walk_positions:
        humanoid.move_to(pos)
        print(f"  Walked to: {pos}")
    
    return humanoid

def demo_multi_robot_coordination():
    """Demonstrate coordination between multiple robots"""
    print("\n🤝 Multi-Robot Coordination Demo")
    print("-" * 40)
    
    # Create multiple robots
    arm = cw.twin("cyberwave/so101", name="Arm")
    spot = cw.twin("spot/spot_mini", name="Spot")
    
    print("Created robot team")
    
    # Coordinated task - Spot brings object, Arm picks it up
    print("Executing coordinated task...")
    
    # Spot moves to object location
    spot.move_to([1, 0, 0])
    print("  Spot: Moved to object")
    
    # Arm prepares for pickup
    arm.move_to([1, 0, 0.2])
    arm.joints.gripper = 0  # Open
    print("  Arm: Ready for pickup")
    
    # Spot delivers object
    spot.move_to([1, 0, 0])  # At pickup location
    print("  Spot: Object delivered")
    
    # Arm picks up
    arm.move(z=0.1)  # Lower
    arm.joints.gripper = 50  # Close
    arm.move(z=0.3)  # Lift
    print("  Arm: Object picked up")
    
    # Spot moves away
    spot.move_to([0, 0, 0])  # Return home
    print("  Spot: Returned home")
    
    print("✅ Coordinated task completed")

def main():
    """Run all robot control demos"""
    print("🤖 Advanced Robot Control Examples")
    print("=" * 60)
    
    # Configure SDK
    cw.configure(base_url="http://localhost:8000")
    print("SDK configured")
    
    # Start simulation
    cw.simulation.play()
    print("Simulation started")
    
    # Run individual robot demos
    arm = demo_robot_arm()
    spot = demo_quadruped() 
    drone = demo_drone()
    humanoid = demo_humanoid()
    
    # Multi-robot coordination
    demo_multi_robot_coordination()
    
    # Final simulation control
    print("\n🎮 Simulation Control")
    print("-" * 40)
    
    print("Running simulation steps...")
    for i in range(3):
        cw.simulation.step()
        print(f"  Step {i+1} completed")
    
    cw.simulation.pause()
    print("Simulation paused")
    
    # Cleanup
    print("\n🧹 Cleanup")
    print("-" * 40)
    
    robots = [arm, spot, drone, humanoid]
    for robot in robots:
        robot.delete()
        print(f"  {robot.name} removed")
    
    cw.simulation.reset()
    print("Simulation reset")
    
    print("\n" + "=" * 60)
    print("🎉 Advanced Robot Control Demo Complete!")
    
    print("\n📚 What You Learned:")
    print("- Creating multiple robot types")
    print("- Individual robot control")
    print("- Joint manipulation")
    print("- Multi-robot coordination")
    print("- Simulation management")
    print("- Resource cleanup")

if __name__ == "__main__":
    main()
