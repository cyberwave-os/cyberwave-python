#!/usr/bin/env python3
"""
Simulation Control Demo
Demonstrates simulation management and physics control
"""

import cyberwave as cw
import time

def setup_environment():
    """Set up a test environment with multiple robots"""
    print("🏗️ Setting up simulation environment...")
    
    # Create various robots for simulation
    robots = {
        'arm': cw.twin("cyberwave/so101", name="Robotic Arm"),
        'spot': cw.twin("spot/spot_mini", name="Quadruped"),
        'humanoid': cw.twin("berkeley/berkeley_humanoid", name="Humanoid"),
        'drone': cw.twin("dji/tello", name="Drone")
    }
    
    # Position robots in the environment
    robots['arm'].move_to([0, 0, 0])
    robots['spot'].move_to([2, 0, 0])
    robots['humanoid'].move_to([0, 2, 0])
    robots['drone'].move_to([2, 2, 1])
    
    print(f"✅ Environment set up with {len(robots)} robots")
    return robots

def demo_simulation_states():
    """Demonstrate different simulation states"""
    print("\n⏯️ Simulation State Control")
    print("-" * 40)
    
    # Start simulation
    print("Starting simulation...")
    cw.simulation.play()
    time.sleep(0.1)  # Brief pause for effect
    
    # Run for several steps
    print("Running simulation steps...")
    for i in range(10):
        cw.simulation.step()
        print(f"  Physics step {i+1}/10")
        time.sleep(0.05)  # Simulate processing time
    
    # Pause simulation
    print("Pausing simulation...")
    cw.simulation.pause()
    
    # Reset simulation
    print("Resetting simulation...")
    cw.simulation.reset()
    
    print("✅ Simulation state demo completed")

def demo_physics_interaction(robots):
    """Demonstrate physics-based interactions"""
    print("\n⚛️ Physics Interaction Demo")
    print("-" * 40)
    
    arm = robots['arm']
    spot = robots['spot']
    
    # Start physics simulation
    cw.simulation.play()
    
    # Simulate object manipulation
    print("Simulating object manipulation...")
    
    # Arm reaches for object
    arm.move_to([0.3, 0, 0.2])
    arm.joints.gripper = 0  # Open
    
    # Simulate physics steps
    for step in range(5):
        cw.simulation.step()
        print(f"  Physics step {step+1}: Arm reaching")
    
    # Grasp object
    arm.joints.gripper = 50  # Close
    arm.move_to([0.3, 0, 0.4])  # Lift
    
    for step in range(5):
        cw.simulation.step()
        print(f"  Physics step {step+1}: Object grasped")
    
    # Spot approaches
    spot.move_to([0.5, 0, 0])
    
    for step in range(10):
        cw.simulation.step()
        print(f"  Physics step {step+1}: Spot approaching")
    
    print("✅ Physics interaction completed")

def demo_real_time_monitoring(robots):
    """Demonstrate real-time state monitoring"""
    print("\n📊 Real-time Monitoring Demo")
    print("-" * 40)
    
    cw.simulation.play()
    
    # Monitor robot states during movement
    arm = robots['arm']
    
    print("Monitoring arm movement...")
    target_positions = [
        [0.2, 0, 0.3],
        [0.4, 0.2, 0.4],
        [0.2, -0.1, 0.3],
        [0, 0, 0.3]  # Home
    ]
    
    for i, target in enumerate(target_positions):
        print(f"\nMovement {i+1}: Target {target}")
        arm.move_to(target)
        
        # Simulate movement with monitoring
        for step in range(3):
            cw.simulation.step()
            current_pos = arm.position
            print(f"  Step {step+1}: Current position {current_pos}")
    
    print("✅ Real-time monitoring completed")

def demo_error_handling():
    """Demonstrate error handling and recovery"""
    print("\n🛠️ Error Handling Demo")
    print("-" * 40)
    
    try:
        # Try to create invalid twin
        invalid_robot = cw.twin("invalid/robot_id")
        print("⚠️ Invalid robot created (should be handled)")
        
    except Exception as e:
        print(f"✅ Error handled: {e}")
    
    try:
        # Try invalid joint control
        arm = cw.twin("cyberwave/so101")
        arm.joints.invalid_joint = 100
        print("⚠️ Invalid joint set (should be handled)")
        
    except Exception as e:
        print(f"✅ Joint error handled: {e}")
    
    # Recovery operations
    print("Performing recovery...")
    cw.simulation.reset()
    print("✅ Simulation reset for recovery")

def demo_performance_optimization():
    """Demonstrate performance optimization techniques"""
    print("\n⚡ Performance Optimization Demo")
    print("-" * 40)
    
    # Batch operations
    print("Creating multiple robots efficiently...")
    robot_configs = [
        ("cyberwave/so101", "Arm1"),
        ("cyberwave/so101", "Arm2"), 
        ("spot/spot_mini", "Spot1"),
        ("dji/tello", "Drone1")
    ]
    
    # Create all robots
    robots = []
    for registry_id, name in robot_configs:
        robot = cw.twin(registry_id, name=name)
        robots.append(robot)
        print(f"  Created: {name}")
    
    # Batch position updates
    print("Batch positioning...")
    positions = [
        [0, 0, 0.3],    # Arm1
        [0.5, 0, 0.3],  # Arm2
        [1, 0, 0],      # Spot1
        [1, 1, 2],      # Drone1
    ]
    
    for robot, pos in zip(robots, positions):
        robot.move_to(pos)
        print(f"  {robot.name} positioned at {pos}")
    
    # Efficient simulation stepping
    print("Running optimized simulation...")
    cw.simulation.play()
    
    # Batch simulation steps
    for batch in range(3):
        print(f"  Batch {batch+1}: Running 10 steps")
        for _ in range(10):
            cw.simulation.step()
    
    # Cleanup
    for robot in robots:
        robot.delete()
    
    print("✅ Performance demo completed")

def main():
    """Run all simulation demos"""
    print("🎮 Cyberwave Simulation Control Demo")
    print("=" * 60)
    
    # Configure SDK
    cw.configure(base_url="http://localhost:8000")
    print("SDK configured for simulation")
    
    # Set up environment
    robots = setup_environment()
    
    # Run demos
    demo_simulation_states()
    demo_physics_interaction(robots)
    demo_real_time_monitoring(robots)
    demo_error_handling()
    demo_performance_optimization()
    
    # Final cleanup
    print("\n🧹 Final Cleanup")
    print("-" * 40)
    
    for name, robot in robots.items():
        robot.delete()
        print(f"  {name.capitalize()} removed")
    
    cw.simulation.reset()
    print("  Simulation reset")
    
    print("\n" + "=" * 60)
    print("🎉 Simulation Demo Complete!")
    
    print("\n📚 Simulation Features Demonstrated:")
    print("- Environment setup and teardown")
    print("- Simulation state control (play/pause/step/reset)")
    print("- Physics-based interactions")
    print("- Real-time state monitoring")
    print("- Error handling and recovery")
    print("- Performance optimization techniques")
    print("- Multi-robot coordination")

if __name__ == "__main__":
    main()
