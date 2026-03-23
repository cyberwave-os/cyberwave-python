"""
Cyberwave SDK Quick Start
"""

import time
from cyberwave import Cyberwave

cw = Cyberwave()

# --- Edit a twin's position and rotation in the environment ---
robot = cw.twin("the-robot-studio/so101")
robot.edit_position(x=1, y=0, z=0.5)
robot.edit_rotation(yaw=90)

# --- Control joints ---
robot.joints.set("1", 30)
print("Joint 1:", robot.joints.get("1"))

# --- Locomotion ---
cw.affect("simulation")   # or "real-world" for the physical robot
rover = cw.twin("unitree/go2")
rover.move_forward(distance=1.0)
rover.turn_left(angle=1.57)

# --- Real-time updates via MQTT ---
robot.subscribe_position(lambda data: print("Position:", data))
robot.subscribe_joints(lambda data: print("Joints:", data))

robot.edit_position(x=2, y=1, z=0.5)
time.sleep(2)

cw.disconnect()
