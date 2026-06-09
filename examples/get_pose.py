"""
Pose reads — joint-space on arms, Cartesian on locomote twins.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# Manipulator: get_pose() is joint-space (alias for get_joints())
arm = cw.twin("the-robot-studio/so101")
print("Joint-space:", arm.get_pose())

# Locomote: get_pose() is world pose from MQTT; prefer pose.get() for typed state
cw.affect("live")
dog = cw.twin("unitree/go2")
print("Cartesian:", dog.pose.get())
print("Shortcut:", dog.get_pose())

cw.disconnect()
