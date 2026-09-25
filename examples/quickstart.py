"""
Cyberwave SDK — Quick Start.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# Scene layout (editor / REST — not MQTT locomotion)
arm = cw.twin("the-robot-studio/so101")
arm.edit_position(x=1, y=0, z=0.5)
arm.edit_rotation(yaw=90)

# Joint commands (names from list(), access by index)
joint_names = arm.joints.list()
if joint_names:
    first_joint_name = joint_names[0]
    arm.set_joints({first_joint_name: -0.2})
    print("First joint", first_joint_name, ":", arm.joints[first_joint_name])

# Locomotion in simulation (burst + stop on edge drivers)
cw.affect("simulation")
quadruped = cw.twin("unitree/go2")
quadruped.move_forward(0.3)

cw.disconnect()
