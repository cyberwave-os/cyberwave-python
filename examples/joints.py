"""
Joints — read and write joint state (MQTT when the twin is connected).

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
arm = cw.twin("the-robot-studio/so101")

joint_names = arm.joints.list()

# Write by index (order matches list())
if len(joint_names) > 1:
    second_joint_name = joint_names[1]
    arm.set_joints({second_joint_name: -0.2})

# Read
print("names:", joint_names)
print("positions:", arm.get_joints())

# Multiple fields: position, velocity, acceleration, effort (torque)
# print(arm.get_joints(what_data=("position", "velocity", "effort")))

cw.disconnect()
