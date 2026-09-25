"""
Controller policy — attach teleop before motion/joint MQTT in live mode.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("live")

arm = cw.twin("the-robot-studio/so101")

# Auto-pick a workspace teleop policy (or no-op when API/auto-attach unavailable)
arm.policy.ensure_attached()
print(arm.policy.attached)

# Explicit assign from workspace list:
# policies = arm.policy.list()
# arm.policy.assign(policies[0])

joint_names = arm.joints.list()
if joint_names:
    last_joint_name = joint_names[-1]
    arm.set_joints({last_joint_name: 0.2})

cw.disconnect()
