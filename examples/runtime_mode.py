"""
Runtime mode — separate live vs simulation MQTT state buckets.

``cw.affect()`` sets ``config.runtime_mode``. Inbound ``source_type`` decides
which bucket is updated; ``get_joints()`` / ``joints.get()`` read the active bucket.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
arm = cw.twin("the-robot-studio/so101")
joint_names = arm.joints.list()

cw.affect("simulation")
if joint_names:
    first_joint_name = joint_names[0]
    arm.set_joints({first_joint_name: -0.2})

cw.affect("live")
if joint_names:
    last_joint_name = joint_names[-1]
    arm.set_joints({last_joint_name: 0.2})

print("simulation data:", arm.get_joints())
print("live data:", arm.get_joints())

cw.disconnect()
