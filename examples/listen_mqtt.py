"""
MQTT listen — catalog-driven inbound topics, then read cached state.

Requirements:
    pip install cyberwave
"""

import time

from cyberwave import Cyberwave

cw = Cyberwave()
arm = cw.twin("the-robot-studio/so101")

session = arm.listen(filters=["joints", "pose"])
time.sleep(2)
print("joints:", arm.joints.list())
print("positions:", arm.get_joints())
session.stop()

cw.disconnect()
