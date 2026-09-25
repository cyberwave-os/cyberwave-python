"""
Compact — shortest joint read.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
print(cw.twin("the-robot-studio/so101").get_joints())
