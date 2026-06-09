"""
Locomotion — move_forward / turn_* with burst + stop (GO2, UGV, etc.).

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")  # use cw.affect("live") for the physical robot

robot = cw.twin("unitree/go2")
robot.locomotion.move_forward(0.3, duration=0.5, rate_hz=10)
robot.turn_left(0.5, duration=0.3)

# Top-level shortcuts delegate to locomotion
# robot.move_forward(0.3, duration=0.5)

cw.disconnect()
