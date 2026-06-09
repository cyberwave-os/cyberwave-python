"""
DJI Mini 4 Pro — takeoff, fly, use gimbal, and land.

Requirements:
    pip install cyberwave
"""

import time

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")  # use cw.affect("live") for actual flight

drone = cw.twin("dji/DJI-Mini-4-Pro")

# Take off
print("Taking off…")
drone.takeoff(altitude=2.0)
time.sleep(4)

# Move forward
print("Moving forward…")
drone.move_forward(1.5)
time.sleep(3)

# Tilt gimbal down
print("Tilting gimbal down…")
drone.gimbal_rotate(pitch=-90.0, duration=1.5)
time.sleep(2)

# Recenter gimbal
print("Recentering gimbal…")
drone.gimbal_recenter()
time.sleep(1)

# Land
print("Landing…")
drone.land()
time.sleep(4)

cw.disconnect()
print("Done.")
