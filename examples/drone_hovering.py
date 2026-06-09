"""
Drone Hovering — takeoff, check hovering status, and land.

Requirements:
    pip install cyberwave
"""

import time

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")

drone = cw.twin("dji/dji-mini-4-pro")
print(f"Drone: {drone.name}")

# Take off and hover
drone.takeoff(altitude=2.0)
print("Took off")

# Check hovering status
if drone.is_hovering():
    status = drone.get_hovering_status()
    print(f"Hovering: {status}")

time.sleep(3)

# Land
drone.land()
print("Landed:", drone.get_hovering_status())

cw.disconnect()
