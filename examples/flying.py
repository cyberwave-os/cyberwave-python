"""
Flying — takeoff / land via ``twin.flight`` (or catalog ``twin.commands``).

Requirements:
    pip install cyberwave
"""

import time

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")

drone = cw.twin("dji/DJI-Mini-4-Pro")
drone.flight.takeoff(altitude=2.0)
time.sleep(2)
drone.flight.land()

# Same envelope on the catalog when listed:
# drone.commands.takeoff(altitude=2.0)

cw.disconnect()
