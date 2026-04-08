"""
Drone hovering status — Cyberwave SDK example

This example shows how to:
  1. Obtain a FlyingTwin from the catalog
  2. Send a takeoff command and record the hovering state in the twin's metadata
  3. Read back the hovering status and altitude
  4. Land and clear the hovering state

The hovering state is stored in twin.metadata.status:
  {
      "status": {
          "controller_requested_hovering": True,
          "controller_requested_hovering_altitude": 2.0,   # metres
      }
  }

In the Cyberwave playground simulate mode this flag prevents gravity from being
applied to the twin so it visually stays at its current altitude.
"""

import time
from cyberwave import Cyberwave
from cyberwave.twin import FlyingTwin

cw = Cyberwave()

# ------------------------------------------------------------------
# 1.  Get a flying twin
#     The SDK automatically returns a FlyingTwin when the asset's
#     capabilities include can_fly=True.
# ------------------------------------------------------------------
drone = cw.twin("cyberwave/px4vision")
cw.affect("simulation")  # use the simulation mode for this example

print(f"Drone: {drone.name}  ({drone.uuid})")

# ------------------------------------------------------------------
# 2.  Take off and persist the hovering state
# ------------------------------------------------------------------

# Send the takeoff command with default altitude
drone.takeoff()

# Send the takeoff command with custom altitude
HOVER_ALTITUDE = 2.0  # metres
drone.takeoff(altitude=HOVER_ALTITUDE)

print(f"Took off — hovering at {HOVER_ALTITUDE} m")

# ------------------------------------------------------------------
# 3.  Read the hovering status back
# ------------------------------------------------------------------
# Quick boolean check
if drone.is_hovering():
    status = drone.get_hovering_status()
    print(
        f"Hovering: {status['controller_requested_hovering']}, "
        f"altitude: {status['controller_requested_hovering_altitude']} m"
    )

# You can also inspect the raw metadata if needed
print(
    "metadata.status:",
    drone._data.metadata.get("status") if hasattr(drone._data, "metadata") else "n/a",
)  # type: ignore[union-attr]

# Simulate the drone holding position for a few seconds
time.sleep(3)

# ------------------------------------------------------------------
# 4.  Land
# ------------------------------------------------------------------
drone.land()

print("Landed — hovering cleared")
print("Final hovering status:", drone.get_hovering_status())

cw.disconnect()
