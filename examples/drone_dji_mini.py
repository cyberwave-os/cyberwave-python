"""
DJI Mini 4 Pro — full flight, locomotion, and gimbal control
=============================================================

End-to-end example of the :class:`FlyingTwin` command surface against
a real DJI Mini 4 Pro tethered to the Cyberwave Android edge driver
(``cyberwave-edge-nodes/cyberwave-edge-dji-mini-android``).

What it covers
--------------

1. Connect, pick the right twin (capabilities resolve to a
   :class:`FlyingCameraTwin`).
2. Take off.
3. Move forward — inherited from :class:`LocomoteTwin`. The Cyberwave
   playground simulator and the Go2 driver execute this directly. The
   DJI Mini driver currently still ignores continuous-stick MQTT
   commands while the physical RC2 owns the sticks (see
   ``DroneCommandManager`` KDoc), so on a real Mini this is a no-op
   that the simulator companion twin still picks up — useful when you
   want a single script to drive both the simulated and the real
   drone for visual diff regression.
4. Tilt the gimbal down for a top-down survey, then sweep using
   ``gimbal_rotate_speed`` for a cinematic move.
5. Recenter the gimbal (the same recentering the keyboard ``N``
   binding emits via ``controller:dji-keyboard:v1``).
6. Land — automatically arms the landing-confirmation flow on the
   driver side (see ``DroneCommandManager`` KDoc), so over water /
   glass / glossy floors the operator gets a Cyberwave alert and a
   second ``land()`` call confirms the touchdown.

Pre-requisites
--------------

- A Cyberwave workspace with a DJI Mini 4 Pro twin (capability
  ``can_fly: true`` and a gimbal camera sensor — picked up
  automatically from the catalog asset).
- Either:
    * The Android edge driver running on a phone tethered to the
      Mini's RC2 controller (live mode), **or**
    * The Cyberwave playground simulator running for this twin (sim
      mode).
- ``CYBERWAVE_API_KEY`` exported in the environment (or a
  ``cyberwave login`` already cached locally).

Run
---

    python examples/drone_dji_mini.py

Switch ``cw.affect(...)`` to ``"simulation"`` for a pure-sim dry run
or ``"real-world"`` to fly the actual aircraft.
"""

import time

from cyberwave import Cyberwave
from cyberwave.twin import FlyingTwin

cw = Cyberwave()


# ---------------------------------------------------------------------------
# Pick simulation or real flight
#
# `cw.affect()` controls the default `source_type` baked into every
# command published below — `"sim_tele"` for simulation, `"tele"` for
# real flight. Edge drivers only act on `tele`, so flipping this is the
# safety knob between dry-runs and live flight.
# ---------------------------------------------------------------------------

cw.affect("simulation")  # change to "real-world" for an actual flight


# ---------------------------------------------------------------------------
# Resolve the twin
#
# `cw.twin("...")` accepts either a UUID, a slug
# (`my-workspace/twins/dji-mini-4-pro-01`), or a catalog-asset slug
# when there's a single twin that uses it. The SDK reads the asset's
# capabilities (`can_fly: true` for a Mini) and returns the right
# subclass — here a `FlyingCameraTwin` (FlyingTwin + camera sensor).
# ---------------------------------------------------------------------------

drone = cw.twin("SZ-DJI-Technology/DJI-Mini-4-Pro")

assert isinstance(drone, FlyingTwin), (
    f"Expected a FlyingTwin (or subclass), got {type(drone).__name__}. "
    "Check the asset capabilities — `can_fly` must be true."
)
print(f"Drone: {drone.name}  (uuid={drone.uuid})")


# ---------------------------------------------------------------------------
# 1. Take off
#
# `altitude` only takes effect in `sim_tele` — the DJI MSDK takeoff
# action is parameter-less and goes to the firmware default (~1.2 m).
# In sim mode the twin metadata flips to `controller_requested_hovering=True`
# so the simulator stops applying gravity to the digital twin.
# ---------------------------------------------------------------------------

print("Taking off…")
drone.takeoff(altitude=2.0)
time.sleep(4.0)  # give the aircraft a moment to stabilise


# ---------------------------------------------------------------------------
# 2. Move forward (inherited from LocomoteTwin)
#
# This is mostly useful in `sim_tele` today — the DJI driver currently
# ignores continuous-stick MQTT commands while the physical RC2 owns
# the sticks. It's still wired through the canonical
# `cyberwave/twin/{uuid}/command` topic, so a future driver upgrade
# (KeyVirtualStickAdvancedParam, off-RC teleop) doesn't need any SDK
# change.
# ---------------------------------------------------------------------------

print("Cruising 1.5 m forward…")
drone.move_forward(1.5)
time.sleep(3.0)


# ---------------------------------------------------------------------------
# 3. Tilt the gimbal down for a top-down shot
#
# Pitch range on the Mini 4 Pro is approximately [-90°, +30°].
# `mode="absolute"` (the default) interprets the angle relative to
# the aircraft heading; `"relative"` applies a delta from the current
# gimbal attitude.
# ---------------------------------------------------------------------------

print("Tilting gimbal down to -45°…")
drone.gimbal_rotate(pitch=-45.0, duration=1.5)
time.sleep(2.0)


# ---------------------------------------------------------------------------
# 4. Cinematic pan with `gimbal_rotate_speed`
#
# Units are 0.1°/s — i.e. `pitch=50` ≈ 5°/s. Each call drives the
# gimbal for a short window influenced by call frequency and airlink
# quality, so sustained motion needs the command re-issued.
# ---------------------------------------------------------------------------

print("Panning gimbal up at ~5°/s for 2 s…")
for _ in range(4):
    drone.gimbal_rotate_speed(pitch=50.0)
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# 5. Recenter the gimbal
#
# Same recentering the keyboard `N` binding emits via
# `controller:dji-keyboard:v1`. Equivalent to:
#   drone.gimbal_rotate(pitch=0.0, mode="absolute")
# ---------------------------------------------------------------------------

print("Recentering gimbal…")
drone.gimbal_recenter()
time.sleep(1.0)


# ---------------------------------------------------------------------------
# 6. Land
#
# On the real DJI driver the first `land()` call kicks off
# auto-landing and arms the landing-confirmation listener. If the
# firmware asks the operator to confirm (over water / glass / glossy
# surfaces) a Cyberwave alert is raised and a second `land()` call
# confirms the touchdown — see `DroneCommandManager` KDoc for the
# full state machine.
#
# In sim mode this also clears `controller_requested_hovering` so the
# simulator re-applies gravity and the twin settles on the ground.
# ---------------------------------------------------------------------------

print("Landing…")
drone.land()
time.sleep(4.0)

# Optional: if the driver raised a `landing_confirmation_required`
# alert, calling `land()` again is the explicit operator confirmation.
# Skipped here for brevity — gate it on the alert in production code.
# drone.land()


# ---------------------------------------------------------------------------
# 7. Other commands available on FlyingTwin
#
# Left commented out so this script is safe to run end-to-end without
# triggering them. Each maps 1:1 onto a DJI MSDK v5 FlightControllerKey
# action — see ``DroneCommandManager`` for the full table.
# ---------------------------------------------------------------------------

# drone.return_to_home()           # KeyStartGoHome (with confirmation flow)
# drone.cancel_return_to_home()    # cancels NORMAL prompt OR in-flight RTH
# drone.set_home_here()            # reset home to current GPS position
# drone.start_compass_calibration()
# drone.stop_compass_calibration()
# drone.reboot()                   # reboot the aircraft
# drone.emergency_stop()           # cancel takeoff/landing/RTH; hand back to RC

cw.disconnect()
print("Done.")
