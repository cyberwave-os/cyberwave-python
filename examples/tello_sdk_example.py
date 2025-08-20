from __future__ import annotations

import os
import time
from cyberwave import Cyberwave


def main():
    """Control a Tello twin via Cyberwave endpoints (teleop + commands)."""

    base = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    token = os.getenv("CYBERWAVE_TOKEN", "")
    tello_uuid = os.getenv("CYBERWAVE_TELLO_TWIN_UUID", "")
    if not (token and tello_uuid):
        raise SystemExit("Set CYBERWAVE_TOKEN and CYBERWAVE_TELLO_TWIN_UUID")

    cw = Cyberwave(base, token)

    # Start a teleop session to log events and (optionally) stream camera
    with cw.teleop.session(tello_uuid, sensors=["camera"]):
        # Takeoff
        cw.twins.command(tello_uuid, "drone.takeoff", {})
        time.sleep(3)

        # Simple hover (no-op delay)
        time.sleep(2)

        # Land
        cw.twins.command(tello_uuid, "drone.land", {})


if __name__ == "__main__":
    main()
