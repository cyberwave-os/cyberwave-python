from __future__ import annotations

import os
from cyberwave import Cyberwave


def main():
    base = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    token = os.getenv("CYBERWAVE_TOKEN", "")
    arm_uuid = os.getenv("CYBERWAVE_ARM_TWIN_UUID", "")
    if not (token and arm_uuid):
        raise SystemExit("Set CYBERWAVE_TOKEN and CYBERWAVE_ARM_TWIN_UUID")

    cw = Cyberwave(base, token)

    # Move arm via unified twin commands (replace with your controller’s verbs)
    cw.twins.command(arm_uuid, "arm.move_joints", {"joints": [0, 10, -5, 45, 0, 90]})
    cw.twins.command(arm_uuid, "gripper.open", {})
    cw.twins.command(arm_uuid, "arm.move_pose", {"pose": {"x": 0.30, "y": 0.10}})


if __name__ == "__main__":
    main()


