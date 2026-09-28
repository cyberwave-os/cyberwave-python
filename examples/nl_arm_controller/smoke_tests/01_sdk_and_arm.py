"""Smoke test 1/4 — Cyberwave SDK + SO-101 twin.

Connects, instantiates an SO-101 twin in the default environment, and sends
one joint command to verify the publish path works. In simulation mode this
animates the 3D twin in the browser viewer; in live mode it also drives the
physical arm (requires `so101-remoteoperate` running on the edge device).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

try:
    from cyberwave import Cyberwave
except ImportError as exc:
    print(f"❌ cyberwave import failed: {exc}")
    sys.exit(1)


JOINTS = ["1", "2", "3", "4", "5", "6"]


def home(robot) -> None:
    """Send every joint to 0° — the canonical zero pose for SO-101."""
    for joint in JOINTS:
        robot.joints.set(joint, 0)


def main() -> None:
    if not os.environ.get("CYBERWAVE_API_KEY"):
        print("❌ CYBERWAVE_API_KEY not set")
        sys.exit(1)

    twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
    if not twin_id or not env_id:
        print("❌ Set CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID in your .env")
        print("   (paste the UUIDs from your Cyberwave dashboard)")
        sys.exit(1)

    print("→ Connecting to Cyberwave...")
    cw = Cyberwave()
    cw.affect(os.environ.get("CW_MODE", "simulation"))

    robot = cw.twin(
        "the-robot-studio/so101",
        twin_id=twin_id,
        environment_id=env_id,
    )
    print(f"  twin: {robot.uuid}")
    print(f"  open in browser: https://cyberwave.com/twin/{robot.uuid}")

    robot.subscribe_joints(
        lambda data: print(f"  [live] {data.get('joint_states') or data}")
    )

    print("→ Homing: every joint → 0°  (start from a known pose)")
    home(robot)
    time.sleep(1.5)

    print("→ Sending joint 1 → +30°  (watch the 3D viewer)")
    robot.joints.set("1", 30)
    time.sleep(1.5)

    print("→ Returning joint 1 → 0°")
    robot.joints.set("1", 0)
    time.sleep(1.5)

    cw.disconnect()
    print("✅ SDK + twin OK")


if __name__ == "__main__":
    main()
