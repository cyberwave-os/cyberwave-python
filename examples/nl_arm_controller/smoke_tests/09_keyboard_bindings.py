"""Smoke test 9 — PiPER keyboard bindings, offline.

No network, no robot, no API keys. Verifies that the key map in `motion.py`
still matches the dashboard's "Keyboard (PiPER)" controller and that every
key routes through the executor's clamping:

  1 / 2  joint1     3 / 4  joint2     5 / 6  joint3     7 / 8  joint4
  9 / 0  joint5     Q / W  joint6     E / R  joint7 (gripper)

Run this after touching `ARM_JOINTS` or `GRIPPER` — it is the guard that the
prompt table, the teleop key map, and the executor clamps stayed in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motion import (  # noqa: E402
    ARM_JOINTS,
    GRIPPER,
    KEY_TO_ACTION,
    MotionExecutor,
)
from teleop import piper_keyboard_bindings, print_bindings  # noqa: E402


EXPECTED_KEYS = {
    "1": ("joint1", +1.0),
    "2": ("joint1", -1.0),
    "3": ("joint2", +1.0),
    "4": ("joint2", -1.0),
    "5": ("joint3", +1.0),
    "6": ("joint3", -1.0),
    "7": ("joint4", +1.0),
    "8": ("joint4", -1.0),
    "9": ("joint5", +1.0),
    "0": ("joint5", -1.0),
    "Q": ("joint6", +1.0),
    "W": ("joint6", -1.0),
    "E": ("joint7", +1.0),
    "R": ("joint7", -1.0),
}


class _NullRobot:
    class _Joints:
        def set(self, joint_name, position, degrees=True):
            return None

    joints = _Joints()

    def get_schema(self, path: str = ""):
        return {}


def check_key_map() -> bool:
    print("→ Key map matches the dashboard panel")
    if KEY_TO_ACTION != EXPECTED_KEYS:
        missing = set(EXPECTED_KEYS) - set(KEY_TO_ACTION)
        extra = set(KEY_TO_ACTION) - set(EXPECTED_KEYS)
        wrong = {
            k: (KEY_TO_ACTION[k], EXPECTED_KEYS[k])
            for k in set(EXPECTED_KEYS) & set(KEY_TO_ACTION)
            if KEY_TO_ACTION[k] != EXPECTED_KEYS[k]
        }
        print(f"❌ missing={sorted(missing)} extra={sorted(extra)} wrong={wrong}")
        return False
    print(f"  {len(KEY_TO_ACTION)} keys, 6 arm joints + gripper")
    return True


def check_sdk_bindings() -> bool:
    print("→ SDK KeyboardBindings round-trip")
    built = piper_keyboard_bindings().build()
    as_map = {
        b["key"]: (b["jointName"], +1.0 if b["direction"] == "increase" else -1.0)
        for b in built
    }
    if as_map != EXPECTED_KEYS:
        print(f"❌ SDK bindings disagree with KEY_TO_ACTION: {as_map}")
        return False
    print(f"  {len(built)} bindings, ready for robot.controller.keyboard(...)")
    return True


def check_clamping() -> bool:
    """Hammer each key past its limit and confirm the executor saturates."""
    print("→ Teleop nudges saturate at the joint limits")
    executor = MotionExecutor(_NullRobot(), dry_run=True, discover_limits=False)

    for spec in ARM_JOINTS:
        for _ in range(200):
            executor.nudge(spec.name, +5.0)
        top = executor.nudge(spec.name, +5.0)
        for _ in range(400):
            executor.nudge(spec.name, -5.0)
        bottom = executor.nudge(spec.name, -5.0)
        if top != spec.upper or bottom != spec.lower:
            print(
                f"❌ {spec.name}: saturated at [{bottom}, {top}], "
                f"expected [{spec.lower}, {spec.upper}]"
            )
            return False
        executor.nudge(spec.name, -bottom)  # back to 0 for the next joint
        print(f"  {spec.name:<8} [{bottom:+7.1f}°, {top:+7.1f}°]")

    for _ in range(50):
        executor.nudge(GRIPPER.name, +10.0)
    wide = executor.gripper_percent
    for _ in range(50):
        executor.nudge(GRIPPER.name, -10.0)
    shut = executor.gripper_percent
    if (wide, shut) != (100.0, 0.0):
        print(f"❌ gripper saturated at [{shut}, {wide}], expected [0.0, 100.0]")
        return False
    print(f"  {GRIPPER.name:<8} [{shut:+7.1f}%, {wide:+7.1f}%]")
    return True


def main() -> None:
    print_bindings()
    print()
    ok = all([check_key_map(), check_sdk_bindings(), check_clamping()])
    print()
    if not ok:
        print("❌ keyboard bindings out of sync")
        sys.exit(1)
    print("✅ Keyboard bindings OK")


if __name__ == "__main__":
    main()
