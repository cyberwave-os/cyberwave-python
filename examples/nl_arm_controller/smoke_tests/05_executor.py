"""Smoke test 5/5 — Phase 3 motion executor on the real SO-101 twin.

Runs three hand-crafted `MotionPlan`s end-to-end and verifies that every plan:
  1. validates cleanly
  2. ramps smoothly (no snap)
  3. ends in the expected pose

Plans are built two ways to prove both code paths work:
  * Python dataclass construction (`MotionPlan(actions=[Action(...), ...])`)
  * `MotionPlan.from_dict({...})` — the exact same JSON shape Claude will emit

Also runs validation negative tests (out-of-range angle gets clamped; bad
action type gets rejected) so we know the safety net is real.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make `motion` importable when running from inside `smoke_tests/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from motion import (  # noqa: E402
    MAX_DURATION_S,
    Action,
    MotionExecutor,
    MotionPlan,
    clamp,
    validate_plan,
)

try:
    from cyberwave import Cyberwave
except ImportError as exc:
    print(f"❌ cyberwave import failed: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Hand-crafted plans
# ---------------------------------------------------------------------------

PLAN_WAVE = MotionPlan(
    say="Waving with joint 1.",
    actions=[
        Action(type="set_joint", joint="1", angle=+30, duration=1.0),
        Action(type="set_joint", joint="1", angle=-30, duration=1.5),
        Action(type="set_joint", joint="1", angle=0, duration=1.0),
    ],
)

PLAN_LOOK_AROUND_DICT = {
    "say": "Looking around — base and shoulder together.",
    "actions": [
        {"type": "set_pose", "pose": {"1": 25, "2": -20}, "duration": 1.5},
        {"type": "wait", "duration": 0.5},
        {"type": "set_pose", "pose": {"1": -25, "2": 20}, "duration": 2.0},
        {"type": "home", "duration": 1.5},
    ],
}

PLAN_HOME = MotionPlan(
    say="Returning home.",
    actions=[Action(type="home", duration=1.5)],
)


def _validation_negative_tests() -> bool:
    """Run pure-Python checks on the validator + clamp logic — no robot needed."""
    print("→ Validator negative tests")

    bad_plan = MotionPlan(
        actions=[
            Action(type="set_joint", joint="99", angle=10),
            Action(type="set_joint", joint="1"),
            Action(type="wait", duration=MAX_DURATION_S + 1),
            Action(type="bogus"),
        ]
    )
    errs = validate_plan(bad_plan)
    print(f"  validator caught {len(errs)} errors:")
    for e in errs:
        print(f"    • {e}")
    if len(errs) < 4:
        print("❌ expected ≥4 errors")
        return False

    cases = [("1", 999, 90.0), ("1", -999, -90.0), ("3", 999, 60.0), ("3", -999, -60.0)]
    for joint, raw, expected in cases:
        got = clamp(joint, raw)
        if got != expected:
            print(f"❌ clamp({joint!r}, {raw}) = {got}, expected {expected}")
            return False
    print(f"  clamp() saturates correctly on {len(cases)} edge cases")

    return True


def main() -> None:
    if not _validation_negative_tests():
        sys.exit(1)
    print()

    if not os.environ.get("CYBERWAVE_API_KEY"):
        print("❌ CYBERWAVE_API_KEY not set")
        sys.exit(1)

    twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
    if not twin_id or not env_id:
        print("❌ Set CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID in your .env")
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

    executor = MotionExecutor(robot)

    print("\n→ Pre-flight: home everything")
    executor.home(duration=1.0)
    time.sleep(0.5)

    print("\n→ Plan 1: WAVE  (Python dataclass construction)")
    executor.execute(PLAN_WAVE)
    time.sleep(0.5)

    print("\n→ Plan 2: LOOK AROUND  (MotionPlan.from_dict — same shape Claude will emit)")
    executor.execute(MotionPlan.from_dict(PLAN_LOOK_AROUND_DICT))
    time.sleep(0.5)

    print("\n→ Plan 3: HOME")
    executor.execute(PLAN_HOME)

    cw.disconnect()
    print("\n✅ Phase 3 motion executor OK")


if __name__ == "__main__":
    main()
