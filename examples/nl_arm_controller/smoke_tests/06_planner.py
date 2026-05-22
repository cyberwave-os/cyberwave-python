"""Smoke test 6 — Phase 4 Claude planner.

Sends 4 fixed utterances to Claude, parses each response into a `MotionPlan`,
and prints the result. Two modes:

  python smoke_tests/06_planner.py            # offline — Claude only, no arm
  python smoke_tests/06_planner.py --execute  # full loop — also drive the twin

The offline mode is what you use to iterate on the prompt.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from motion import MotionExecutor  # noqa: E402
from planner import plan_from_utterance  # noqa: E402


UTTERANCES = [
    "wave at the audience",
    "look up and to the right",
    "do a small bow",
    "stop and go home",
]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        sys.exit(1)

    do_execute = "--execute" in sys.argv

    executor: MotionExecutor | None = None
    cw = None
    if do_execute:
        try:
            from cyberwave import Cyberwave
        except ImportError as exc:
            print(f"❌ cyberwave import failed: {exc}")
            sys.exit(1)

        twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
        env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
        if not twin_id or not env_id:
            print("❌ Need CYBERWAVE_TWIN_ID + CYBERWAVE_ENVIRONMENT_ID for --execute")
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
        executor = MotionExecutor(robot)
        executor.home(duration=1.0)
        time.sleep(0.5)
    else:
        print("(offline mode — re-run with --execute to also drive the arm)\n")

    failures = 0
    for utterance in UTTERANCES:
        print("─" * 64)
        print(f"  utterance: {utterance!r}")
        result = plan_from_utterance(utterance)

        preview = result.raw_response.replace("\n", " ")[:180]
        print(f"  model:     {result.model}")
        print(f"  raw:       {preview}{'…' if len(result.raw_response) > 180 else ''}")

        if not result.ok or result.plan is None:
            print(f"  ❌ {result.error}")
            failures += 1
            continue

        print(f"  say:       {result.plan.say!r}")
        print(f"  actions:   {len(result.plan.actions)}")
        for i, a in enumerate(result.plan.actions, 1):
            extras = []
            if a.joint is not None:
                extras.append(f"joint={a.joint}")
            if a.angle is not None:
                extras.append(f"angle={a.angle:+.1f}")
            if a.pose is not None:
                extras.append(f"pose={a.pose}")
            extras.append(f"dur={a.duration:.2f}s")
            print(f"     {i}. {a.type:<10} {' '.join(extras)}")

        if executor is not None:
            print()
            executor.execute(result.plan)
            time.sleep(0.5)

    print("─" * 64)
    if cw is not None:
        cw.disconnect()

    if failures:
        print(f"❌ {failures}/{len(UTTERANCES)} planner calls failed")
        sys.exit(1)
    print(f"✅ All {len(UTTERANCES)} utterances produced valid plans")


if __name__ == "__main__":
    main()
