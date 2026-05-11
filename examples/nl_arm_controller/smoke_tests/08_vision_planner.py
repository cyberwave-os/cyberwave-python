"""Smoke test 8 — Phase 7 Claude-Vision planner.

Pulls one frame from the camera twin (so smoke test 07 must already pass),
then sends it to Claude with three different prompts to exercise all three
modes of the vision planner:

  1. pure description    → "what do you see?"
  2. find / Q&A          → "is there a red cup on the table?"
  3. visually-grounded motion → "look at the [object you're holding up]"

Offline by default — does not drive the arm. Add --execute to run any
returned plan on the live SO-101.
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
from planner import plan_from_utterance_with_image  # noqa: E402
from vision import open_camera_from_env  # noqa: E402


UTTERANCES = [
    "what do you see in front of you? describe the scene briefly.",
    "is there a red cup on the table?",
    "look at whatever is most prominent in the scene",
]


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        sys.exit(1)

    do_execute = "--execute" in sys.argv

    print("→ Opening webcam…")
    try:
        cam = open_camera_from_env()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    print(
        f"  ✓ camera ready: {cam.info.width}x{cam.info.height} "
        f"@ {cam.info.fps:.0f} fps  (index {cam.info.index})"
    )

    print("→ Capturing frame…")
    frame_b64 = cam.grab_frame_b64(quality=80)
    if frame_b64 is None:
        print("❌ Could not read a frame from the webcam.")
        sys.exit(1)
    print(f"  base64 JPEG: {len(frame_b64):,} chars")

    executor = None
    cw = None
    if do_execute:
        from cyberwave import Cyberwave

        arm_twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
        env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
        if not arm_twin_id or not env_id:
            print("❌ Need CYBERWAVE_TWIN_ID + CYBERWAVE_ENVIRONMENT_ID for --execute")
            sys.exit(1)
        cw = Cyberwave()
        cw.affect(os.environ.get("CW_MODE", "live"))
        robot = cw.twin(
            "the-robot-studio/so101",
            twin_id=arm_twin_id,
            environment_id=env_id,
        )
        executor = MotionExecutor(robot)
        executor.home(duration=1.0)
        time.sleep(0.5)
    else:
        print("(offline mode — re-run with --execute to also drive the arm)")

    failures = 0
    for utterance in UTTERANCES:
        print()
        print("─" * 64)
        print(f"  utterance: {utterance!r}")
        t0 = time.monotonic()
        result = plan_from_utterance_with_image(utterance, frame_b64)
        dt = (time.monotonic() - t0) * 1000

        preview = result.raw_response.replace("\n", " ")[:200]
        print(f"  model:    {result.model}  ({dt:.0f} ms)")
        print(f"  raw:      {preview}{'…' if len(result.raw_response) > 200 else ''}")

        if not result.ok or result.plan is None:
            print(f"  ❌ {result.error}")
            failures += 1
            continue

        print(f"  say:      {result.plan.say!r}")
        print(f"  actions:  {len(result.plan.actions)}")
        for i, a in enumerate(result.plan.actions, 1):
            print(f"     {i}. {a.type:<10} {a}")

        if executor is not None and result.plan.actions:
            print()
            executor.execute(result.plan)
            time.sleep(0.5)

    print()
    print("─" * 64)
    cam.close()
    if cw is not None:
        cw.disconnect()
    if failures:
        print(f"❌ {failures}/{len(UTTERANCES)} planner calls failed")
        sys.exit(1)
    print(f"✅ all {len(UTTERANCES)} vision-aware plans validated")


if __name__ == "__main__":
    main()
