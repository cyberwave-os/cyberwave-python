"""Voice/Text-driven SO-101 controller — natural-language motion agent.

Workshop demo: speak or type a command in plain English, and Claude translates
it into a structured joint-motion plan that the Cyberwave SDK runs on the
SO-101 arm (or its digital twin in simulation). With --vision, every prompt
also feeds the latest webcam frame from the Pi-side camera publisher so the
agent can describe the scene or act on visual context.

Phase 7 — voice + text + vision agent loop:

    python nl_arm_controller.py                       # text REPL, drives the twin
    python nl_arm_controller.py --voice               # voice REPL (hold SPACE)
    python nl_arm_controller.py --vision              # text + scene awareness
    python nl_arm_controller.py --voice --vision      # full demo: voice + scene
    python nl_arm_controller.py --dry-run             # plan only, no robot motion
    python nl_arm_controller.py --check               # env + deps self-check

Examples to say or type:
    wave at the audience
    look up and to the right
    do a small bow
    what do you see?                              # vision only
    is there a red cup?                           # vision only
    look at the red cup                           # vision-grounded motion

`exit`, `quit`, `bye`, or `Ctrl+C` to leave (the arm is always homed first).
In voice mode, Esc cancels the *current* recording without exiting.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from planner_config import get_openrouter_model

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------

CYBERWAVE_API_KEY = os.environ.get("CYBERWAVE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

CW_MODE = os.environ.get("CW_MODE", "live")
CW_ENV_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
CW_TWIN_ID = os.environ.get("CYBERWAVE_TWIN_ID")
CW_CAMERA_INDEX = int(os.environ.get("CW_CAMERA_INDEX", "0"))

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
MISTRAL_STT_MODEL = os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-latest")

TWIN_ASSET_KEY = "the-robot-studio/so101"

VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "false").lower() == "true"
SAMPLE_RATE = 16000

EXIT_WORDS = {"exit", "quit", "bye", "stop the demo", "shutdown"}


# ---------------------------------------------------------------------------
# Phase 1 self-check (kept for diagnostics)
# ---------------------------------------------------------------------------


def _check_secret(name: str, value: str | None) -> tuple[str, bool]:
    if not value:
        return f"  {name:<24} ❌ not set", False
    return f"  {name:<24} ✅ {value[:8]}…  (len {len(value)})", True

def _check_secret_any(*names: str) -> tuple[str, bool]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return f"  {name:<24} ✅ {value[:8]}…  (len {len(value)})", True
    return f"  {' or '.join(names):<24} ❌ none set", False


def _active_planner_label() -> str:
    """Which LLM/provider `planner.py` will actually use, for banner/self-check display."""
    if OPENROUTER_API_KEY:
        return f"{get_openrouter_model()} (via OpenRouter / agents SDK)"
    if ANTHROPIC_API_KEY:
        return f"{ANTHROPIC_MODEL} (via Anthropic)"
    return "(no ANTHROPIC_API_KEY or OPENROUTER_API_KEY set)"



def run_self_check() -> int:
    print("  NL → SO-101 Controller — environment self-check")
    print("─" * 64)

    rows = [
        _check_secret("CYBERWAVE_API_KEY", CYBERWAVE_API_KEY),
        _check_secret_any("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"),
        _check_secret("MISTRAL_API_KEY", MISTRAL_API_KEY),
    ]
    for line, _ in rows:
        print(line)
    keys_ok = all(ok for _, ok in rows)

    print()
    print(f"  active planner           = {_active_planner_label()}")
    print(f"  CW_MODE                  = {CW_MODE}")
    print(f"  CYBERWAVE_TWIN_ID        = {CW_TWIN_ID or '(unset)'}")
    print(f"  CYBERWAVE_ENVIRONMENT_ID = {CW_ENV_ID or '(unset)'}")
    print(f"  ANTHROPIC_MODEL          = {ANTHROPIC_MODEL}")
    print(f"  OPENROUTER_MODEL         = {get_openrouter_model()}")
    print(f"  MISTRAL_STT_MODEL        = {MISTRAL_STT_MODEL}")
    print(f"  VOICE_ENABLED            = {VOICE_ENABLED}")

    print()
    deps_ok = True
    for mod_name in ("cyberwave", "anthropic", "agents", "httpx", "sounddevice", "soundfile", "pynput"):
        try:
            __import__(mod_name)
            print(f"  import {mod_name:<14} ✅")
        except ImportError as exc:
            print(f"  import {mod_name:<14} ❌  ({exc})")
            deps_ok = False

    print("─" * 64)
    if keys_ok and deps_ok:
        print("  ✅ Environment ready.")
        return 0
    print("  ❌ Fix the items marked ❌ above and re-run.")
    return 1


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _print_banner(
    twin_uuid: str | None,
    dry_run: bool,
    voice: bool,
    vision: bool,
    camera_info: str | None = None,
) -> None:
    print("─" * 64)
    inputs = []
    if voice:
        inputs.append("voice")
    else:
        inputs.append("text")
    if vision:
        inputs.append("vision")
    print(f"  NL → SO-101 controller  ({' + '.join(inputs)})")
    print("─" * 64)
    print(f"  mode:        {'DRY-RUN (no arm)' if dry_run else CW_MODE}")
    print(f"  planner:     {_active_planner_label()}")
    if voice:
        print(f"  STT model:   {MISTRAL_STT_MODEL}")
    if vision and camera_info:
        print(f"  camera:      {camera_info}")
    if twin_uuid:
        print(f"  arm twin:    {twin_uuid}")
        print(f"  viewer:      https://cyberwave.com/twin/{twin_uuid}")
    print()
    print("  Examples:")
    print("    • wave at the audience")
    print("    • look up and to the right")
    print("    • do a small bow")
    if vision:
        print("    • what do you see?")
        print("    • is there a red cup?")
        print("    • look at the [object]")
    if voice:
        print("  Hold SPACE while speaking, release to send. Esc cancels a turn.")
    print(f"  Exit: {'say' if voice else 'type'} {sorted(EXIT_WORDS)} or press Ctrl+C.")
    print("─" * 64)


def _read_text() -> str | None:
    try:
        return input("\n  you ▸ ").strip()
    except EOFError:
        print()
        return None


def _read_voice() -> str | None:
    from voice import capture_utterance

    transcript, err = capture_utterance()
    if err:
        return ""  # keep the loop alive; user retries by holding SPACE again
    return transcript


def run_agent(dry_run: bool, voice: bool, vision: bool) -> int:
    if not (OPENROUTER_API_KEY or ANTHROPIC_API_KEY):
        print("❌ Need either OPENROUTER_API_KEY or ANTHROPIC_API_KEY set in .env")
        return 1

    if voice and not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not set in .env (required for --voice)")
        return 1

    from planner import plan_from_utterance, plan_from_utterance_with_image  # noqa: F401

    executor = None
    cw = None
    twin_uuid = None

    if not dry_run:
        if not CYBERWAVE_API_KEY:
            print("❌ CYBERWAVE_API_KEY not set in .env")
            return 1
        if not CW_TWIN_ID or not CW_ENV_ID:
            print("❌ CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID must be set in .env")
            return 1

        from cyberwave import Cyberwave  # noqa: WPS433

        from motion import MotionExecutor  # noqa: WPS433

        print("→ Connecting to Cyberwave…")
        cw = Cyberwave()
        cw.affect(CW_MODE)
        robot = cw.twin(TWIN_ASSET_KEY, twin_id=CW_TWIN_ID, environment_id=CW_ENV_ID)
        twin_uuid = robot.uuid
        executor = MotionExecutor(robot)

        print("→ Homing…")
        executor.home(duration=1.0)
        time.sleep(0.5)

    camera = None
    camera_info_str: str | None = None
    if vision:
        from vision import open_camera_from_env  # noqa: WPS433

        print(f"→ Opening webcam (index {CW_CAMERA_INDEX})…")
        try:
            camera = open_camera_from_env()
        except RuntimeError as exc:
            print(f"  ❌ {exc}")
            return 1
        camera_info_str = (
            f"index {camera.info.index} — "
            f"{camera.info.width}x{camera.info.height} @ {camera.info.fps:.0f} fps"
        )
        print(f"  ✓ {camera_info_str}")

    _print_banner(twin_uuid, dry_run, voice, vision, camera_info_str)

    try:
        while True:
            utterance = _read_voice() if voice else _read_text()
            if utterance is None:
                break

            if not utterance:
                continue
            if utterance.lower().rstrip(".!?") in EXIT_WORDS:
                break

            t0 = time.monotonic()
            frame_b64: str | None = None
            if camera is not None:
                t_frame = time.monotonic()
                frame_b64 = camera.grab_frame_b64(quality=80)
                if frame_b64 is None:
                    print("  ⚠️  webcam read failed; falling back to text-only plan")
                else:
                    print(
                        f"  📷 frame {camera.info.width}x{camera.info.height}  "
                        f"({(time.monotonic() - t_frame) * 1000:.0f} ms grab+encode, "
                        f"{len(frame_b64) // 1024} KB b64)"
                    )

            try:
                if camera is not None:
                    result = plan_from_utterance_with_image(utterance, frame_b64)
                else:
                    result = plan_from_utterance(utterance)
            except Exception as exc:  # network / API failure
                print(f"  ❌ planner call failed: {exc}")
                continue

            dt = (time.monotonic() - t0) * 1000

            if not result.ok or result.plan is None:
                print(f"  ❌ planner: {result.error}  ({dt:.0f} ms)")
                preview = (result.raw_response or "").replace("\n", " ")[:160]
                if preview:
                    print(f"     raw: {preview}")
                continue

            print(f"  🤖  ({dt:.0f} ms, {len(result.plan.actions)} actions)")
            if executor is None:
                print(f"  💬  {result.plan.say}")
                for i, a in enumerate(result.plan.actions, 1):
                    print(f"     {i}. {a}")
                continue

            try:
                executor.execute(result.plan)
            except Exception:
                print("  ❌ executor crashed:")
                traceback.print_exc()
                try:
                    print("  → safety home")
                    executor.home(duration=1.0)
                except Exception:
                    pass
                continue
    except KeyboardInterrupt:
        print("\n  (Ctrl+C — shutting down)")
    finally:
        if executor is not None:
            try:
                print("\n→ Homing before exit…")
                executor.home(duration=1.0)
                time.sleep(0.5)
            except Exception:
                pass
        if cw is not None:
            try:
                cw.disconnect()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass

    print("👋 bye")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(run_self_check())

    dry_run = "--dry-run" in sys.argv
    voice = "--voice" in sys.argv
    vision = "--vision" in sys.argv
    sys.exit(run_agent(dry_run=dry_run, voice=voice, vision=vision))


if __name__ == "__main__":
    main()
