"""Keyboard teleop for the PiPER — the dashboard's bindings, in the terminal.

The key map is `motion.KEY_TO_ACTION`, which mirrors the "Keyboard (PiPER)"
controller in the Cyberwave environment view one-for-one:

      1 / 2   joint1  base rotation        7 / 8   joint4  forearm roll
      3 / 4   joint2  shoulder pitch       9 / 0   joint5  wrist pitch
      5 / 6   joint3  elbow                Q / W   joint6  wrist roll
                                           E / R   joint7  gripper open/close

Odd digit / Q / E increases, even digit / W / R decreases — same as the panel,
so muscle memory transfers between the browser and this REPL.

Every keypress goes through `MotionExecutor.nudge()`, so terminal teleop gets
the same clamping the LLM plans get, and the gripper still goes out in native
units with `joint8` left to the platform's mimic.

Two keys are added on top of the dashboard set, both unbound there:
  H  — home the arm (gripper untouched)
  X  — leave teleop
"""

from __future__ import annotations

import select
import sys
import termios
import tty

from motion import ARM_JOINTS, GRIPPER, KEY_TO_ACTION, MotionExecutor

STEP_DEG: float = 5.0
STEP_GRIPPER_PCT: float = 10.0
HOME_KEY: str = "H"
EXIT_KEY: str = "X"
POLL_INTERVAL_S: float = 0.02


def piper_keyboard_bindings():
    """The same bindings as a `cyberwave.KeyboardBindings`, for the platform.

    Handy for pushing this map to a twin's controller
    (`robot.controller.keyboard(piper_keyboard_bindings(), step=5.0)`) so the
    dashboard panel and the CLI stay in sync. Note that SDK teleop's default
    `stop_key="q"` collides with joint6 here — pass `stop_key="x"` to match
    this module.
    """
    from cyberwave import KeyboardBindings

    bindings = KeyboardBindings()
    for spec in ARM_JOINTS:
        bindings.bind(spec.key_increase, spec.name, "increase")
        bindings.bind(spec.key_decrease, spec.name, "decrease")
    bindings.bind(GRIPPER.key_increase, GRIPPER.name, "increase")
    bindings.bind(GRIPPER.key_decrease, GRIPPER.name, "decrease")
    return bindings


def print_bindings() -> None:
    print("─" * 64)
    print("  PiPER keyboard teleop — same bindings as the dashboard panel")
    print("─" * 64)
    for spec in ARM_JOINTS:
        print(
            f"   {spec.key_increase} / {spec.key_decrease}   {spec.name}  "
            f"{spec.label:<15} [{spec.lower:+.0f}°, {spec.upper:+.0f}°]"
        )
    print(
        f"   {GRIPPER.key_increase} / {GRIPPER.key_decrease}   {GRIPPER.name}  "
        f"{GRIPPER.label:<15} [0%, 100%]"
    )
    print()
    print(f"   {HOME_KEY}       home the arm")
    print(f"   {EXIT_KEY}       back to the natural-language prompt")
    print("─" * 64)


def run_keyboard_teleop(
    executor: MotionExecutor,
    *,
    step_deg: float = STEP_DEG,
    step_gripper_pct: float = STEP_GRIPPER_PCT,
) -> None:
    """Read single keypresses and nudge joints until EXIT_KEY or Ctrl+C."""
    if not sys.stdin.isatty():
        print("  ⚠️  keyboard teleop needs an interactive terminal (stdin is not a tty)")
        return

    print_bindings()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], POLL_INTERVAL_S)
            if not ready:
                continue

            key = sys.stdin.read(1).upper()
            if key == EXIT_KEY or key == "\x03":  # X or Ctrl+C
                break
            if key == HOME_KEY:
                executor.home(duration=1.0)
                print("\r  home                                   ", flush=True)
                continue

            action = KEY_TO_ACTION.get(key)
            if action is None:
                continue

            joint, sign = action
            step = step_gripper_pct if joint == GRIPPER.name else step_deg
            value = executor.nudge(joint, sign * step)
            unit = "%" if joint == GRIPPER.name else "°"
            print(f"\r  {joint:<8} {value:+7.1f}{unit}        ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\n  (leaving teleop)")
