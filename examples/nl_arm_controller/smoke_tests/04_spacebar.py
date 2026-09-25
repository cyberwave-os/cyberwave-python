"""Smoke test 4/4 — Spacebar listener.

Asks you to press and release SPACEBAR three times, printing PRESS/RELEASE
events. Proves macOS Accessibility / Input Monitoring permissions are granted
for your terminal app — the #1 day-of failure for live demos.
"""

from __future__ import annotations

import sys

try:
    from pynput import keyboard
except ImportError as exc:
    print(f"❌ pynput import failed: {exc}")
    sys.exit(1)


def main() -> None:
    print("→ Press and release SPACEBAR three times.")
    print("  Press ESC to abort.\n")

    state = {"down": False, "cycles": 0}

    def on_press(key):
        if key == keyboard.Key.esc:
            print("  aborted")
            return False
        if key == keyboard.Key.space and not state["down"]:
            state["down"] = True
            print("  PRESS")

    def on_release(key):
        if key == keyboard.Key.space and state["down"]:
            state["down"] = False
            state["cycles"] += 1
            print(f"  RELEASE  ({state['cycles']}/3)")
            if state["cycles"] >= 3:
                return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if state["cycles"] >= 3:
        print("\n✅ Spacebar listener OK")
    else:
        print(
            "\n❌ Did not complete 3 cycles. "
            "Check System Settings → Privacy & Security → Accessibility "
            "AND Input Monitoring (allow your terminal app in both)."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
