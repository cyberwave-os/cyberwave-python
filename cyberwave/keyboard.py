"""
Keyboard teleoperation for controlling twins from the terminal.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .twin import Twin


class KeyboardBindings:
    """Builder for keyboard teleoperation bindings."""

    def __init__(self) -> None:
        self._bindings: List[Dict[str, Any]] = []

    def bind(
        self, key: str, joint_name: str, direction: str = "increase"
    ) -> "KeyboardBindings":
        """
        Bind a key to a joint action.

        Args:
            key: Keyboard key (e.g., "W", "S")
            joint_name: Name of the joint to control
            direction: "increase" or "decrease"

        Returns:
            Self for chaining
        """
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        if not key:
            raise ValueError("key is required")
        if not joint_name:
            raise ValueError("joint_name is required")
        self._bindings.append(
            {
                "key": key.upper(),
                "jointName": joint_name,
                "direction": direction,
            }
        )
        return self

    def build(self) -> List[Dict[str, Any]]:
        """Return the bindings as a list of dictionaries."""
        return list(self._bindings)


@dataclass
class _Binding:
    joint_name: str
    direction: str


class KeyboardTeleop:
    """
    CLI keyboard teleoperation helper.

    Reads keyboard input and updates joint positions via the twin's joints controller.
    """

    def __init__(
        self,
        twin: "Twin",
        bindings: List[Dict[str, Any]],
        *,
        step: float = 0.05,
        rate_hz: int = 20,
        fetch_initial: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initialize keyboard teleop.

        Args:
            twin: Twin instance to control
            bindings: List of key bindings from KeyboardBindings.build()
            step: Position change per keypress (degrees)
            rate_hz: Polling rate in Hz
            fetch_initial: Whether to fetch initial joint positions
            verbose: Whether to print status messages
        """
        self._twin = twin
        self._bindings = self._normalize_bindings(bindings)
        self._step = float(step)
        self._interval = 1.0 / float(rate_hz) if rate_hz > 0 else 0.05
        self._fetch_initial = fetch_initial
        self._verbose = verbose
        self._positions: Dict[str, float] = {}

    def run(self, stop_key: str = "q") -> None:
        """
        Run the keyboard teleop loop.

        Args:
            stop_key: Key to press to stop (default: 'q')
        """
        if not self._bindings:
            raise ValueError("No keyboard bindings configured")
        if self._fetch_initial:
            self._positions = self._get_initial_positions()

        stop_key = stop_key.lower()
        if self._verbose:
            print(f"Keyboard teleop active. Press '{stop_key}' to stop.")
            print("Bindings:")
            for key, bindings in self._bindings.items():
                for b in bindings:
                    print(f"  {key}: {b.joint_name} ({b.direction})")

        if os.name == "nt":
            self._run_windows(stop_key)
        else:
            self._run_posix(stop_key)

    def _normalize_bindings(self, bindings: List[Dict[str, Any]]) -> Dict[str, List[_Binding]]:
        normalized: Dict[str, List[_Binding]] = {}
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            key = str(binding.get("key") or "").upper()
            joint_name = binding.get("jointName") or binding.get("joint_name")
            direction = binding.get("direction") or "increase"
            if not key or not joint_name:
                continue
            normalized.setdefault(key, []).append(
                _Binding(joint_name=str(joint_name), direction=str(direction))
            )
        return normalized

    def _get_initial_positions(self) -> Dict[str, float]:
        positions: Dict[str, float] = {}
        try:
            positions = self._twin.joints.get_all()
        except Exception:
            pass
        return positions

    def _apply_binding(self, binding: _Binding) -> None:
        delta = self._step if binding.direction == "increase" else -self._step
        current = self._positions.get(binding.joint_name, 0.0)
        next_pos = current + delta
        self._positions[binding.joint_name] = next_pos
        try:
            self._twin.joints.set(binding.joint_name, next_pos, degrees=True)
            if self._verbose:
                print(f"  {binding.joint_name}: {next_pos:.2f}°")
        except Exception as e:
            if self._verbose:
                print(f"  Error setting {binding.joint_name}: {e}")

    def _run_windows(self, stop_key: str) -> None:
        import msvcrt

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if not key:
                    continue
                if key == stop_key:
                    break
                bindings = self._bindings.get(key.upper(), [])
                for binding in bindings:
                    self._apply_binding(binding)
            time.sleep(self._interval)

    def _run_posix(self, stop_key: str) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], self._interval)
                if rlist:
                    key = sys.stdin.read(1).lower()
                    if key == stop_key:
                        break
                    bindings = self._bindings.get(key.upper(), [])
                    for binding in bindings:
                        self._apply_binding(binding)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

