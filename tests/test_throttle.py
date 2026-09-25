"""Throttle: clock-injectable rate gate (first call fires; then once per interval)."""

from __future__ import annotations

from cyberwave.driver.support.throttle import Throttle


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_first_call_is_ready() -> None:
    assert Throttle(5.0, now=_Clock()).ready() is True


def test_within_interval_not_ready() -> None:
    clock = _Clock()
    t = Throttle(5.0, now=clock)
    assert t.ready() is True
    clock.advance(4.9)
    assert t.ready() is False


def test_after_interval_ready_again() -> None:
    clock = _Clock()
    t = Throttle(5.0, now=clock)
    assert t.ready() is True
    clock.advance(5.0)
    assert t.ready() is True


def test_last_fire_advances_only_on_true() -> None:
    clock = _Clock()
    t = Throttle(5.0, now=clock)
    assert t.ready() is True   # fire at t=0
    clock.advance(3.0)
    assert t.ready() is False  # t=3: suppressed, no advance
    clock.advance(2.0)         # t=5 measured from the first fire
    assert t.ready() is True
