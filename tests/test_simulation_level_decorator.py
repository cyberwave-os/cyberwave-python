"""SimLevel enum + simulation_level decorator behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.twin.simulation_support import SimLevel, simulation_level


class _Handle:
    def __init__(self, twin):
        self._twin = twin

    @simulation_level(SimLevel.MUJOCO)
    def get_frame(self):
        return "frame"

    @simulation_level()  # defaults to PLAYGROUND
    def get_state(self):
        return "state"


def _twin():
    calls = []
    twin = SimpleNamespace()
    twin._ensure_simulation_support = lambda level, *, method: calls.append((level, method))
    twin._calls = calls
    return twin


def test_default_level_is_playground() -> None:
    assert _Handle.get_state.__cw_sim_level__ == SimLevel.PLAYGROUND


def test_mujoco_level_tag() -> None:
    assert _Handle.get_frame.__cw_sim_level__ == SimLevel.MUJOCO


def test_decorator_calls_ensure_with_level_and_method() -> None:
    twin = _twin()
    handle = _Handle(twin)
    assert handle.get_frame() == "frame"
    assert twin._calls == [(SimLevel.MUJOCO, "_Handle.get_frame")]


def test_decorator_resolves_self_when_no_twin_attr() -> None:
    # When the decorated object has no _twin, `self` is used as the twin.
    calls = []

    class _TwinLike:
        def _ensure_simulation_support(self, level, *, method):
            calls.append((level, method))

        @simulation_level(SimLevel.MUJOCO)
        def op(self):
            return "ok"

    assert _TwinLike().op() == "ok"
    assert len(calls) == 1
    level, method = calls[0]
    assert level == SimLevel.MUJOCO
    assert method.endswith("_TwinLike.op")


def test_level_ordering() -> None:
    assert SimLevel.UNSUPPORTED < SimLevel.PLAYGROUND < SimLevel.MUJOCO
