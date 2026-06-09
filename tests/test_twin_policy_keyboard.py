"""Policy keyboard teleop; twin.controller removed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.twin import JointTwin, Twin
from cyberwave.twin.factory import create_twin


def test_twin_has_no_controller_property() -> None:
    client = SimpleNamespace(mqtt=MagicMock(), config=SimpleNamespace(runtime_mode="live"))
    twin = Twin(client, SimpleNamespace(uuid="t", name="T"))
    assert not hasattr(type(twin), "controller") or getattr(type(twin), "controller", None) is None
    with pytest.raises(AttributeError):
        _ = twin.controller


def test_policy_keyboard_returns_keyboard_teleop() -> None:
    from cyberwave.keyboard import KeyboardBindings, KeyboardTeleop

    client = SimpleNamespace(mqtt=MagicMock(), config=SimpleNamespace(runtime_mode="live"))
    twin = JointTwin(
        client,
        SimpleNamespace(
            uuid="t",
            name="T",
            capabilities={"has_joints": True, "can_locomote": False},
        ),
    )
    bindings = KeyboardBindings()
    teleop = twin.policy.keyboard(bindings, step=0.1, verbose=False)
    assert isinstance(teleop, KeyboardTeleop)
