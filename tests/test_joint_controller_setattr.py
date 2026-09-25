"""Regression tests for ``JointsHandle.__setattr__`` typing."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import MagicMock

import pytest

from cyberwave.twin.capabilities.joints import JointsHandle


def _make_fake_twin() -> SimpleNamespace:
    return SimpleNamespace(
        uuid="twin-uuid",
        asset_id=None,
        _data=SimpleNamespace(
            universal_schema={
                "joints": [{"name": "shoulder", "type": "revolute"}],
            },
        ),
        client=SimpleNamespace(
            mqtt=MagicMock(),
            config=SimpleNamespace(topic_prefix=""),
        ),
        _prepare_outbound_command=lambda: None,
        _resolve_topic_and_payload=lambda **kwargs: None,
        _publish_resolved=lambda resolved: None,
        _outbound_log=[],
    )


def test_setattr_accepts_twin_object_for_bookkeeping_attribute() -> None:
    twin = _make_fake_twin()
    handle = JointsHandle(twin)
    assert handle._twin is twin
    assert handle._curr_joints_by_mode == {}


def test_setattr_accepts_dict_for_curr_joints_bookkeeping_attribute() -> None:
    twin = _make_fake_twin()
    handle = JointsHandle(twin)
    handle._curr_joints_by_mode = {"live": {"shoulder": {"position": 0.0}}}
    assert handle._curr_joints_by_mode["live"]["shoulder"]["position"] == 0.0


def test_setattr_value_annotation_is_not_a_numeric_type() -> None:
    hints = get_type_hints(JointsHandle.__setattr__)
    value_hint = hints.get("value", Any)
    forbidden = {float, int, complex}
    assert value_hint not in forbidden


def test_setattr_signature_keeps_dunder_dispatchable() -> None:
    sig = inspect.signature(JointsHandle.__setattr__)
    assert list(sig.parameters) == ["self", "name", "value"]


def test_getattr_unknown_name_does_not_loop_on_twin() -> None:
    twin = _make_fake_twin()
    handle = JointsHandle(twin)
    with pytest.raises(AttributeError, match="no attribute 'twin'"):
        _ = handle.twin
