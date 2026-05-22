"""Regression tests for ``JointController.__setattr__`` typing.

The dunder is used both for joint position writes (``j.shoulder = 45``)
and for two bookkeeping attributes set in ``__init__``: ``twin`` (a
``Twin`` instance) and ``_joint_states`` (a ``dict`` or ``None``).
Annotating ``value`` as ``float`` was a lie for those two attributes
and — once the SDK is Cython-compiled with ``language_level=3`` for
the worker images — it crashed every workflow tick with::

    TypeError: must be real number, not CameraTwin

at ``twin.py:241`` (the function definition line — Cython enforces the
annotation as a C-level coercion *at function entry*, before any of the
``if name in ["twin", "_joint_states"]`` guard runs).

These tests pin two contracts that together prevent the regression:

1. ``JointController.__setattr__`` accepts ``Twin``-like and ``dict``
   values for the bookkeeping names without raising. (Catches the
   source-level intent regression even on uncompiled Python.)
2. The annotation on ``value`` is not a numeric type. (Catches the
   Cython-strict regression at source review time, since uncompiled
   CPython doesn't enforce annotations and would otherwise let a
   ``: float`` slip back in unnoticed.)
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import MagicMock

from cyberwave.twin import JointController


def _make_fake_twin() -> SimpleNamespace:
    """Minimal stand-in that satisfies ``JointController.__init__``.

    ``__init__`` only stores ``twin`` on the instance — the field is
    consumed lazily by ``refresh()``/``set()``. We intentionally don't
    use a real ``Twin`` so the test stays fast and dependency-free.
    """

    return SimpleNamespace(
        uuid="twin-uuid",
        client=SimpleNamespace(mqtt=MagicMock()),
    )


def test_setattr_accepts_twin_object_for_bookkeeping_attribute() -> None:
    twin = _make_fake_twin()

    jc = JointController(twin)

    assert jc.twin is twin
    assert jc._joint_states is None


def test_setattr_accepts_dict_for_joint_states_bookkeeping_attribute() -> None:
    twin = _make_fake_twin()
    jc = JointController(twin)

    jc._joint_states = {"shoulder": 0.0}

    assert jc._joint_states == {"shoulder": 0.0}


def test_setattr_value_annotation_is_not_a_numeric_type() -> None:
    """Pin the SDK contract that survives Cython compilation.

    Cython's ``language_level=3`` turns on ``annotation_typing`` by
    default, which would coerce ``value`` to a Python ``float`` at
    function entry. Keeping the annotation as ``Any`` (or omitting it)
    means the dunder stays a plain Python method even when compiled,
    so ``self.twin = twin`` and ``self._joint_states = {}`` keep
    working.
    """

    hints = get_type_hints(JointController.__setattr__)
    value_hint = hints.get("value", Any)
    forbidden = {float, int, complex}
    assert value_hint not in forbidden, (
        "JointController.__setattr__ value annotation must not be a "
        "numeric type — Cython would enforce it at call entry and "
        "break ``self.twin = twin`` in __init__. Use Any instead."
    )


def test_setattr_signature_keeps_dunder_dispatchable() -> None:
    """Belt-and-braces: ensure the parameter list still matches the
    dunder protocol so future refactors don't accidentally drop the
    ``value`` parameter (which would also re-introduce the bug, just
    with a different traceback).
    """

    sig = inspect.signature(JointController.__setattr__)
    assert list(sig.parameters) == ["self", "name", "value"]
