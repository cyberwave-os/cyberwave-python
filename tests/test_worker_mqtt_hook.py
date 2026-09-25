"""Tests for the ``@cw.on_mqtt`` decorator in :class:`HookRegistry`.

Mirrors :mod:`tests.test_worker_alert_hook` so the behaviour stays
parallel with the alert hook (the workflow MQTT trigger emitter relies
on ``cw.on_mqtt`` having the same registration shape).
"""

from __future__ import annotations

import pytest

from cyberwave.workers.hooks import HookRegistry


def test_on_mqtt_registers_channel_hook_with_subtopic_and_qos():
    registry = HookRegistry()

    @registry.on_mqtt("twin-uuid", subtopic="status", qos=1)
    def handle_mqtt(payload, topic, ctx):
        return None

    hooks = registry.hooks
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.channel == "mqtt/status"
    assert hook.hook_type == "mqtt"
    assert hook.twin_uuid == "twin-uuid"
    assert hook.callback is handle_mqtt
    assert hook.options == {"subtopic": "status", "qos": 1}


def test_on_mqtt_normalises_subtopic_whitespace():
    registry = HookRegistry()

    @registry.on_mqtt("twin-uuid", subtopic="  position  ")
    def handler(payload, topic, ctx):
        pass

    hook = registry.hooks[0]
    assert hook.options["subtopic"] == "position"
    assert hook.channel == "mqtt/position"
    # Defaults to qos=0 when not specified
    assert hook.options["qos"] == 0


@pytest.mark.parametrize(
    "subtopic",
    ["", "   ", "/abs", "cyberwave/twin/x", "wild+", "wild#", "double//slash"],
)
def test_on_mqtt_rejects_invalid_subtopic(subtopic):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_mqtt("twin-uuid", subtopic=subtopic)


def test_on_mqtt_rejects_non_string_subtopic():
    registry = HookRegistry()
    with pytest.raises(TypeError):
        registry.on_mqtt("twin-uuid", subtopic=42)  # type: ignore[arg-type]


@pytest.mark.parametrize("qos", [-1, 3, "0", 0.5, True])
def test_on_mqtt_rejects_invalid_qos(qos):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_mqtt("twin-uuid", subtopic="status", qos=qos)
