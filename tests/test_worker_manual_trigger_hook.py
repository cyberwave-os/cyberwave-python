"""Tests for the ``@cw.on_manual_trigger`` decorator in :class:`HookRegistry`.

``on_manual_trigger`` is a workflow-scoped sibling of ``on_mqtt`` (see
:mod:`tests.test_worker_mqtt_hook`). It registers with the same
``hook_type == "mqtt"`` shape so it rides the worker runtime's existing
MQTT subscribe/dispatch path, but its ``scope="workflow"`` option makes
the runtime subscribe under ``cyberwave/workflow/<uuid>/run`` rather than
the twin base — the inbound command sibling of the ``execution/*``
telemetry the worker already publishes.
"""

from __future__ import annotations

import pytest

from cyberwave.workers.hooks import (
    MANUAL_TRIGGER_SUBTOPIC,
    HookRegistry,
    manual_trigger_topic,
)


def test_on_manual_trigger_registers_workflow_scoped_mqtt_hook():
    registry = HookRegistry()

    @registry.on_manual_trigger("twin-uuid", workflow_uuid="wf-123")
    def handle_manual(payload, topic, ctx):
        return None

    hooks = registry.hooks
    assert len(hooks) == 1
    hook = hooks[0]
    # Same registration shape as on_mqtt so the runtime dispatch,
    # lifecycle and monitor stats treat it identically...
    assert hook.hook_type == "mqtt"
    assert hook.twin_uuid == "twin-uuid"
    assert hook.callback is handle_manual
    assert hook.channel == "mqtt/workflow/wf-123/run"
    # ...but the workflow scope routes it to the workflow base topic.
    assert hook.options == {
        "subtopic": MANUAL_TRIGGER_SUBTOPIC,
        "qos": 1,
        "scope": "workflow",
        "workflow_uuid": "wf-123",
    }


def test_manual_trigger_topic_is_under_workflow_base():
    assert manual_trigger_topic("wf-123") == "cyberwave/workflow/wf-123/run"


def test_on_manual_trigger_defaults_to_qos_1():
    registry = HookRegistry()

    @registry.on_manual_trigger("twin-uuid", workflow_uuid="wf-123")
    def handler(payload, topic, ctx):
        pass

    # Manual commands should arrive at-least-once, unlike the qos=0
    # default for the general on_mqtt sensor hook.
    assert registry.hooks[0].options["qos"] == 1


def test_on_manual_trigger_honours_explicit_qos():
    registry = HookRegistry()

    @registry.on_manual_trigger("twin-uuid", workflow_uuid="wf-123", qos=2)
    def handler(payload, topic, ctx):
        pass

    assert registry.hooks[0].options["qos"] == 2


@pytest.mark.parametrize("workflow_uuid", ["", "   "])
def test_on_manual_trigger_rejects_empty_workflow_uuid(workflow_uuid):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_manual_trigger("twin-uuid", workflow_uuid=workflow_uuid)


def test_on_manual_trigger_rejects_non_string_workflow_uuid():
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_manual_trigger("twin-uuid", workflow_uuid=42)  # type: ignore[arg-type]


@pytest.mark.parametrize("qos", [3, -1, 0.5, True])
def test_on_manual_trigger_rejects_invalid_qos(qos):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_manual_trigger(
            "twin-uuid", workflow_uuid="wf-123", qos=qos  # type: ignore[arg-type]
        )
