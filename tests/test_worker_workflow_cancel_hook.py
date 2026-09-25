"""Tests for the ``@cw.on_workflow_cancel`` decorator in :class:`HookRegistry`.

Sibling of ``on_manual_trigger`` (see :mod:`tests.test_worker_manual_trigger_hook`):
same ``hook_type == "mqtt"`` / ``scope="workflow"`` shape, but subscribed
under ``cyberwave/workflow/<uuid>/cancel`` instead of ``.../run``.
"""

from __future__ import annotations

import pytest

from cyberwave.workers.hooks import (
    WORKFLOW_CANCEL_SUBTOPIC,
    HookRegistry,
    workflow_cancel_topic,
)


def test_on_workflow_cancel_registers_workflow_scoped_mqtt_hook():
    registry = HookRegistry()

    @registry.on_workflow_cancel("twin-uuid", workflow_uuid="wf-123")
    def handle_cancel(payload, topic, ctx):
        return None

    hooks = registry.hooks
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.hook_type == "mqtt"
    assert hook.twin_uuid == "twin-uuid"
    assert hook.callback is handle_cancel
    assert hook.channel == "mqtt/workflow/wf-123/cancel"
    assert hook.options == {
        "subtopic": WORKFLOW_CANCEL_SUBTOPIC,
        "qos": 1,
        "scope": "workflow",
        "workflow_uuid": "wf-123",
    }


def test_workflow_cancel_topic_is_under_workflow_base():
    assert workflow_cancel_topic("wf-123") == "cyberwave/workflow/wf-123/cancel"


def test_on_workflow_cancel_defaults_to_qos_1():
    registry = HookRegistry()

    @registry.on_workflow_cancel("twin-uuid", workflow_uuid="wf-123")
    def handler(payload, topic, ctx):
        pass

    assert registry.hooks[0].options["qos"] == 1


def test_on_workflow_cancel_honours_explicit_qos():
    registry = HookRegistry()

    @registry.on_workflow_cancel("twin-uuid", workflow_uuid="wf-123", qos=2)
    def handler(payload, topic, ctx):
        pass

    assert registry.hooks[0].options["qos"] == 2


@pytest.mark.parametrize("workflow_uuid", ["", "   "])
def test_on_workflow_cancel_rejects_empty_workflow_uuid(workflow_uuid):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_workflow_cancel("twin-uuid", workflow_uuid=workflow_uuid)


def test_on_workflow_cancel_rejects_non_string_workflow_uuid():
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_workflow_cancel("twin-uuid", workflow_uuid=42)  # type: ignore[arg-type]


@pytest.mark.parametrize("qos", [3, -1, 0.5, True])
def test_on_workflow_cancel_rejects_invalid_qos(qos):
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.on_workflow_cancel(
            "twin-uuid", workflow_uuid="wf-123", qos=qos  # type: ignore[arg-type]
        )


def test_manual_trigger_and_workflow_cancel_use_distinct_channels():
    """Guards against the two workflow-scoped hooks accidentally colliding
    on the same MQTT channel, which would make the runtime's dedup/ack
    dispatch logic (keyed on subtopic) misbehave for both."""
    registry = HookRegistry()

    @registry.on_manual_trigger("twin-uuid", workflow_uuid="wf-123")
    def handle_run(payload, topic, ctx):
        pass

    @registry.on_workflow_cancel("twin-uuid", workflow_uuid="wf-123")
    def handle_cancel(payload, topic, ctx):
        pass

    channels = {hook.channel for hook in registry.hooks}
    assert channels == {"mqtt/workflow/wf-123/run", "mqtt/workflow/wf-123/cancel"}
