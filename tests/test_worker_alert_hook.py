from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.decode import decode_sample_payload


def test_on_alert_registers_alert_channel_hook():
    registry = HookRegistry()

    @registry.on_alert("twin-uuid")
    def handle_alert(alert, ctx):
        return None

    hooks = registry.hooks
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook.channel == "alert"
    assert hook.hook_type == "alert"
    assert hook.twin_uuid == "twin-uuid"
    assert hook.callback is handle_alert


def test_decode_sample_payload_parses_raw_json_alert_payload():
    sample = type(
        "Sample",
        (),
        {
            "payload": b'{"uuid":"alert-uuid","alert_type":"person_detected"}',
            "timestamp": 123.0,
        },
    )()

    payload, timestamp = decode_sample_payload(sample)

    assert payload == {"uuid": "alert-uuid", "alert_type": "person_detected"}
    assert timestamp == 123.0
