"""PR3 M4 — twin.listen() session API."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.manifest.driver_config import (
    TWIN_CAMERA_PHOTO_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    TWIN_POSITION_TOPIC_SLUG,
    TWIN_ROTATION_TOPIC_SLUG,
    TWIN_TELEMETRY_TOPIC_SLUG,
)
from cyberwave.mqtt.listen import MqttMessage, noop_handler
from cyberwave.twin import LocomoteTwin


def _catalog_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {
                TWIN_COMMAND_TOPIC_SLUG: {"direction": "both"},
                TWIN_POSITION_TOPIC_SLUG: {"direction": "publish"},
                TWIN_ROTATION_TOPIC_SLUG: {"direction": "publish"},
                TWIN_TELEMETRY_TOPIC_SLUG: {"direction": "publish"},
            },
            "commands": {"supported": ["move_forward", "stop"]},
        }
    }


def _make_twin() -> LocomoteTwin:
    mqtt = MagicMock()
    mqtt.connected = True
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="dev/", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    return LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="twin-1",
            name="Go2",
            asset_uuid="a",
            metadata=_catalog_metadata(),
            capabilities={"can_locomote": True},
        ),
    )


def test_listen_dry_run_returns_topic_listen_specs() -> None:
    twin = _make_twin()
    specs = twin.listen(dry_run=True)
    assert TWIN_POSITION_TOPIC_SLUG in specs
    assert "twin-1" in specs[TWIN_POSITION_TOPIC_SLUG].topic
    assert specs[TWIN_POSITION_TOPIC_SLUG].topic.startswith("dev/")


def test_listen_dry_run_default_excludes_telemetry() -> None:
    twin = _make_twin()
    specs = twin.listen(dry_run=True)
    assert all("/telemetry" not in slug for slug in specs)


def test_listen_dry_run_include_telemetry() -> None:
    twin = _make_twin()
    specs = twin.listen(dry_run=True, include_telemetry=True)
    assert TWIN_TELEMETRY_TOPIC_SLUG in specs


def test_listen_dry_run_camera_filter_subscribes_photo_slug() -> None:
    twin = _make_twin()
    specs = twin.listen(dry_run=True, filters=["camera"])
    assert list(specs) == [TWIN_CAMERA_PHOTO_TOPIC_SLUG]
    assert "twin-1" in specs[TWIN_CAMERA_PHOTO_TOPIC_SLUG].topic


def test_listen_dry_run_unknown_filter_raises() -> None:
    twin = _make_twin()
    with pytest.raises(ValueError, match="Unknown listen filter"):
        twin.listen(dry_run=True, filters=["not_a_stream"])


def test_listen_session_dispatches_handler_by_slug() -> None:
    twin = _make_twin()
    received: list[MqttMessage] = []

    def _handler(msg: MqttMessage) -> None:
        received.append(msg)

    session = twin.listen(
        dry_run=False,
        filters=["pose"],
        handlers={TWIN_POSITION_TOPIC_SLUG: _handler},
    )
    assert session is not None
    call = twin.client.mqtt.subscribe.call_args_list[0]
    topic = call[0][0]
    callback = call[0][1]
    callback({"position": {"x": 1.0, "y": 0.0, "z": 0.0}})
    assert received[0].slug == TWIN_POSITION_TOPIC_SLUG
    assert "twin-1" in received[0].topic
    session.stop()


def test_listen_default_handler_is_noop() -> None:
    twin = _make_twin()
    session = twin.listen(dry_run=False, filters=["pose"])
    call = twin.client.mqtt.subscribe.call_args_list[0]
    call[0][1]({"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
    session.stop()


def test_listen_does_not_call_prepare_outbound_command() -> None:
    twin = _make_twin()
    with patch.object(twin, "_prepare_outbound_command") as gate:
        twin.listen(dry_run=True)
    gate.assert_not_called()


def test_subscribe_is_alias_for_listen() -> None:
    assert LocomoteTwin.subscribe is LocomoteTwin.listen
