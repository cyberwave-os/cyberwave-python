"""Tests for twin.get_frame(source='remote_edge') (MQTT take_photo)."""

import base64
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.twin import Twin, TwinCameraHandle

FAKE_JPEG = b"\xff\xd8fake-jpeg-payload\xff\xd9"
FAKE_B64 = base64.b64encode(FAKE_JPEG).decode()


def _make_mqtt_client():
    """Build a mock MQTT client with subscribe/publish/unsubscribe."""
    mqtt = MagicMock()
    mqtt.connected = True

    _handlers: dict[str, callable] = {}

    def _subscribe(topic, handler, qos=0):
        _handlers[topic] = handler

    def _unsubscribe(topic):
        _handlers.pop(topic, None)

    mqtt.subscribe = MagicMock(side_effect=_subscribe)
    mqtt.unsubscribe = MagicMock(side_effect=_unsubscribe)
    mqtt._test_handlers = _handlers
    return mqtt


def _take_photo_catalog_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
            "commands": {
                "supported": ["take_photo"],
                "specs": {"take_photo": {}},
            },
        },
    }


def _make_twin_with_mqtt():
    """Create a Twin with a mock client that has MQTT support.

    Mirrors the real SDK architecture: the outer mqtt_client.py wrapper
    delegates to an inner _client that actually has subscribe/unsubscribe.
    """
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = FAKE_JPEG

    inner_mqtt = _make_mqtt_client()

    outer_mqtt = MagicMock()
    outer_mqtt.connected = True
    outer_mqtt._client = inner_mqtt
    outer_mqtt.subscribe = MagicMock(side_effect=inner_mqtt.subscribe)
    outer_mqtt.unsubscribe = MagicMock(side_effect=inner_mqtt.unsubscribe)
    outer_mqtt.publish = MagicMock(side_effect=inner_mqtt.publish)

    config = SimpleNamespace(topic_prefix="", runtime_mode="live", source_type="tele")
    client = SimpleNamespace(
        twins=twins_manager,
        mqtt=outer_mqtt,
        config=config,
        assets=MagicMock(),
    )
    twin = Twin(
        client,
        SimpleNamespace(
            uuid="twin-uuid",
            name="TestTwin",
            asset_uuid="asset-uuid",
            metadata=_take_photo_catalog_metadata(),
        ),
    )
    twin._prepare_outbound_command = MagicMock()
    return twin, inner_mqtt


# ---------------------------------------------------------------------------
# edge_photo tests
# ---------------------------------------------------------------------------


class TestEdgePhoto:
    def test_simulation_runtime_rejects_non_cloud_source(self):
        twin, _ = _make_twin_with_mqtt()
        twin.client.config.runtime_mode = "simulation"
        camera = TwinCameraHandle(twin)

        with pytest.raises(ValueError, match="simulation runtime"):
            camera.get_frame(source="remote_edge")

    def test_raises_when_take_photo_not_in_catalog(self):
        inner_mqtt = _make_mqtt_client()
        outer_mqtt = MagicMock()
        outer_mqtt.connected = True
        outer_mqtt._client = inner_mqtt
        client = SimpleNamespace(
            twins=MagicMock(),
            mqtt=outer_mqtt,
            config=SimpleNamespace(topic_prefix="", runtime_mode="live", source_type="tele"),
            assets=MagicMock(),
        )
        twin = Twin(
            client,
            SimpleNamespace(
                uuid="twin-uuid",
                name="TestTwin",
                asset_uuid="asset-uuid",
                metadata={
                    "mqtt": {
                        "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                        "commands": {"supported": ["stop"], "specs": {"stop": {}}},
                    },
                },
            ),
        )
        camera = TwinCameraHandle(twin)

        with pytest.raises(CyberwaveError, match="cannot support take photo"):
            camera.get_frame("bytes", source="remote_edge")

    def test_sends_take_photo_command(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({
                    "image": FAKE_B64,
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                })

        t = threading.Thread(target=respond)
        t.start()

        result = camera.get_frame("bytes", source="remote_edge")
        t.join()

        mqtt.publish.assert_called_once()
        call_args = mqtt.publish.call_args
        topic = call_args[0][0]
        payload = call_args[0][1]
        assert topic == "cyberwave/twin/twin-uuid/command"
        assert payload["command"] == "take_photo"
        assert payload["source_type"] == "tele"
        assert result == FAKE_JPEG

    def test_returns_numpy_frame(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        mock_np = MagicMock()
        mock_cv2 = MagicMock()
        sentinel = MagicMock(name="numpy_frame")
        mock_np.frombuffer.return_value = MagicMock()
        mock_np.uint8 = "uint8"
        mock_cv2.imdecode.return_value = sentinel
        mock_cv2.IMREAD_COLOR = 1

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({"image": FAKE_B64, "format": "jpeg"})

        t = threading.Thread(target=respond)
        t.start()

        with patch.dict("sys.modules", {"numpy": mock_np, "cv2": mock_cv2}):
            result = camera.get_frame("numpy", source="remote_edge")

        t.join()
        assert result is sentinel

    def test_raises_on_timeout(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        with pytest.raises(CyberwaveError, match="Timed out.*take_photo"):
            camera.get_frame(source="remote_edge", edge_timeout_s=0.2)

    def test_raises_on_edge_error_response(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({
                    "status": "error",
                    "message": "No camera frame available",
                })

        t = threading.Thread(target=respond)
        t.start()

        with pytest.raises(CyberwaveError, match="No camera frame available"):
            camera.get_frame(source="remote_edge", edge_timeout_s=2.0)

        t.join()

    def test_raises_on_missing_image_field(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({"format": "jpeg"})

        t = threading.Thread(target=respond)
        t.start()

        with pytest.raises(CyberwaveError, match="missing"):
            camera.get_frame(source="remote_edge", edge_timeout_s=2.0)

        t.join()

    def test_unsubscribes_after_response(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({"image": FAKE_B64, "format": "jpeg"})

        t = threading.Thread(target=respond)
        t.start()

        camera.get_frame("bytes", source="remote_edge")
        t.join()

        mqtt.unsubscribe.assert_called_once_with(
            "cyberwave/twin/twin-uuid/camera/photo"
        )

    def test_unsubscribes_on_timeout(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        with pytest.raises(CyberwaveError):
            camera.get_frame(source="remote_edge", edge_timeout_s=0.1)

        mqtt.unsubscribe.assert_called_once_with(
            "cyberwave/twin/twin-uuid/camera/photo"
        )

    def test_respects_topic_prefix(self):
        twin, mqtt = _make_twin_with_mqtt()
        twin.client.config.topic_prefix = "custom/"
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "custom/cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler({"image": FAKE_B64, "format": "jpeg"})

        t = threading.Thread(target=respond)
        t.start()

        camera.get_frame("bytes", source="remote_edge")
        t.join()

        call_args = mqtt.publish.call_args
        assert call_args[0][0] == "custom/cyberwave/twin/twin-uuid/command"


# ---------------------------------------------------------------------------
# edge_photos tests
# ---------------------------------------------------------------------------


class TestEdgePhotos:
    def test_captures_multiple_frames(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        call_count = 0

        def respond():
            nonlocal call_count
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                call_count += 1
                handler({"image": FAKE_B64, "format": "jpeg"})

        frames = []
        for _ in range(3):
            t = threading.Thread(target=respond)
            t.start()
            frames.append(
                camera.get_frame("bytes", source="remote_edge", edge_timeout_s=2.0)
            )
            t.join()

        assert len(frames) == 3
        assert all(f == FAKE_JPEG for f in frames)

    def test_edge_photos_convenience(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def poll_respond(stop_event):
            while not stop_event.is_set():
                handler = mqtt._test_handlers.get(
                    "cyberwave/twin/twin-uuid/camera/photo"
                )
                if handler:
                    handler({"image": FAKE_B64, "format": "jpeg"})
                time.sleep(0.01)

        stop = threading.Event()
        t = threading.Thread(target=poll_respond, args=(stop,))
        t.start()

        try:
            frames = []
            for _ in range(2):
                frames.append(
                    camera.get_frame("bytes", source="remote_edge", edge_timeout_s=2.0)
                )
                time.sleep(0.05)
            assert len(frames) == 2
        finally:
            stop.set()
            t.join()
