"""Unified twin.camera.get_frame() — cloud / local / zenoh sources."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cyberwave.twin import CameraTwin

_CAMERA_CAPS = {
    "sensors": [{"id": "cam", "type": "rgb", "name": "cam"}],
}


def _camera_twin(client):
    return CameraTwin(
        client,
        SimpleNamespace(uuid="t", name="T", capabilities=_CAMERA_CAPS),
    )


def test_get_frame_cloud_default_calls_rest() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
    )
    client.twins.get_latest_frame.return_value = b"jpeg"
    twin = _camera_twin(client)

    result = twin.camera.get_frame()

    assert result == b"jpeg"
    client.twins.get_latest_frame.assert_called_once()


def test_get_frame_cloud_returns_none_on_rest_failure() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
    )
    client.twins.get_latest_frame.side_effect = RuntimeError("network")
    twin = _camera_twin(client)

    assert twin.camera.get_frame(source="cloud") is None


def test_get_frame_local_never_calls_rest() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
    )
    twin = _camera_twin(client)
    fake = np.zeros((4, 4, 3), dtype=np.uint8)
    cam = twin.camera[0]

    with patch.object(cam, "_capture_local_array", return_value=fake):
        result = cam.get_frame("numpy", source="local")

    assert result.shape == (4, 4, 3)
    client.twins.get_latest_frame.assert_not_called()


def test_get_frame_zenoh_uses_subscribe_fetch() -> None:
    fake = np.zeros((2, 2, 3), dtype=np.uint8)
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
        fetch_zenoh_frame=MagicMock(return_value=fake),
    )
    twin = CameraTwin(
        client,
        SimpleNamespace(
            uuid="twin-uuid",
            name="T",
            capabilities={"sensors": [{"id": "cam1", "name": "front", "type": "rgb"}]},
        ),
    )

    result = twin.camera.get_frame("numpy", source="zenoh", sensor_id="cam1")

    assert result.shape == (2, 2, 3)
    client.fetch_zenoh_frame.assert_called_once_with(
        "twin-uuid",
        sensor_name="cam1",
        timeout_s=3.0,
        max_age_ms=None,
    )
    client.twins.get_latest_frame.assert_not_called()


def test_get_frame_zenoh_uses_sensor_id_not_display_name() -> None:
    """Wire keys use sensor id (driver convention), not human-readable name."""
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
        fetch_zenoh_frame=MagicMock(
            return_value=np.zeros((2, 2, 3), dtype=np.uint8)
        ),
    )
    twin = CameraTwin(
        client,
        SimpleNamespace(
            uuid="twin-uuid",
            name="T",
            capabilities={
                "sensors": [
                    {
                        "id": "color_camera",
                        "name": "RGB Camera",
                        "type": "rgb",
                    }
                ]
            },
        ),
    )

    twin.camera.get_frame("numpy", source="zenoh", sensor_id="color_camera")

    client.fetch_zenoh_frame.assert_called_once()
    assert client.fetch_zenoh_frame.call_args.kwargs["sensor_name"] == "color_camera"


def test_twin_get_frame_defaults_to_first_sensor() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
    )
    client.twins.get_latest_frame.return_value = b"jpeg"
    twin = CameraTwin(
        client,
        SimpleNamespace(
            uuid="t",
            name="T",
            capabilities={
                "sensors": [
                    {"id": "cam_a", "type": "rgb", "name": "front"},
                    {"id": "cam_b", "type": "rgb", "name": "rear"},
                ]
            },
        ),
    )

    twin.get_frame()

    client.twins.get_latest_frame.assert_called_once()
    assert client.twins.get_latest_frame.call_args.kwargs["sensor_id"] == "cam_a"


def test_get_frame_rejects_unknown_source() -> None:
    twin = _camera_twin(
        SimpleNamespace(twins=MagicMock(), config=SimpleNamespace(source_type="tele")),
    )
    with pytest.raises(ValueError, match="cloud.*local.*zenoh.*remote_edge"):
        twin.camera.get_frame(source="mqtt")
