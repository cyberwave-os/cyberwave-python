from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveAPIError, CyberwaveError
from cyberwave.resources import TwinManager
from cyberwave.twin import CameraTwin, Twin

_CAMERA_CAPS = {"sensors": [{"id": "front_camera", "type": "rgb", "name": "front"}]}


def _make_manager() -> tuple[TwinManager, MagicMock]:
    mock_api = MagicMock()
    manager = TwinManager(mock_api)
    return manager, mock_api


def test_get_latest_frame_returns_bytes_and_passes_query_params():
    manager, mock_api = _make_manager()
    response_data = MagicMock()
    response_data.data = b"jpeg-bytes"
    mock_api.api_client.param_serialize.return_value = (
        "GET",
        "/api/v1/twins/twin-uuid/latest-frame",
        {},
        None,
        [],
        {},
        [],
        {},
    )
    mock_api.api_client.call_api.return_value = response_data

    result = manager.get_latest_frame(
        "twin-uuid",
        sensor_id="wrist_camera",
        mock=True,
        source_type="simulation",
    )

    assert result == b"jpeg-bytes"
    call_kwargs = mock_api.api_client.param_serialize.call_args.kwargs
    assert call_kwargs["method"] == "GET"
    assert call_kwargs["resource_path"] == "/api/v1/twins/{uuid}/latest-frame"
    assert call_kwargs["path_params"] == {"uuid": "twin-uuid"}
    assert ("sensor_id", "wrist_camera") in call_kwargs["query_params"]
    assert ("mock", "true") in call_kwargs["query_params"]
    assert ("source_type", "sim") in call_kwargs["query_params"]
    response_data.read.assert_called_once()


def test_get_latest_frame_encodes_string_payload():
    manager, mock_api = _make_manager()
    response_data = MagicMock()
    response_data.data = "frame-text"
    mock_api.api_client.param_serialize.return_value = (
        "GET",
        "/api/v1/twins/twin-uuid/latest-frame",
        {},
        None,
        [],
        {},
        [],
        {},
    )
    mock_api.api_client.call_api.return_value = response_data

    result = manager.get_latest_frame("twin-uuid")

    assert result == b"frame-text"


def test_get_latest_frame_raises_for_unexpected_payload():
    manager, mock_api = _make_manager()
    response_data = MagicMock()
    response_data.data = {"not": "bytes"}
    response_data.raw_data = None
    mock_api.api_client.param_serialize.return_value = (
        "GET",
        "/api/v1/twins/twin-uuid/latest-frame",
        {},
        None,
        [],
        {},
        [],
        {},
    )
    mock_api.api_client.call_api.return_value = response_data

    with pytest.raises(CyberwaveAPIError, match="get latest frame"):
        manager.get_latest_frame("twin-uuid")


def test_twin_get_latest_frame_delegates_to_manager():
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"frame"
    client = SimpleNamespace(twins=twins_manager)
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="twin-uuid", name="Twin", capabilities=_CAMERA_CAPS),
    )

    result = twin.get_latest_frame(sensor_id="front_camera")

    assert result == b"frame"
    twins_manager.get_latest_frame.assert_called_once_with(
        "twin-uuid",
        sensor_id="front_camera",
        mock=False,
    )


def test_twin_get_latest_frame_wraps_errors():
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.side_effect = RuntimeError("boom")
    client = SimpleNamespace(twins=twins_manager)
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="twin-uuid", name="Twin", capabilities=_CAMERA_CAPS),
    )

    with pytest.raises(CyberwaveError, match="Failed to get latest frame"):
        twin.get_latest_frame()


def test_twin_get_latest_frame_uses_client_affect_source_type():
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"frame"
    client = SimpleNamespace(
        twins=twins_manager,
        config=SimpleNamespace(source_type="sim"),
    )
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="twin-uuid", name="Twin", capabilities=_CAMERA_CAPS),
    )

    result = twin.get_latest_frame()

    assert result == b"frame"
    twins_manager.get_latest_frame.assert_called_once_with(
        "twin-uuid",
        sensor_id="front_camera",
        mock=False,
        source_type="sim",
    )


def test_twin_get_latest_frame_maps_edge_source_type_to_tele():
    """When affect('real-world') sets source_type='edge', capture_frame should use 'tele'."""
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"frame"
    client = SimpleNamespace(
        twins=twins_manager,
        config=SimpleNamespace(source_type="edge"),
    )
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="twin-uuid", name="Twin", capabilities=_CAMERA_CAPS),
    )

    twin.get_latest_frame()

    twins_manager.get_latest_frame.assert_called_once_with(
        "twin-uuid",
        sensor_id="front_camera",
        mock=False,
        source_type="tele",
    )


def test_camera_twin_capture_frame_uses_local_capture_not_rest():
    """capture_frame delegates to camera.capture (local path), not latest-frame REST."""
    import numpy as np
    from unittest.mock import patch

    twins_manager = MagicMock()
    client = SimpleNamespace(twins=twins_manager)
    camera_twin = CameraTwin(
        client,
        SimpleNamespace(uuid="cam-twin", name="CamTwin", capabilities=_CAMERA_CAPS),
    )
    fake = np.zeros((8, 8, 3), dtype=np.uint8)

    with patch.object(camera_twin.camera, "_capture_local_array", return_value=fake):
        result = camera_twin.capture_frame("numpy", sensor_id="front")

    assert result.shape == (8, 8, 3)
    twins_manager.get_latest_frame.assert_not_called()
