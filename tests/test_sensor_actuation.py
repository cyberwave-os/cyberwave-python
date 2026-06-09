from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.twin import CameraTwin

_CAMERA_CAPS = {"sensors": [{"id": "cam", "type": "rgb", "name": "cam"}]}


def test_get_frame_cloud_returns_none_on_rest_failure() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(source_type="tele"),
    )
    client.twins.get_latest_frame.side_effect = RuntimeError("network")
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="t", name="T", capabilities=_CAMERA_CAPS),
    )
    assert twin.camera.get_frame(source="cloud") is None


def test_get_frame_local_never_calls_rest() -> None:
    client = SimpleNamespace(
        twins=MagicMock(),
        config=SimpleNamespace(source_type="tele"),
    )
    twin = CameraTwin(
        client,
        SimpleNamespace(uuid="t", name="T", capabilities=_CAMERA_CAPS),
    )
    with patch.object(twin.camera, "_capture_local_array", return_value=None):
        twin.camera.get_frame(source="local")
    client.twins.get_latest_frame.assert_not_called()
