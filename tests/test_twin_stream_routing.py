from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave.twin.classes import CameraTwin


def test_twin_stream_defaults_to_camera() -> None:
    twin = CameraTwin(
        SimpleNamespace(mqtt=MagicMock(), config=SimpleNamespace(topic_prefix="")),
        SimpleNamespace(
            uuid="t",
            capabilities={"sensors": [{"id": "cam1", "type": "rgb", "name": "front"}]},
        ),
    )
    # twin.stream() routes through the default imaging handle (mixin path),
    # which is distinct from the twin.camera family collection.
    handle = twin._default_imaging_handle()
    with patch.object(handle, "stream") as mock_stream:
        twin.stream(fps=10, idx=1)
    mock_stream.assert_called_once_with(fps=10, camera_id=1)
