"""Camera APIs duplicated by the driver camera stack warn ahead of removal in 1.0.0."""

from __future__ import annotations

import importlib
import sys
import threading
import warnings
from unittest.mock import MagicMock

import pytest


def test_client_video_stream_warns_and_delegates() -> None:
    from cyberwave.client import Cyberwave

    client = MagicMock(spec=Cyberwave)
    client._build_video_streamer.return_value = "streamer"

    with pytest.warns(DeprecationWarning, match="1.0.0"):
        result = Cyberwave.video_stream(client, "twin-uuid", camera_id=2, fps=15)

    assert result == "streamer"
    kwargs = client._build_video_streamer.call_args.kwargs
    assert (kwargs["twin_uuid"], kwargs["camera_id"], kwargs["fps"]) == ("twin-uuid", 2, 15)


def test_camera_stream_manager_warns() -> None:
    pytest.importorskip("aiortc")
    from cyberwave.sensor.manager import CameraStreamManager

    with pytest.warns(DeprecationWarning, match="BaseCameraDriver"):
        CameraStreamManager(MagicMock(), [], threading.Event(), MagicMock())


def test_edge_camera_config_warns() -> None:
    from cyberwave.sensor.config import EdgeCameraConfig

    with pytest.warns(DeprecationWarning, match="CameraSourceConfig"):
        EdgeCameraConfig()


def test_cyberwave_camera_module_warns() -> None:
    pytest.importorskip("aiortc")
    sys.modules.pop("cyberwave.camera", None)

    with pytest.warns(DeprecationWarning, match="cyberwave.sensor"):
        importlib.import_module("cyberwave.camera")


def test_twin_streaming_uses_the_internal_factory_without_warning() -> None:
    import asyncio

    from cyberwave.twin.classes import CameraTwin

    streamer = MagicMock()
    streamer.start = MagicMock(return_value=asyncio.sleep(0))
    twin = MagicMock()
    twin._is_simulation_mode.return_value = False
    twin.uuid = "twin-uuid"
    twin.default_camera_name = "color_camera"
    twin.client._build_video_streamer.return_value = streamer

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        asyncio.run(CameraTwin.stream_video_background.__wrapped__(twin, fps=15))

    twin.client._build_video_streamer.assert_called_once()
    twin.client.video_stream.assert_not_called()
