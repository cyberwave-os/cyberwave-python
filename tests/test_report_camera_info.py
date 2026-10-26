"""``TwinManager.report_camera_info``: a driver reports its camera's calibration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.resources import TwinManager


def _manager() -> tuple[TwinManager, MagicMock]:
    api = MagicMock()
    api.api_client.param_serialize.return_value = ("POST", "url", {}, None, [])
    api.api_client.response_deserialize.return_value = SimpleNamespace(
        data={"changed": True, "sensor": "wrist_color_camera", "parameters": {}}
    )
    return TwinManager(api), api


def test_report_posts_the_calibration_to_the_sensor() -> None:
    manager, api = _manager()

    result = manager.report_camera_info(
        "twin-1",
        "wrist_color_camera",
        width=640,
        height=480,
        fx=653.68229,
        fy=651.855994,
        cx=311.753415,
        cy=232.400954,
    )

    kwargs = api.api_client.param_serialize.call_args.kwargs
    assert kwargs["method"] == "POST"
    assert kwargs["resource_path"] == "/api/v1/twins/{uuid}/sensors/{name}/camera-info"
    assert kwargs["path_params"] == {"uuid": "twin-1", "name": "wrist_color_camera"}
    assert kwargs["body"] == {
        "width": 640,
        "height": 480,
        "fx": 653.68229,
        "fy": 651.855994,
        "cx": 311.753415,
        "cy": 232.400954,
    }
    api.api_client.call_api.assert_called_once_with(
        *api.api_client.param_serialize.return_value
    )
    assert result["changed"] is True


def test_a_failed_report_raises() -> None:
    manager, api = _manager()
    api.api_client.call_api.side_effect = RuntimeError("backend down")

    with pytest.raises(Exception):
        manager.report_camera_info(
            "twin-1",
            "wrist_color_camera",
            width=640,
            height=480,
            fx=1,
            fy=1,
            cx=0,
            cy=0,
        )
