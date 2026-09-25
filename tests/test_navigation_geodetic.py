"""Geodetic (GPS) ``goto`` payload tests.

The Move To GPS workflow node emits ``twin.navigation.goto`` with a
``geodetic_position`` + ``orientation`` + ``coordinate_frame`` instead of a
Cartesian ``position``. These verify the SDK builds that command shape (the
one ``NavigationService`` and the DJI Wayline driver consume) and rejects
mixing the two representations.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.motion import TwinNavigationHandle


def _build_twin_nav() -> tuple[TwinNavigationHandle, MagicMock]:
    api_client = MagicMock()
    api_client.param_serialize.return_value = ()
    response = MagicMock()
    response.data = b"{}"
    api_client.call_api.return_value = response
    client = SimpleNamespace(
        config=SimpleNamespace(source_type="edge"),
        api=SimpleNamespace(api_client=api_client),
    )
    twin = SimpleNamespace(uuid="twin-uuid", client=client)
    return TwinNavigationHandle(twin), api_client


_GEODETIC_FRAME = {
    "kind": "geodetic",
    "horizontal_units": "deg",
    "vertical_units": "m",
    "altitude_reference": "relative_to_takeoff",
    "yaw_units": "deg",
    "yaw_reference": "true_north",
}


def test_geodetic_goto_builds_gps_command() -> None:
    nav, api_client = _build_twin_nav()

    nav.goto(
        geodetic_position={
            "latitude": 47.3769,
            "longitude": 8.5417,
            "altitude": 30.0,
        },
        orientation={"heading": 90.0},
        coordinate_frame=_GEODETIC_FRAME,
    )

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["command"] == "goto"
    assert body["geodetic_position"] == {
        "latitude": 47.3769,
        "longitude": 8.5417,
        "altitude": 30.0,
    }
    assert body["orientation"] == {"heading": 90.0}
    assert body["coordinate_frame"]["kind"] == "geodetic"
    # A geodetic command must never carry a Cartesian position.
    assert "position" not in body


def test_goto_rejects_position_and_geodetic_position_together() -> None:
    nav, _ = _build_twin_nav()
    with pytest.raises(ValueError, match="not both"):
        nav.goto(
            [1.0, 2.0, 0.0],
            geodetic_position={
                "latitude": 47.0,
                "longitude": 8.0,
                "altitude": 10.0,
            },
        )


def test_goto_requires_a_target() -> None:
    nav, _ = _build_twin_nav()
    with pytest.raises(ValueError, match="position or geodetic_position"):
        nav.goto()


def test_cartesian_goto_still_builds_position_command() -> None:
    nav, api_client = _build_twin_nav()

    nav.goto([1.0, 2.0, 0.0])

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["position"] == [1.0, 2.0, 0.0]
    assert "geodetic_position" not in body
    assert "coordinate_frame" not in body


def test_orientation_is_forwarded_on_the_cartesian_branch() -> None:
    # Regression: orientation used to be added only inside the geodetic branch,
    # so a Cartesian goto silently dropped it instead of letting the server
    # reject an unsupported combination.
    nav, api_client = _build_twin_nav()

    nav.goto([1.0, 2.0, 0.0], orientation={"heading": 90.0})

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["position"] == [1.0, 2.0, 0.0]
    assert body["orientation"] == {"heading": 90.0}


def test_geodetic_position_and_heading_coerce_editor_strings() -> None:
    nav, api_client = _build_twin_nav()

    nav.goto(
        geodetic_position={
            "latitude": "47.3769",
            "longitude": "8.5417",
            "altitude": "30",
        },
        orientation={"heading": "90"},
        coordinate_frame=_GEODETIC_FRAME,
    )

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["geodetic_position"] == {
        "latitude": 47.3769,
        "longitude": 8.5417,
        "altitude": 30.0,
    }
    assert body["orientation"] == {"heading": 90.0}


def test_goto_rejects_orientation_together_with_yaw_or_rotation() -> None:
    nav, _ = _build_twin_nav()
    with pytest.raises(ValueError, match="not both"):
        nav.goto([1.0, 2.0, 0.0], orientation={"heading": 90.0}, yaw=1.0)
    with pytest.raises(ValueError, match="not both"):
        nav.goto(
            [1.0, 2.0, 0.0],
            orientation={"heading": 90.0},
            rotation=[1.0, 0.0, 0.0, 0.0],
        )


def test_geodetic_position_requires_a_geodetic_coordinate_frame() -> None:
    # The server infers "geodetic" only from coordinate_frame.kind, so a
    # geodetic_position without one would be read as a malformed Cartesian goto.
    nav, _ = _build_twin_nav()
    with pytest.raises(ValueError, match="coordinate_frame"):
        nav.goto(
            geodetic_position={
                "latitude": 47.0,
                "longitude": 8.0,
                "altitude": 10.0,
            }
        )
    with pytest.raises(ValueError, match="coordinate_frame"):
        nav.goto(
            geodetic_position={
                "latitude": 47.0,
                "longitude": 8.0,
                "altitude": 10.0,
            },
            coordinate_frame={"kind": "cartesian"},
        )
