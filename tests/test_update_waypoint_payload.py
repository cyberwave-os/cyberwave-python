"""``EnvironmentManager.update_waypoint`` refuses a pose with a missing axis.

The generated REST model reads the OpenAPI default and substitutes ``0.0``
for an axis the caller left out, so a dropped pose would reach the server as
three finite numbers and be written as the waypoint's frame origin — the
server cannot tell that from a measurement. The refusal has to happen before
the request is built, which means here (CYB-3809).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cyberwave.resources import EnvironmentManager

WAYPOINT_METHOD = "src_app_api_environments_update_environment_waypoint_position"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"position": {}}, "missing x, y, z", id="dropped-pose"),
        pytest.param({"position": {"x": 0.5}}, "missing y, z", id="partial-position"),
        pytest.param(
            {"rotation": {"w": 1.0, "x": 0.0}}, "missing y, z", id="partial-rotation"
        ),
        pytest.param(
            {"position": [1.0, 2.0, 3.0]}, "must be a dict", id="sequence-position"
        ),
    ],
)
def test_update_waypoint_refuses_a_partial_position(kwargs, expected) -> None:
    api = MagicMock()
    manager = EnvironmentManager(api)

    with pytest.raises(ValueError, match=expected):
        manager.update_waypoint("env-1", "centroid", **kwargs)

    getattr(api, WAYPOINT_METHOD).assert_not_called()


def test_update_waypoint_sends_a_complete_pose() -> None:
    api = MagicMock()
    manager = EnvironmentManager(api)

    manager.update_waypoint(
        "env-1", "centroid", position={"x": 1.0, "y": 2.0, "z": 3.0}
    )

    _, _, payload = getattr(api, WAYPOINT_METHOD).call_args.args
    assert payload.position.x == 1.0
    assert payload.position.y == 2.0
    assert payload.position.z == 3.0
    # Not sent, so left untouched server-side rather than reset.
    assert payload.rotation is None


class _FramedPositionUpdate:
    """A generated model new enough to carry ``reference_frame``."""

    model_fields = {
        "position": None,
        "rotation": None,
        "reference_frame": None,
        "joint_state_source": None,
    }

    @classmethod
    def from_dict(cls, data):
        return dict(data)


def test_update_waypoint_forwards_the_reference_frame(monkeypatch) -> None:
    import cyberwave.rest

    monkeypatch.setattr(
        cyberwave.rest,
        "EnvironmentWaypointPositionUpdateSchema",
        _FramedPositionUpdate,
        raising=False,
    )
    api = MagicMock()
    frame = {"type": "twin_link", "twin_uuid": "twin-1", "link": None}

    EnvironmentManager(api).update_waypoint(
        "env-1",
        "centroid",
        position={"x": 1.0, "y": 2.0, "z": 3.0},
        reference_frame=frame,
    )

    _, _, payload = getattr(api, WAYPOINT_METHOD).call_args.args
    assert payload["reference_frame"] == frame
    assert "joint_state_source" not in payload


def test_update_waypoint_forwards_the_joint_state_source(monkeypatch) -> None:
    import cyberwave.rest

    monkeypatch.setattr(
        cyberwave.rest,
        "EnvironmentWaypointPositionUpdateSchema",
        _FramedPositionUpdate,
        raising=False,
    )
    api = MagicMock()

    EnvironmentManager(api).update_waypoint(
        "env-1",
        "centroid",
        position={"x": 0.0, "y": 0.0, "z": 0.35},
        reference_frame={
            "type": "twin_link",
            "twin_uuid": "twin-1",
            "link": "camera_color_frame",
        },
        joint_state_source="simulation",
    )

    _, _, payload = getattr(api, WAYPOINT_METHOD).call_args.args
    assert payload["joint_state_source"] == "simulation"


def test_update_waypoint_refuses_an_unknown_joint_state_source() -> None:
    api = MagicMock()

    with pytest.raises(ValueError, match="joint_state_source"):
        EnvironmentManager(api).update_waypoint(
            "env-1",
            "centroid",
            position={"x": 1.0, "y": 2.0, "z": 3.0},
            joint_state_source="measured",
        )

    getattr(api, WAYPOINT_METHOD).assert_not_called()


def test_update_waypoint_refuses_an_unknown_reference_frame() -> None:
    api = MagicMock()

    with pytest.raises(ValueError, match="reference_frame"):
        EnvironmentManager(api).update_waypoint(
            "env-1",
            "centroid",
            position={"x": 1.0, "y": 2.0, "z": 3.0},
            reference_frame={"type": "world"},
        )

    getattr(api, WAYPOINT_METHOD).assert_not_called()


def test_update_waypoint_refuses_a_frame_its_rest_client_would_drop(
    monkeypatch,
) -> None:
    """An outdated generated model would silently store the pose unconverted."""
    import cyberwave.rest

    class _LegacyPositionUpdate(_FramedPositionUpdate):
        model_fields = {"position": None, "rotation": None}

    monkeypatch.setattr(
        cyberwave.rest,
        "EnvironmentWaypointPositionUpdateSchema",
        _LegacyPositionUpdate,
        raising=False,
    )
    api = MagicMock()

    with pytest.raises(Exception, match="predates waypoint reference frames"):
        EnvironmentManager(api).update_waypoint(
            "env-1",
            "centroid",
            position={"x": 1.0, "y": 2.0, "z": 3.0},
            reference_frame={"type": "environment"},
        )

    getattr(api, WAYPOINT_METHOD).assert_not_called()
