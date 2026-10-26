"""``EnvironmentManager.resolve_waypoint_poses``: the poses a Move Twin node drives to.

Compiled edge workflows call it when a Move Twin node runs, so a waypoint moved
since compilation (e.g. by Move Waypoint earlier in the run) is driven to where
it is now.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cyberwave.rest
from cyberwave.resources import EnvironmentManager

POSES_METHOD = "src_app_api_environments_resolve_waypoint_poses"


class _PosesRequest:
    """A generated request model: records what the SDK builds it from."""

    @classmethod
    def from_dict(cls, data):
        return dict(data)


def test_resolve_waypoint_poses_sends_only_what_was_given(monkeypatch) -> None:
    monkeypatch.setattr(
        cyberwave.rest,
        "EnvironmentWaypointPosesRequestSchema",
        _PosesRequest,
        raising=False,
    )
    api = MagicMock()
    getattr(api, POSES_METHOD).return_value = [
        SimpleNamespace(
            to_dict=lambda: {"id": "dock-a", "position": {"x": 9.0, "y": 9.0, "z": 0.0}}
        ),
        {
            "id": "dock-b",
            "position": {"x": 7.0, "y": 8.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    ]

    poses = EnvironmentManager(api).resolve_waypoint_poses(
        "env-1", ["dock-a", "dock-b"], mode="root", twin_uuid="twin-1", root_link="base_link"
    )

    environment_id, payload = getattr(api, POSES_METHOD).call_args.args
    assert environment_id == "env-1"
    assert payload == {
        "waypoint_ids": ["dock-a", "dock-b"],
        "mode": "root",
        "twin_uuid": "twin-1",
        "root_link": "base_link",
    }
    # Plain dicts, and a rotation key even when the server left it out.
    assert poses[0] == {
        "id": "dock-a",
        "position": {"x": 9.0, "y": 9.0, "z": 0.0},
        "rotation": None,
    }
    assert poses[1]["rotation"] == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def test_resolve_waypoint_poses_refuses_an_unknown_mode() -> None:
    api = MagicMock()

    with pytest.raises(ValueError, match="mode"):
        EnvironmentManager(api).resolve_waypoint_poses("env-1", ["a"], mode="orbit")

    getattr(api, POSES_METHOD).assert_not_called()
