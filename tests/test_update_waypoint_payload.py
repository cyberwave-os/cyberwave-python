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
