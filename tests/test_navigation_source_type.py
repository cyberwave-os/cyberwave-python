"""Navigation source_type defaulting tests.

These verify that the NavigationHelper defaults its command ``source_type``
from ``client.config.source_type`` (which is set by ``cw.affect(...)``) when
the caller doesn't pin one explicitly. This keeps navigation consistent with
locomotion helpers and lets generated mission workers route commands to sim
vs live by calling ``client.affect(execution_target)`` once up front.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.motion import TwinNavigationHandle
from cyberwave.navigation import NavigationPlan


def _build_twin_nav(
    *, source_type: str | None
) -> tuple[TwinNavigationHandle, MagicMock]:
    api_client = MagicMock()
    api_client.param_serialize.return_value = ()
    response = MagicMock()
    response.data = b"{}"
    api_client.call_api.return_value = response
    client = SimpleNamespace(
        config=SimpleNamespace(source_type=source_type),
        api=SimpleNamespace(api_client=api_client),
    )
    twin = SimpleNamespace(uuid="twin-uuid", client=client)
    return TwinNavigationHandle(twin), api_client


@pytest.mark.parametrize(
    "method_kwargs",
    [
        {"method": "goto", "args": ([1.0, 2.0, 0.0],)},
        {
            "method": "follow_path",
            "args": ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],),
        },
        {"method": "stop", "args": ()},
        {"method": "pause", "args": ()},
        {"method": "resume", "args": ()},
    ],
)
def test_navigation_commands_default_source_type_from_client_config(
    method_kwargs: dict,
) -> None:
    nav, api_client = _build_twin_nav(source_type="sim")

    getattr(nav, method_kwargs["method"])(*method_kwargs["args"])

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["source_type"] == "sim"


def test_navigation_explicit_source_type_wins_over_client_config() -> None:
    nav, api_client = _build_twin_nav(source_type="sim")

    nav.goto([1.0, 2.0, 0.0], source_type="edge")

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["source_type"] == "edge"


def test_navigation_omits_source_type_when_config_has_none() -> None:
    nav, api_client = _build_twin_nav(source_type=None)

    nav.goto([1.0, 2.0, 0.0])

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert "source_type" not in body


def test_navigation_plan_waypoint_actions_are_sent_in_path_payload() -> None:
    nav, api_client = _build_twin_nav(source_type=None)
    plan = NavigationPlan(name="inspect")
    plan.waypoint(
        x=1.0,
        y=2.0,
        z=0.0,
        waypoint_id="dock-a",
        actions=[
            {
                "plugin": "capture_image",
                "params": {"workflow_execution_uuid": "exec-123"},
            }
        ],
    )

    nav.follow_path(plan)

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["waypoints"][0]["actions"] == [
        {
            "plugin": "capture_image",
            "params": {"workflow_execution_uuid": "exec-123"},
        }
    ]
