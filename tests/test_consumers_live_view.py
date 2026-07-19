"""Shared consumer primitives: hub, mapping view, pose view."""

from __future__ import annotations

from cyberwave.consumers.callback_hub import CallbackHub, StateSubscription
from cyberwave.consumers.mqtt_live_view import LiveMappingView, PoseView


def test_hub_notify_invokes_subscribers_with_fresh_snapshot() -> None:
    hub = CallbackHub(label="test")
    seen: list[dict] = []
    sub = hub.subscribe(lambda snap: seen.append(snap))
    assert isinstance(sub, StateSubscription)
    hub.notify(lambda: {"a": 1})
    hub.notify(lambda: {"a": 2})
    assert seen == [{"a": 1}, {"a": 2}]


def test_hub_cancel_stops_delivery() -> None:
    hub = CallbackHub(label="test")
    seen: list[int] = []
    sub = hub.subscribe(lambda snap: seen.append(snap))
    hub.notify(lambda: 1)
    sub.cancel()
    hub.notify(lambda: 2)
    assert seen == [1]


def test_hub_swallows_callback_exceptions() -> None:
    hub = CallbackHub(label="test")
    ok: list[int] = []

    def boom(_snap: object) -> None:
        raise RuntimeError("boom")

    hub.subscribe(boom)
    hub.subscribe(lambda snap: ok.append(snap))
    hub.notify(lambda: 5)  # must not raise
    assert ok == [5]


def test_mapping_view_refreshes_in_place_on_notify() -> None:
    hub = CallbackHub(label="map")
    state = {"x": 1}
    view = LiveMappingView(hub, lambda: dict(state))
    assert view == {"x": 1}
    assert isinstance(view, dict)
    state["x"] = 2
    state["y"] = 9
    hub.notify(lambda: dict(state))
    assert view == {"x": 2, "y": 9}  # same object, refreshed in place


def test_mapping_view_stop_freezes_contents() -> None:
    hub = CallbackHub(label="map")
    state = {"x": 1}
    view = LiveMappingView(hub, lambda: dict(state))
    view.stop()
    state["x"] = 2
    hub.notify(lambda: dict(state))
    assert view == {"x": 1}  # frozen at last value


def test_mapping_view_on_update_delivers_snapshot() -> None:
    hub = CallbackHub(label="map")
    seen: list[dict] = []
    view = LiveMappingView(hub, lambda: {})
    view.on_update(lambda snap: seen.append(snap))
    hub.notify(lambda: {"k": 1})
    assert seen == [{"k": 1}]


def test_pose_view_is_none_before_data() -> None:
    hub = CallbackHub(label="pose")
    current: list[object] = [None]
    view = PoseView(hub, lambda: current[0])
    assert view.pose is None
    assert view.position is None
    assert view.orientation is None
    assert view.frame_id() is None
    assert view.translation_dict() is None
    assert view.orientation_dict() is None
    assert view.to_legacy_pose() is None


def test_pose_view_refreshes_on_notify() -> None:
    from cyberwave.data.state_representation import CartesianPose
    from cyberwave.data.state_representation.geometry.primitives import (
        Quaterniond,
        Vector3d,
    )
    from cyberwave.data.state_representation.space.spatial_state import SpatialState

    pose = CartesianPose(
        spatial_state=SpatialState(),
        position=Vector3d(1.0, 2.0, 3.0),
        orientation=Quaterniond(),
    )
    hub = CallbackHub(label="pose")
    current: list[object] = [None]
    view = PoseView(hub, lambda: current[0])
    current[0] = pose
    hub.notify(lambda: pose)
    assert view.pose is pose
    assert view.position.x == 1.0
    assert view.to_legacy_pose()["position"]["x"] == 1.0
