"""Live MQTT state views: auto-refreshing mapping + pose presenters.

Shared by joints/pose/imu/gps/power handles so every MQTT `.get()` returns a
view that refreshes in place and supports `on_update()` callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from .callback_hub import CallbackHub, StateSubscription

if TYPE_CHECKING:
    from ..data.state_representation import CartesianPose

__all__ = ["LiveMappingView", "PoseView"]


class LiveMappingView(dict):
    """Live ``dict`` view that refreshes in place on every hub notify.

    A real ``dict`` subclass, so ``json.dumps(view)``, ``view.copy()`` and
    ``isinstance(view, dict)`` all work. Call :meth:`stop` to freeze it.
    """

    def __init__(self, hub: CallbackHub, compute: Callable[[], dict]) -> None:
        super().__init__()
        self._hub = hub
        self._compute = compute
        self._subscriptions: List[StateSubscription] = []
        self._refresh()
        self._attach(hub.subscribe(self._refresh))

    def _refresh(self, _snapshot: Any = None) -> None:
        data = self._compute()
        dict.clear(self)
        dict.update(self, data)

    def _attach(self, sub: StateSubscription) -> None:
        self._subscriptions.append(sub)

    def on_update(self, callback: Callable[[dict], None]) -> StateSubscription:
        return self._hub.subscribe(callback)

    def stop(self) -> None:
        for sub in self._subscriptions:
            sub.cancel()
        self._subscriptions.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"


class PoseView:
    """Live proxy over the frozen ``CartesianPose``.

    Delegates the pose API to an internal current pose it swaps on each notify.
    Every accessor returns ``None`` while no pose has arrived.
    """

    def __init__(
        self, hub: CallbackHub, compute: Callable[[], "Optional[CartesianPose]"]
    ) -> None:
        self._hub = hub
        self._compute = compute
        self._subscriptions: List[StateSubscription] = []
        self._current: "Optional[CartesianPose]" = None
        self._refresh()
        self._attach(hub.subscribe(self._refresh))

    def _refresh(self, _snapshot: Any = None) -> None:
        self._current = self._compute()

    def _attach(self, sub: StateSubscription) -> None:
        self._subscriptions.append(sub)

    @property
    def pose(self) -> "Optional[CartesianPose]":
        return self._current

    @property
    def position(self) -> Any:
        return self._current.position if self._current is not None else None

    @property
    def orientation(self) -> Any:
        return self._current.orientation if self._current is not None else None

    def frame_id(self) -> Optional[str]:
        return self._current.frame_id() if self._current is not None else None

    def translation_dict(self) -> Optional[dict]:
        return self._current.translation_dict() if self._current is not None else None

    def orientation_dict(self) -> Optional[dict]:
        return self._current.orientation_dict() if self._current is not None else None

    def to_legacy_pose(self) -> Optional[dict]:
        return self._current.to_legacy_pose() if self._current is not None else None

    def on_update(
        self, callback: Callable[["Optional[CartesianPose]"], None]
    ) -> StateSubscription:
        return self._hub.subscribe(callback)

    def stop(self) -> None:
        for sub in self._subscriptions:
            sub.cancel()
        self._subscriptions.clear()

    def __repr__(self) -> str:
        return f"PoseView(pose={self._current!r})"
