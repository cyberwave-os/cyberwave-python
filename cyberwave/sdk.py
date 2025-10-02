from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Awaitable, Callable, Tuple

from .async_http import AsyncHttpClient
from .assets_api import AssetsAPI
from .environments import EnvironmentsAPI
from .http import HttpClient
from .missions import MissionsAPI, Mission
from .projects import ProjectsAPI
from .runtime import CyberwaveTask, run as run_async
from .runs import RunsAPI
from .sensors import SensorsAPI
from .teleop import TeleopAPI
from .twins import TwinsAPI
from .events import EventsAPI

_BASE_TYPES: Tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    dict,
    list,
    tuple,
    set,
    frozenset,
    int,
    float,
    bool,
    type(None),
)


def _wrap_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return run_async(value)  # returns value or CyberwaveTask
    if isinstance(value, (CyberwaveTask, _SyncNamespace)):
        return value
    if isinstance(value, _BASE_TYPES):
        return value
    if _should_wrap_object(value):
        return _SyncNamespace(value)
    return value


def _wrap_callable(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _wrap_result(func(*args, **kwargs))

    return wrapper


def _wrap_attribute(attr: Any) -> Any:
    if inspect.iscoroutinefunction(attr) or inspect.ismethod(attr) or inspect.isfunction(attr):
        return _wrap_callable(attr)
    if inspect.isclass(attr) or inspect.ismodule(attr) or inspect.isbuiltin(attr):
        return attr
    if callable(attr):
        return _wrap_callable(attr)
    return _wrap_result(attr)


def _should_wrap_object(obj: Any) -> bool:
    if isinstance(obj, (CyberwaveTask, _SyncNamespace)):
        return False
    if isinstance(obj, _BASE_TYPES):
        return False
    if inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
        return False
    if inspect.isclass(obj) or inspect.ismodule(obj):
        return False
    # Inspect public attributes to see if any coroutine functions exist
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            member = getattr(obj, name)
        except AttributeError:
            continue
        if inspect.iscoroutinefunction(member):
            return True
    return False


class _SyncNamespace:
    """Wrap an async namespace with synchronous dispatch helpers."""

    def __init__(self, target: Any):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        return _wrap_attribute(attr)

    def __dir__(self) -> list[str]:  # pragma: no cover - passthrough helper
        return dir(self._target)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<CyberwaveSync {self._target!r}>"


class _CyberwaveAsync:
    """Internal async-first facade used by :class:`Cyberwave`."""

    def __init__(self, base_url: str, token: str):
        http = AsyncHttpClient(base_url, access_token_getter=lambda: token)
        self._http = http
        self._http_sync = HttpClient(http.base_url, token)

        self.missions = MissionsAPI(http)
        self.runs = RunsAPI(http)
        self.environments = EnvironmentsAPI(http)
        self.projects = ProjectsAPI(http)
        self.assets = AssetsAPI(http)
        self.twins = TwinsAPI(http, self._http_sync, assets_api=self.assets)
        self.teleop = TeleopAPI(http, self._http_sync)
        self.sensors = SensorsAPI(http)
        self.events = EventsAPI(http)


class Cyberwave:
    """High-level SDK facade with automatic sync/async bridging."""

    def __init__(self, base_url: str, token: str):
        self._async = _CyberwaveAsync(base_url, token)

        # Synchronous namespaces wrap the async APIs transparently
        self.missions = _SyncNamespace(self._async.missions)
        self.runs = _SyncNamespace(self._async.runs)
        self.environments = _SyncNamespace(self._async.environments)
        self.projects = _SyncNamespace(self._async.projects)
        self.assets = _SyncNamespace(self._async.assets)
        self.twins = _SyncNamespace(self._async.twins)
        self.teleop = _SyncNamespace(self._async.teleop)
        self.sensors = _SyncNamespace(self._async.sensors)
        self.events = _SyncNamespace(self._async.events)

        # Expose async facade for opt-in advanced usage
        self.async_client = self._async

    def run(self, awaitable: Awaitable[Any]) -> Any:
        """Run an awaitable in a synchronous context or get a CyberwaveTask."""
        return run_async(awaitable)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough helper
        return getattr(self._async, name)


__all__ = ["Cyberwave", "Mission", "CyberwaveTask"]
