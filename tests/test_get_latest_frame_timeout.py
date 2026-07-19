"""``get_latest_frame`` must forward a per-request HTTP timeout.

Regression guard for the ``policy_depth`` freeze: without a request timeout the
underlying urllib3 client blocks forever on a stalled socket, wedging the
controller's background camera poll thread. Both the ``TwinManager`` REST call
and the deprecated ``Twin.get_latest_frame`` wrapper must pass ``_request_timeout``
straight through so callers can bound the fetch.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.resources import TwinManager
from cyberwave.twin import Twin


def _make_manager() -> tuple[TwinManager, MagicMock]:
    mgr = TwinManager.__new__(TwinManager)  # bypass __init__ (no live client)
    api = MagicMock()
    api.api_client.param_serialize.return_value = ("GET", "url", {}, None, [])
    resp = MagicMock()
    resp.data = b"png-bytes"
    api.api_client.call_api.return_value = resp
    mgr.api = api  # type: ignore[attr-defined]
    return mgr, api


def test_manager_forwards_request_timeout_to_call_api() -> None:
    mgr, api = _make_manager()
    result = mgr.get_latest_frame(
        "twin-uuid", frame_bucket="policy_depth", _request_timeout=2.0
    )
    assert result == b"png-bytes"
    _, kwargs = api.api_client.call_api.call_args
    assert kwargs.get("_request_timeout") == 2.0


def test_manager_defaults_request_timeout_to_none() -> None:
    mgr, api = _make_manager()
    mgr.get_latest_frame("twin-uuid")
    _, kwargs = api.api_client.call_api.call_args
    assert kwargs.get("_request_timeout") is None


def test_twin_get_latest_frame_forwards_timeout() -> None:
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"x"
    client = SimpleNamespace(
        twins=twins_manager, config=SimpleNamespace(source_type="sim")
    )
    twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="T"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        twin.get_latest_frame(source_type="sim", _request_timeout=2.0)
    kwargs = twins_manager.get_latest_frame.call_args.kwargs
    assert kwargs.get("_request_timeout") == 2.0


def test_twin_get_latest_frame_omits_timeout_when_unset() -> None:
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"x"
    client = SimpleNamespace(
        twins=twins_manager, config=SimpleNamespace(source_type="sim")
    )
    twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="T"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        twin.get_latest_frame(source_type="sim")
    kwargs = twins_manager.get_latest_frame.call_args.kwargs
    assert "_request_timeout" not in kwargs
