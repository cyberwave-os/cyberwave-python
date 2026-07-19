"""``get_latest_frame(return_headers=True)`` surfaces the render generation.

The sim stamps a per-render generation on every frame; the backend returns it as
the ``X-Frame-Generation`` header. The SDK exposes it via ``return_headers`` so a
polling consumer (the controller stale-frame guard) can tell a legitimately
static scene from a dead producer. The header is read case-insensitively across
the header shapes the generated HTTP client can expose; a missing/invalid value
yields ``generation=None`` and never raises.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.resources import TwinManager, _extract_frame_generation
from cyberwave.twin import Twin


class _RespGetheaders:
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        self.data = data
        self._headers = headers

    def read(self) -> None:
        return None

    def getheaders(self):
        return dict(self._headers)


class _RespGetheader:
    def __init__(self, data: bytes, value: str | None) -> None:
        self.data = data
        self._value = value

    def read(self) -> None:
        return None

    def getheader(self, name: str):
        return self._value if name.lower() == "x-frame-generation" else None


class _RespHeadersMapping:
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        self.data = data
        self.headers = headers

    def read(self) -> None:
        return None


def test_extract_generation_from_getheaders_case_insensitive() -> None:
    assert (
        _extract_frame_generation(_RespGetheaders(b"", {"X-Frame-Generation": "5"}))
        == 5
    )
    assert (
        _extract_frame_generation(_RespGetheaders(b"", {"x-frame-generation": "6"}))
        == 6
    )


def test_extract_generation_from_getheader_and_headers_mapping() -> None:
    assert _extract_frame_generation(_RespGetheader(b"", "9")) == 9
    assert (
        _extract_frame_generation(
            _RespHeadersMapping(b"", {"X-Frame-Generation": "11"})
        )
        == 11
    )


def test_extract_generation_absent_or_invalid_is_none() -> None:
    assert _extract_frame_generation(_RespGetheader(b"", None)) is None
    assert _extract_frame_generation(_RespHeadersMapping(b"", {})) is None
    assert (
        _extract_frame_generation(_RespGetheaders(b"", {"X-Frame-Generation": "abc"}))
        is None
    )


def _make_manager(resp) -> TwinManager:
    mgr = TwinManager.__new__(TwinManager)  # bypass __init__ (no live client)
    api = MagicMock()
    api.api_client.param_serialize.return_value = ("GET", "url", {}, None, [])
    api.api_client.call_api.return_value = resp
    mgr.api = api  # type: ignore[attr-defined]
    return mgr


def test_return_headers_true_yields_bytes_and_generation() -> None:
    resp = _RespGetheaders(b"png-bytes", {"X-Frame-Generation": "42"})
    mgr = _make_manager(resp)
    result = mgr.get_latest_frame(
        "twin-uuid", frame_bucket="policy_depth", return_headers=True
    )
    assert result == (b"png-bytes", {"generation": 42})


def test_return_headers_default_is_bare_bytes() -> None:
    resp = _RespGetheaders(b"png-bytes", {"X-Frame-Generation": "42"})
    mgr = _make_manager(resp)
    assert mgr.get_latest_frame("twin-uuid") == b"png-bytes"


def test_return_headers_true_generation_none_when_absent() -> None:
    resp = _RespHeadersMapping(b"jpeg-bytes", {})
    mgr = _make_manager(resp)
    assert mgr.get_latest_frame("twin-uuid", return_headers=True) == (
        b"jpeg-bytes",
        {"generation": None},
    )


def test_twin_wrapper_does_not_forward_return_headers() -> None:
    # The generation heartbeat is consumed via the NON-deprecated manager
    # (client.twins.get_latest_frame), not the deprecated Twin wrapper. The
    # wrapper is intentionally left un-extended: it never forwards return_headers
    # and returns bare bytes.
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = b"png"
    client = SimpleNamespace(
        twins=twins_manager, config=SimpleNamespace(source_type="sim")
    )
    twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="T"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = twin.get_latest_frame(source_type="sim")
    kwargs = twins_manager.get_latest_frame.call_args.kwargs
    assert "return_headers" not in kwargs
    assert result == b"png"
