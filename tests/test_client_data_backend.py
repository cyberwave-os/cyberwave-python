"""``Cyberwave.data`` must not invent a Zenoh endpoint (CYB-3201).

``ZenohBackend`` is peer mode by design: with no ``connect`` endpoint it takes
Zenoh's peer default and finds same-host peers by multicast scouting. The client
used to override that by injecting ``tcp/127.0.0.1:7447`` whenever ``ZENOH_CONNECT``
and ``ZENOH_LISTEN`` were both unset — which is every driver's default state. That
dialled the *worker* container (the only one given ``ZENOH_LISTEN=tcp/0.0.0.0:7447``,
for the CLI monitor), silently electing it hub of a star topology nobody designed,
with edge-core restarting it on every ``worker-files-changed``.

These tests pin the contract in both directions: unconfigured stays peer-to-peer,
and an explicit ``ZENOH_CONNECT`` is still honoured verbatim. Darwin is the one
exception — multicast scouting links no two sessions there, so the loopback
fallback survives for the dev loop and is pinned separately.

Mirrors ``cyberwave-sim/tests/test_zenoh_databus.py::test_open_databus_does_not_synthesize_a_router_endpoint``.
"""

from __future__ import annotations

from typing import Any

import pytest

from cyberwave.client import Cyberwave


@pytest.fixture(autouse=True)
def _clean_zenoh_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ZENOH_CONNECT",
        "ZENOH_LISTEN",
        "ZENOH_ROUTER_HOST",
        "ZENOH_ROUTER_PORT",
        "CYBERWAVE_DATA_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


class _FakeConfig:
    """Stands in for ``BackendConfig`` without reading the environment."""

    def __init__(self, connect: list[str] | None = None) -> None:
        self.backend = "zenoh"
        self.zenoh_connect: list[str] = list(connect or [])
        self.zenoh_listen: list[str] = []


def _resolve(
    monkeypatch: pytest.MonkeyPatch, cfg: _FakeConfig, *, system: str = "Linux"
) -> dict[str, Any]:
    """Run ``_get_data_backend`` against *cfg* and capture what it passed on.

    *system* pins the host platform so the edge-host contract is asserted the
    same way on a developer's Mac as in CI.
    """
    captured: dict[str, Any] = {}
    monkeypatch.setattr("cyberwave.client.platform.system", lambda: system)

    def _fake_get_backend(resolved: Any) -> Any:
        captured["connect"] = list(resolved.zenoh_connect)
        captured["listen"] = list(resolved.zenoh_listen)
        return object()

    import cyberwave.data.config as cw_config

    monkeypatch.setattr(cw_config, "BackendConfig", lambda: cfg)
    monkeypatch.setattr(cw_config, "get_backend", _fake_get_backend)

    # Bypass __init__: the data-backend seam needs no credentials or transport.
    client = Cyberwave.__new__(Cyberwave)
    client._data_backend = None
    client._get_data_backend()
    return captured


def test_unconfigured_stays_peer_to_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _resolve(monkeypatch, _FakeConfig())

    assert captured["connect"] == []
    assert captured["listen"] == []


def test_stale_router_hint_in_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZENOH_ROUTER_HOST", "10.0.0.9")
    monkeypatch.setenv("ZENOH_ROUTER_PORT", "9999")

    captured = _resolve(monkeypatch, _FakeConfig())

    assert captured["connect"] == []


def test_explicit_connect_endpoint_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented opt-out for hosts where multicast is unavailable."""
    captured = _resolve(monkeypatch, _FakeConfig(connect=["tcp/10.0.0.5:7447"]))

    assert captured["connect"] == ["tcp/10.0.0.5:7447"]


def test_peer_discovery_is_logged_as_such(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The old line advertised an "edge router" that was never deployed."""
    with caplog.at_level("INFO", logger="cyberwave.client"):
        _resolve(monkeypatch, _FakeConfig())

    assert "peer-to-peer discovery" in caplog.text
    assert "edge router" not in caplog.text


def test_macos_keeps_the_loopback_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multicast links nothing on Darwin, and edge-core publishes the worker's
    7447 so host-side sessions can attach. Without this the dev loop reaches
    nobody at all.
    """
    captured = _resolve(monkeypatch, _FakeConfig(), system="Darwin")

    assert captured["connect"] == ["tcp/127.0.0.1:7447"]


def test_macos_fallback_never_overrides_an_explicit_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _resolve(
        monkeypatch, _FakeConfig(connect=["tcp/10.0.0.5:7447"]), system="Darwin"
    )

    assert captured["connect"] == ["tcp/10.0.0.5:7447"]
