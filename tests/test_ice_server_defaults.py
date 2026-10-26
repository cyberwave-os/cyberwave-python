"""The SDK's shared ICE defaults use TURN over TLS without optional imports."""

from __future__ import annotations

import pytest

from cyberwave.sensor.ice import DEFAULT_TURN_SERVERS


def test_base_video_ice_defaults_match_shared_default() -> None:
    pytest.importorskip("aiortc")
    from cyberwave.sensor.base_video import DEFAULT_TURN_SERVERS as video_defaults

    assert video_defaults is DEFAULT_TURN_SERVERS


def test_microphone_ice_defaults_match_shared_default() -> None:
    pytest.importorskip("aiortc")
    from cyberwave.sensor.microphone import _AUDIO_TURN_SERVERS

    assert _AUDIO_TURN_SERVERS is DEFAULT_TURN_SERVERS


def test_audio_microphone_ice_defaults_match_shared_default() -> None:
    pytest.importorskip("aiortc")
    from cyberwave.sensor.audio_microphone import _AUDIO_TURN_SERVERS

    assert _AUDIO_TURN_SERVERS is DEFAULT_TURN_SERVERS


def test_public_ice_defaults_do_not_require_media_extras() -> None:
    from cyberwave.sensor import DEFAULT_TURN_SERVERS as public_defaults

    assert public_defaults is DEFAULT_TURN_SERVERS


def _turn_entries(servers: list[dict]) -> list[str]:
    urls: list[str] = []
    for server in servers:
        raw = server.get("urls", [])
        for url in [raw] if isinstance(raw, str) else raw:
            if url.startswith(("turn:", "turns:")):
                urls.append(url)
    return urls


@pytest.mark.parametrize(
    "servers",
    [DEFAULT_TURN_SERVERS],
    ids=["shared"],
)
def test_first_turn_uri_is_tls(servers: list[dict]) -> None:
    """aiortc honours only the FIRST turn:/turns: URI and discards the rest.

    So the relay transport is decided by ordering, not by the list as a whole. If a
    plain `turn:` entry is ever placed ahead of the `turns:` one, restricted networks
    silently lose the TLS relay path.
    """
    turn_urls = _turn_entries(servers)
    assert turn_urls, "no turn:/turns: entry configured"
    assert turn_urls[0].startswith("turns:"), (
        f"first TURN URI is {turn_urls[0]!r}; aiortc uses only the first one, so a "
        "plain turn: entry here disables the TLS relay path"
    )


def test_aiortc_resolves_defaults_to_tls() -> None:
    """Lock in the actual resolution, not just the string ordering."""
    aiortc = pytest.importorskip("aiortc")
    from aiortc.rtcicetransport import connection_kwargs

    kwargs = connection_kwargs(
        [aiortc.RTCIceServer(**server) for server in DEFAULT_TURN_SERVERS]
    )
    assert kwargs["turn_ssl"] is True
    assert kwargs["turn_transport"] == "tcp"
    assert kwargs["turn_server"][1] == 443


# -- LAN-only ICE profile (CYBERWAVE_WEBRTC_ICE_MODE=lan) ---------------------------


def test_ice_mode_defaults_and_lan_aliases() -> None:
    from cyberwave.sensor.ice import ICE_MODE_DEFAULT, ICE_MODE_LAN, ice_mode

    assert ice_mode({}) == ICE_MODE_DEFAULT
    assert ice_mode({"CYBERWAVE_WEBRTC_ICE_MODE": ""}) == ICE_MODE_DEFAULT
    assert ice_mode({"CYBERWAVE_WEBRTC_ICE_MODE": "default"}) == ICE_MODE_DEFAULT
    for value in ("lan", "LAN", " lan ", "lan-only", "host"):
        assert ice_mode({"CYBERWAVE_WEBRTC_ICE_MODE": value}) == ICE_MODE_LAN


def test_unknown_ice_mode_keeps_the_relay(caplog: pytest.LogCaptureFixture) -> None:
    from cyberwave.sensor.ice import effective_ice_servers, ice_mode

    env = {"CYBERWAVE_WEBRTC_ICE_MODE": "offline-ish"}
    assert ice_mode(env) == "default"
    assert effective_ice_servers(None, environ=env) == DEFAULT_TURN_SERVERS
    assert "Unknown CYBERWAVE_WEBRTC_ICE_MODE" in caplog.text


def test_default_mode_uses_defaults_or_explicit_servers() -> None:
    from cyberwave.sensor.ice import effective_ice_servers

    assert effective_ice_servers(None, environ={}) == DEFAULT_TURN_SERVERS
    explicit = [{"urls": "turn:edge.local:3478"}]
    assert effective_ice_servers(explicit, environ={}) == explicit
    assert effective_ice_servers([], environ={}) == []


def test_lan_mode_drops_stun_and_turn_even_when_configured() -> None:
    from cyberwave.sensor.ice import effective_ice_servers

    env = {"CYBERWAVE_WEBRTC_ICE_MODE": "lan"}
    assert effective_ice_servers(None, environ=env) == []
    assert effective_ice_servers([{"urls": "turn:edge.local:3478"}], environ=env) == []


def test_lan_mode_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from cyberwave.sensor.ice import effective_ice_servers, is_lan_ice_mode

    monkeypatch.setenv("CYBERWAVE_WEBRTC_ICE_MODE", "lan")
    assert is_lan_ice_mode()
    assert effective_ice_servers(DEFAULT_TURN_SERVERS) == []
    monkeypatch.delenv("CYBERWAVE_WEBRTC_ICE_MODE")
    assert effective_ice_servers(None) == DEFAULT_TURN_SERVERS


def test_lan_mode_gathers_host_candidates_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty list must reach aiortc: ``None`` would make it add Google STUN."""
    aiortc = pytest.importorskip("aiortc")
    from aiortc.rtcicetransport import RTCIceGatherer, connection_kwargs

    from cyberwave.sensor.ice import effective_ice_servers

    monkeypatch.setenv("CYBERWAVE_WEBRTC_ICE_MODE", "lan")
    servers = [aiortc.RTCIceServer(**s) for s in effective_ice_servers(DEFAULT_TURN_SERVERS)]
    assert servers == []
    kwargs = connection_kwargs(servers)
    assert kwargs.get("stun_server") is None
    assert kwargs.get("turn_server") is None
    gatherer = RTCIceGatherer(iceServers=servers)
    assert gatherer.getDefaultIceServers()  # aiortc's fallback exists...
    assert gatherer._connection.stun_server is None  # ...but is not used
    assert gatherer._connection.turn_server is None


def test_video_consumer_has_no_ice_servers_in_lan_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("aiortc")
    from cyberwave.consumers.video import IncomingVideoStream

    consumer = IncomingVideoStream.__new__(IncomingVideoStream)
    consumer._turn_servers = None
    monkeypatch.setenv("CYBERWAVE_WEBRTC_ICE_MODE", "lan")
    assert consumer._ice_servers() == []
    monkeypatch.delenv("CYBERWAVE_WEBRTC_ICE_MODE")
    assert len(consumer._ice_servers()) == len(DEFAULT_TURN_SERVERS)
