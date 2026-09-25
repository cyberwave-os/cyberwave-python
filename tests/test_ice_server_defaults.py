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
