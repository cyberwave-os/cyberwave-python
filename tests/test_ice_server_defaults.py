"""The SDK's ICE defaults are duplicated across three modules; keep them identical.

`microphone` and `audio_microphone` each carry their own copy of the TURN list to
avoid importing `base_video` (which pulls in aiortc/av) at import time. Comments
asked them to stay in sync; these tests enforce it.
"""

from __future__ import annotations

import pytest

from cyberwave.sensor.audio_microphone import _AUDIO_TURN_SERVERS as AUDIO_MIC_SERVERS
from cyberwave.sensor.base_video import DEFAULT_TURN_SERVERS
from cyberwave.sensor.microphone import _AUDIO_TURN_SERVERS as MIC_SERVERS


def test_microphone_ice_defaults_match_base_video() -> None:
    assert MIC_SERVERS == DEFAULT_TURN_SERVERS


def test_audio_microphone_ice_defaults_match_base_video() -> None:
    assert AUDIO_MIC_SERVERS == DEFAULT_TURN_SERVERS


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
    [DEFAULT_TURN_SERVERS, MIC_SERVERS, AUDIO_MIC_SERVERS],
    ids=["base_video", "microphone", "audio_microphone"],
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
