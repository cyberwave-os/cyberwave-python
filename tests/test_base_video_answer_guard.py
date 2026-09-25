"""Regression tests for producer-side WebRTC answer handling.

A duplicate or stale WebRTC answer on the shared, twin-scoped ``webrtc-answer``
topic used to drive ``pc.setRemoteDescription()`` a second time, after the peer
connection was already ``stable`` — which raises aiortc
``InvalidStateError: Cannot handle answer in signaling state "stable"`` and tears
down the live camera tile. These tests pin the two guards that prevent it:

1. ``_on_answer_message`` is idempotent per offer — only the first matching
   answer is captured; later ones are ignored until ``_reset_state``.
2. ``_wait_for_answer`` only applies the remote answer while the connection is
   ``have-local-offer``; a ``stable``/closed connection is left alone.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.sensor.base_video import BaseVideoStreamer  # noqa: E402


class _FakeMQTT:
    topic_prefix = ""
    client_id = "test-client"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))

    def subscribe(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakePeerConnection:
    def __init__(self, signaling_state: str) -> None:
        self.signalingState = signaling_state
        self.set_remote_calls: List[Any] = []

    async def setRemoteDescription(self, desc: Any) -> None:
        self.set_remote_calls.append(desc)


class _Streamer(BaseVideoStreamer):
    """Minimal concrete streamer; the track is never initialized in these tests."""

    def initialize_track(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _make_streamer() -> _Streamer:
    return _Streamer(
        client=_FakeMQTT(),
        twin_uuid="twin-1",
        camera_name="wrist",
        auto_reconnect=False,
        enable_health_check=False,
    )


def _answer_payload(sdp_tag: str) -> Dict[str, Any]:
    return {
        "type": "answer",
        "target": "edge",
        "sensor": "wrist",
        "sdp": f"v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n{sdp_tag}",
        # matching stream identity (defaults)
    }


def test_first_answer_captured() -> None:
    streamer = _make_streamer()
    streamer._reset_state()
    streamer._on_answer_message(_answer_payload("first"))
    assert streamer._answer_received is True
    assert streamer._answer_data is not None
    assert "first" in streamer._answer_data["sdp"]


def test_duplicate_answer_ignored() -> None:
    streamer = _make_streamer()
    streamer._reset_state()
    streamer._on_answer_message(_answer_payload("first"))
    # A second answer for the same offer must NOT overwrite the captured one.
    streamer._on_answer_message(_answer_payload("second"))
    assert "first" in streamer._answer_data["sdp"]
    assert "second" not in streamer._answer_data["sdp"]


def test_reset_state_allows_new_answer() -> None:
    streamer = _make_streamer()
    streamer._reset_state()
    streamer._on_answer_message(_answer_payload("first"))
    # A fresh offer (start/reconnect) resets the guard so the next answer applies.
    streamer._reset_state()
    assert streamer._answer_received is False
    streamer._on_answer_message(_answer_payload("second"))
    assert "second" in streamer._answer_data["sdp"]


def test_wait_for_answer_applies_when_have_local_offer() -> None:
    streamer = _make_streamer()
    streamer._reset_state()
    streamer.pc = _FakePeerConnection("have-local-offer")
    streamer._answer_received = True
    streamer._answer_data = json.dumps(_answer_payload("apply"))
    asyncio.run(streamer._wait_for_answer(timeout=1.0))
    assert len(streamer.pc.set_remote_calls) == 1


def test_wait_for_answer_skips_when_stable() -> None:
    # The core regression: a stable connection must not get a second
    # setRemoteDescription (which raises InvalidStateError in aiortc).
    streamer = _make_streamer()
    streamer._reset_state()
    streamer.pc = _FakePeerConnection("stable")
    streamer._answer_received = True
    streamer._answer_data = json.dumps(_answer_payload("stale"))
    asyncio.run(streamer._wait_for_answer(timeout=1.0))
    assert streamer.pc.set_remote_calls == []
