"""Incoming-media consumers (WebRTC video, future audio) for the Cyberwave SDK."""

from __future__ import annotations

from typing import Any

__all__ = ["IncomingVideoStream"]


def __getattr__(name: str) -> Any:
    # Deferring behind __getattr__ keeps this package import cheap; the aiortc
    # cost is only paid when ``IncomingVideoStream`` is actually accessed.
    if name == "IncomingVideoStream":
        from .video import IncomingVideoStream

        globals()[name] = IncomingVideoStream
        return IncomingVideoStream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
