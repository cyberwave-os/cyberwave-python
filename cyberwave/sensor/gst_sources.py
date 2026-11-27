"""GStreamer source fragments that mux an existing H.264 stream without decoding it.

Each fragment ends in H.264 elementary stream, ready for the tail that
:mod:`cyberwave.sensor.camera_gstreamer` appends (``h264parse`` and the video
sync probe). Nothing here decodes or re-encodes: RTP is depayloaded, MPEG-TS is
demuxed, and the camera's own H.264 is forwarded.

=========================  ==================================================
``rtsp://`` / ``rtsps://``  IP camera or NVR (``rtspsrc``, TCP interleaved)
``rtp://host:port``         RTP/H.264 pushed over UDP to this host (listen)
``udp://host:port``         MPEG-TS over UDP (listen; multicast addresses join)
``tcp://host:port``         MPEG-TS over TCP (connect to a TS server)
``appsrc``                  H.264 pushed in-process with ``GstH264Source.push()``
=========================  ==================================================

This module only builds strings; it imports neither GStreamer nor PyAV.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

#: Source value selecting push mode (:meth:`GstH264Source.push`).
PUSH_SOURCE = "appsrc"

#: The appsrc the push mode builds. Live and timestamped by the pusher; when the
#: pipeline stalls the oldest buffers are dropped rather than blocking the
#: caller's thread (e.g. a robot SDK's receive loop).
APPSRC_FRAGMENT = (
    "appsrc name=cw_appsrc is-live=true format=time do-timestamp=false block=false "
    "max-buffers=60 leaky-type=downstream "
    'caps="video/x-h264,stream-format=byte-stream"'
)

_RTP_H264_CAPS = "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000"

#: URL schemes :func:`h264_source_for_url` accepts.
SUPPORTED_URL_SCHEMES = ("rtsp", "rtsps", "rtp", "udp", "tcp")

# Up to the last ``@`` of the URL token: a raw password may itself contain ``@``
# or ``/``. A path containing ``@`` is over-redacted, which is the safe side.
_CREDENTIALS_RE = re.compile(r"(://)[^\s\"']*@")


def redact_credentials(text: str) -> str:
    """Mask ``user:password@`` in any URL inside *text* (for logs and errors)."""
    return _CREDENTIALS_RE.sub(r"\1***@", text)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _host_port(url: str, scheme: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.port is None:
        raise ValueError(f"{scheme}:// source needs a port, e.g. {scheme}://0.0.0.0:5600")
    return parts.hostname or "0.0.0.0", parts.port


def h264_source_for_url(url: str, *, latency_ms: int = 50) -> str:
    """GStreamer fragment that yields the H.264 carried by *url*, without decoding.

    Raises:
        ValueError: unsupported scheme, or a missing port where one is required.
    """
    url = url.strip()
    scheme = urlsplit(url).scheme.lower()
    if scheme in ("rtsp", "rtsps"):
        return f"rtspsrc location={_quote(url)} latency={latency_ms} protocols=tcp ! rtph264depay"
    if scheme == "rtp":
        host, port = _host_port(url, scheme)
        return (
            f"udpsrc address={host} port={port} caps={_quote(_RTP_H264_CAPS)} "
            f"! rtpjitterbuffer latency={latency_ms} ! rtph264depay"
        )
    if scheme == "udp":
        host, port = _host_port(url, scheme)
        return f"udpsrc address={host} port={port} ! tsdemux"
    if scheme == "tcp":
        host, port = _host_port(url, scheme)
        if host == "0.0.0.0":
            raise ValueError("tcp:// connects to a server; give its host, e.g. tcp://10.0.0.5:8000")
        return f"tcpclientsrc host={host} port={port} ! tsdemux"
    raise ValueError(
        f"Unsupported H.264 passthrough URL scheme {scheme!r}; "
        f"supported: {', '.join(SUPPORTED_URL_SCHEMES)}"
    )
