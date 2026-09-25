"""Dependency-free WebRTC ICE server defaults.

Keep transport configuration here so callers and tests can inspect it without
importing optional WebRTC dependencies such as :mod:`aiortc` and :mod:`av`.
"""

from __future__ import annotations

from typing import Any

# TURN relay defaults to TLS on 443.
#
# This is a CHOICE, not a preference list: aiortc honours only the FIRST
# turn:/turns: URI it encounters (aiortc.rtcicetransport.connection_kwargs) and
# ignores the rest, unlike a browser which gathers from every URL. Listing a
# turn:3478 fallback here would therefore be dead config, so there isn't one.
#
# 443/TLS is the default because it is the only port that reliably survives
# corporate and industrial firewalls, which routinely block 3478 and outbound UDP
# to high ports. The tradeoff is deliberate and applies to every deployment: the
# relay path is now TCP, so it carries head-of-line blocking and TCP retransmit
# semantics instead of the loss tolerance video codecs expect. Deployments with
# working UDP can pass turn_servers= explicitly, or set
# CYBERWAVE_WEBRTC_TURN_URL=turn:turn.cyberwave.com:3478 on edge nodes
# (see edge_driver_env.resolve_ice_servers).
#
# Note the different hostname. TLS terminates at a GCP SSL proxy load balancer on
# its own global IP (tls.turn.cyberwave.com), while turn.cyberwave.com keeps
# pointing at the coturn VM because STUN and the UDP relay range are UDP and that
# load balancer is TCP-only. See devops/terraform-turn-service/gcloud-turn.tf.
#
# STUN stays on 3478 against the VM: aiortc silently drops any "stuns:" URI
# (connection_kwargs only matches scheme == "stun"), so there is no TLS STUN
# option. On a network that blocks 3478 this yields no srflx candidate, which is
# fine - the relay candidate from turns:443 is what connects.
DEFAULT_TURN_SERVERS: list[dict[str, Any]] = [
    {"urls": ["stun:turn.cyberwave.com:3478"]},
    {
        "urls": "turns:tls.turn.cyberwave.com:443",
        "username": "cyberwave-user",
        "credential": "cyberwave-admin",
    },
]
