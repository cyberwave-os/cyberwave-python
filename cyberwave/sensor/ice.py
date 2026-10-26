"""Dependency-free WebRTC ICE server defaults.

Keep transport configuration here so callers and tests can inspect it without
importing optional WebRTC dependencies such as :mod:`aiortc` and :mod:`av`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

#: Environment variable selecting the ICE profile for every SDK peer connection.
ICE_MODE_ENV = "CYBERWAVE_WEBRTC_ICE_MODE"
#: Default profile: STUN + TURN (``DEFAULT_TURN_SERVERS`` or what the caller passes).
ICE_MODE_DEFAULT = "default"
#: LAN-only profile: no STUN, no TURN; host candidates only.
ICE_MODE_LAN = "lan"
ICE_MODES = (ICE_MODE_DEFAULT, ICE_MODE_LAN)

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


# LAN-only ICE (on-prem / offline sites).
#
# CYBERWAVE_WEBRTC_ICE_MODE=lan makes every SDK peer connection gather host
# candidates only: no STUN server is asked for a server-reflexive address and
# no TURN relay is allocated, so nothing leaves the site and no internet
# reachability is required. Both peers (the edge and the viewer / media SFU)
# must then be on networks that can reach each other's host addresses directly.
#
# It overrides explicitly configured servers too (turn_servers=..., edge
# WEBRTC_TURN_URL): "lan" is a site policy, and a TURN entry that cannot be
# reached offline would only delay ICE until it times out. aiortc falls back to
# Google's public STUN server when iceServers is None, so LAN mode passes an
# explicit empty list.
_warned_lan_override = False


def ice_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """The configured ICE profile: ``"default"`` or ``"lan"``.

    Unknown values fall back to ``"default"`` with a warning, so a typo never
    silently cuts a deployment off from its relay.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(ICE_MODE_ENV) or "").strip().lower()
    if not raw:
        return ICE_MODE_DEFAULT
    if raw in {"lan", "local", "host", "lan-only", "lan_only"}:
        return ICE_MODE_LAN
    if raw != ICE_MODE_DEFAULT:
        logger.warning(
            "Unknown %s=%r; using the default ICE profile (STUN + TURN)",
            ICE_MODE_ENV,
            raw,
        )
    return ICE_MODE_DEFAULT


def effective_ice_servers(
    servers: Optional[list[dict[str, Any]]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """The ICE server list a peer connection should use.

    ``servers`` is what the caller configured (``None`` for the SDK defaults).
    In LAN mode the result is always ``[]`` (host candidates only); otherwise
    ``servers`` if given, else :data:`DEFAULT_TURN_SERVERS`.
    """
    global _warned_lan_override
    if ice_mode(environ) == ICE_MODE_LAN:
        if servers and not _warned_lan_override:
            _warned_lan_override = True
            logger.info(
                "%s=lan: ignoring %d configured STUN/TURN server(s); "
                "gathering host candidates only",
                ICE_MODE_ENV,
                len(servers),
            )
        return []
    return list(DEFAULT_TURN_SERVERS if servers is None else servers)


def is_lan_ice_mode(environ: Optional[Mapping[str, str]] = None) -> bool:
    return ice_mode(environ) == ICE_MODE_LAN
