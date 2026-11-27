"""Tolerant ICE consent freshness for the SDK's WebRTC peer connections.

aioice checks consent (RFC 7675) with one STUN binding request every 4-6 s,
sent with ``retransmissions=0``: one packet, about 0.5 s to answer. After
``CONSENT_FAILURES`` (6) misses in a row it closes ICE; DTLS then closes and
aiortc's ``RTCPeerConnection`` closes itself. The driver sees
``iceConnectionState`` and ``connectionState`` go straight to ``closed``, with
no ``failed`` first, and its reconnect monitor re-offers.

On an edge device that streams video, a single STUN request is easy to lose:
it can queue behind video on a busy uplink, wait on an event loop that is
encoding a frame, or drop on a lossy return path. Media keeps flowing all the
while, so these closes are spurious.

:func:`apply_ice_consent_patch` keeps the RFC 7675 check and makes it tolerant:

* each check sends up to :data:`CONSENT_RETRANSMISSIONS` retransmissions
  (STUN RTO 0.5 s, doubling: packets at 0, 0.5 and 1.5 s, timeout at 3.5 s,
  inside the 4-6 s check interval), and
* consent expires after :data:`CONSENT_FAILURES` failed checks in a row
  (about 60 s without any answer, instead of about 30 s).

The patch replaces ``aioice.Connection.query_consent`` with a copy of the
aioice 0.9-0.12 loop that differs only in those two values and in calling the
watches registered with :func:`watch_consent_expiry` when consent expires. It
also reports an ICE socket lost while connected, the other way aioice before
0.11 closes a working connection (see :func:`_wrap_socket_loss`).
It is idempotent, and it checks the installed aioice: on a version or shape it
was not written for it logs one WARNING and changes nothing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Consecutive failed consent checks before ICE is closed (aioice: 6).
CONSENT_FAILURES = 12
#: STUN retransmissions per consent check (aioice: 0).
CONSENT_RETRANSMISSIONS = 2
#: aioice ``major.minor`` releases whose ``query_consent`` this module copies
#: (the loop is the same from 0.9 to 0.12, apart from an assert in 0.12).
SUPPORTED_AIOICE_VERSIONS = ((0, 9), (0, 10), (0, 11), (0, 12))

#: Fragments the original ``query_consent`` must contain for the copy to match.
_SOURCE_FINGERPRINT = (
    "CONSENT_INTERVAL",
    "CONSENT_FAILURES",
    "_nominated",
    "build_request",
    "retransmissions=0",
    "TransactionError",
    "_query_consent_task",
)

_PATCH_MARKER = "_cyberwave_tolerant_consent"
_HANDLERS_ATTR = "_cyberwave_consent_handlers"

_state: dict[str, Any] = {"result": None}

ConsentExpiredHandler = Callable[[], None]


def _version_tuple(version: str) -> Optional[tuple[int, int]]:
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _unsupported(reason: str) -> bool:
    logger.warning(
        "ICE consent patch not applied (%s); WebRTC connections keep aioice's "
        "default consent check, which closes after a few lost STUN packets",
        reason,
    )
    _state["result"] = False
    return False


def apply_ice_consent_patch() -> bool:
    """Install the tolerant consent check once per process.

    Returns:
        ``True`` if the patch is in place, ``False`` if aioice is missing or
        does not match (one WARNING the first time, then silent).
    """
    if _state["result"] is not None:
        return bool(_state["result"])

    try:
        import aioice
        from aioice import ice, stun
    except ImportError:
        _state["result"] = False
        return False

    connection_cls = getattr(ice, "Connection", None)
    original = getattr(connection_cls, "query_consent", None)
    if getattr(original, _PATCH_MARKER, False):
        _state["result"] = True
        return True

    version = getattr(aioice, "__version__", "")
    if _version_tuple(version) not in SUPPORTED_AIOICE_VERSIONS:
        return _unsupported(f"aioice {version or 'unknown'} is not a tested version")
    if (
        connection_cls is None
        or original is None
        or not hasattr(connection_cls, "build_request")
        or not hasattr(ice, "CONSENT_INTERVAL")
        or not hasattr(ice, "CONSENT_FAILURES")
        or not hasattr(stun, "TransactionError")
    ):
        return _unsupported(f"aioice {version} internals changed")
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        source = None
    if source is not None and not all(part in source for part in _SOURCE_FINGERPRINT):
        return _unsupported(f"aioice {version} query_consent changed")

    async def query_consent(self: Any) -> None:
        """RFC 7675 consent check, tolerant of single lost STUN packets."""
        failures = 0
        while True:
            # randomize between 0.8 and 1.2 times CONSENT_INTERVAL
            await asyncio.sleep(ice.CONSENT_INTERVAL * (0.8 + 0.4 * random.random()))

            for pair in self._nominated.values():
                request = self.build_request(pair, nominate=False)
                try:
                    await pair.protocol.request(
                        request,
                        pair.remote_addr,
                        integrity_key=self.remote_password.encode("utf8"),
                        retransmissions=CONSENT_RETRANSMISSIONS,
                    )
                    failures = 0
                except stun.TransactionError:
                    failures += 1
                if failures >= ice.CONSENT_FAILURES:
                    _notify_consent_expired(self, failures)
                    self._query_consent_task = None
                    return await self.close()

    setattr(query_consent, _PATCH_MARKER, True)
    query_consent.__wrapped__ = original  # type: ignore[attr-defined]
    connection_cls.query_consent = query_consent
    ice.CONSENT_FAILURES = CONSENT_FAILURES
    _wrap_socket_loss(ice.StunProtocol)
    logger.debug(
        "ICE consent patch applied (aioice %s): %d checks, %d retransmissions each",
        version,
        CONSENT_FAILURES,
        CONSENT_RETRANSMISSIONS,
    )
    _state["result"] = True
    return True


def _wrap_socket_loss(protocol_cls: Any) -> None:
    """Report an ICE socket lost while the connection is up.

    aioice before 0.11 ends ``Connection.recv()`` when *any* of its sockets is
    lost, a TURN allocation socket included, even if media uses another pair;
    DTLS then closes and the peer connection closes itself, with the same
    ``closed``/``closed`` signature as a consent expiry. Only the report is
    added; aioice's handling is unchanged. A normal close clears the nominated
    pairs before closing the sockets, so it is not reported.
    """
    original = getattr(protocol_cls, "connection_lost", None)
    if original is None or getattr(original, _PATCH_MARKER, False):
        return

    def connection_lost(self: Any, exc: Optional[BaseException]) -> None:
        try:
            connection = getattr(self, "receiver", None)
            nominated = getattr(connection, "_nominated", None)
            if nominated:
                on_pair = any(pair.protocol is self for pair in nominated.values())
                _notify(connection, "socket_lost", self, on_pair, exc)
        except Exception:
            logger.debug("ICE socket-loss report failed", exc_info=True)
        return original(self, exc)

    setattr(connection_lost, _PATCH_MARKER, True)
    connection_lost.__wrapped__ = original  # type: ignore[attr-defined]
    protocol_cls.connection_lost = connection_lost


def _notify(connection: Any, event: str, *args: Any) -> None:
    watches = list(getattr(connection, _HANDLERS_ATTR, ()))
    if not watches and event == "consent_expired":
        logger.warning(
            "ICE consent expired after %d failed checks; closing the connection",
            args[0],
        )
    for watch in watches:
        try:
            getattr(watch, event)(*args)
        except Exception:
            logger.exception("ICE %s handler failed", event)


def _notify_consent_expired(connection: Any, failures: int) -> None:
    _notify(connection, "consent_expired", failures)


@dataclass
class ConsentWatch:
    """Records why ICE went away on one peer connection, for log messages.

    It sees two causes that both close the peer connection with no ``failed``
    state first: consent expiry, and the loss of an ICE socket while
    connected (which aioice before 0.11 treats as the end of the connection).
    """

    label: str
    #: Whether this connection is watched (patch in place, connection found);
    #: if not, a close says nothing about either cause.
    tracked: bool = False
    expired_at: Optional[float] = None
    socket_lost_at: Optional[float] = None
    socket_lost_detail: str = ""
    on_expired: Optional[ConsentExpiredHandler] = None

    def consent_expired(self, failures: int) -> None:
        self.expired_at = time.monotonic()
        logger.warning(
            "ICE consent expired for %s: no STUN answer for %d checks in a row "
            "(each with %d retransmissions); the peer connection will close",
            self.label,
            failures,
            CONSENT_RETRANSMISSIONS,
        )
        if self.on_expired is not None:
            self.on_expired()

    def socket_lost(
        self, protocol: Any, on_nominated_pair: bool, exc: Optional[BaseException]
    ) -> None:
        candidate = getattr(protocol, "local_candidate", None)
        kind = getattr(candidate, "type", None) or "unknown"
        self.socket_lost_at = time.monotonic()
        self.socket_lost_detail = (
            f"{kind} socket, {'the' if on_nominated_pair else 'not the'} media pair"
        )
        logger.warning(
            "ICE socket lost for %s while connected (%s): %s",
            self.label,
            self.socket_lost_detail,
            exc or "closed",
        )

    def describe_close(self, now: Optional[float] = None) -> str:
        """Say what preceded a close, for the reconnect monitor's warning."""
        if not self.tracked:
            return "ICE consent expiry not tracked"
        now = time.monotonic() if now is None else now
        causes = []
        if self.expired_at is not None:
            causes.append(f"ICE consent expired {now - self.expired_at:.1f}s earlier")
        if self.socket_lost_at is not None:
            causes.append(
                f"ICE {self.socket_lost_detail} lost {now - self.socket_lost_at:.1f}s earlier"
            )
        if not causes:
            return "no ICE consent expiry or socket loss recorded"
        return "after " + " and ".join(causes)


def _ice_connections(pc: Any) -> list[Any]:
    """The aioice connections behind ``pc``'s transceivers (aiortc 1.x)."""
    found: list[Any] = []
    try:
        transceivers = list(pc.getTransceivers())
    except Exception:
        return found
    for transceiver in transceivers:
        try:
            gatherer = transceiver.sender.transport.transport.iceGatherer
            connection = gatherer._connection
        except AttributeError:
            continue
        if connection is not None and all(connection is not c for c in found):
            found.append(connection)
    return found


def watch_consent_expiry(
    pc: Any,
    label: str,
    on_expired: Optional[ConsentExpiredHandler] = None,
) -> ConsentWatch:
    """Log a WARNING naming ``label`` when ``pc``'s ICE consent expires.

    Also warns when one of ``pc``'s ICE sockets is lost while connected. Call
    it after the tracks are added (aiortc creates the ICE transports in
    ``addTrack``). The returned :class:`ConsentWatch` keeps the times so a
    reconnect monitor can say whether a close followed either event.
    """
    watch = ConsentWatch(label=label, on_expired=on_expired)
    if not _state["result"]:
        return watch
    for connection in _ice_connections(pc):
        watches = getattr(connection, _HANDLERS_ATTR, None)
        if watches is None:
            watches = []
            try:
                setattr(connection, _HANDLERS_ATTR, watches)
            except AttributeError:
                continue
        watches.append(watch)
        watch.tracked = True
    return watch
