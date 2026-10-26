"""DataBackend abstract base class and core types.

This module defines the transport-agnostic contract that all data backends
(Zenoh, filesystem, etc.) must satisfy.  Payloads are raw ``bytes`` at this
layer — encoding / decoding lives in the wire-format and public-API layers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class Sample:
    """A single data sample flowing through the bus."""

    channel: str
    payload: bytes
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] | None = None


class Subscription:
    """Handle returned by :meth:`DataBackend.subscribe`.

    Call :meth:`close` to stop receiving samples.
    """

    def close(self) -> None:
        """Unsubscribe and release resources.  Safe to call multiple times."""


class DataBackend(ABC):
    """Transport-agnostic data bus contract.

    Concrete implementations (``ZenohBackend``, ``FilesystemBackend``, …) move
    raw bytes between publishers and subscribers.  Higher-level layers handle
    serialization and the public ``cw.data.*`` API.
    """

    @abstractmethod
    def publish(
        self,
        channel: str,
        payload: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish *payload* to *channel*."""

    @abstractmethod
    def subscribe(
        self,
        channel: str,
        callback: Callable[[Sample], None],
        *,
        policy: str = "latest",
    ) -> Subscription:
        """Subscribe to *channel*.

        *callback* is invoked on an internal thread for each incoming sample.

        Args:
            channel: Channel name (e.g. ``"frames/default"``).
            callback: Function called with each :class:`Sample`.
            policy: ``"latest"`` keeps only the most recent sample between
                dispatches; ``"fifo"`` delivers every sample in order.

        Returns:
            A :class:`Subscription` handle — call ``.close()`` to stop.
        """

    @abstractmethod
    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
    ) -> Sample | None:
        """Return the most recent sample on *channel*, or ``None``."""

    # Subscriber presence -----------------------------------------------------
    #
    # Optional capability.  Not abstract so existing third-party backends keep
    # working: the defaults say "this transport cannot tell", which callers
    # must treat as *unknown* (fail closed), never as "no subscribers".

    def has_subscribers(self, channel: str, *, local_only: bool = False) -> bool | None:
        """Whether at least one subscriber currently matches *channel*.

        Returns ``None`` when this backend cannot report subscriber presence
        (the default).  Queryables / ``latest()`` readers do **not** count as
        subscribers.

        Args:
            channel: Channel name, as passed to :meth:`publish`.
            local_only: Count only subscribers that live in this backend's own
                transport session (e.g. the same process).  See the concrete
                backend for the exact meaning.
        """
        return None

    def on_subscribers_changed(
        self,
        channel: str,
        callback: Callable[[bool], None],
        *,
        local_only: bool = False,
    ) -> Subscription:
        """Call ``callback(has_subscribers)`` whenever presence on *channel* flips.

        *callback* runs on an internal transport thread: keep it short and
        non-blocking (set a flag or event).  It is **not** called for the
        state at registration time; read :meth:`has_subscribers` after
        registering to get the initial state without a race.

        Raises:
            NotImplementedError: This backend cannot report subscriber presence.

        Returns:
            A :class:`Subscription` handle — call ``.close()`` to stop.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support subscriber-presence callbacks"
        )

    @abstractmethod
    def close(self) -> None:
        """Tear down transport resources."""

    # Helpers -----------------------------------------------------------------

    VALID_POLICIES: tuple[str, ...] = ("latest", "fifo")
    """Subscribe policies accepted by all backends."""

    @staticmethod
    def _validate_policy(policy: str) -> None:
        if policy not in DataBackend.VALID_POLICIES:
            raise ValueError(
                f"Invalid subscribe policy '{policy}'.  "
                f"Must be one of: {', '.join(repr(p) for p in DataBackend.VALID_POLICIES)}."
            )

    # Context-manager support -------------------------------------------------

    def __enter__(self) -> DataBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
