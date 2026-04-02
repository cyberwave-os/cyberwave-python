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
