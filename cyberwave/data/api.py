"""Public ``cw.data`` API — typed publish / subscribe / latest.

:class:`DataBus` is the user-facing facade.  It accepts typed payloads
(numpy arrays, dicts, raw bytes), handles wire-format encoding/decoding
via :mod:`~.header`, and delegates transport to a
:class:`~.backend.DataBackend`.

Usage::

    cw = Cyberwave(api_key="...")
    cw.data.publish("frames", frame_array)
    depth = cw.data.latest("depth", max_age_ms=50)
    sub = cw.data.subscribe("joint_states", on_joints)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from .backend import DataBackend, Sample, Subscription
from .exceptions import DataBackendError
from .header import (
    CONTENT_TYPE_BYTES,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_NUMPY,
    HeaderMeta,
    HeaderTemplate,
    decode,
)
from .keys import build_key

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def _numpy() -> Any:
    """Lazy-import numpy so the module is usable without it for bytes/dict paths."""
    import numpy as np

    return np


def _decode_sample(header: HeaderMeta, payload: bytes) -> Any:
    """Decode *payload* based on *header.content_type*."""
    if header.content_type == CONTENT_TYPE_NUMPY:
        if header.shape is None or header.dtype is None:
            raise DataBackendError(
                "numpy/ndarray sample is missing shape or dtype in header"
            )
        np = _numpy()
        return np.frombuffer(payload, dtype=header.dtype).reshape(header.shape).copy()
    if header.content_type == CONTENT_TYPE_JSON:
        return json.loads(payload)
    return payload


class DataBus:
    """Public API for the data layer.  Accessed as ``cw.data``."""

    def __init__(
        self,
        backend: DataBackend,
        twin_uuid: str,
        *,
        sensor_name: str | None = None,
        key_prefix: str = "cw",
    ) -> None:
        self._backend = backend
        self._twin_uuid = twin_uuid
        self._sensor_name = sensor_name
        self._key_prefix = key_prefix
        self._templates: dict[str, HeaderTemplate] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> DataBackend:
        """The underlying transport backend (raw :class:`~.backend.DataBackend`)."""
        return self._backend

    @property
    def key_prefix(self) -> str:
        """Key prefix used when building Zenoh key expressions (default ``"cw"``)."""
        return self._key_prefix

    # ------------------------------------------------------------------
    # publish
    # ------------------------------------------------------------------

    def publish(
        self,
        channel: str,
        sample: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish a typed sample to *channel*.

        Accepted *sample* types (see :mod:`~.header` for the canonical
        ``CONTENT_TYPE_*`` constants):

        * ``numpy.ndarray`` → :data:`~.header.CONTENT_TYPE_NUMPY`,
          payload = ``array.tobytes()``
        * ``dict`` → :data:`~.header.CONTENT_TYPE_JSON`,
          payload = ``json.dumps(dict).encode()``
        * ``bytes`` → :data:`~.header.CONTENT_TYPE_BYTES`,
          payload as-is

        On the first call per channel a :class:`~.header.HeaderTemplate` is
        created and cached.  Subsequent calls reuse it — only ``ts`` and
        ``seq`` are updated per sample.  **Metadata is bound at template
        creation time**; passing a different *metadata* on later calls to
        the same channel is a no-op (a warning is logged).
        """
        key = build_key(
            self._twin_uuid,
            channel,
            self._sensor_name,
            prefix=self._key_prefix,
        )
        tmpl = self._get_or_create_template(channel, sample, metadata=metadata)
        payload_bytes = self._sample_to_bytes(sample)
        wire_bytes = tmpl.pack(payload_bytes)
        self._backend.publish(key, wire_bytes)

    # ------------------------------------------------------------------
    # subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        channel: str,
        callback: Callable[..., None],
        *,
        policy: str = "latest",
        raw: bool = False,
    ) -> Subscription:
        """Subscribe to *channel*.

        *callback* receives decoded data (numpy array, dict, or bytes)
        unless *raw* is ``True``, in which case it receives the raw
        :class:`~.backend.Sample`.
        """
        key = build_key(
            self._twin_uuid,
            channel,
            self._sensor_name,
            prefix=self._key_prefix,
        )

        if raw:
            return self._backend.subscribe(key, callback, policy=policy)

        def _on_sample(sample: Sample) -> None:
            try:
                header, payload = decode(sample.payload)
                decoded = _decode_sample(header, payload)
            except Exception:
                logger.warning(
                    "Failed to decode sample on channel '%s'",
                    channel,
                    exc_info=True,
                )
                return
            try:
                callback(decoded)
            except Exception:
                logger.warning(
                    "Subscriber callback error on channel '%s'",
                    channel,
                    exc_info=True,
                )

        return self._backend.subscribe(key, _on_sample, policy=policy)

    # ------------------------------------------------------------------
    # latest
    # ------------------------------------------------------------------

    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
        max_age_ms: float | None = None,
        raw: bool = False,
    ) -> Any:
        """Return the latest value on *channel*, decoded.

        If *max_age_ms* is set, returns ``None`` when the sample's
        acquisition timestamp (``header.ts``) is older than *max_age_ms*
        milliseconds — Phase 1 temporal synchronisation primitive.

        .. note::

            The staleness check compares the publisher's ``time.time()``
            against the local clock.  On a single machine this is exact;
            across machines the accuracy is bounded by NTP drift, so very
            small thresholds (< ~20 ms) may be unreliable.

        Raises:
            WireFormatError: If the sample's wire header is corrupt.
            ValueError: If *max_age_ms* is negative.
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError(
                f"max_age_ms must be >= 0, got {max_age_ms}"
            )

        key = build_key(
            self._twin_uuid,
            channel,
            self._sensor_name,
            prefix=self._key_prefix,
        )
        sample = self._backend.latest(key, timeout_s=timeout_s)
        if sample is None:
            return None

        if raw:
            return sample

        header, payload = decode(sample.payload)

        if max_age_ms is not None:
            age_s = time.time() - header.ts
            if age_s > max_age_ms / 1000.0:
                return None

        return _decode_sample(header, payload)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying backend."""
        self._backend.close()

    def __enter__(self) -> DataBus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_template(
        self,
        channel: str,
        sample: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> HeaderTemplate:
        with self._lock:
            existing = self._templates.get(channel)
            if existing is not None:
                # numpy shape/dtype change → invalidate so we re-create below.
                # Setting existing = None here means both guard checks fail and
                # execution falls through to the creation block.
                np = _numpy() if self._is_numpy(sample) else None
                if np is not None and isinstance(sample, np.ndarray) and (
                    sample.shape != existing.shape
                    or str(sample.dtype) != existing.dtype
                ):
                    existing = None

            if existing is not None:
                if metadata is not None:
                    logger.warning(
                        "metadata= ignored on channel '%s': template was "
                        "already created on first publish. Metadata is bound "
                        "at template creation time.",
                        channel,
                    )
                return existing

            np = _numpy() if self._is_numpy(sample) else None
            if np is not None and isinstance(sample, np.ndarray):
                tmpl = HeaderTemplate(
                    CONTENT_TYPE_NUMPY,
                    shape=sample.shape,
                    dtype=str(sample.dtype),
                    metadata=metadata,
                )
            elif isinstance(sample, dict):
                tmpl = HeaderTemplate(CONTENT_TYPE_JSON, metadata=metadata)
            elif isinstance(sample, bytes):
                tmpl = HeaderTemplate(CONTENT_TYPE_BYTES, metadata=metadata)
            else:
                raise TypeError(f"Unsupported sample type: {type(sample)}")

            self._templates[channel] = tmpl
            return tmpl

    @staticmethod
    def _is_numpy(sample: Any) -> bool:
        return type(sample).__module__ == "numpy" or type(sample).__name__ == "ndarray"

    @staticmethod
    def _sample_to_bytes(sample: Any) -> bytes:
        np = _numpy() if DataBus._is_numpy(sample) else None
        if np is not None and isinstance(sample, np.ndarray):
            return sample.tobytes()
        if isinstance(sample, dict):
            return json.dumps(sample, separators=(",", ":")).encode()
        if isinstance(sample, bytes):
            return sample
        raise TypeError(f"Unsupported sample type: {type(sample)}")
