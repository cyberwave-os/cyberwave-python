"""Model loading API, runtime backends, and prediction output types.

Quick start — unified local + cloud surface::

    import cyberwave as cw

    # Local (edge) model — resolves via the on-node weights cache:
    detector = cw.models.load("yolov8n")
    result = detector.predict(frame, confidence=0.5)

    # Cloud (Playground) model — same API, routes via the Playground API:
    segmenter = cw.models.load("acme/models/sam-3.1")
    result = segmenter.predict("scene.jpg", prompt="cup", structured_task="segment")

    # Cloud slug, but run it locally: download the checkpoint once and
    # keep reusing the cached copy. The second call below skips the
    # download entirely.
    local_sam = cw.models.load("acme/models/sam-3.1", download=True)
    local_sam.predict("scene.jpg")
    cw.models.load("acme/models/sam-3.1").predict("scene.jpg")  # cache hit

The module-level :func:`load` lazily constructs (and caches) a
:class:`~cyberwave.Cyberwave` client using ``CYBERWAVE_API_KEY`` /
``CYBERWAVE_BASE_URL``, matching the ergonomics of the edge-worker
runtime where ``cw`` is injected pre-configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cyberwave.models.cloud import CloudLoadedModel
from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.manager import ModelManager
from cyberwave.models.runtimes import available_runtimes, register_runtime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult

if TYPE_CHECKING:
    from cyberwave.client import Cyberwave


_default_client: "Cyberwave | None" = None


def _get_default_client() -> "Cyberwave":
    """Return a process-wide :class:`Cyberwave` client, constructing once.

    Used by the module-level :func:`load` so that the snippet
    ``import cyberwave as cw; cw.models.load(slug)`` works outside the
    edge-worker runtime (where ``cw`` is pre-injected). The client is
    cached so repeated ``cw.models.load(...)`` calls don't re-auth on
    every invocation.
    """
    global _default_client
    if _default_client is None:
        from cyberwave.client import Cyberwave

        _default_client = Cyberwave()
    return _default_client


def load(model_id: str, **kwargs: Any) -> LoadedModel | CloudLoadedModel:
    """Module-level convenience matching ``cw.models.load(...)`` on a client.

    Routes cloud slugs (``workspace/models/name`` / UUIDs) through a
    lazily-constructed default :class:`~cyberwave.Cyberwave` client and
    local catalog ids through the on-node weights cache — the same
    dispatch as :meth:`ModelManager.load`. Exists so the unified
    snippet in the Playground docs is copy-pasteable as written.
    """
    return _get_default_client().models.load(model_id, **kwargs)


__all__ = [
    "BoundingBox",
    "CloudLoadedModel",
    "Detection",
    "LoadedModel",
    "ModelManager",
    "PredictionResult",
    "available_runtimes",
    "load",
    "register_runtime",
]
