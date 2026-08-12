"""Deprecated alias for the structured-action catalog.

This package used to hold a second, parallel implementation of cloud model
inference (``MLModelsClient``) alongside :mod:`cyberwave.models.playground`.
Nothing instantiated it — ``cw.mlmodels`` resolves to
:class:`~cyberwave.models.manager.ModelManager` (see
:attr:`cyberwave.Cyberwave.mlmodels`), which runs through
:class:`~cyberwave.models.playground.PlaygroundHandle`. The duplicate shipped
its own copy of this catalog, so the two could silently drift.

Only the catalog re-exports remain, so the documented
``from cyberwave.mlmodels import STRUCTURED_ACTIONS`` keeps working. Prefer the
canonical top-level import::

    from cyberwave import STRUCTURED_ACTIONS

``MLModelsClient``, ``MLModelRunResult``, and ``MLModelSummary`` are gone — they
were only reachable through the unused client. To run a cloud model use
``cw.models.run(...)`` / ``cw.mlmodels.run(...)``, which posts to the
authenticated, credit-gated ``POST /api/v1/mlmodels/{uuid}/run``.
"""

from cyberwave.models.playground import (
    STRUCTURED_ACTIONS,
    StructuredAction,
    get_action,
    list_actions,
)

__all__ = [
    "STRUCTURED_ACTIONS",
    "StructuredAction",
    "get_action",
    "list_actions",
]
