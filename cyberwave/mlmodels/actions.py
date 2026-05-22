"""Slim local view of the backend structured-action catalog.

The authoritative catalog lives in
``cyberwave-backend/src/lib/structured_actions.py`` and is served over HTTP
at ``GET /api/v1/mlmodels/structured-actions``. The SDK used to ship a full
mirror including prompt templates and JSON schemas; that created a
maintenance burden and a drift risk for every prompt tweak on the backend.

What stays in the SDK is intentionally minimal: the set of action **ids**
plus a short label / description / output-format tuple for autocomplete and
docstring help. The backend is the source of truth for:

* the prompt wording (the ``/run`` endpoint rewrites the user prompt for
  structured tasks),
* per-model ``allowed_structured_tasks`` (surfaced on
  :class:`cyberwave.mlmodels.types.MLModelSummary`),
* the JSON schema of the result (the backend validates provider output and
  stamps ``output_format`` before returning).

Callers that want the live contract can use
:meth:`cyberwave.mlmodels.MLModelsClient.fetch_structured_actions_catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredAction:
    """Slim descriptor of a playground structured task.

    Only stable, public-facing fields are mirrored. Prompt templates and
    output schemas live on the backend and are fetched lazily via the live
    catalog endpoint when callers need them.
    """

    id: str
    label: str
    description: str
    output_format: str
    requires_image: bool = True


STRUCTURED_ACTIONS: tuple[StructuredAction, ...] = (
    StructuredAction(
        id="free",
        label="Free prompt",
        description="Pass the user prompt through unchanged.",
        output_format="text",
        requires_image=False,
    ),
    StructuredAction(
        id="caption",
        label="Caption / describe",
        description="Describe the input image in one concise sentence.",
        output_format="text",
    ),
    StructuredAction(
        id="detect_points",
        label="Detect points",
        description="Per-object 2D points ([y, x] normalized to 0-1000).",
        output_format="points",
    ),
    StructuredAction(
        id="detect_boxes",
        label="Detect bounding boxes",
        description="Per-object axis-aligned boxes ([ymin, xmin, ymax, xmax] normalized to 0-1000).",
        output_format="boxes",
    ),
    StructuredAction(
        id="segment",
        label="Segment",
        description="Per-object segmentation masks (box + base64 PNG cut-out).",
        output_format="masks",
    ),
    StructuredAction(
        id="point_trajectory",
        label="Predict image-plane trajectory",
        description="Ordered 2D waypoints matching the detect_points shape.",
        output_format="trajectory",
    ),
    StructuredAction(
        id="plan_steps",
        label="Plan steps",
        description="Decompose a goal into an ordered list of sub-tasks.",
        output_format="plan_steps",
        requires_image=False,
    ),
    StructuredAction(
        id="detect_objects_3d",
        label="Detect objects in 3D",
        description="Metric-3D detections (label + pose + size in the camera frame).",
        output_format="detections_3d",
    ),
    StructuredAction(
        id="predict_grasps",
        label="Predict grasps",
        description="Ranked parallel-jaw grasps (pose + width + score).",
        output_format="grasps",
    ),
    StructuredAction(
        id="detect_relations",
        label="Detect spatial relations (scene graph)",
        description="Scene graph of visible objects as subject / predicate / object triples.",
        output_format="relations",
    ),
)


_ACTIONS_BY_ID: dict[str, StructuredAction] = {a.id: a for a in STRUCTURED_ACTIONS}


def get_action(task_id: str) -> StructuredAction | None:
    """Return the :class:`StructuredAction` with ``id == task_id`` or None."""
    return _ACTIONS_BY_ID.get(task_id)


def list_actions() -> list[StructuredAction]:
    """Return all structured actions in declaration order."""
    return list(STRUCTURED_ACTIONS)


__all__ = [
    "STRUCTURED_ACTIONS",
    "StructuredAction",
    "get_action",
    "list_actions",
]
