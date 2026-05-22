"""Cloud Playground client for the Cyberwave `/api/v1/mlmodels` API.

``cw.mlmodels`` is the companion to ``cw.models`` (edge runtime models):

* ``cw.models``    — load an edge model (ultralytics / ONNX / TFLite) and
  call ``.predict(frame)`` locally on a worker or laptop.
* ``cw.mlmodels``  — resolve a catalog entry by slug / UUID and call
  ``.run(...)`` against the backend Playground, the same endpoint the
  browser UI uses.

Example::

    import cyberwave as cw

    client = cw.Cyberwave(api_key="...")
    result = client.mlmodels.run(
        "acme/models/gemini-robotics-er",
        image="scene.jpg",
        prompt="cups",
        structured_task="detect_points",
    )
    result.save_annotated_image("scene.jpg", "scene.annotated.png")

See also:
    * :mod:`cyberwave.image` — base64 + annotation helpers reused by ``run``.
    * :mod:`cyberwave.mlmodels.actions` — local catalog of structured tasks.
"""

from cyberwave.mlmodels.actions import (
    STRUCTURED_ACTIONS,
    StructuredAction,
    get_action,
    list_actions,
)
from cyberwave.mlmodels.client import MLModelsClient
from cyberwave.mlmodels.types import MLModelRunResult, MLModelSummary

__all__ = [
    "MLModelRunResult",
    "MLModelSummary",
    "MLModelsClient",
    "STRUCTURED_ACTIONS",
    "StructuredAction",
    "get_action",
    "list_actions",
]
