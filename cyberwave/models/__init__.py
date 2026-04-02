"""Model loading API, runtime backends, and prediction output types.

Quick start::

    from cyberwave.models import ModelManager

    mgr = ModelManager()
    model = mgr.load("yolov8n")
    result = model.predict(frame)
"""

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.manager import ModelManager
from cyberwave.models.runtimes import available_runtimes, register_runtime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult

__all__ = [
    "BoundingBox",
    "Detection",
    "LoadedModel",
    "ModelManager",
    "PredictionResult",
    "available_runtimes",
    "register_runtime",
]
