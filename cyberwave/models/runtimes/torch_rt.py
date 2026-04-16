"""PyTorch native inference backend (stub).

Loads ``.pt`` / ``.pth`` files via ``torch.jit.load`` (TorchScript) or
``torch.load`` (pickled state-dicts require the model class to be
available).  The ``predict()`` method is not yet implemented.
"""

from __future__ import annotations

import logging
from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult

logger = logging.getLogger(__name__)


class TorchRuntime(ModelRuntime):
    """Runtime backend for native PyTorch models."""

    name = "torch"

    @property
    def supports_predict(self) -> bool:
        return False

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        map_location = device or "cpu"
        try:
            model = torch.jit.load(model_path, map_location=map_location)
        except Exception:
            logger.debug(
                "torch.jit.load failed, falling back to torch.load",
                exc_info=True,
            )
            model = torch.load(model_path, map_location=map_location, weights_only=True)

        if hasattr(model, "eval"):
            model.eval()
        return model

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        raise NotImplementedError(
            "PyTorch native predict() is not yet implemented. "
            "Use load() to inspect the model or contribute an implementation."
        )
