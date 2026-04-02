"""Abstract base class for ML runtime backends.

Each supported inference engine (Ultralytics, ONNX Runtime, OpenCV, …)
implements this interface so that ``ModelManager`` and ``LoadedModel``
can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cyberwave.models.types import PredictionResult


class ModelRuntime(ABC):
    """Abstract interface for an ML runtime backend."""

    name: str

    @abstractmethod
    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load model weights from *model_path*.

        Returns a runtime-specific opaque model handle.
        """
        ...

    @abstractmethod
    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference and return a normalised ``PredictionResult``."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if the runtime's dependencies are importable."""
        ...
