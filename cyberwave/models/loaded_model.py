"""User-facing wrapper around a loaded ML model.

Returned by ``ModelManager.load()`` / ``ModelManager.load_from_file()``.
Provides a stable ``.predict()`` API regardless of the underlying
runtime backend.
"""

from __future__ import annotations

from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


class LoadedModel:
    """A loaded ML model ready for inference."""

    def __init__(
        self,
        *,
        name: str,
        runtime: ModelRuntime,
        model_handle: Any,
        device: str = "cpu",
        model_path: str = "",
    ) -> None:
        self._name = name
        self._runtime = runtime
        self._model_handle = model_handle
        self._device = device
        self._model_path = model_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> str:
        return self._runtime.name

    @property
    def device(self) -> str:
        return self._device

    def predict(
        self,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference on *input_data*."""
        return self._runtime.predict(
            self._model_handle,
            input_data,
            confidence=confidence,
            classes=classes,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"LoadedModel(name={self._name!r}, "
            f"runtime={self.runtime!r}, "
            f"device={self._device!r})"
        )
