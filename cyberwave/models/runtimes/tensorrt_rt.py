"""TensorRT inference backend (stub).

Loads serialised ``.engine`` / ``.trt`` plans for NVIDIA GPU-optimised
inference.  The ``predict()`` method is not yet implemented — loading
and inspecting engines works today.
"""

from __future__ import annotations

import logging
from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult

logger = logging.getLogger(__name__)


class TensorRTRuntime(ModelRuntime):
    """Runtime backend for TensorRT serialised engines."""

    name = "tensorrt"

    @property
    def supports_predict(self) -> bool:
        return False

    def is_available(self) -> bool:
        try:
            import tensorrt  # noqa: F401

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
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(model_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialise TensorRT engine: {model_path}")
        logger.info("TensorRT engine loaded with %d bindings", engine.num_bindings)
        return engine

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
            "TensorRT predict() is not yet implemented. "
            "Use load() to inspect the engine or contribute an implementation."
        )
