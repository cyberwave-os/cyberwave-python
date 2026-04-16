"""Runtime registry — maps runtime names to implementations.

Use ``register_runtime`` to add new backends and ``get_runtime`` to
obtain a ready-to-use instance.
"""

from __future__ import annotations

from cyberwave.models.runtimes.base import ModelRuntime

_RUNTIME_REGISTRY: dict[str, type[ModelRuntime]] = {}


def register_runtime(runtime_class: type[ModelRuntime]) -> None:
    """Register a ``ModelRuntime`` subclass by its ``name`` attribute."""
    _RUNTIME_REGISTRY[runtime_class.name] = runtime_class


def get_runtime(name: str) -> ModelRuntime:
    """Return an instance of the runtime registered under *name*.

    Raises ``ValueError`` if the name is unknown, or ``ImportError`` if
    the runtime's dependencies are not installed.
    """
    if name not in _RUNTIME_REGISTRY:
        available = ", ".join(sorted(_RUNTIME_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown model runtime '{name}'. Available: {available}. "
            f"Install the required package and ensure it is registered."
        )
    cls = _RUNTIME_REGISTRY[name]
    instance = cls()
    if not instance.is_available():
        raise ImportError(
            f"Runtime '{name}' is registered but its dependencies are not "
            f"installed. Install with: pip install cyberwave[ml]"
        )
    return instance


def available_runtimes() -> list[str]:
    """Return names of runtimes whose dependencies are currently importable."""
    return [name for name, cls in _RUNTIME_REGISTRY.items() if cls().is_available()]


# Auto-register built-in runtimes.  Each runtime module defers its heavy
# third-party import (onnxruntime, cv2, torch, tensorrt, tflite_runtime)
# to load()/predict() method bodies, so the module-level imports below
# always succeed regardless of which optional dependencies are installed.
# (numpy *is* imported eagerly in onnxruntime_rt and opencv_rt, but it is
# a core SDK dependency so that is fine.)

from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime  # noqa: E402

register_runtime(UltralyticsRuntime)

from cyberwave.models.runtimes.onnxruntime_rt import OnnxRuntime  # noqa: E402

register_runtime(OnnxRuntime)

from cyberwave.models.runtimes.opencv_rt import OpenCVRuntime  # noqa: E402

register_runtime(OpenCVRuntime)

from cyberwave.models.runtimes.tflite_rt import TFLiteRuntime  # noqa: E402

register_runtime(TFLiteRuntime)

from cyberwave.models.runtimes.tensorrt_rt import TensorRTRuntime  # noqa: E402

register_runtime(TensorRTRuntime)

from cyberwave.models.runtimes.torch_rt import TorchRuntime  # noqa: E402

register_runtime(TorchRuntime)
