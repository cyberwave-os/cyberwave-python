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

    @property
    def supports_predict(self) -> bool:
        """Whether ``predict()`` is implemented (not just a stub)."""
        return True


def resolve_torch_device(
    device: str | None,
    torch_mod: Any,
    *,
    runtime_label: str,
) -> str:
    """Resolve a caller-supplied device string to one ``torch`` accepts.

    ``"auto"`` (and ``None``) picks CUDA when available, else CPU. Anything else
    is validated by ``torch.device()`` and returned verbatim, so indexed and
    non-CUDA accelerators (``cuda:1``, ``mps``) work.

    Shared rather than per-runtime because ``ModelManager._detect_device()``
    returns ``"cuda:0"``, not ``"cuda"`` — a ``{"cpu", "cuda"}`` whitelist
    rejects every GPU host.
    """
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    try:
        torch_mod.device(requested)
    except Exception as exc:
        raise ValueError(
            f"Unsupported {runtime_label} device {device!r}. Use 'auto', 'cpu', "
            "'cuda', an indexed CUDA device such as 'cuda:0', or any other "
            "device string accepted by torch.device()."
        ) from exc
    return requested
