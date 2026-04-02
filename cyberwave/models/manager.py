"""Model manager — loading, caching, and runtime/device detection.

Exposed as ``cw.models`` on the ``Cyberwave`` client.  Searches the
local model cache (populated by Edge Core), detects the appropriate
runtime backend, and returns a ``LoadedModel`` with a stable
``.predict()`` API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.runtimes import get_runtime

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "/app/models"
FALLBACK_MODEL_DIR = os.path.expanduser("~/.cyberwave/models")


class ModelManager:
    """Manages model loading, caching, and runtime selection."""

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_device: str | None = None,
    ) -> None:
        dir_from_env = os.environ.get("CYBERWAVE_MODEL_DIR")
        if model_dir:
            self._model_dir = Path(model_dir)
        elif dir_from_env:
            self._model_dir = Path(dir_from_env)
        elif Path(DEFAULT_MODEL_DIR).is_dir():
            self._model_dir = Path(DEFAULT_MODEL_DIR)
        else:
            self._model_dir = Path(FALLBACK_MODEL_DIR)

        self._default_device = default_device or os.environ.get(
            "CYBERWAVE_MODEL_DEVICE"
        )
        self._loaded: dict[str, LoadedModel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model by catalog ID.

        Returns a cached instance on repeated calls with the same
        arguments.  The runtime is auto-detected from the model ID when
        not specified.
        """
        effective_device = device or self._default_device or self._detect_device()
        resolved_runtime = runtime or self._detect_runtime(model_id)

        cache_key = f"{model_id}:{resolved_runtime}:{effective_device}"
        if cache_key in self._loaded:
            return self._loaded[cache_key]

        rt = get_runtime(resolved_runtime)
        model_path = self._resolve_model_path(model_id, resolved_runtime)

        logger.info(
            "Loading model '%s' with runtime '%s' on device '%s'",
            model_id,
            resolved_runtime,
            effective_device,
        )
        handle = rt.load(str(model_path), device=effective_device, **kwargs)

        loaded = LoadedModel(
            name=model_id,
            runtime=rt,
            model_handle=handle,
            device=effective_device,
            model_path=str(model_path),
        )
        self._loaded[cache_key] = loaded
        return loaded

    def load_from_file(
        self,
        path: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model directly from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        resolved_runtime = runtime or self._detect_runtime_from_extension(p.suffix)
        rt = get_runtime(resolved_runtime)
        effective_device = device or self._default_device or self._detect_device()

        handle = rt.load(str(p), device=effective_device, **kwargs)
        return LoadedModel(
            name=p.stem,
            runtime=rt,
            model_handle=handle,
            device=effective_device,
            model_path=str(p),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_id: str, runtime: str) -> Path:
        """Resolve a catalog model ID to a local file path."""
        for ext in self._runtime_extensions(runtime):
            candidate = self._model_dir / f"{model_id}{ext}"
            if candidate.exists():
                return candidate

        model_subdir = self._model_dir / model_id
        if model_subdir.is_dir():
            for ext in self._runtime_extensions(runtime):
                for f in sorted(model_subdir.iterdir()):
                    if f.suffix == ext:
                        return f

        # Ultralytics convenience: pass model_id directly so the library
        # can auto-download from hub (development path only).
        if runtime == "ultralytics":
            return Path(model_id)

        raise FileNotFoundError(
            f"Model '{model_id}' not found in {self._model_dir}. "
            f"Ensure edge core has downloaded the model weights, "
            f"or use load_from_file()."
        )

    @staticmethod
    def _detect_runtime(model_id: str) -> str:
        """Heuristic: detect runtime from model ID."""
        lower = model_id.lower()
        if any(k in lower for k in ("yolo", "yolov5", "yolov8", "yolov11")):
            return "ultralytics"
        if any(k in lower for k in ("background-subtraction", "haar", "cascade")):
            return "opencv"
        raise ValueError(
            f"Cannot auto-detect runtime for model '{model_id}'. "
            f"Pass runtime= explicitly, e.g.: "
            f"cw.models.load('{model_id}', runtime='ultralytics')"
        )

    @staticmethod
    def _detect_runtime_from_extension(ext: str) -> str:
        """Map a file extension to a runtime name."""
        mapping: dict[str, str] = {
            ".pt": "ultralytics",
            ".onnx": "onnxruntime",
            ".tflite": "tflite",
            ".xml": "opencv",
            ".engine": "tensorrt",
        }
        return mapping.get(ext.lower(), "ultralytics")

    @staticmethod
    def _runtime_extensions(runtime: str) -> list[str]:
        mapping: dict[str, list[str]] = {
            "ultralytics": [".pt", ".onnx", ".engine"],
            "onnxruntime": [".onnx"],
            "opencv": [".xml", ".caffemodel"],
            "tflite": [".tflite"],
            "torch": [".pt", ".pth"],
            "tensorrt": [".engine", ".trt"],
        }
        return mapping.get(runtime, [".pt"])

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"
