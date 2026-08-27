"""PyTorch native inference backend.

Loads ``.pt`` / ``.pth`` files via ``torch.jit.load`` (TorchScript) or
``torch.load`` (pickled state-dicts / full modules).  ``predict()`` runs
a general-purpose forward pass and dispatches the output to a
``PredictionResult`` subtype:

* **1-D / 2-D tensor** → :class:`~cyberwave.models.types.ClassificationResult`
  (softmax is applied; top-K candidates are returned).
* **Anything else** → :class:`~cyberwave.models.types.CustomResult` carrying
  the raw NumPy array so the workflow can continue and process it.

This runtime is the catch-all for arbitrary user-supplied checkpoints — no
seeded catalog model declares ``edge_runtime="torch"`` — so it deliberately
infers as little as possible about output layout. In particular it does **not**
decode a 3-D output as detections: that shape is equally consistent with
segmentation logits, per-token embeddings and batched depth, and guessing
produces confidently-wrong boxes. Detector checkpoints belong in the
``onnxruntime`` or ``ultralytics`` runtimes, which know their output contract.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import (
    ClassificationCandidate,
    ClassificationResult,
    CustomResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class TorchRuntime(ModelRuntime):
    """Runtime backend for native PyTorch models."""

    name = "torch"

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
        """Run forward pass and return the best-matching ``PredictionResult``.

        ``classes`` and ``kwargs`` are accepted (and ignored) so that callers
        can pass detector-oriented context like ``twin_uuid`` without breaking
        the interface — this runtime never decodes detections, see the module
        docstring. ``confidence`` applies to the classification path only.

        Raises :class:`TypeError` with a clear message when ``model_handle`` is
        a plain state-dict (loaded via ``torch.load`` from a checkpoint file)
        rather than a callable TorchScript module — those files carry weights
        only and cannot be used for inference without the model class definition.
        Export the model with ``torch.jit.save(torch.jit.script(model), path)``
        to produce a self-contained ``.pt`` file.
        """
        import torch

        if not callable(model_handle):
            raise TypeError(
                f"TorchRuntime: model_handle is a {type(model_handle).__name__!r}, not a "
                "callable model.  The file was loaded as a plain state-dict (weights only) "
                "and cannot be used for inference without the model class definition.  "
                "Export the model as a TorchScript module instead:\n"
                "    torch.jit.save(torch.jit.script(model), 'model.pt')"
            )

        img = np.asarray(input_data)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3]  # drop alpha channel when present

        tensor = torch.from_numpy(
            np.ascontiguousarray(img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)  # HWC -> 1CHW

        device = _model_device(model_handle)
        tensor = tensor.to(device)

        with torch.no_grad():
            raw_output = model_handle(tensor)

        # Normalise compound outputs (tuple / list / dict) to the first tensor.
        primary = _primary_tensor(raw_output)

        if isinstance(primary, torch.Tensor):
            arr = primary.cpu().float().numpy()
        else:
            arr = np.asarray(primary, dtype=np.float32)

        # NOTE: a 3-D output is deliberately NOT decoded as YOLO detections.
        # This runtime is the catch-all for arbitrary user-supplied .pt/.pth
        # files (no seeded catalog model uses edge_runtime="torch"), so shape
        # alone is not evidence of layout: a segmentation logit map, per-token
        # embeddings and a batched depth map are all (B, C, N). Running YOLO box
        # decoding + NMS over one of those yields confidently-wrong boxes, which
        # is worse than no answer. Models that really are detectors should be
        # exported to ONNX or loaded via the ultralytics runtime, both of which
        # know their own output contract.

        # Classification-like 1-D / 2-D output: apply softmax and return top-K.
        if arr.ndim <= 2:
            flat = arr.flatten()
            if flat.size > 0:
                shifted = flat - flat.max()
                probs = np.exp(shifted) / np.exp(shifted).sum()
                top_indices = np.argsort(-probs)
                candidates = [
                    ClassificationCandidate(
                        label=str(int(idx)),
                        confidence=float(probs[idx]),
                        index=int(idx),
                    )
                    for idx in top_indices[:10]
                    if float(probs[idx]) >= confidence
                ]
                logger.debug(
                    "TorchRuntime: classification top-1 label=%s conf=%.3f",
                    candidates[0].label if candidates else "none",
                    candidates[0].confidence if candidates else 0.0,
                )
                return ClassificationResult(candidates, raw=raw_output)

        # Fallback: hand back the raw tensor so the workflow can continue and
        # a downstream node can apply the model's real output contract.
        logger.info(
            "TorchRuntime: output shape %s has no generic interpretation — "
            "returning CustomResult with the raw tensor. Export to ONNX or use "
            "the ultralytics runtime if this model needs decoded detections.",
            arr.shape,
        )
        return CustomResult(data=arr, label="torch", raw=raw_output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_device(model: Any) -> Any:
    """Return the device the model lives on, defaulting to CPU."""
    import torch

    try:
        params = list(model.parameters())
        if params:
            return params[0].device
    except Exception:
        pass
    return torch.device("cpu")


def _primary_tensor(output: Any) -> Any:
    """Extract the first meaningful tensor from a compound model output."""
    if isinstance(output, (list, tuple)):
        for item in output:
            if item is not None:
                return item
        return output
    if isinstance(output, dict):
        for v in output.values():
            if v is not None:
                return v
        return output
    return output
