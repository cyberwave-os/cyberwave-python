"""Depth Anything V2 inference runtime.

Reconstructs the ``DepthAnythingV2`` model class from the vendored
``video_depth_anything`` library at ``/opt/video-depth-anything``, which
ships the same ``DINOv2`` backbone and ``DPTHead`` components used by DA2.
No separate ``depth_anything_v2`` package is required on the device.

Supported checkpoints
---------------------
* ``depth_anything_v2_vits.pth``
* ``depth_anything_v2_vitb.pth``
* ``depth_anything_v2_vitl.pth``

All three are plain PyTorch state-dicts; the encoder variant (``vits`` /
``vitb`` / ``vitl``) is auto-detected from the filename when not passed
explicitly via ``encoder=``.

Output
------
Returns a :class:`~cyberwave.models.types.DepthResult` with
``metric=False`` (relative disparity).  Downstream consumers that need
metric depth should use the Video-Depth-Anything metric checkpoints instead.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime, resolve_torch_device
from cyberwave.models.types import DepthResult, PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_REPOSITORY_PATH = "/opt/video-depth-anything"
DEFAULT_INPUT_SIZE = 518
DINOV2_PATCH_SIZE = 14

_ENCODER_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_INTERMEDIATE_LAYERS: dict[str, list[int]] = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
}


@dataclass
class _Dav2Handle:
    model: Any
    torch: Any
    device: str
    encoder: str
    input_size: int


class DepthAnythingV2Runtime(ModelRuntime):
    """Runtime backend for Depth Anything V2 (image monocular depth)."""

    name = "depth_anything_v2"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return os.path.isdir(DEFAULT_REPOSITORY_PATH)

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        encoder: str | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        repository_path: str = DEFAULT_REPOSITORY_PATH,
        **kwargs: Any,
    ) -> Any:
        del kwargs

        if repository_path and repository_path not in sys.path:
            sys.path.insert(0, repository_path)

        try:
            import torch
            from video_depth_anything.dpt import DPTHead
            from video_depth_anything.video_depth import DINOv2
        except Exception as exc:
            raise RuntimeError(
                "Failed to import video_depth_anything components. "
                f"Ensure the vendored library at '{repository_path}' is present."
            ) from exc

        import torch as _torch
        import torch.nn as nn
        import torch.nn.functional as F

        resolved_encoder = encoder or _detect_encoder(model_path)
        if resolved_encoder not in _ENCODER_CONFIGS:
            raise ValueError(
                f"Unsupported DA2 encoder '{resolved_encoder}'. "
                f"Use one of {sorted(_ENCODER_CONFIGS)}."
            )

        cfg = _ENCODER_CONFIGS[resolved_encoder]

        class _DepthAnythingV2(nn.Module):
            def __init__(self, enc: str) -> None:
                super().__init__()
                self.intermediate_layer_idx = _INTERMEDIATE_LAYERS[enc]
                self.enc = enc
                self.pretrained = DINOv2(model_name=enc)
                self.depth_head = DPTHead(
                    self.pretrained.embed_dim,
                    cfg["features"],
                    out_channels=cfg["out_channels"],
                )

            def forward(self, x: Any) -> Any:
                patch_h = x.shape[-2] // 14
                patch_w = x.shape[-1] // 14
                features = self.pretrained.get_intermediate_layers(
                    x, self.intermediate_layer_idx, return_class_token=True
                )
                depth = self.depth_head(features, patch_h, patch_w)
                depth = F.interpolate(
                    depth,
                    size=(x.shape[-2], x.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )
                return F.relu(depth).squeeze(1)

        effective_device = _resolve_device(device, _torch)
        effective_input_size = _resolve_input_size(input_size)

        logger.info(
            "Loading Depth Anything V2 (encoder=%s, device=%s, input_size=%d, "
            "checkpoint=%s)",
            resolved_encoder,
            effective_device,
            effective_input_size,
            model_path,
        )

        model = _DepthAnythingV2(resolved_encoder)
        state_dict = _torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(effective_device).eval()

        return _Dav2Handle(
            model=model,
            torch=_torch,
            device=effective_device,
            encoder=resolved_encoder,
            input_size=effective_input_size,
        )

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run DA2 depth estimation. ``confidence``/``classes`` are ignored."""
        del confidence, classes, kwargs

        handle: _Dav2Handle = model_handle
        torch = handle.torch
        frame_bgr = np.asarray(input_data)
        if frame_bgr.ndim != 3 or frame_bgr.shape[-1] not in (3, 4):
            raise ValueError(
                f"DA2 predict expects HWC BGR image, got shape {frame_bgr.shape}"
            )

        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, :3][:, :, ::-1].copy()

        tensor = _preprocess(frame_rgb, handle.input_size, torch)
        tensor = tensor.to(handle.device)

        with torch.no_grad():
            depth = handle.model(tensor)

        depth_np = depth.squeeze().cpu().float().numpy()

        # Resize depth back to the original camera resolution so that depth
        # pixel (u, v) corresponds to camera pixel (u, v) — required for
        # correct 3-D point-cloud reconstruction downstream.
        if depth_np.shape != (frame_h, frame_w):
            try:
                import cv2

                depth_np = cv2.resize(
                    depth_np, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
                )
            except ImportError:
                from PIL import Image as _PIL_Image

                depth_np = np.asarray(
                    _PIL_Image.fromarray(depth_np).resize(
                        (frame_w, frame_h), _PIL_Image.BILINEAR
                    )
                )

        return DepthResult(
            depth_map=depth_np,
            metric=False,
            h=frame_h,
            w=frame_w,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_encoder(model_path: str) -> str:
    """Guess encoder variant from checkpoint filename."""
    lower = os.path.basename(model_path).lower()
    for enc in ("vitl", "vitb", "vits"):
        if enc in lower:
            return enc
    return "vits"


def _resolve_device(device: str | None, torch_mod: Any) -> str:
    return resolve_torch_device(device, torch_mod, runtime_label="DA2")


def _resolve_input_size(input_size: int) -> int:
    """Validate the DINOv2 inference resolution.

    Cost is roughly quadratic in this value (518 = 37x37 patches, 322 = 23x23),
    so it is the main speed dial. Must stay a multiple of the patch size:
    ``forward()`` derives its grid as ``x.shape[-2] // 14``, and a non-multiple
    fails in the DPT head mid-inference rather than here at load.
    """
    size = int(input_size)
    if size <= 0 or size % DINOV2_PATCH_SIZE != 0:
        raise ValueError(
            f"DA2 input_size must be a positive multiple of {DINOV2_PATCH_SIZE} "
            f"(the DINOv2 patch size), got {input_size!r}. "
            f"Common values: 518 (default), 392, 322, 266."
        )
    return size


def _preprocess(frame_rgb: Any, input_size: int, torch_mod: Any) -> Any:
    """Resize, normalise, and convert to NCHW tensor (standard ImageNet stats)."""
    try:
        import cv2

        resized = cv2.resize(frame_rgb, (input_size, input_size))
    except ImportError:
        from PIL import Image as _PIL_Image

        resized = np.asarray(
            _PIL_Image.fromarray(frame_rgb).resize((input_size, input_size))
        )

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    return torch_mod.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)
