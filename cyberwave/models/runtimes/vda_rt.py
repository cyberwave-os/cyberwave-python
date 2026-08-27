"""Video-Depth-Anything runtime.

Wraps the vendored ``video_depth_anything`` library at
``/opt/video-depth-anything``. VDA maintains a temporal frame window inside
the handle, so per-worker state is carried on the returned :class:`_VdaHandle`.
Two checkpoint families are supported: relative (default) and metric — the
metric flag is surfaced on :class:`DepthResult` so downstream consumers like
``object_pose`` can gate on it.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime, resolve_torch_device
from cyberwave.models.types import DepthResult, PredictionResult

logger = logging.getLogger(__name__)


DEFAULT_REPOSITORY_PATH = "/opt/video-depth-anything"
DEFAULT_CHECKPOINT_DIR = "/app/checkpoints"
DEFAULT_INPUT_SIZE = 518
DINOV2_PATCH_SIZE = 14

_ENCODER_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

_RELATIVE_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true",
}

_METRIC_CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth?download=true",
    "vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth?download=true",
}


@dataclass
class _VdaHandle:
    """Opaque model handle returned by :meth:`VideoDepthAnythingRuntime.load`."""

    model: Any
    torch: Any
    device: str
    encoder: str
    metric: bool
    input_size: int
    fp32: bool
    lock: threading.Lock = field(default_factory=threading.Lock)


class VideoDepthAnythingRuntime(ModelRuntime):
    """Monocular-depth runtime backed by Video-Depth-Anything."""

    name = "video_depth_anything"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        encoder: str = "vits",
        metric: bool = False,
        input_size: int = DEFAULT_INPUT_SIZE,
        fp32: bool = False,
        repository_path: str = DEFAULT_REPOSITORY_PATH,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        auto_download: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Load a Video-Depth-Anything checkpoint.

        When ``model_path`` is empty, the path is resolved from
        ``checkpoint_dir`` + ``metric`` + ``encoder``. Missing files are
        pulled from Hugging Face when ``auto_download`` is True.
        """
        del kwargs

        if encoder not in _ENCODER_CONFIGS:
            raise ValueError(
                f"Unsupported VDA encoder '{encoder}'. "
                f"Use one of {sorted(_ENCODER_CONFIGS)}."
            )

        if repository_path and repository_path not in sys.path:
            sys.path.insert(0, repository_path)

        try:
            import torch  # noqa: F401
            from video_depth_anything.video_depth_stream import (
                VideoDepthAnything as VideoDepthAnythingModel,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Video-Depth-Anything. Ensure the vendored "
                f"library at '{repository_path}' is present and torch is installed."
            ) from exc

        import torch as _torch

        resolved_path = _resolve_checkpoint_path(
            model_path=model_path,
            checkpoint_dir=checkpoint_dir,
            encoder=encoder,
            metric=metric,
        )
        if not os.path.exists(resolved_path):
            if not auto_download:
                raise FileNotFoundError(
                    f"VDA checkpoint not found at '{resolved_path}' and "
                    "auto_download is disabled."
                )
            _download_checkpoint(resolved_path, encoder=encoder, metric=metric)

        effective_device = _resolve_device(device, _torch)
        effective_input_size = _resolve_input_size(input_size)

        logger.info(
            "Loading Video-Depth-Anything (encoder=%s, metric=%s, device=%s, "
            "input_size=%d, checkpoint=%s)",
            encoder,
            metric,
            effective_device,
            effective_input_size,
            resolved_path,
        )

        model = VideoDepthAnythingModel(**_ENCODER_CONFIGS[encoder])
        model.load_state_dict(
            _torch.load(resolved_path, map_location="cpu"), strict=True
        )
        model = model.to(effective_device).eval()

        return _VdaHandle(
            model=model,
            torch=_torch,
            device=effective_device,
            encoder=encoder,
            metric=metric,
            input_size=effective_input_size,
            fp32=bool(fp32),
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
        """Run per-frame VDA inference. ``confidence``/``classes`` are ignored."""
        del confidence, classes, kwargs

        handle: _VdaHandle = model_handle
        frame_bgr = np.asarray(input_data)
        if frame_bgr.ndim != 3 or frame_bgr.shape[-1] not in (3, 4):
            raise ValueError(
                f"VDA predict expects HWC BGR image, got shape {frame_bgr.shape}"
            )

        frame_h, frame_w = frame_bgr.shape[:2]

        # BGR -> RGB (VDA vendored library is trained on RGB inputs).
        frame_rgb = frame_bgr[:, :, :3][:, :, ::-1]
        with handle.lock:
            # VDA is a stateful video model: it initialises its temporal cache
            # (frame_height, frame_width, transform, frame_cache_list, …) on
            # the first call and hard-asserts the size stays constant.  If the
            # camera resolution changes — or the warmup used a synthetic frame
            # with a different resolution — we must reset the temporal state so
            # the next call is treated as a fresh "first frame".
            model = handle.model
            stored_h = getattr(model, "frame_height", None)
            if stored_h is not None and (
                stored_h != frame_h or getattr(model, "frame_width", None) != frame_w
            ):
                logger.debug(
                    "VDA frame size changed (%dx%d → %dx%d); resetting temporal state",
                    getattr(model, "frame_width", "?"),
                    stored_h,
                    frame_w,
                    frame_h,
                )
                model.transform = None
                model.frame_id_list = []
                model.frame_cache_list = []
                model.id = -1

            depth = model.infer_video_depth_one(
                frame_rgb,
                input_size=handle.input_size,
                device=handle.device,
                fp32=handle.fp32,
            )

        depth = np.asarray(depth, dtype=np.float32)
        h, w = frame_h, frame_w

        return DepthResult(
            depth_map=depth,
            metric=bool(handle.metric),
            h=h,
            w=w,
        )


def _resolve_checkpoint_path(
    *,
    model_path: str,
    checkpoint_dir: str,
    encoder: str,
    metric: bool,
) -> str:
    if model_path:
        return model_path
    prefix = "metric_video_depth_anything" if metric else "video_depth_anything"
    return os.path.join(checkpoint_dir, f"{prefix}_{encoder}.pth")


def _resolve_device(device: str | None, torch_mod: Any) -> str:
    return resolve_torch_device(device, torch_mod, runtime_label="VDA")


def _resolve_input_size(input_size: int) -> int:
    """Validate the DINOv2 inference resolution.

    VDA shares the DA2 backbone, so the same patch-grid constraint and the same
    quadratic cost curve apply — see the twin in ``depth_anything_v2_rt``.
    """
    size = int(input_size)
    if size <= 0 or size % DINOV2_PATCH_SIZE != 0:
        raise ValueError(
            f"VDA input_size must be a positive multiple of {DINOV2_PATCH_SIZE} "
            f"(the DINOv2 patch size), got {input_size!r}. "
            f"Common values: 518 (default), 392, 322, 266."
        )
    return size


def _download_checkpoint(target_path: str, *, encoder: str, metric: bool) -> None:
    urls = _METRIC_CHECKPOINT_URLS if metric else _RELATIVE_CHECKPOINT_URLS
    url = urls.get(encoder)
    if not url:
        raise ValueError(
            f"No checkpoint URL for VDA encoder '{encoder}' (metric={metric})."
        )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info("Downloading VDA checkpoint from %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:
        with open(target_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    logger.info("Downloaded VDA checkpoint to %s", target_path)
