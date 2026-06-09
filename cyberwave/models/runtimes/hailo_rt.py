"""Hailo inference backend.

Runs ``.hef`` binaries on a Hailo accelerator (Hailo-8, Hailo-8L,
Hailo-10H, …) through the ``hailo_platform`` Python bindings shipped
with HailoRT.

The HEF format is hardware-architecture-locked: a Hailo-8 binary
will not run on a Hailo-8L (and vice versa). This backend tries to
verify the HEF arch against the device arch at load time, but the
HailoRT 4.23.0 Python binding does not expose the HEF's compiled
arch via any of the historical attr names — so the pre-load guard
is best-effort and silently no-ops when neither side reports an
arch string. In that case we rely on
``vdevice.configure(hef, params)`` to raise HailoRT's native
incompatibility error when the user mismatches a Hailo-8L HEF
against a Hailo-8 device.

Three NMS output layouts are supported, all produced by HEFs
compiled with the Hailo Model Zoo's ``*_nms_postprocess`` op:

* **Flat NMS** — a single output tensor of shape
  ``[batch, max_dets, 6]`` with columns
  ``(x1, y1, x2, y2, score, class_id)`` in normalized ``[0, 1]``
  coordinates relative to the *model input* (letterboxed) image.

* **Batched per-class NMS** (Hailo Model Zoo YOLOv8 HEFs on
  HailoRT 4.23.0+) — a Python ``list`` of length ``batch_size``
  where each batch element is either an ndarray of shape
  ``(num_classes, max_per_class_K, 5)`` (fixed *K*, e.g.
  ``yolov8s.hef``) or a ``list[num_classes]`` of variable-length
  ``[K, 5]`` arrays (e.g. ``yolov8m.hef``). Columns are
  ``(x1, y1, x2, y2, score)``; class id is implicit in axis 0.

* **Per-class NMS** (older / non-batched variant) — a single
  output whose first axis indexes class IDs (length =
  ``num_classes``); each entry is a 2-D array ``[K, 5]`` with
  columns ``(x1, y1, x2, y2, score)``. The HailoRT binding sometimes
  returns this as a list and sometimes as a ragged object array.

Raw (non-NMS) feature-map outputs are now detected automatically at
load time: :func:`_infer_model_kind` returns ``"detection_raw"`` and
:func:`_postprocess_det_raw` decodes them on the CPU.  Both combined
``(H, W, 4+nc)`` / ``(H, W, 64+nc)`` tensors and separate box + class
tensor pairs ``(H, W, 4)`` + ``(H, W, nc)`` are supported, across one or
more scales.  HEFs in the Hailo Model Zoo *do* ship with NMS on-chip by
default, so this path is only needed for custom HEFs compiled without the
``yolov8_nms_postprocess`` op.

**Instance segmentation models** (``yolov8n/s/m_seg.hef``) are
handled by passing ``model_kind="instance_segmentation"`` to
:meth:`HailoRuntime.load`. The Hailo Model Zoo seg HEFs include the
``yolov8_bbox_decoding`` op (bounding-box decoding on-chip) but *not*
NMS, so the outputs are raw decoded feature maps. :func:`_postprocess_seg`
runs the full CPU-side pipeline: multi-scale head decode → greedy NMS
→ mask coefficient × prototype decode → per-detection
:class:`~cyberwave.models.types.Mask` objects stamped into original-image
space.

Two channel layouts are handled:

* **C = 4 + num_classes + num_masks** (116 for COCO): boxes already
  decoded to ``(x1, y1, x2, y2)`` in normalised ``[0, 1]`` coords by
  the on-chip ``yolov8_bbox_decoding`` postprocess op.
* **C = 4 * reg_max + num_classes + num_masks** (176): raw DFL
  distribution; :func:`_decode_one_seg_head` applies soft-argmax and
  converts to normalised coords.

**Embedding / encoder models** (e.g. CLIP image encoder)
are handled by passing ``model_kind="embedding"`` to
:meth:`HailoRuntime.load`. The :meth:`HailoRuntime.predict`
method then skips the NMS postprocess path and instead calls
:func:`_extract_embedding`, returning a
:class:`~cyberwave.models.types.PredictionResult` whose
``output`` is an :class:`~cyberwave.models.types.EmbeddingResult`.
Image preprocessing for embedding models uses :func:`_resize_for_embedding`
(plain stretch resize, no letterbox padding) rather than the
YOLO letterbox convention.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    EmbeddingResult,
    InstanceSegmentationResult,
    Mask,
    PredictionResult,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Letterbox parameters carried from preprocess to postprocess so the
# inverse transform can un-letterbox detection coordinates back to
# the original image's pixel space.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _Letterbox:
    """Forward letterbox params used to undo the transform on outputs."""

    scale: float      # uniform resize factor applied to (orig_w, orig_h)
    pad_left: int     # pixels of padding added on the left
    pad_top: int      # pixels of padding added on top
    target_w: int     # model input width after letterboxing
    target_h: int     # model input height after letterboxing


# ------------------------------------------------------------------
# Handle
# ------------------------------------------------------------------


@dataclass
class _HailoHandle:
    """Opaque model handle returned by :meth:`HailoRuntime.load`.

    The ``InferVStreams`` pipeline and the network_group activation
    are entered **once** at load time and exited on :meth:`close`.
    Opening them per-:meth:`predict` call would tank throughput by
    an order of magnitude (DMA buffer setup on every frame).
    """

    vdevice: Any                       # hailo_platform.VDevice
    network_group: Any                 # hailo_platform.ConfiguredNetwork
    pipeline: Any                      # entered hailo_platform.InferVStreams
    activation: Any                    # entered ConfiguredNetwork.activate()
    exit_stack: ExitStack              # owns pipeline + activation lifetimes
    input_name: str
    input_shape_hw: tuple[int, int]    # (H, W) regardless of internal order
    output_names: list[str]
    class_names: dict[int, str]
    hw_arch: str
    # "detection" | "detection_raw" | "instance_segmentation" | "embedding"
    # Set from the ``model_kind`` kwarg passed to :meth:`HailoRuntime.load`.
    model_kind: str = "detection"
    lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: bool = False

    def close(self) -> None:
        """Release the activation, vstreams, and VDevice in order."""
        if self._closed:
            return
        self._closed = True
        try:
            self.exit_stack.close()
        finally:
            _safe_release(self.vdevice)


# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------


class HailoRuntime(ModelRuntime):
    """Runtime backend for Hailo ``.hef`` models via ``hailo_platform``."""

    name = "hailo"

    def is_available(self) -> bool:
        try:
            import hailo_platform  # noqa: F401

            return True
        except ImportError:
            return False

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        labels: list[str] | dict[int, str] | None = None,
        model_kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a ``.hef`` and configure it on the Hailo device.

        Args:
            model_path: path to the ``.hef`` file.
            device: ignored — Hailo binaries pick the accelerator
                automatically via ``VDevice``. Kept for API parity
                with the other runtimes.
            labels: optional class-id → label mapping. When omitted
                the runtime tries to read it from a sidecar
                ``<model>.labels.json`` and finally falls back to
                ``str(class_id)``.
            model_kind: ``"detection"`` (YOLO NMS output),
                ``"detection_raw"`` (YOLO feature-map output, no on-chip NMS),
                ``"instance_segmentation"`` (yolov8_seg raw feature maps),
                or ``"embedding"`` (CLIP/encoder models). When omitted
                (the default) the runtime auto-detects the kind from the
                HEF's output-stream shapes so callers do not need to set
                this explicitly.
        """
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        hef = HEF(model_path)
        hef_arch = _normalize_arch(_hef_hw_arch(hef))

        vdevice = VDevice()
        device_arch = _normalize_arch(_vdevice_hw_arch(vdevice))
        # Best-effort pre-load mismatch guard. On HailoRT 4.23.0 both
        # ``hef_arch`` and ``device_arch`` come back empty because the
        # Python binding does not expose either via the historical
        # attr names; in that case we rely on ``vdevice.configure(...)``
        # below to raise HailoRT's native incompatibility error.
        # TODO: re-enable once the 4.23+ attr names are confirmed.
        if device_arch and hef_arch and device_arch != hef_arch:
            _safe_release(vdevice)
            raise RuntimeError(
                f"HEF '{model_path}' was compiled for {hef_arch!r}, but the "
                f"connected accelerator is {device_arch!r}. Use the "
                f"{device_arch}-compiled variant."
            )

        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_group = vdevice.configure(hef, configure_params)[0]

        input_info = hef.get_input_vstream_infos()[0]
        output_infos = hef.get_output_vstream_infos()
        input_h, input_w = _input_hw(input_info)

        input_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8
        )
        output_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        # Enter the pipeline + activation contexts once and stash
        # them. ExitStack guarantees both are released in reverse
        # order on close()/error, even if one of them raises on exit.
        stack = ExitStack()
        try:
            pipeline = stack.enter_context(
                InferVStreams(network_group, input_params, output_params)
            )
            activation = stack.enter_context(network_group.activate())
        except Exception:
            stack.close()
            _safe_release(vdevice)
            raise

        effective_kind = model_kind or _infer_model_kind(output_infos)

        logger.info(
            "Loaded Hailo HEF '%s' (arch=%s, input=%dx%d, %d output(s), kind=%s)",
            model_path,
            hef_arch or "unknown",
            input_h,
            input_w,
            len(output_infos),
            effective_kind,
        )

        return _HailoHandle(
            vdevice=vdevice,
            network_group=network_group,
            pipeline=pipeline,
            activation=activation,
            exit_stack=stack,
            input_name=input_info.name,
            input_shape_hw=(input_h, input_w),
            output_names=[o.name for o in output_infos],
            class_names=_resolve_class_names(model_path, labels),
            hw_arch=hef_arch,
            model_kind=effective_kind,
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
        import time

        handle: _HailoHandle = model_handle
        target_h, target_w = handle.input_shape_hw

        img = np.asarray(input_data)
        img_h, img_w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else (0, 0)

        if handle.model_kind == "embedding":
            t0 = time.perf_counter()
            tensor = _resize_for_embedding(img, target_h=target_h, target_w=target_w)
            t1 = time.perf_counter()

            with handle.lock:
                outputs = handle.pipeline.infer({handle.input_name: tensor})
            t2 = time.perf_counter()

            vec = _extract_embedding(outputs)
            t3 = time.perf_counter()

            logger.debug(
                "hailo embed: pre=%.1fms infer=%.1fms post=%.1fms total=%.1fms "
                "dim=%d",
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t3 - t0) * 1000,
                len(vec),
            )
            return EmbeddingResult(vector=vec, raw=outputs)

        t0 = time.perf_counter()
        tensor, letterbox = _preprocess(img, target_h=target_h, target_w=target_w)
        t1 = time.perf_counter()

        with handle.lock:
            outputs = handle.pipeline.infer({handle.input_name: tensor})
        t2 = time.perf_counter()

        if handle.model_kind == "instance_segmentation":
            seg_result = _postprocess_seg(
                outputs,
                class_names=handle.class_names,
                confidence=confidence,
                classes=classes,
                letterbox=letterbox,
                orig_w=img_w,
                orig_h=img_h,
            )
            t3 = time.perf_counter()
            logger.debug(
                "hailo seg: pre=%.1fms infer=%.1fms post=%.1fms total=%.1fms dets=%d",
                (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
                (t3 - t0) * 1000, len(seg_result.detections),
            )
            return seg_result.__class__(seg_result.detections, raw=outputs)

        if handle.model_kind == "detection_raw":
            detections = _postprocess_det_raw(
                outputs,
                class_names=handle.class_names,
                confidence=confidence,
                classes=classes,
                letterbox=letterbox,
                orig_w=img_w,
                orig_h=img_h,
            )
            t3 = time.perf_counter()
            logger.debug(
                "hailo raw det: pre=%.1fms infer=%.1fms post=%.1fms total=%.1fms "
                "detections=%d",
                (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
                (t3 - t0) * 1000, len(detections),
            )
            return DetectionResult(detections, raw=outputs)

        # Auto-detect raw spatial outputs on the first "detection" frame.
        # Static HEF shape metadata is unreliable for this — HailoRT reports
        # the underlying conv-layer shapes even for models with on-chip NMS.
        # Checking the actual inference output is definitive: NMS results are
        # Python lists or ragged object arrays; raw feature maps are float
        # ndarrays with H > 1 and W > 1.
        if handle.model_kind == "detection" and _outputs_look_spatial(outputs):
            handle.model_kind = "detection_raw"
            logger.info(
                "hailo: auto-detected raw spatial feature-map outputs "
                "(HEF compiled without on-chip NMS); switching to "
                "detection_raw decoder for all subsequent frames.",
            )
            detections = _postprocess_det_raw(
                outputs,
                class_names=handle.class_names,
                confidence=confidence,
                classes=classes,
                letterbox=letterbox,
                orig_w=img_w,
                orig_h=img_h,
            )
            t3 = time.perf_counter()
            logger.debug(
                "hailo raw det: pre=%.1fms infer=%.1fms post=%.1fms total=%.1fms "
                "detections=%d",
                (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
                (t3 - t0) * 1000, len(detections),
            )
            return DetectionResult(detections, raw=outputs)

        detections = _postprocess(
            outputs,
            class_names=handle.class_names,
            confidence=confidence,
            classes=classes,
            letterbox=letterbox,
            orig_w=img_w,
            orig_h=img_h,
        )
        t3 = time.perf_counter()

        logger.debug(
            "hailo predict: pre=%.1fms infer=%.1fms post=%.1fms total=%.1fms "
            "detections=%d",
            (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
            (t3 - t0) * 1000, len(detections),
        )

        return DetectionResult(detections, raw=outputs)


# ------------------------------------------------------------------
# Hailo binding helpers (defensive against version drift)
# ------------------------------------------------------------------


def _hef_hw_arch(hef: Any) -> str:
    """Best-effort extraction of the HEF's compiled hardware arch.

    On HailoRT 4.23.0's Python binding none of these attribute names
    exist on the ``HEF`` object, so the function returns ``""`` and
    the caller falls back to letting ``vdevice.configure(...)`` raise
    HailoRT's native incompatibility error at load time. Re-enable
    once the 4.23+ binding's attr name is confirmed.
    """
    for attr in ("get_hef_device_arch", "get_target_arch", "device_arch"):
        fn = getattr(hef, attr, None)
        if fn is None:
            continue
        try:
            return str(fn() if callable(fn) else fn)
        except Exception:
            continue
    return ""


def _vdevice_hw_arch(vdevice: Any) -> str:
    """Best-effort extraction of the physical device's hardware arch."""
    try:
        devs = vdevice.get_physical_devices()
    except Exception:
        return ""
    if not devs:
        return ""
    dev = devs[0]
    for attr in ("get_arch", "device_arch", "arch"):
        v = getattr(dev, attr, None)
        if v is None:
            continue
        try:
            return str(v() if callable(v) else v)
        except Exception:
            continue
    return ""


def _normalize_arch(raw: str) -> str:
    """Canonicalize HEF / device arch strings for comparison.

    Inputs we've seen across HailoRT versions::

        "HAILO8"               → "hailo8"
        "Hailo-8"              → "hailo8"
        "HAILO_ARCH_HAILO8"    → "hailo8"
        "HAILO_ARCH_HAILO_8L"  → "hailo8l"
        ""                     → ""

    Strategy: lowercase, drop ``hailo_arch_`` prefix, strip non-alnum.
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    s = re.sub(r"^hailo[_\-]?arch[_\-]?", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _safe_release(vdevice: Any) -> None:
    """Call VDevice.release() if available; never raise."""
    try:
        release = getattr(vdevice, "release", None)
        if callable(release):
            release()
    except Exception:
        logger.debug("VDevice.release() raised on cleanup", exc_info=True)


def _input_hw(input_info: Any) -> tuple[int, int]:
    """Return ``(H, W)`` from a Hailo input vstream info.

    ``input_info.shape`` order depends on ``input_info.format.order``.
    We support the two layouts seen in the wild (NHWC / HWC and
    NCHW / CHW); other orders fall back to assuming HWC and log a
    warning. On HailoRT 4.23.0 + Hailo Model Zoo YOLOv8 HEFs the
    shape is ``(640, 640, 3)`` (HWC, channel-last).
    """
    shape = tuple(input_info.shape)
    if len(shape) == 3:
        # Heuristic: the channel axis is the small one (1, 3, 4).
        if shape[-1] in (1, 3, 4):           # HWC
            return shape[0], shape[1]
        if shape[0] in (1, 3, 4):            # CHW
            return shape[1], shape[2]
    if len(shape) == 4:
        # NHWC or NCHW
        if shape[-1] in (1, 3, 4):
            return shape[1], shape[2]
        if shape[1] in (1, 3, 4):
            return shape[2], shape[3]
    logger.warning(
        "Could not infer (H, W) from Hailo input shape %s; assuming HWC.",
        shape,
    )
    return shape[0], shape[1]


# Standard COCO 80-class labels in index order. Used as the default for
# any YOLO HEF that ships without a .labels.json sidecar — every model
# in the Hailo catalog is compiled from a COCO-pretrained YOLO checkpoint
# so this is always the right fallback. Callers can override via the
# ``labels`` argument or a sidecar file.
_COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _resolve_class_names(
    model_path: str,
    labels: list[str] | dict[int, str] | None,
) -> dict[int, str]:
    """Resolve class-id → name from caller-provided labels or a sidecar.

    Falls back to the standard COCO 80-class list when no labels are
    supplied and no ``.labels.json`` sidecar exists alongside the HEF.
    Every YOLO HEF in the Hailo catalog is COCO-pretrained, so this
    default is always correct for catalog models.
    """
    if isinstance(labels, dict):
        return {int(k): str(v) for k, v in labels.items()}
    if isinstance(labels, list):
        return {i: str(name) for i, name in enumerate(labels)}

    import json
    from pathlib import Path

    sidecar = Path(model_path).with_suffix(".labels.json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text())
            if isinstance(data, list):
                return {i: str(n) for i, n in enumerate(data)}
            if isinstance(data, dict):
                return {int(k): str(v) for k, v in data.items()}
        except Exception:
            logger.warning("Could not parse class-names sidecar %s", sidecar)

    return {i: name for i, name in enumerate(_COCO_CLASSES)}


# ------------------------------------------------------------------
# Preprocess: letterbox (detection) and plain resize (embedding)
# ------------------------------------------------------------------


def _resize_for_embedding(
    img: np.ndarray,
    *,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize HWC uint8 image → NHWC uint8 tensor for encoder/embedding models.

    Unlike :func:`_preprocess` this performs a plain stretch resize without
    letterbox padding. CLIP and similar vision encoders are trained on
    square-cropped inputs; padding the image with grey bars would introduce
    blank regions that degrade the embedding quality. A simple resize gives
    a better approximation of the center-crop pre-processing the model was
    trained with when the input aspect ratio is reasonably close to 1:1.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image, got shape {img.shape}")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Empty input image")

    try:
        import cv2

        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR)
        )

    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)

    return np.expand_dims(np.ascontiguousarray(resized), 0)  # NHWC, batch=1


def _extract_embedding(
    outputs: dict[str, Any] | Any,
) -> np.ndarray:
    """Extract a flat feature vector from an encoder/CLIP HEF output.

    Encoder HEFs (e.g. ``clip_resnet_50x4.hef``) produce a single output
    tensor of shape ``(1, D)`` or ``(D,)`` — a batch-of-one embedding
    vector. This helper unwraps that tensor into a 1-D float32 ndarray.

    If the output doesn't match the expected shape (e.g. more than one
    output stream, unexpected ndim) the raw values are still returned by
    ravelling the first available tensor, so callers always get *something*
    numeric. An empty array is returned only when the output dict is empty.
    """
    if isinstance(outputs, dict):
        if not outputs:
            logger.warning("Hailo embedding: inference returned empty outputs dict.")
            return np.zeros(0, dtype=np.float32)
        tensors = list(outputs.values())
    else:
        tensors = [outputs]

    if len(tensors) > 1:
        logger.warning(
            "Hailo embedding: expected 1 output tensor, got %d. "
            "Using the first tensor. Check that the correct HEF is loaded.",
            len(tensors),
        )

    raw = tensors[0]
    arr = np.asarray(raw, dtype=np.float32)

    # Shape (1, D) → (D,)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    # Already flat
    if arr.ndim == 1:
        return arr
    # Unexpected shape — ravel and warn
    logger.warning(
        "Hailo embedding: unexpected output shape %s; ravelling to 1-D.",
        arr.shape,
    )
    return arr.ravel()


def _preprocess(
    img: np.ndarray,
    *,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, _Letterbox]:
    """Letterbox-resize HWC uint8 → NHWC uint8 tensor + transform params.

    Hailo HEFs compiled by the Dataflow Compiler accept uint8 input
    directly; the quantization to int8 happens inside the chip using
    the per-tensor scale baked into the HEF. Letterboxing preserves
    aspect ratio with a constant 114-grey pad (the YOLO convention)
    so detection coordinates can be un-letterboxed back to the
    original image's pixel space in postprocess.
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image, got shape {img.shape}")

    src_h, src_w = img.shape[:2]
    if src_w == 0 or src_h == 0:
        raise ValueError("Empty input image")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    try:
        import cv2

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        )

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    canvas = np.full((target_h, target_w, img.shape[2]), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8)

    tensor = np.expand_dims(np.ascontiguousarray(canvas), 0)  # NHWC, batch=1
    return tensor, _Letterbox(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        target_w=target_w,
        target_h=target_h,
    )


# ------------------------------------------------------------------
# Postprocess: NMS-on-chip output → Detection objects
# ------------------------------------------------------------------


def _postprocess(
    outputs: dict[str, np.ndarray] | list[Any] | Any,
    *,
    class_names: dict[int, str],
    confidence: float,
    classes: list[str] | None,
    letterbox: _Letterbox,
    orig_w: int,
    orig_h: int,
) -> list[Detection]:
    """Decode Hailo NMS-on-chip outputs into ``Detection`` objects.

    Handles flat ``[batch, N, 6]``, batched per-class
    ``list[batch][ndarray(num_classes, K, 5)]``, and per-class
    ``[num_classes][max_per_class, 5]`` layouts (see module
    docstring). Coordinates are assumed to be in ``[0, 1]`` of the
    letterboxed model input; we un-letterbox them back to the
    original image's pixel space using ``letterbox``.
    """
    rows = _flatten_nms_output(outputs)
    if not rows:
        return []

    # rows is list[(class_id, score, x1, y1, x2, y2)] in normalized
    # letterboxed coordinates. Filter by confidence first to keep the
    # un-letterbox loop short.
    rows = [r for r in rows if r[1] >= confidence]
    if not rows:
        return []

    frame_w = orig_w if orig_w > 0 else letterbox.target_w
    frame_h = orig_h if orig_h > 0 else letterbox.target_h
    frame_area = max(frame_w * frame_h, 1)

    detections: list[Detection] = []
    for cid, score, nx1, ny1, nx2, ny2 in rows:
        label = class_names.get(cid, str(cid))
        if classes and label not in classes:
            continue
        # Un-letterbox: normalized model-input coords → original-image pixels.
        x1 = (nx1 * letterbox.target_w - letterbox.pad_left) / letterbox.scale
        y1 = (ny1 * letterbox.target_h - letterbox.pad_top) / letterbox.scale
        x2 = (nx2 * letterbox.target_w - letterbox.pad_left) / letterbox.scale
        y2 = (ny2 * letterbox.target_h - letterbox.pad_top) / letterbox.scale
        # Clip to the original frame.
        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detections.append(
            Detection(
                label=label,
                confidence=float(score),
                bbox=bbox,
                area_ratio=bbox.area / frame_area,
            )
        )
    return detections


# ------------------------------------------------------------------
# Instance segmentation postprocessor (yolov8_seg HEFs)
# ------------------------------------------------------------------

_SEG_REG_MAX = 16     # DFL distribution bins (reg_max in YOLOv8 head)
_SEG_NM = 32          # mask prototype count
_SEG_IOU_THRESH = 0.45
#: Detection-head channel counts seen in the wild for YOLOv8-seg HEFs.
#: 116 = 4 + 80 + 32 (pre-decoded boxes), 176 = 64 + 80 + 32 (raw DFL).
_SEG_DET_CHANNELS: frozenset[int] = frozenset({116, 176})

# Raw detection (no on-chip NMS) parameters
_DET_RAW_REG_MAX = 16   # default YOLOv8 DFL reg_max
_DET_RAW_IOU_THRESH = 0.45


def _infer_model_kind(output_infos: list[Any]) -> str:
    """Best-effort model-kind detection from HEF output-stream shapes.

    Examines the last dimension (channel axis for HWC tensors) of each output
    to distinguish:

    * **instance_segmentation** — detected in two layouts:

      - *Combined* (classic): one tensor per scale with last dim in
        ``{116, 176}`` = ``4 + num_classes + 32`` or ``64 + num_classes + 32``.
      - *Split* (Hailo-specific): Hailo compiles seg HEFs with separate
        box / class / mask-coefficient tensors per scale plus one proto tensor;
        the mask-coefficient and proto tensors share ``C = _SEG_NM = 32``.
        Two or more outputs with ``last_dim == 32`` is the discriminating
        signal (a plain detection model only gets ``C = 32`` if it has exactly
        32 classes, which is handled by requesting ``model_kind`` explicitly).

    * **embedding** — single 1-D (or batch-1 2-D) output, no spatial dims.

    * **detection** — everything else, including on-chip NMS outputs and
      raw spatial feature-map outputs.  Raw outputs are identified and
      re-classified to ``"detection_raw"`` at inference time by
      :func:`_outputs_look_spatial`, which inspects the actual tensors
      returned by the Hailo pipeline rather than the static HEF metadata.
      Static shape heuristics are unreliable for that distinction because
      HailoRT can expose the underlying conv-layer shapes even for models
      compiled with an on-chip NMS post-process.
    """
    # Extract last dim of every output (= C in HWC, or the "row width" in
    # per-class NMS tensors).  Using only the last dim avoids false matches
    # from spatial dimensions coincidentally equalling channel counts.
    last_dims: list[int] = []
    for info in output_infos:
        s = tuple(getattr(info, "shape", ()))
        if s:
            last_dims.append(s[-1])
        # For 4-D NCHW (N, C, H, W) also collect s[1].
        if len(s) == 4:
            last_dims.append(s[1])

    # --- Instance segmentation -----------------------------------------
    # Combined layout: last dim encodes box + classes + mask-coeff together.
    has_combined_seg = any(c in _SEG_DET_CHANNELS for c in last_dims)
    # Split layout (Hailo): separate mask-coeff tensors (C=32) per scale
    # plus one proto tensor (also C=32) → at least 2 outputs with last dim 32.
    last_dim_32_count = last_dims.count(_SEG_NM)
    has_split_seg = last_dim_32_count >= 2

    if has_combined_seg or has_split_seg:
        return "instance_segmentation"

    # --- Embedding ---------------------------------------------------------
    if len(output_infos) == 1:
        s = tuple(getattr(output_infos[0], "shape", ()))
        if len(s) <= 2:   # (D,) or (1, D) — flat embedding
            return "embedding"

    return "detection"


def _split_seg_outputs(
    outputs: dict[str, Any],
    num_masks: int = _SEG_NM,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Split yolov8-seg HEF outputs into (proto, [det_s8, det_s16, det_s32]).

    Proto is identified as the tensor with the largest spatial area whose
    channel count equals ``num_masks`` (32).  Detection tensors are the
    remaining 4-D tensors sorted largest→smallest by spatial area (stride-8
    first).

    Two output layouts are handled:

    * **Combined** (classic): one tensor per scale with C in ``{116, 176}``.
      Detection tensors are returned as-is.

    * **Split** (Hailo-specific): Hailo compiles seg HEFs with separate
      box, class, and mask-coefficient tensors per scale plus one proto
      tensor.  When multiple C=``num_masks`` tensors are found (indicating
      per-scale mask-coefficient outputs rather than a single proto), the
      tensors at each scale are grouped by spatial resolution and
      concatenated as ``[box | cls | coef]``.  This reconstructs the
      combined C=176 (DFL) or C=116 (pre-decoded) layout that
      ``_decode_one_seg_head`` expects, so no change is needed downstream.
    """
    tensors: list[np.ndarray] = []
    for v in outputs.values():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 3:
            a = a[np.newaxis]
        if a.ndim == 4:
            tensors.append(a[0])   # drop batch dim → (H, W, C)

    tensors.sort(key=lambda t: t.shape[0] * t.shape[1], reverse=True)

    proto: np.ndarray | None = None
    remaining: list[np.ndarray] = []
    for t in tensors:
        if t.shape[2] == num_masks and proto is None:
            proto = t              # (ph, pw, nm)  — largest spatial, nm channels
        else:
            remaining.append(t)

    # Split format: any leftover C=num_masks tensors are per-scale
    # mask-coefficient outputs (not detection heads).  Reconstruct a
    # combined tensor per scale by concatenating [box | cls | coef] so
    # that _decode_one_seg_head can handle them with its existing paths.
    if any(t.shape[2] == num_masks for t in remaining):
        by_scale: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        for t in remaining:
            by_scale[(t.shape[0], t.shape[1])].append(t)

        combined: list[np.ndarray] = []
        for _hw, group in sorted(
            by_scale.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True
        ):
            coef_t: np.ndarray | None = None
            other: list[np.ndarray] = []
            for t in group:
                if t.shape[2] == num_masks and coef_t is None:
                    coef_t = t
                else:
                    other.append(t)
            if coef_t is None or not other:
                continue
            # Sort non-coef tensors ascending by C so box (C=4 or C=64)
            # comes before class (C=nc), matching the [box|cls|coef] layout
            # that _decode_one_seg_head expects.
            other.sort(key=lambda t: t.shape[2])
            combined.append(np.concatenate([*other, coef_t], axis=-1))
        return proto, combined[:3]

    return proto, remaining[:3]


def _decode_one_seg_head(
    feat: np.ndarray,    # (H, W, C)
    input_h: int,
    input_w: int,
    num_classes: int,
    num_masks: int,
    conf_thresh: float,
    reg_max: int = _SEG_REG_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one YOLOv8-seg detection head → (boxes_norm, scores, class_ids, coefs).

    Handles two channel layouts:
    - C == 4 + nc + nm  (116): boxes already decoded by on-chip ``yolov8_bbox_decoding``
    - C == 4*reg_max + nc + nm  (176): raw DFL distribution, decoded here via soft-argmax
    """
    H, W, C = feat.shape
    dfl_ch = 4 * reg_max   # 64

    if C == 4 + num_classes + num_masks:
        boxes_hw = np.clip(feat[:, :, :4], 0.0, 1.0)
        cls_logits = feat[:, :, 4:4 + num_classes]
        coefs_hw = feat[:, :, 4 + num_classes:]

    elif C == dfl_ch + num_classes + num_masks:
        # Soft-argmax DFL decode: (H*W, 4, reg_max) → (H*W, 4) l/t/r/b distances
        dfl = feat[:, :, :dfl_ch].reshape(-1, 4, reg_max)
        dfl = np.exp(dfl - dfl.max(axis=-1, keepdims=True))
        dfl = dfl / dfl.sum(axis=-1, keepdims=True)
        ltrb = (dfl * np.arange(reg_max, dtype=np.float32)).sum(axis=-1)  # (H*W, 4)

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        stride_h, stride_w = input_h / H, input_w / W
        cx = (xs.ravel() + 0.5) * stride_w
        cy = (ys.ravel() + 0.5) * stride_h
        x1 = np.clip((cx - ltrb[:, 0] * stride_w) / input_w, 0.0, 1.0)
        y1 = np.clip((cy - ltrb[:, 1] * stride_h) / input_h, 0.0, 1.0)
        x2 = np.clip((cx + ltrb[:, 2] * stride_w) / input_w, 0.0, 1.0)
        y2 = np.clip((cy + ltrb[:, 3] * stride_h) / input_h, 0.0, 1.0)
        boxes_hw = np.stack([x1, y1, x2, y2], axis=-1).reshape(H, W, 4)
        cls_logits = feat[:, :, dfl_ch:dfl_ch + num_classes]
        coefs_hw = feat[:, :, dfl_ch + num_classes:]

    else:
        logger.warning(
            "Hailo seg head: unexpected channel count %d (expected %d or %d).",
            C, 4 + num_classes + num_masks, dfl_ch + num_classes + num_masks,
        )
        return (
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.int32),
            np.empty((0, num_masks), np.float32),
        )

    N = H * W
    cls_prob = 1.0 / (1.0 + np.exp(-cls_logits.reshape(N, num_classes)))
    class_ids = cls_prob.argmax(axis=1).astype(np.int32)
    scores = cls_prob[np.arange(N), class_ids]
    keep = scores >= conf_thresh
    return (
        boxes_hw.reshape(N, 4)[keep],
        scores[keep],
        class_ids[keep],
        coefs_hw.reshape(N, num_masks)[keep],
    )


def _decode_one_det_head(
    feat: np.ndarray,             # (H, W, C) — combined [box+cls] OR cls-only tensor
    box_feat: np.ndarray | None,  # (H, W, 4) or (H, W, 4*reg_max) — separate box tensor
    input_h: int,
    input_w: int,
    num_classes: int,
    conf_thresh: float,
    reg_max: int = _DET_RAW_REG_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one raw YOLOv8 detection head → (boxes_norm, scores, class_ids).

    Handles two combined channel layouts (``box_feat is None``):

    * ``C == 4 + num_classes``: boxes already decoded to ``(x1, y1, x2, y2)``
      in normalised ``[0, 1]`` coords.
    * ``C == 4 * reg_max + num_classes``: raw DFL distribution; soft-argmax
      is applied to convert to normalised coords.

    And the separate-tensor layout (``box_feat is not None``):

    * ``box_feat`` shape ``(H, W, 4)`` — pre-decoded box coordinates.
    * ``box_feat`` shape ``(H, W, 4 * reg_max)`` — raw DFL distribution.
    * ``feat`` shape ``(H, W, num_classes)`` — class logits.

    Returns empty arrays when the channel count does not match any known layout.
    """
    H, W, C = feat.shape
    dfl_ch = 4 * reg_max

    _empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        np.empty((0, 4), np.float32),
        np.empty(0, np.float32),
        np.empty(0, np.int32),
    )

    if box_feat is not None:
        cls_logits = feat
        box_raw = box_feat
    elif C == 4 + num_classes:
        box_raw = feat[:, :, :4]
        cls_logits = feat[:, :, 4:]
    elif C == dfl_ch + num_classes:
        box_raw = feat[:, :, :dfl_ch]
        cls_logits = feat[:, :, dfl_ch:]
    else:
        return _empty

    bH, bW, bC = box_raw.shape
    if bC == 4:
        boxes_hw = np.clip(box_raw, 0.0, 1.0)
    elif bC == dfl_ch:
        dfl = box_raw.reshape(-1, 4, reg_max)
        dfl = np.exp(dfl - dfl.max(axis=-1, keepdims=True))
        dfl = dfl / dfl.sum(axis=-1, keepdims=True)
        ltrb = (dfl * np.arange(reg_max, dtype=np.float32)).sum(axis=-1)
        ys, xs = np.meshgrid(np.arange(bH), np.arange(bW), indexing="ij")
        stride_h, stride_w = input_h / bH, input_w / bW
        cx = (xs.ravel() + 0.5) * stride_w
        cy = (ys.ravel() + 0.5) * stride_h
        x1 = np.clip((cx - ltrb[:, 0] * stride_w) / input_w, 0.0, 1.0)
        y1 = np.clip((cy - ltrb[:, 1] * stride_h) / input_h, 0.0, 1.0)
        x2 = np.clip((cx + ltrb[:, 2] * stride_w) / input_w, 0.0, 1.0)
        y2 = np.clip((cy + ltrb[:, 3] * stride_h) / input_h, 0.0, 1.0)
        boxes_hw = np.stack([x1, y1, x2, y2], axis=-1).reshape(bH, bW, 4)
    else:
        return _empty

    if (bH, bW) != (H, W):
        logger.warning(
            "hailo det head: box tensor spatial dims (%d×%d) do not match "
            "class tensor dims (%d×%d); skipping head.",
            bH, bW, H, W,
        )
        return _empty

    N = H * W
    cls_prob = 1.0 / (1.0 + np.exp(-cls_logits.reshape(N, num_classes)))
    class_ids = cls_prob.argmax(axis=1).astype(np.int32)
    scores = cls_prob[np.arange(N), class_ids]
    keep = scores >= conf_thresh
    return (
        boxes_hw.reshape(N, 4)[keep],
        scores[keep],
        class_ids[keep],
    )


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    """Greedy IoU NMS. ``boxes``: (N, 4) x1y1x2y2. Returns kept index array."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-7)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def _decode_instance_masks(
    coefs: np.ndarray,       # (N, nm)
    proto: np.ndarray,       # (ph, pw, nm)
    boxes_norm: np.ndarray,  # (N, 4) x1y1x2y2 normalized in letterboxed input coords
    letterbox: _Letterbox,
    orig_h: int,
    orig_w: int,
) -> list[Mask]:
    """Decode mask coefficients + prototype tensors into per-detection Masks.

    1. matrix-multiply coefs × proto → sigmoid → full proto-res masks (ph×pw each)
    2. For each detection, crop the proto-space mask to the bbox region
    3. Resize crop to the bbox pixel size in the original image
    4. Un-letterbox and stamp into a full-image binary canvas
    """
    ph, pw, nm = proto.shape
    proto_flat = proto.reshape(ph * pw, nm).T          # (nm, ph*pw)
    masks_full = (1.0 / (1.0 + np.exp(-(coefs @ proto_flat)))).reshape(-1, ph, pw)

    frame_w = orig_w if orig_w > 0 else letterbox.target_w
    frame_h = orig_h if orig_h > 0 else letterbox.target_h

    try:
        import cv2 as _cv2

        def _resize(crop: np.ndarray, w: int, h: int) -> np.ndarray:
            return _cv2.resize(crop, (w, h), interpolation=_cv2.INTER_LINEAR)

    except ImportError:
        from PIL import Image as _Image

        def _resize(crop: np.ndarray, w: int, h: int) -> np.ndarray:  # type: ignore[misc]
            return np.asarray(
                _Image.fromarray((crop * 255).astype(np.uint8)).resize((w, h), _Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

    result: list[Mask] = []
    for mask, box_n in zip(masks_full, boxes_norm):
        # Un-letterbox box coords → original-image pixels
        x1p = (box_n[0] * letterbox.target_w - letterbox.pad_left) / letterbox.scale
        y1p = (box_n[1] * letterbox.target_h - letterbox.pad_top) / letterbox.scale
        x2p = (box_n[2] * letterbox.target_w - letterbox.pad_left) / letterbox.scale
        y2p = (box_n[3] * letterbox.target_h - letterbox.pad_top) / letterbox.scale
        x1p = max(0.0, min(float(frame_w), x1p))
        y1p = max(0.0, min(float(frame_h), y1p))
        x2p = max(0.0, min(float(frame_w), x2p))
        y2p = max(0.0, min(float(frame_h), y2p))

        bw = max(1, int(round(x2p - x1p)))
        bh = max(1, int(round(y2p - y1p)))

        # Crop in proto space
        px1, py1 = int(max(0, box_n[0] * pw)), int(max(0, box_n[1] * ph))
        px2, py2 = int(min(pw, box_n[2] * pw)), int(min(ph, box_n[3] * ph))
        if px2 <= px1 or py2 <= py1:
            result.append(Mask(data=np.zeros((frame_h, frame_w), np.uint8), h=frame_h, w=frame_w))
            continue

        binary = (_resize(mask[py1:py2, px1:px2], bw, bh) > 0.5).astype(np.uint8)
        canvas = np.zeros((frame_h, frame_w), np.uint8)
        ys, xs = int(round(y1p)), int(round(x1p))
        ye, xe = min(frame_h, ys + bh), min(frame_w, xs + bw)
        if ye > ys and xe > xs:
            canvas[ys:ye, xs:xe] = binary[:ye - ys, :xe - xs]
        result.append(Mask(data=canvas, h=frame_h, w=frame_w))

    return result


def _postprocess_seg(
    outputs: dict[str, Any],
    *,
    class_names: dict[int, str],
    confidence: float,
    classes: list[str] | None,
    letterbox: _Letterbox,
    orig_w: int,
    orig_h: int,
    num_classes: int = 80,
    num_masks: int = _SEG_NM,
    iou_thresh: float = _SEG_IOU_THRESH,
) -> InstanceSegmentationResult:
    """Full CPU-side decode for yolov8-seg HEF outputs.

    Identifies proto + detection tensors, runs per-scale head decode,
    greedy NMS, mask decode, and returns an InstanceSegmentationResult.
    """
    proto, det_tensors = _split_seg_outputs(outputs, num_masks)
    if proto is None or not det_tensors:
        logger.warning(
            "Hailo seg: could not identify proto/detection tensors in outputs %s.",
            {k: getattr(v, "shape", type(v).__name__) for k, v in outputs.items()},
        )
        return InstanceSegmentationResult([])

    input_h, input_w = letterbox.target_h, letterbox.target_w
    all_boxes, all_scores, all_class_ids, all_coefs = [], [], [], []

    for feat in det_tensors:
        b, s, c, co = _decode_one_seg_head(
            feat, input_h, input_w, num_classes, num_masks, confidence,
        )
        if b.size:
            all_boxes.append(b)
            all_scores.append(s)
            all_class_ids.append(c)
            all_coefs.append(co)

    if not all_boxes:
        return InstanceSegmentationResult([])

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_class_ids)
    coefs = np.concatenate(all_coefs)

    keep = _nms_numpy(boxes, scores, iou_thresh)
    boxes, scores, class_ids, coefs = boxes[keep], scores[keep], class_ids[keep], coefs[keep]

    masks = _decode_instance_masks(coefs, proto, boxes, letterbox, orig_h, orig_w)

    frame_w = orig_w if orig_w > 0 else input_w
    frame_h = orig_h if orig_h > 0 else input_h
    frame_area = max(frame_w * frame_h, 1)

    detections: list[Detection] = []
    for box_n, score, cid, mask in zip(boxes, scores, class_ids, masks):
        label = class_names.get(int(cid), str(cid))
        if classes and label not in classes:
            continue
        x1 = (box_n[0] * input_w - letterbox.pad_left) / letterbox.scale
        y1 = (box_n[1] * input_h - letterbox.pad_top) / letterbox.scale
        x2 = (box_n[2] * input_w - letterbox.pad_left) / letterbox.scale
        y2 = (box_n[3] * input_h - letterbox.pad_top) / letterbox.scale
        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detections.append(Detection(
            label=label,
            confidence=float(score),
            bbox=bbox,
            mask=mask,
            area_ratio=bbox.area / frame_area,
        ))
    return InstanceSegmentationResult(detections)


def _outputs_look_spatial(outputs: dict[str, Any]) -> bool:
    """Return True if every value in *outputs* is a 3-D or 4-D spatial ndarray.

    This distinguishes raw feature-map outputs (HEF compiled without on-chip
    NMS) from on-chip NMS outputs at inference time, where we have the actual
    tensor data rather than the unreliable static HEF metadata.

    On-chip NMS outputs are Python lists (per-class detections) or 2-D / 3-D
    ndarrays (flat NMS rows with 5 or 6 columns).  Raw spatial outputs are
    always 3-D (H, W, C) or 4-D (N, H, W, C) float ndarrays where both
    spatial dimensions are > 1.
    """
    if not isinstance(outputs, dict) or not outputs:
        return False
    for v in outputs.values():
        if isinstance(v, list):
            return False          # per-class NMS list
        try:
            a = np.asarray(v)
        except Exception:
            return False
        if a.dtype == object:
            return False          # ragged / per-class object array (NMS)
        # Spatial: ndim 3 (H,W,C) or 4 (N,H,W,C / N,C,H,W) with both
        # middle dims > 1.  NMS flat arrays are ndim ≤ 3 with last dim 5 or 6.
        if a.ndim == 4 and a.shape[1] > 1 and a.shape[2] > 1:
            continue
        elif a.ndim == 3 and a.shape[0] > 1 and a.shape[1] > 1 and a.shape[2] > 6:
            # Also require last dim > 6 for 3-D tensors to exclude per-class
            # NMS arrays like (num_classes, max_per_class, 5).
            continue
        else:
            return False
    return True


def _postprocess_det_raw(
    outputs: dict[str, Any],
    *,
    class_names: dict[int, str],
    confidence: float,
    classes: list[str] | None,
    letterbox: _Letterbox,
    orig_w: int,
    orig_h: int,
    num_classes: int | None = None,
    reg_max: int = _DET_RAW_REG_MAX,
    iou_thresh: float = _DET_RAW_IOU_THRESH,
) -> list[Detection]:
    """CPU-side decode for YOLOv8 HEFs compiled without on-chip NMS.

    Groups all 4-D output tensors by spatial size ``(H, W)`` to form per-scale
    groups.  For each scale the function identifies:

    * A **combined** tensor whose channel count equals ``4 + nc`` (pre-decoded
      boxes) or ``4 * reg_max + nc`` (raw DFL distribution).
    * Or **separate** box ``(H, W, 4)`` / ``(H, W, 4*reg_max)`` and class
      ``(H, W, nc)`` tensors.

    All decoded detections from every scale are merged, greedy NMS is applied,
    and the surviving boxes are un-letterboxed to original-image pixel space.

    ``num_classes`` defaults to ``len(class_names)``; pass it explicitly when
    the labels file has a different length than the actual model output.
    """
    nc = num_classes if num_classes is not None else len(class_names)
    if nc == 0:
        logger.warning(
            "hailo raw det: no class names found and num_classes not set; "
            "defaulting to nc=80. Pass num_classes explicitly or provide a "
            "labels file to suppress this warning.",
        )
        nc = 80  # last-resort fallback for unlabelled COCO models
    dfl_ch = 4 * reg_max

    # Group tensors by spatial resolution (H, W).
    scale_map: dict[tuple[int, int], list[np.ndarray]] = {}
    for v in outputs.values():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 3:
            a = a[np.newaxis]
        if a.ndim != 4:
            continue
        t = a[0]  # (H, W, C)
        scale_map.setdefault((t.shape[0], t.shape[1]), []).append(t)

    input_h, input_w = letterbox.target_h, letterbox.target_w
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []
    decoded_scales = 0

    for (H, W), scale_tensors in sorted(scale_map.items()):
        box_t: np.ndarray | None = None
        cls_t: np.ndarray | None = None
        combined_t: np.ndarray | None = None

        for t in scale_tensors:
            C = t.shape[2]
            if C == 4 + nc or C == dfl_ch + nc:
                combined_t = t
            elif C == nc:
                # Note: when nc happens to equal 4 or dfl_ch this is
                # ambiguous with the box tensor.  In that case the combined
                # tensor should be present; if not, we skip the scale below.
                cls_t = t
            elif C == 4 or C == dfl_ch:
                box_t = t

        if combined_t is not None:
            b, s, c = _decode_one_det_head(
                combined_t, None, input_h, input_w, nc, confidence, reg_max,
            )
        elif cls_t is not None and box_t is not None:
            b, s, c = _decode_one_det_head(
                cls_t, box_t, input_h, input_w, nc, confidence, reg_max,
            )
        else:
            logger.debug(
                "hailo raw det: scale %dx%d — tensors %s do not match any "
                "known layout for nc=%d reg_max=%d; skipping.",
                H, W,
                [t.shape for t in scale_tensors],
                nc,
                reg_max,
            )
            continue

        decoded_scales += 1
        if b.size:
            all_boxes.append(b)
            all_scores.append(s)
            all_class_ids.append(c)

    if not all_boxes:
        if scale_map and decoded_scales == 0:
            all_shapes = {k: [t.shape for t in v] for k, v in scale_map.items()}
            logger.warning(
                "hailo raw det: could not decode any scale. nc=%d reg_max=%d. "
                "Output shapes: %s. If your model has a different number of "
                "classes, provide the correct labels file or pass "
                "model_kind='detection_raw' with matching labels.",
                nc, reg_max, all_shapes,
            )
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_class_ids)

    keep = _nms_numpy(boxes, scores, iou_thresh)
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    frame_w = orig_w if orig_w > 0 else input_w
    frame_h = orig_h if orig_h > 0 else input_h
    frame_area = max(frame_w * frame_h, 1)

    detections: list[Detection] = []
    for box_n, score, cid in zip(boxes, scores, class_ids):
        label = class_names.get(int(cid), str(cid))
        if classes and label not in classes:
            continue
        x1 = (box_n[0] * input_w - letterbox.pad_left) / letterbox.scale
        y1 = (box_n[1] * input_h - letterbox.pad_top) / letterbox.scale
        x2 = (box_n[2] * input_w - letterbox.pad_left) / letterbox.scale
        y2 = (box_n[3] * input_h - letterbox.pad_top) / letterbox.scale
        x1 = max(0.0, min(float(frame_w), x1))
        y1 = max(0.0, min(float(frame_h), y1))
        x2 = max(0.0, min(float(frame_w), x2))
        y2 = max(0.0, min(float(frame_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detections.append(Detection(
            label=label,
            confidence=float(score),
            bbox=bbox,
            area_ratio=bbox.area / frame_area,
        ))
    return detections


def _rows_from_per_class_iter(
    per_class_iter: list[Any],
) -> list[tuple[int, float, float, float, float, float]]:
    """Flatten ``list[num_classes][K, 5]`` NMS rows (variable *K* per class).

    Hailo's ``yolov8_nms_postprocess`` op emits per-detection rows in
    ``(y_min, x_min, y_max, x_max, score)`` order (TF/Keras convention,
    height-axis first). We swap to ``(x1, y1, x2, y2)`` on the way out
    so the rest of the pipeline uses the standard (x-first) layout.
    """
    out_pc: list[tuple[int, float, float, float, float, float]] = []
    for cid, arr in enumerate(per_class_iter):
        a = np.asarray(arr)
        if a.size == 0 or a.ndim != 2 or a.shape[-1] != 5:
            continue
        for row in a:
            y1, x1, y2, x2, score = row[:5]  # Hailo: y-first
            out_pc.append(
                (
                    cid,
                    float(score),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                )
            )
    return out_pc


def _flatten_nms_output(
    outputs: dict[str, np.ndarray] | list[Any] | Any,
) -> list[tuple[int, float, float, float, float, float]]:
    """Normalise a Hailo NMS output into a flat list of detection rows.

    Returns ``[(class_id, score, x1, y1, x2, y2), ...]`` with
    coordinates in normalized ``[0, 1]`` of the letterboxed input.
    Returns an empty list (and logs a warning) for layouts we don't
    recognise — including raw (non-NMS) feature-map outputs, which
    would require a CPU-side YOLO decoder we deliberately don't ship.
    """
    if isinstance(outputs, dict):
        if not outputs:
            return []
        raw = next(iter(outputs.values()))
    else:
        raw = outputs

    # ---- Flat NMS: [1, N, 6] or [N, 6] -----------------------------
    if isinstance(raw, np.ndarray) and raw.ndim in (2, 3) and raw.shape[-1] == 6:
        preds = raw[0] if raw.ndim == 3 else raw
        out: list[tuple[int, float, float, float, float, float]] = []
        for row in preds:
            x1, y1, x2, y2, score, cls = row[:6]
            out.append(
                (int(cls), float(score), float(x1), float(y1), float(x2), float(y2))
            )
        return out

    # ---- Batched per-class NMS (batch=1) --------------------------------
    # HailoRT 4.23.0+ may return either:
    # * ``list[batch][ndarray(num_classes, K, 5)]`` — fixed *K* (yolov8s), or
    # * ``list[batch][list[num_classes][K, 5]]`` — variable *K* per class
    #   (yolov8m on Pi 5 + AI HAT+); ``np.asarray`` on the inner list fails
    #   with "inhomogeneous shape" so we must not coerce the whole batch slab.
    if isinstance(raw, list) and len(raw) >= 1:
        batch0 = raw[0]
        if isinstance(batch0, list):
            return _rows_from_per_class_iter(batch0)
        # HailoRT sometimes wraps the per-class list as a 1-D object
        # ndarray (length num_classes) instead of a bare Python list.
        if isinstance(batch0, np.ndarray) and batch0.dtype == object:
            return _rows_from_per_class_iter(list(batch0))
        if (
            isinstance(batch0, np.ndarray)
            and batch0.ndim == 3
            and batch0.shape[-1] == 5
        ):
            out_b: list[tuple[int, float, float, float, float, float]] = []
            for cid in range(batch0.shape[0]):
                per_class = batch0[cid]
                if per_class.size == 0:
                    continue
                for row in per_class:
                    y1, x1, y2, x2, score = row[:5]  # Hailo: y-first
                    out_b.append(
                        (
                            cid,
                            float(score),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        )
                    )
            return out_b

    # ---- Per-class NMS: outer axis = class_id, inner shape [K, 5] ----
    # The HailoRT binding sometimes returns this as a list and
    # sometimes as a ragged object array; handle both.
    per_class_iter: list[Any] | None = None
    if isinstance(raw, list):
        per_class_iter = raw
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        per_class_iter = list(raw)
    elif (
        isinstance(raw, np.ndarray)
        and raw.ndim == 4
        and raw.shape[-1] == 5
        and raw.shape[0] == 1
    ):
        # Shape (1, num_classes, max_per_class, 5)
        per_class_iter = list(raw[0])

    if per_class_iter is not None:
        return _rows_from_per_class_iter(per_class_iter)

    shape_str = getattr(raw, "shape", type(raw).__name__)
    logger.warning(
        "Hailo output shape %s is not a recognised NMS layout. The HEF "
        "may have been compiled without on-chip NMS; CPU YOLO decode is "
        "not implemented in this backend.",
        shape_str,
    )
    return []
