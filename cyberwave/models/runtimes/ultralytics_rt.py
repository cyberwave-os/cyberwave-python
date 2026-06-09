"""Ultralytics (YOLOv8 / YOLOv11) runtime adapter.

Supports all Ultralytics task types:

* **detect** — :class:`DetectionResult` with plain :class:`Detection` objects.
* **segment** — each :class:`Detection` carries a :class:`Mask` in ``.mask``.
* **pose** — each :class:`Detection` carries a :class:`KeypointSet` in
  ``.keypoint_set`` (raw numpy array also kept in ``.keypoints`` for
  backward compatibility).
* **obb** — each :class:`Detection` carries an :class:`OrientedBoundingBox`
  in ``.obb``; ``.bbox`` is the tightest axis-aligned bounding box.
* **classify** — :class:`ClassificationResult` (top-K class scores).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePath
from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import (
    BoundingBox,
    ClassificationCandidate,
    ClassificationResult,
    Detection,
    DetectionResult,
    InstanceSegmentationResult,
    KeypointSet,
    Mask,
    OBBResult,
    OrientedBoundingBox,
    PoseResult,
    PredictionResult,
)

_logger = logging.getLogger("cyberwave.models.runtimes.ultralytics")


class UltralyticsRuntime(ModelRuntime):
    """Runtime backend for Ultralytics YOLO models."""

    name = "ultralytics"

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401

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
        from ultralytics import YOLO

        p = Path(model_path)

        # Defensive guard for the ``IsADirectoryError`` wedge: if
        # ``p`` is a directory we must NOT forward it to ``YOLO()``
        # because Ultralytics would then call ``torch.load(<dir>)`` and
        # crash on every worker start. The SDK's
        # ``ModelManager._resolve_model_path`` is the authoritative
        # place that decides whether a staging directory is orphan
        # cruft (prune + auto-download) or operator-staged content
        # (raise with an actionable error). Reaching this branch means
        # a caller bypassed the manager and handed us a raw directory
        # path — surface a clear error rather than silently destroying
        # whatever the operator put there.
        if p.is_dir():
            try:
                contents = sorted(item.name for item in p.iterdir())
            except OSError:
                contents = []
            raise FileNotFoundError(
                f"UltralyticsRuntime.load() received a directory at "
                f"{p}, not a weight file. This is usually an orphan "
                f"staging directory left by a previously failed Edge "
                f"Core download. Resolve it via "
                f"``ModelManager.load(model_id)`` (the SDK manager "
                f"handles this case) or manually remove the directory "
                f"if it only contains stale partial-download cruft. "
                f"Found inside: {contents}."
            )

        # Ultralytics resolves missing weights against CWD; chdir into a
        # writable dir so auto-downloads don't land in an unwritable
        # container WORKDIR. The same dir is stashed on the handle for
        # ``_apply_text_prompt`` to reuse for the lazy MobileCLIP text
        # encoder download triggered by open-vocab heads (YOLOE,
        # YOLO-World) at the first ``predict(prompt=...)`` call.
        # ``os.chdir`` is process-global — load() is a one-time startup
        # op so the race is acceptable.
        writable_dir = self._writable_model_dir(p)
        needs_primary_download = not p.is_file()
        old_cwd = os.getcwd() if needs_primary_download else None
        try:
            if needs_primary_download:
                os.chdir(writable_dir)
            model = YOLO(p.name if needs_primary_download else model_path)
        finally:
            if old_cwd is not None:
                os.chdir(old_cwd)

        model._cw_writable_dir = str(writable_dir)  # type: ignore[assignment]

        if device:
            # ``model.to(device)`` raises ``TypeError`` for any non-PyTorch
            # backend (ONNX, TensorRT, OpenVINO, …) loaded through the
            # Ultralytics ``YOLO`` wrapper — those formats are inference-
            # only and pin the device at export time. We still want
            # ``cw.models.load('foo.onnx', runtime='ultralytics')`` to
            # succeed (Ultralytics provides letterboxing + NMS that the
            # raw onnxruntime adapter does not), so swallow the format
            # mismatch and rely on the per-call ``device=`` kwarg in
            # ``predict()`` instead.
            try:
                model.to(device)
            except TypeError:
                pass
        return model

    @staticmethod
    def _normalize_prompt(
        prompt: str | list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        """Coerce a prompt input into a clean, stripped list of class strings.

        Accepts three shapes:

        - ``None`` → ``[]`` (caller treats as "no prompt configured").
        - ``str`` → **split on commas**. The workflow editor surfaces
          ``prompt`` as a single STRING input today, so the only way to
          author a multi-class YOLOE prompt from the inspector is to
          type ``"helmet, safety vest"`` — we parse that into the list
          form the open-vocab head wants. A bare ``"helmet"`` (no comma)
          becomes ``["helmet"]``.
        - ``list`` / ``tuple`` → already a class list (e.g. an upstream
          node whose output is a sequence of strings).

        In every case the result is **stripped** and empty entries are
        dropped. This keeps the per-handle cache key stable
        (``" helmet "`` ≡ ``"helmet"`` ≡ ``"helmet, "``) and prevents a
        default-empty inspector field from triggering a spurious head
        reset on every frame.
        """
        if prompt is None:
            return []
        if isinstance(prompt, str):
            parts: list[str] = prompt.split(",")
        elif isinstance(prompt, list | tuple):
            parts = [str(p) for p in prompt]
        else:
            return []
        return [p.strip() for p in parts if isinstance(p, str) and p.strip()]

    @staticmethod
    def _apply_text_prompt(
        model_handle: Any,
        prompt: str | list[str] | tuple[str, ...] | None,
    ) -> None:
        """Configure an open-vocab YOLO head from a text prompt, with caching.

        Single string (``"helmet"``), comma-separated string
        (``"helmet, safety vest"``), or list/tuple of strings — all
        normalize to a clean class list via :meth:`_normalize_prompt`.

        Cache strategy: the **normalized** prompt tuple is stored on
        the model handle. Subsequent calls with the same logical
        prompts (regardless of whitespace / single-string vs list
        encoding) skip the (cheap but not free) re-parameterization.

        Failure modes:

        - Handle lacks ``set_classes`` / ``get_text_pe`` (closed-set
          YOLOv8 / classifier nets): silently skip. The backend
          compile-time gate (``text_prompt_unsupported``) already
          rejects this case at workflow sync time, so reaching here
          means a hand-crafted API caller — nothing to do.
        - ``clip`` (or another optional dep) is not importable in the
          worker image: ``ultralytics.nn.text_model`` does a lazy
          ``import clip`` inside ``get_text_pe``, so the failure
          surfaces here as ``ModuleNotFoundError`` rather than at
          ``load`` time. Edge worker images are supposed to bundle
          ``ultralytics/CLIP`` (see ``edge-ml-worker/Dockerfile``);
          this branch only fires on stripped-down or hand-built
          worker images where that bake step was skipped. Log a
          warning with the missing module name and keep the previous
          class set so the worker keeps emitting detections instead
          of crashing the predict loop on every frame.
        - The Ultralytics call itself raises (bad tokenizer state,
          OOM during text encoding, GPU disconnect): log a **warning**
          and continue. The previous class set stays active so the
          worker keeps producing detections instead of crashing — but
          the operator gets a loud signal that the new prompt isn't
          live. Silently no-op'ing here was the original behaviour and
          was a footgun ("I changed the prompt and nothing happened").
        """
        prompts = UltralyticsRuntime._normalize_prompt(prompt)
        if not prompts:
            return
        if not hasattr(model_handle, "set_classes") or not hasattr(
            model_handle, "get_text_pe"
        ):
            return
        key = tuple(prompts)
        if getattr(model_handle, "_cw_active_prompt", None) == key:
            return
        # ``get_text_pe`` triggers Ultralytics' lazy MobileCLIP download
        # which writes to a bare filename → CWD. Sandbox it to the dir
        # stashed by :meth:`load` so the write lands somewhere writable.
        # Narrowed to str/PurePath rather than os.PathLike because
        # MagicMock implements __fspath__ by default.
        writable_dir = getattr(model_handle, "_cw_writable_dir", None)
        if not isinstance(writable_dir, (str, PurePath)):
            writable_dir = None
        old_cwd = os.getcwd() if writable_dir else None
        try:
            try:
                if writable_dir:
                    os.chdir(writable_dir)
                model_handle.set_classes(prompts, model_handle.get_text_pe(prompts))
            finally:
                if old_cwd is not None:
                    os.chdir(old_cwd)
        except ModuleNotFoundError as exc:
            _logger.warning(
                "Failed to apply YOLOE text prompt %s because %r is not "
                "installed in this worker image. Open-vocab YOLO heads "
                "(YOLOE text/visual, YOLO-World) need the 'ultralytics/CLIP' "
                "fork to be importable for `get_text_pe`; rebuild the edge "
                "worker image so CLIP is bundled, or switch this workflow "
                "to a prompt-free model variant (e.g. yoloe-26*-seg-pf). "
                "Previous class set (%s) remains active.",
                prompts,
                exc.name,
                getattr(model_handle, "_cw_active_prompt", None),
            )
            return
        except (AttributeError, RuntimeError) as exc:
            _logger.warning(
                "Failed to apply YOLOE text prompt %s; previous class set "
                "(%s) remains active. Underlying error: %s",
                prompts,
                getattr(model_handle, "_cw_active_prompt", None),
                exc,
            )
            return
        model_handle._cw_active_prompt = key

    @staticmethod
    def _writable_model_dir(model_path: Path) -> Path:
        """Find a writable directory for Ultralytics auto-downloads."""
        candidates = []

        if model_path.is_absolute() and model_path.parent != Path("/"):
            candidates.append(model_path.parent)

        env_dir = os.environ.get("CYBERWAVE_MODELS_DIR") or os.environ.get(
            "CYBERWAVE_MODEL_DIR"
        )
        if env_dir:
            candidates.append(Path(env_dir))

        candidates.extend([Path("/app/models"), Path.home() / ".cyberwave" / "models"])

        for d in candidates:
            try:
                d.mkdir(parents=True, exist_ok=True)
                if os.access(d, os.W_OK):
                    return d
            except OSError:
                continue

        return Path("/tmp")

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        prompt: str | list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        # Open-vocab YOLO heads (YOLOE / YOLO-World) are steered by a text
        # prompt re-parameterized into the classification head via
        # ``model.set_classes(prompts, model.get_text_pe(prompts))``. The
        # closed-set YOLOv8/YOLOv11 models don't expose either method, so
        # the ``hasattr`` guards inside :meth:`_apply_text_prompt` keep
        # the legacy path untouched.
        #
        # Authoring contract: a single string is split on commas, so
        # ``prompt="helmet, safety vest"`` and ``prompt=["helmet",
        # "safety vest"]`` are equivalent. ``set_classes`` is cheap but
        # not free, so the normalized prompt tuple is cached on the
        # handle and re-parameterization is skipped when the next call
        # uses the same prompts — the common case for an edge worker
        # chewing through 10–30 frames per second.
        self._apply_text_prompt(model_handle, prompt)
        results = model_handle(input_data, conf=confidence, verbose=False)
        detections: list[Detection] = []
        class_result: ClassificationResult | None = None
        # Resolved during the loop; last result wins (all frames share one task).
        task = "detect"

        for result in results:
            frame_h, frame_w = result.orig_shape or (0, 0)
            frame_area = frame_h * frame_w

            if getattr(result, "probs", None) is not None:
                task = "classify"
                class_result = _parse_classification(result)
                continue

            if getattr(result, "obb", None) is not None:
                task = "obb"
                detections.extend(_parse_obb(result, frame_area, classes))
                continue

            if result.boxes is None:
                continue

            # Detect / segment / pose — all share result.boxes
            names = _names_dict(result)
            kp_data = _tensor_attr(result, "keypoints", "data")
            mask_data = _tensor_attr(result, "masks", "data")

            if kp_data is not None:
                task = "pose"
            elif mask_data is not None:
                task = "segment"

            for i, box in enumerate(result.boxes):
                label = names.get(_box_cls(box), str(_box_cls(box)))
                if classes and label not in classes:
                    continue

                bbox = _box_bbox(box)

                msk: Mask | None = None
                if mask_data is not None and i < len(mask_data):
                    msk = Mask(data=mask_data[i], h=frame_h, w=frame_w)

                raw_kps = None
                kp_set: KeypointSet | None = None
                if kp_data is not None and i < len(kp_data):
                    raw_kps = kp_data[i]
                    kp_set = KeypointSet.from_array(raw_kps)

                detections.append(
                    Detection(
                        label=label,
                        confidence=_box_conf(box),
                        bbox=bbox,
                        area_ratio=bbox.area / frame_area if frame_area else 0.0,
                        mask=msk,
                        keypoints=raw_kps,
                        keypoint_set=kp_set,
                    )
                )

        # Return the most specific result type matching the detected task.
        if task == "classify":
            return ClassificationResult(top=class_result.top, raw=results)
        if task == "obb":
            return OBBResult(detections, raw=results)
        if task == "pose":
            return PoseResult(detections, raw=results)
        if task == "segment":
            return InstanceSegmentationResult(detections, raw=results)
        return DetectionResult(detections, raw=results)


# ---------------------------------------------------------------------------
# Module-level helpers (same convention as onnxruntime_rt.py)
# ---------------------------------------------------------------------------


def _tensor_attr(obj: Any, attr: str, sub: str) -> Any | None:
    """Return ``getattr(obj, attr).<sub>.cpu().numpy()``, or ``None``."""
    container = getattr(obj, attr, None)
    if container is None:
        return None
    try:
        return getattr(container, sub).cpu().numpy()
    except AttributeError:
        return None


def _names_dict(result: Any) -> dict[int, str]:
    """Normalize ``result.names`` to ``dict[int, str]`` across YOLO versions."""
    raw = getattr(result, "names", None) or {}
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def _box_cls(box: Any) -> int:
    """Extract class index from a box tensor — robust to v8/v11/v26 layouts."""
    try:
        return int(box.cls[0])
    except (IndexError, TypeError):
        # Scalar tensor (no batch dim) introduced in some export variants
        return int(box.cls)


def _box_conf(box: Any) -> float:
    """Extract confidence score from a box tensor."""
    try:
        return float(box.conf[0])
    except (IndexError, TypeError):
        return float(box.conf)


def _box_bbox(box: Any) -> BoundingBox:
    """Extract axis-aligned bbox from a box tensor as a :class:`BoundingBox`."""
    try:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
    except (IndexError, TypeError):
        # Newer versions may expose xyxy already as a 1-D tensor
        try:
            x1, y1, x2, y2 = box.xyxy.tolist()
        except TypeError:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy]
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def _parse_classification(result: Any) -> ClassificationResult:
    probs = result.probs
    names = _names_dict(result)
    top_indices: list[int] = []
    top_confs: list[float] = []
    try:
        top_indices = [int(i) for i in probs.top5]
        top_confs = [float(c) for c in probs.top5conf]
    except (AttributeError, TypeError):
        try:
            top_indices = [int(probs.top1)]
            top_confs = [float(probs.top1conf)]
        except (AttributeError, TypeError):
            pass
    candidates = [
        ClassificationCandidate(
            label=names.get(idx, str(idx)),
            confidence=min(max(float(conf), 0.0), 1.0),
            index=idx,
        )
        for idx, conf in zip(top_indices, top_confs)
    ]
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return ClassificationResult(top=candidates)


def _parse_obb(
    result: Any,
    frame_area: int,
    classes: list[str] | None,
) -> list[Detection]:
    obb = result.obb
    names = _names_dict(result)
    detections: list[Detection] = []
    try:
        # ``xywhr`` is the canonical Ultralytics name (r = angle in radians).
        # Older checkpoints may expose ``xywha`` instead.
        # Use explicit `is None` — never `or` on a tensor (ambiguous bool).
        _xywhr = getattr(obb, "xywhr", None)
        xywhr = (_xywhr if _xywhr is not None else obb.xywha).cpu().numpy()
        confs = obb.conf.cpu().numpy()
        clss = obb.cls.cpu().numpy()
        xyxy = obb.xyxy.cpu().numpy()  # (N, 4) axis-aligned bbox
    except AttributeError:
        return detections
    for i in range(len(xywhr)):
        label = names.get(int(clss[i]), str(int(clss[i])))
        if classes and label not in classes:
            continue
        cx, cy, w, h, angle = (float(v) for v in xywhr[i])
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        aabb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detections.append(
            Detection(
                label=label,
                confidence=float(confs[i]),
                bbox=aabb,
                area_ratio=aabb.area / frame_area if frame_area else 0.0,
                obb=OrientedBoundingBox(cx=cx, cy=cy, w=w, h=h, angle_rad=angle),
            )
        )
    return detections
