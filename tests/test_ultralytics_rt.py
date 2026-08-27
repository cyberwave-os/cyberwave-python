"""Tests for cyberwave.models.runtimes.ultralytics_rt — Ultralytics backend."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime
from cyberwave.models.types import DepthResult, PredictionResult
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestUltralyticsRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = UltralyticsRuntime


def _box_mock(xyxy: list[float], cls: int, conf: float) -> MagicMock:
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = xyxy
    box.cls = [cls]
    box.conf = [conf]
    return box


def _result_mock(
    *,
    boxes: list[MagicMock],
    names: dict[int, str],
    orig_shape: tuple[int, int] = (480, 640),
    keypoints_data: np.ndarray | None = None,
) -> MagicMock:
    result = MagicMock()
    result.orig_shape = orig_shape
    result.boxes = boxes
    result.names = names
    # Explicit None for every task-dispatch attribute so the runtime does not
    # mistake a MagicMock truthy value for a real probs/obb/masks/depth object.
    result.probs = None
    result.obb = None
    result.masks = None
    result.depth = None
    if keypoints_data is None:
        result.keypoints = None
    else:
        result.keypoints = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.cpu.return_value.numpy.return_value = keypoints_data
        result.keypoints.data = tensor_mock
    return result


class TestUltralyticsPredictDetection:
    def test_returns_prediction_result(self):
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=1, conf=0.8),
            ],
            names={0: "person", 1: "car"},
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), confidence=0.5
        )

        assert isinstance(pred, PredictionResult)
        assert len(pred.detections) == 2
        labels = [d.label for d in pred.detections]
        assert labels == ["person", "car"]
        # No pose data → keypoints should be None.
        assert all(d.keypoints is None for d in pred.detections)

    def test_filters_by_class(self):
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=1, conf=0.8),
            ],
            names={0: "person", 1: "car"},
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), classes=["person"]
        )

        assert len(pred.detections) == 1
        assert pred.detections[0].label == "person"


class TestUltralyticsPredictPose:
    def test_keypoints_attached_to_detections(self):
        rt = UltralyticsRuntime()
        # 2 detections × 17 keypoints × 3 (x, y, vis).
        kp = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=0, conf=0.8),
            ],
            names={0: "person"},
            keypoints_data=kp,
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(pred.detections) == 2
        assert pred.detections[0].keypoints is not None
        assert pred.detections[0].keypoints.shape == (17, 3)
        np.testing.assert_array_equal(pred.detections[0].keypoints, kp[0])
        np.testing.assert_array_equal(pred.detections[1].keypoints, kp[1])

    def test_keypoints_omitted_when_class_filter_skips_box(self):
        # Two pose detections; user filters to "dog" only — none survive.
        rt = UltralyticsRuntime()
        kp = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=0, conf=0.8),
            ],
            names={0: "person"},
            keypoints_data=kp,
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), classes=["dog"]
        )
        assert pred.detections == []

    def test_handles_missing_keypoints_gracefully(self):
        # `result.keypoints.data` raises AttributeError → we return no keypoints.
        rt = UltralyticsRuntime()

        class _NoData:
            """Stand-in for a result.keypoints object without ``.data``."""

        result = MagicMock()
        result.orig_shape = (480, 640)
        result.boxes = [_box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9)]
        result.names = {0: "person"}
        result.probs = None
        result.obb = None
        result.masks = None
        result.depth = None
        result.keypoints = _NoData()
        model = MagicMock(return_value=[result])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(pred.detections) == 1
        assert pred.detections[0].keypoints is None


class TestUltralyticsYoloePrompt:
    """``predict(prompt=...)`` reconfigures the open-vocab head once and caches.

    Mirrors the Ultralytics YOLOE / YOLO-World API where the
    classification head is re-parameterized from a text embedding via
    ``model.set_classes(prompts, model.get_text_pe(prompts))``.

    The cache is keyed by the prompt tuple so repeated calls at 10-30
    fps with a constant prompt skip the (cheap but not free)
    re-parameterization. Changing the prompt triggers another call.
    """

    @staticmethod
    def _yoloe_handle(result_obj: MagicMock) -> MagicMock:
        handle = MagicMock(return_value=[result_obj])
        # Default MagicMock would give truthy hasattr for *anything* —
        # set the open-vocab API explicitly so the runtime sees them and
        # leave _cw_active_prompt unset so the cache misses on the first
        # call.
        handle.set_classes = MagicMock()
        handle.get_text_pe = MagicMock(return_value="text_pe_tensor")
        del handle._cw_active_prompt
        del handle._cw_writable_dir  # opt out of the chdir sandbox, covered separately
        # ``spec=[]`` blocks MagicMock's auto-attribute creation so the
        # prompt-free probe (``hasattr(layer, "lrpc")``) stays False by
        # default — prompt-free tests reassign ``handle.model`` with an
        # ``lrpc``-bearing layer to opt in.
        handle.model = [MagicMock(spec=[])]
        return handle

    @staticmethod
    def _single_box_result() -> MagicMock:
        return _result_mock(
            boxes=[_box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9)],
            names={0: "helmet"},
        )

    def test_string_prompt_calls_set_classes_once_per_unique_prompt(self):
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")

        handle.set_classes.assert_called_once_with(["helmet"], "text_pe_tensor")
        handle.get_text_pe.assert_called_once_with(["helmet"])

    def test_list_prompt_passes_through_as_list(self):
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt=["helmet", "safety vest"],
        )

        handle.set_classes.assert_called_once_with(
            ["helmet", "safety vest"], "text_pe_tensor"
        )

    def test_prompt_change_re_parameterizes_head(self):
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
        rt.predict(
            handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="safety vest"
        )

        assert handle.set_classes.call_count == 2
        assert handle.set_classes.call_args_list[0][0][0] == ["helmet"]
        assert handle.set_classes.call_args_list[1][0][0] == ["safety vest"]

    def test_none_prompt_does_not_touch_set_classes(self):
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8))

        handle.set_classes.assert_not_called()
        handle.get_text_pe.assert_not_called()

    def test_blank_prompt_strings_are_ignored(self):
        # An editor default of "" should not trigger a head reset on every
        # frame just because the field is wired but empty.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="")
        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt=["", "   "],
        )

        handle.set_classes.assert_not_called()

    def test_closed_set_yolo_handle_without_set_classes_is_ignored(self):
        # Plain YOLOv8: no set_classes attribute → prompt is silently
        # ignored. The runtime must not crash.
        rt = UltralyticsRuntime()
        result_obj = self._single_box_result()
        handle = MagicMock(return_value=[result_obj], spec=["__call__"])

        pred = rt.predict(
            handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet"
        )

        assert isinstance(pred, PredictionResult)
        assert not hasattr(handle, "set_classes")

    def test_comma_separated_string_is_split_into_class_list(self):
        # Editor surfaces ``prompt`` as a single STRING input; the only
        # way to author a multi-class YOLOE prompt without dropping into
        # the SDK is to type a comma-separated value. The runtime must
        # split it so the open-vocab head sees real classes.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt="helmet, safety vest, hard hat",
        )

        handle.set_classes.assert_called_once_with(
            ["helmet", "safety vest", "hard hat"], "text_pe_tensor"
        )

    def test_whitespace_only_differences_do_not_thrash_the_cache(self):
        # Operators copy-pasting or hand-editing prompts often leave
        # stray spaces. Three encodings of the same logical prompt
        # ("helmet" + "safety vest") must share one cache slot — i.e.
        # set_classes is called exactly once.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())

        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt="helmet, safety vest",
        )
        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt="  helmet ,safety vest  ",
        )
        rt.predict(
            handle,
            np.zeros((480, 640, 3), dtype=np.uint8),
            prompt=["helmet", " safety vest "],
        )

        handle.set_classes.assert_called_once_with(
            ["helmet", "safety vest"], "text_pe_tensor"
        )

    def test_prompt_free_yoloe_skips_set_classes_with_warning(self, caplog):
        # Prompt-free YOLOE (``*-seg-pf``) exposes ``set_classes`` but
        # asserts on ``lrpc`` inside it — proactive detection must skip
        # the call instead of letting it raise on every frame.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())
        head_layer = MagicMock()
        head_layer.lrpc = MagicMock()
        handle.model = [MagicMock(), MagicMock(), head_layer]

        with caplog.at_level("WARNING", logger="cyberwave.models.runtimes.ultralytics"):
            pred = rt.predict(
                handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet"
            )

        handle.set_classes.assert_not_called()
        handle.get_text_pe.assert_not_called()
        assert isinstance(pred, PredictionResult)

        warning_records = [r for r in caplog.records if "prompt-free" in r.getMessage()]
        assert warning_records, caplog.records
        msg = warning_records[0].getMessage()
        assert "helmet" in msg
        assert "clear the prompt" in msg.lower() or "Clear the prompt" in msg
        assert "yoloe-26" in msg

    def test_prompt_free_yoloe_warns_once_per_unique_prompt(self, caplog):
        # The skip path must update ``_cw_active_prompt`` so the cache
        # short-circuits subsequent same-prompt frames — at 30 fps a
        # warning per frame would drown the log.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())
        head_layer = MagicMock()
        head_layer.lrpc = MagicMock()
        handle.model = [head_layer]

        with caplog.at_level("WARNING", logger="cyberwave.models.runtimes.ultralytics"):
            rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
            rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
            rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")

        warning_records = [r for r in caplog.records if "prompt-free" in r.getMessage()]
        assert len(warning_records) == 1, (
            f"Expected exactly one prompt-free warning, got "
            f"{len(warning_records)}: {[r.getMessage() for r in warning_records]}"
        )

    def test_assertion_error_from_set_classes_is_caught(self, caplog):
        # Defensive fallback when the proactive ``lrpc`` probe misses a
        # variant — AssertionError must degrade to a warning, not a
        # per-frame crash.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())
        handle.set_classes.side_effect = AssertionError(
            "Prompt-free model does not support setting classes."
        )

        with caplog.at_level("WARNING", logger="cyberwave.models.runtimes.ultralytics"):
            pred = rt.predict(
                handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet"
            )

        assert isinstance(pred, PredictionResult)
        assert any(
            "Failed to apply YOLOE text prompt" in r.getMessage()
            for r in caplog.records
        ), caplog.records

    @pytest.mark.parametrize(
        "build_handle",
        [
            # Each case targets one defensive exit in _is_prompt_free_yoloe.
            pytest.param(lambda h: setattr(h, "model", None), id="model_is_none"),
            pytest.param(lambda h: setattr(h, "model", []), id="empty_list_indexerror"),
            pytest.param(lambda h: setattr(h, "model", 42), id="int_not_subscriptable"),
            pytest.param(
                lambda h: setattr(h, "model", {0: MagicMock(spec=[])}),
                id="dict_missing_minus_one_key",
            ),
        ],
    )
    def test_is_prompt_free_yoloe_returns_false_on_malformed_handles(
        self, build_handle
    ) -> None:
        handle = MagicMock()
        build_handle(handle)
        assert UltralyticsRuntime._is_prompt_free_yoloe(handle) is False

    def test_is_prompt_free_yoloe_true_only_when_lrpc_present(self) -> None:
        handle = MagicMock()
        # Without lrpc → prompt-driven.
        handle.model = [MagicMock(spec=[])]
        assert UltralyticsRuntime._is_prompt_free_yoloe(handle) is False
        # With lrpc on the *last* layer → prompt-free.
        head = MagicMock()
        head.lrpc = MagicMock()
        handle.model = [MagicMock(spec=[]), head]
        assert UltralyticsRuntime._is_prompt_free_yoloe(handle) is True

    def test_set_classes_failure_logs_warning_and_does_not_cache(self, caplog):
        # If Ultralytics raises (bad tokenizer state, OOM during text
        # encoding, GPU disconnect), the runtime must NOT silently
        # swallow the failure: the operator needs a loud signal that
        # their new prompt isn't live. The previous class set stays
        # active and we must not poison the cache, so the next predict()
        # call retries set_classes.
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())
        handle.set_classes.side_effect = RuntimeError("text encoder OOM")

        with caplog.at_level("WARNING", logger="cyberwave.models.runtimes.ultralytics"):
            rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")

        assert any(
            "Failed to apply YOLOE text prompt" in rec.getMessage()
            for rec in caplog.records
        ), caplog.records
        # No cache poisoning: a follow-up call must attempt set_classes again.
        handle.set_classes.side_effect = None
        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
        assert handle.set_classes.call_count == 2

    def test_missing_clip_module_logs_warning_and_does_not_cache(self, caplog):
        # ``ultralytics.nn.text_model`` lazy-imports ``clip`` inside
        # ``get_text_pe``, so a worker image that forgot to bundle
        # ``ultralytics/CLIP`` only fails on the first prompted frame —
        # not at model load. Without explicit handling, the
        # ``ModuleNotFoundError`` propagates out of every ``predict()``
        # call and the worker either crash-loops or burns CPU on a
        # detection hook that never produces output. We want a single
        # loud warning naming the missing module + the recovery path,
        # and the cache must NOT be poisoned so a follow-up call still
        # re-attempts ``set_classes`` (useful for tests that patch CLIP
        # in mid-flight, and for parity with the generic-failure
        # branch below).
        rt = UltralyticsRuntime()
        handle = self._yoloe_handle(self._single_box_result())
        handle.get_text_pe.side_effect = ModuleNotFoundError(
            "No module named 'clip'", name="clip"
        )

        with caplog.at_level("WARNING", logger="cyberwave.models.runtimes.ultralytics"):
            rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")

        matching = [
            rec
            for rec in caplog.records
            if "Failed to apply YOLOE text prompt" in rec.getMessage()
            and "'clip'" in rec.getMessage()
        ]
        assert matching, caplog.records
        # The recovery hint must point at the worker-image rebuild path,
        # not at "pip install clip" (the runtime has no pip).
        assert "edge worker image" in matching[0].getMessage()

        # Recovery: ``get_text_pe`` is evaluated first as the second
        # argument to ``set_classes``, so the failing-import call never
        # reached ``set_classes`` (call_count == 0). Once CLIP is back,
        # the next predict should actually invoke ``set_classes`` for
        # the first time — confirming the cache wasn't poisoned with a
        # success marker for a prompt that never made it through.
        assert handle.set_classes.call_count == 0
        handle.get_text_pe.side_effect = None
        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), prompt="helmet")
        assert handle.set_classes.call_count == 1
        handle.set_classes.assert_called_with(["helmet"], "text_pe_tensor")


class TestUltralyticsAvailable:
    def test_is_available_returns_bool(self):
        assert isinstance(UltralyticsRuntime().is_available(), bool)


class TestUltralyticsLoadDeviceCompat:
    """``load()`` must tolerate the ``TypeError`` Ultralytics raises from
    ``model.to(device)`` for non-PyTorch backends (ONNX, TensorRT, …).

    The wrapped model still produces predictions through Ultralytics'
    own ``__call__`` path — losing only the early device move — so the
    SDK should swallow the format-mismatch error and return the handle
    instead of crashing the worker at module import time.
    """

    def _install_fake_ultralytics(
        self, monkeypatch: pytest.MonkeyPatch, *, yolo_factory
    ) -> None:
        """Install a stub ``ultralytics`` module exposing *yolo_factory* as ``YOLO``."""
        fake = types.ModuleType("ultralytics")
        fake.YOLO = yolo_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake)

    def test_load_swallows_typeerror_from_to_for_onnx_handle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model_path = tmp_path / "yolov8n.onnx"
        model_path.write_bytes(
            b""
        )  # exists() must be True so load() skips chdir-download

        handle = MagicMock(name="onnx_yolo_handle")
        handle.to.side_effect = TypeError(
            "model='yolov8n.onnx' should be a *.pt PyTorch model"
        )
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        returned = rt.load(str(model_path), device="cpu")

        assert returned is handle
        handle.to.assert_called_once_with("cpu")

    def test_load_still_calls_to_for_pt_handle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The compatibility shim must not regress the happy path: a real
        PyTorch handle should still receive ``model.to(device)``."""
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_bytes(b"")

        handle = MagicMock(name="pt_yolo_handle")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        returned = rt.load(str(model_path), device="cpu")

        assert returned is handle
        handle.to.assert_called_once_with("cpu")

    def test_load_does_not_swallow_unrelated_errors_from_to(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the ``TypeError`` raised by Ultralytics' format guard is
        absorbed — any other failure (e.g. CUDA OOM, invalid device string)
        must surface so the worker can log a real diagnostic instead of
        silently running on the wrong device."""
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_bytes(b"")

        handle = MagicMock(name="pt_yolo_handle")
        handle.to.side_effect = RuntimeError("CUDA out of memory")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            rt.load(str(model_path), device="cuda:0")

    def test_load_stashes_device_on_handle_after_successful_to(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``predict()`` re-asserts device on every call by reading this
        stashed value — without it, the fix in ``predict()`` has nothing
        to forward and the lazily-built Ultralytics predictor silently
        falls back to auto-detected device instead of ``model.to()``'s."""
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_bytes(b"")

        handle = MagicMock(name="pt_yolo_handle")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        rt.load(str(model_path), device="cuda:0")

        assert handle._cw_device == "cuda:0"

    def test_load_does_not_stash_device_when_to_raises_typeerror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """For non-PyTorch backends where ``.to()`` is a no-op format
        mismatch, nothing should be stashed — ``predict()`` must fall
        back to Ultralytics' own device handling for those formats."""
        model_path = tmp_path / "yolov8n.onnx"
        model_path.write_bytes(b"")

        handle = MagicMock(name="onnx_yolo_handle")
        handle.to.side_effect = TypeError(
            "model='yolov8n.onnx' should be a *.pt PyTorch model"
        )
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        rt.load(str(model_path), device="cpu")

        # Check ``__dict__`` directly rather than ``hasattr``/``getattr`` —
        # either of those would trigger MagicMock's auto-vivification and
        # falsely "find" an attribute that was never actually set.
        assert "_cw_device" not in handle.__dict__


class TestUltralyticsPredictDevice:
    """Ultralytics builds its ``Predictor`` lazily on the *first*
    inference call and resolves ``device`` from that call's own
    ``overrides``, not from wherever ``model.to(device)`` moved the
    weights at ``load()`` time. ``predict()`` must therefore re-assert
    the device stashed by ``load()`` (or an explicit per-call override)
    on every single call — otherwise a GPU-loaded model silently runs
    its first inference (and predictor-caching) pass on CPU.
    """

    @staticmethod
    def _handle(return_value: list) -> MagicMock:
        handle = MagicMock(name="yolo_handle", return_value=return_value)
        del handle._cw_device  # undo MagicMock auto-vivification
        return handle

    def test_predict_forwards_device_stashed_by_load(self) -> None:
        rt = UltralyticsRuntime()
        handle = self._handle([])
        handle._cw_device = "cuda:0"

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8))

        _, kwargs = handle.call_args
        assert kwargs["device"] == "cuda:0"
        assert kwargs["conf"] == 0.5
        assert kwargs["verbose"] is False

    def test_predict_explicit_device_overrides_stashed_device(self) -> None:
        rt = UltralyticsRuntime()
        handle = self._handle([])
        handle._cw_device = "cuda:0"

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8), device="cpu")

        _, kwargs = handle.call_args
        assert kwargs["device"] == "cpu"

    def test_predict_omits_device_when_none_was_ever_stashed(self) -> None:
        rt = UltralyticsRuntime()
        handle = self._handle([])

        rt.predict(handle, np.zeros((480, 640, 3), dtype=np.uint8))

        _, kwargs = handle.call_args
        assert "device" not in kwargs


class TestUltralyticsLoadOrphanDirSelfHeal:
    """``load()`` must never feed an orphan staging directory to ``torch.load``.

    Regression for the ``IsADirectoryError`` wedge: a failed Edge Core
    download leaves the per-model directory behind on disk; if the SDK
    runtime ever forwards that path to ``YOLO()``, ``torch.load`` dies
    with ``IsADirectoryError`` on every worker start.

    The authoritative recovery lives in
    :class:`cyberwave.models.manager.ModelManager._resolve_model_path`
    (which prunes cruft-only orphans and raises actionable errors for
    operator-staged content). The runtime is the **defensive backstop**
    — if a caller bypasses the manager and hands us a raw directory
    path, we raise a clear ``FileNotFoundError`` rather than silently
    destroying operator data or letting Ultralytics crash later inside
    ``torch.load``.
    """

    def _install_fake_ultralytics(
        self, monkeypatch: pytest.MonkeyPatch, *, yolo_factory
    ) -> None:
        fake = types.ModuleType("ultralytics")
        fake.YOLO = yolo_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake)

    def test_directory_at_model_path_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Direct ``UltralyticsRuntime().load("/some/dir")`` raises with
        a message that names the offending path and points at the
        manager. YOLO is never called and the directory is preserved
        intact so any operator-staged content survives."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        orphan = models_dir / "yoloe-26m-seg.pt"
        orphan.mkdir()
        (orphan / "README.txt").write_text("hand-staged by operator")

        called = {"yolo": False}

        def fake_yolo(path: str) -> MagicMock:
            called["yolo"] = True
            return MagicMock(name="yolo_handle")

        self._install_fake_ultralytics(monkeypatch, yolo_factory=fake_yolo)

        rt = UltralyticsRuntime()
        with pytest.raises(FileNotFoundError) as excinfo:
            rt.load(str(orphan), device=None)
        msg = str(excinfo.value)
        assert str(orphan) in msg
        assert "ModelManager" in msg, (
            "error must direct callers at the manager which holds the "
            "authoritative recovery logic"
        )
        assert "README.txt" in msg, "error must surface the directory contents"
        assert called["yolo"] is False, "YOLO must not be called for a directory path"
        # Operator content survives — "human always wins".
        assert orphan.exists()
        assert (orphan / "README.txt").exists()

    def test_nonexistent_path_routes_through_download_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The expected post-manager-prune state: ``p`` does not exist
        as a file. The runtime chdirs into a writable model dir and
        invokes ``YOLO(p.name)`` so the hub client can fetch the
        weights. Verifies the ``not p.is_file()`` gate (vs the old
        ``not p.exists()``) still routes correctly when ``p`` simply
        does not exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        target = models_dir / "yoloe-26m-seg.pt"  # does not exist

        captured: dict[str, object] = {}

        def fake_yolo(path: str) -> MagicMock:
            captured["yolo_path"] = path
            captured["cwd_at_call"] = Path.cwd()
            return MagicMock(name="yolo_handle")

        self._install_fake_ultralytics(monkeypatch, yolo_factory=fake_yolo)

        rt = UltralyticsRuntime()
        cwd_before = Path.cwd()
        try:
            rt.load(str(target), device=None)
        finally:
            os.chdir(cwd_before)

        assert captured["yolo_path"] == "yoloe-26m-seg.pt"
        assert captured["cwd_at_call"] == models_dir
        assert Path.cwd() == cwd_before


class TestUltralyticsWritableModelDirContract:
    """``load()`` stashes a writable dir on the handle so
    ``_apply_text_prompt`` can sandbox Ultralytics' lazy MobileCLIP
    download into a writable location instead of the worker's
    read-only WORKDIR.
    """

    @staticmethod
    def _install_fake_ultralytics(
        monkeypatch: pytest.MonkeyPatch, *, yolo_factory
    ) -> None:
        fake = types.ModuleType("ultralytics")
        fake.YOLO = yolo_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake)

    def test_load_stashes_writable_dir_on_handle_for_existing_weight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        weight = models_dir / "yoloe-26m-seg-pf.onnx"
        weight.write_bytes(b"")

        handle = MagicMock(name="yolo_handle")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        returned = rt.load(str(weight), device=None)

        assert returned is handle
        assert handle._cw_writable_dir == str(models_dir)

    def test_load_stashes_writable_dir_on_handle_for_missing_weight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        target = models_dir / "yoloe-26m-seg.pt"

        handle = MagicMock(name="yolo_handle")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        cwd_before = Path.cwd()
        try:
            rt.load(str(target), device=None)
        finally:
            os.chdir(cwd_before)

        assert handle._cw_writable_dir == str(models_dir)

    def test_apply_text_prompt_chdirs_into_writable_dir(self, tmp_path: Path) -> None:
        writable_dir = tmp_path / "models"
        writable_dir.mkdir()

        observed_cwds: list[Path] = []

        handle = MagicMock()
        handle.set_classes = MagicMock()
        handle.get_text_pe = MagicMock(
            side_effect=lambda _: observed_cwds.append(Path.cwd()) or "text_pe_tensor"
        )
        del handle._cw_active_prompt
        handle._cw_writable_dir = str(writable_dir)
        handle.model = [
            MagicMock(spec=[])
        ]  # prompt-driven (no lrpc); see _yoloe_handle

        cwd_before = Path.cwd()
        try:
            UltralyticsRuntime._apply_text_prompt(handle, "helmet")
        finally:
            os.chdir(cwd_before)

        assert observed_cwds == [writable_dir]
        assert Path.cwd() == cwd_before
        handle.set_classes.assert_called_once_with(["helmet"], "text_pe_tensor")

    def test_apply_text_prompt_restores_cwd_when_get_text_pe_raises(
        self, tmp_path: Path
    ) -> None:
        writable_dir = tmp_path / "models"
        writable_dir.mkdir()

        handle = MagicMock()
        handle.set_classes = MagicMock()
        handle.get_text_pe = MagicMock(side_effect=RuntimeError("text encoder OOM"))
        del handle._cw_active_prompt
        handle._cw_writable_dir = str(writable_dir)
        handle.model = [MagicMock(spec=[])]  # prompt-driven (no lrpc)

        cwd_before = Path.cwd()
        UltralyticsRuntime._apply_text_prompt(handle, "helmet")

        assert Path.cwd() == cwd_before

    def test_apply_text_prompt_skips_chdir_for_legacy_handle_without_stash(
        self,
    ) -> None:
        handle = MagicMock()
        handle.set_classes = MagicMock()
        handle.get_text_pe = MagicMock(return_value="text_pe_tensor")
        del handle._cw_active_prompt
        del handle._cw_writable_dir
        handle.model = [MagicMock(spec=[])]  # prompt-driven (no lrpc)

        cwd_before = Path.cwd()
        UltralyticsRuntime._apply_text_prompt(handle, "helmet")

        assert Path.cwd() == cwd_before
        handle.set_classes.assert_called_once_with(["helmet"], "text_pe_tensor")

    def test_apply_text_prompt_ignores_non_path_stash_value(self) -> None:
        # MagicMock auto-creates _cw_writable_dir as a child MagicMock
        # (truthy, implements __fspath__) — runtime must reject it.
        handle = MagicMock()
        handle.set_classes = MagicMock()
        handle.get_text_pe = MagicMock(return_value="text_pe_tensor")
        del handle._cw_active_prompt
        handle.model = [MagicMock(spec=[])]  # prompt-driven (no lrpc)

        cwd_before = Path.cwd()
        UltralyticsRuntime._apply_text_prompt(handle, "helmet")

        assert Path.cwd() == cwd_before
        handle.set_classes.assert_called_once_with(["helmet"], "text_pe_tensor")


class TestUltralyticsPredictDepth:
    """``yolo26*-depth`` returns metric depth, not detections.

    A depth result carries no boxes, so before the depth branch existed it fell
    through every dispatch check to ``result.boxes is None -> continue`` and the
    caller got an empty ``DetectionResult`` — no error, no depth, nothing
    downstream to notice. These pin the branch and the resolution guard.
    """

    @staticmethod
    def _depth_result(depth: np.ndarray, orig_shape=(480, 640)) -> MagicMock:
        result = MagicMock()
        result.orig_shape = orig_shape
        result.boxes = None
        result.probs = None
        result.obb = None
        result.masks = None
        result.keypoints = None
        tensor = MagicMock()
        tensor.cpu.return_value.numpy.return_value = depth
        result.depth = MagicMock()
        result.depth.data = tensor
        return result

    def test_returns_metric_depth_result(self):
        rt = UltralyticsRuntime()
        depth = np.full((480, 640), 3.5, dtype=np.float32)
        model = MagicMock(return_value=[self._depth_result(depth)])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert isinstance(pred, DepthResult)
        # Metric matters twice over: it suppresses the disparity inversion in
        # send_depth and satisfies object_pose's metric depth-source guard.
        assert pred.metric is True
        assert pred.depth_map.shape == (480, 640)
        assert pred.depth_map.dtype == np.float32
        np.testing.assert_allclose(pred.depth_map, 3.5)
        assert (pred.h, pred.w) == (480, 640)

    def test_resizes_head_output_back_to_the_frame(self):
        """Trained at imgsz=768, so the head may not emit at frame resolution.

        Depth pixel (u, v) has to line up with camera pixel (u, v) or the
        pinhole back-projection downstream puts every point in the wrong place.
        """
        pytest.importorskip("cv2")
        rt = UltralyticsRuntime()
        depth = np.full((768, 768), 2.0, dtype=np.float32)
        model = MagicMock(return_value=[self._depth_result(depth, (480, 640))])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert pred.depth_map.shape == (480, 640)
        np.testing.assert_allclose(pred.depth_map, 2.0, atol=1e-5)

    def test_squeezes_a_leading_batch_dimension(self):
        rt = UltralyticsRuntime()
        depth = np.full((1, 480, 640), 1.25, dtype=np.float32)
        model = MagicMock(return_value=[self._depth_result(depth)])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert pred.depth_map.shape == (480, 640)

    def test_malformed_depth_falls_through_instead_of_propagating(self):
        """A non-2-D depth payload must not reach a consumer as a depth map."""
        rt = UltralyticsRuntime()
        model = MagicMock(
            return_value=[self._depth_result(np.zeros((3, 4, 5), dtype=np.float32))]
        )

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert not isinstance(pred, DepthResult)

    def test_detection_results_are_untouched(self):
        """The depth branch runs first, so guard the task it could shadow."""
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[_box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9)],
            names={0: "person"},
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert not isinstance(pred, DepthResult)
        assert len(pred.detections) == 1


# --------------------------------------------------------------------------
# Bulk box parsing (CYB-3248)
# --------------------------------------------------------------------------


class _CountingTensor:
    """Minimal stand-in for a torch tensor that records host transfers.

    ``cpu().numpy()`` is the only path the runtime may use to pull box
    attributes across; every call bumps the shared counter so a test can
    assert the whole batch moved in one transfer.
    """

    def __init__(self, arr: np.ndarray, counter: dict[str, int], name: str):
        self._arr = arr
        self._counter = counter
        self._name = name

    def cpu(self) -> "_CountingTensor":
        return self

    def numpy(self) -> np.ndarray:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        return self._arr

    # Per-box access, i.e. what the old ``for box in result.boxes`` path used.
    def __getitem__(self, idx):
        self._counter[f"{self._name}_item"] = (
            self._counter.get(f"{self._name}_item", 0) + 1
        )
        return self._arr[idx]

    def __len__(self) -> int:
        return len(self._arr)

    def tolist(self):
        return self._arr.tolist()


class _BulkBoxes:
    """Tensor-batch ``Boxes`` stand-in, as Ultralytics hands back on GPU.

    Iterating yields per-box wrappers, so a regression to the per-box form
    still produces correct output — it just shows up as a transfer count
    above one, which is exactly what the assertions here pin.
    """

    def __init__(self, cls_arr, conf_arr, xyxy_arr, counter):
        self.cls = _CountingTensor(cls_arr, counter, "cls")
        self.conf = _CountingTensor(conf_arr, counter, "conf")
        self.xyxy = _CountingTensor(xyxy_arr, counter, "xyxy")
        self._n = len(cls_arr)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        for i in range(self._n):
            box = MagicMock()
            box.cls = [self.cls[i]]
            box.conf = [self.conf[i]]
            box.xyxy = [MagicMock()]
            box.xyxy[0].tolist.return_value = list(self.xyxy[i])
            yield box


def _bulk_result(n: int, *, masks: bool = False, kps: bool = False):
    """Synthetic N-detection result with deterministic, distinguishable values."""
    rng = np.random.default_rng(3248)
    cls_arr = np.arange(n, dtype=np.float32) % 3
    conf_arr = np.linspace(0.51, 0.99, n, dtype=np.float32)
    x1 = np.arange(n, dtype=np.float32)
    xyxy_arr = np.stack([x1, x1 + 1, x1 + 11, x1 + 21], axis=1)

    counter: dict[str, int] = {}
    result = MagicMock()
    result.orig_shape = (480, 640)
    result.boxes = _BulkBoxes(cls_arr, conf_arr, xyxy_arr, counter)
    result.names = {0: "person", 1: "car", 2: "dog"}
    # Explicit None for every task-dispatch attribute, same convention as
    # _result_mock. Depth is checked before the detect branch, and a bare
    # MagicMock is truthy there — it only fails to shadow this because
    # np.asarray() coerces it to a 0-length array that trips the ndim guard.
    # Pinning it None keeps these tests off that accident.
    result.probs = None
    result.obb = None
    result.depth = None

    if masks:
        mask_arr = rng.random((n, 8, 8), dtype=np.float32)
        result.masks = MagicMock()
        result.masks.data = _CountingTensor(mask_arr, counter, "masks")
    else:
        result.masks = None
        mask_arr = None

    if kps:
        kp_arr = rng.random((n, 17, 3), dtype=np.float32)
        result.keypoints = MagicMock()
        result.keypoints.data = _CountingTensor(kp_arr, counter, "kps")
    else:
        result.keypoints = None
        kp_arr = None

    return result, counter, (cls_arr, conf_arr, xyxy_arr, mask_arr, kp_arr)


class TestUltralyticsBulkBoxParsing:
    """Pins the bulk (whole-batch) read of result.boxes — see CYB-3248.

    The per-box form this replaced cost three device->host copies and three
    implicit syncs per detection on a GPU worker.
    """

    def test_output_matches_source_arrays_at_n100_with_masks_and_keypoints(self):
        n = 100
        rt = UltralyticsRuntime()
        result, _, (cls_arr, conf_arr, xyxy_arr, mask_arr, kp_arr) = _bulk_result(
            n, masks=True, kps=True
        )
        names = {0: "person", 1: "car", 2: "dog"}

        pred = rt.predict(
            MagicMock(return_value=[result]),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

        assert len(pred.detections) == n
        frame_area = 480 * 640
        for i, det in enumerate(pred.detections):
            assert det.label == names[int(cls_arr[i])]
            assert det.confidence == pytest.approx(float(conf_arr[i]))
            assert det.bbox.x1 == pytest.approx(float(xyxy_arr[i][0]))
            assert det.bbox.y1 == pytest.approx(float(xyxy_arr[i][1]))
            assert det.bbox.x2 == pytest.approx(float(xyxy_arr[i][2]))
            assert det.bbox.y2 == pytest.approx(float(xyxy_arr[i][3]))
            assert det.area_ratio == pytest.approx(det.bbox.area / frame_area)
            # Index alignment: mask[i] / keypoints[i] must still line up with
            # box i now that the loop indexes bulk arrays instead of zipping.
            assert np.array_equal(det.mask.data, mask_arr[i])
            assert np.array_equal(det.keypoints, kp_arr[i])
            assert len(det.keypoint_set.points) == 17

    def test_reads_each_box_attribute_once_not_per_box(self):
        rt = UltralyticsRuntime()
        result, counter, _ = _bulk_result(100)

        rt.predict(
            MagicMock(return_value=[result]),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

        # One whole-batch transfer per attribute, regardless of N.
        assert counter.get("cls") == 1
        assert counter.get("conf") == 1
        assert counter.get("xyxy") == 1
        # And no per-box indexing into the device tensors.
        assert counter.get("conf_item") is None
        assert counter.get("cls_item") is None

    def test_class_filter_still_applies_over_bulk_arrays(self):
        rt = UltralyticsRuntime()
        result, counter, (cls_arr, _, _, _, _) = _bulk_result(30)

        pred = rt.predict(
            MagicMock(return_value=[result]),
            np.zeros((480, 640, 3), dtype=np.uint8),
            classes=["dog"],
        )

        expected = int((cls_arr == 2).sum())
        assert len(pred.detections) == expected
        assert {d.label for d in pred.detections} == {"dog"}
        assert counter.get("conf") == 1

    def test_empty_boxes_produce_no_detections(self):
        rt = UltralyticsRuntime()
        result, _, _ = _bulk_result(0)

        pred = rt.predict(
            MagicMock(return_value=[result]),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

        assert pred.detections == []

    def test_falls_back_to_per_box_helpers_for_non_tensor_boxes(self):
        # Mocked handles in the rest of this suite hand back a plain list of
        # box mocks, which has no .cls/.conf/.xyxy — the fallback must keep
        # those working.
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=1, conf=0.8),
            ],
            names={0: "person", 1: "car"},
        )

        pred = rt.predict(
            MagicMock(return_value=[result_obj]),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

        assert [d.label for d in pred.detections] == ["person", "car"]
        assert [round(d.confidence, 2) for d in pred.detections] == [0.9, 0.8]
        assert pred.detections[0].bbox.x2 == 110.0
