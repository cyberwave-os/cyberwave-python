"""Extra ``predict()`` kwargs must reach the Ultralytics model call.

``UltralyticsRuntime.predict`` accepted ``**kwargs`` and then built its call
from ``conf``/``verbose``/``device`` alone, so ``predict(frame, imgsz=512)``
was a silent no-op. These tests use a fake handle — a plain callable that
records what it was called with — so no ultralytics install is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime


class _FakeYolo:
    """Stands in for the ``YOLO`` handle: callable, records its kwargs.

    Deliberately has no ``set_classes``/``get_text_pe``, so the open-vocab
    prompt path short-circuits and only the call kwargs are under test.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, _input: Any, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return []


def _frame() -> Any:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _predict(**kwargs: Any) -> dict[str, Any]:
    handle = _FakeYolo()
    UltralyticsRuntime().predict(handle, _frame(), **kwargs)
    assert len(handle.calls) == 1
    return handle.calls[0]


def test_forwards_imgsz() -> None:
    """The regression: this used to be dropped between signature and call."""
    assert _predict(imgsz=512)["imgsz"] == 512


def test_forwards_arbitrary_ultralytics_args() -> None:
    call = _predict(iou=0.45, max_det=10, half=True)
    assert call["iou"] == 0.45
    assert call["max_det"] == 10
    assert call["half"] is True


def test_defaults_are_still_applied() -> None:
    call = _predict(confidence=0.7)
    assert call["conf"] == 0.7
    assert call["verbose"] is False


def test_explicit_kwarg_overrides_our_default() -> None:
    """Highest priority on the right, matching Ultralytics' own convention."""
    call = _predict(confidence=0.7, conf=0.9, verbose=True)
    assert call["conf"] == 0.9
    assert call["verbose"] is True


def test_device_is_forwarded_and_overridable() -> None:
    assert _predict(device="cuda:0")["device"] == "cuda:0"


def test_device_falls_back_to_the_handle_stash() -> None:
    handle = _FakeYolo()
    handle._cw_device = "cuda:1"  # what load() stashes after model.to(device)
    UltralyticsRuntime().predict(handle, _frame())
    assert handle.calls[0]["device"] == "cuda:1"


def test_no_extra_kwargs_leaves_the_call_minimal() -> None:
    """Nothing invented: an unconfigured call must stay exactly as before."""
    assert set(_predict().keys()) == {"conf", "verbose"}


def test_classes_and_prompt_are_not_forwarded() -> None:
    """Both are handled by the runtime itself, not by Ultralytics' predictor.

    ``classes`` filters results post-hoc (Ultralytics expects integer indices,
    not our label strings) and ``prompt`` is applied via ``set_classes``.
    Forwarding either would make ``get_cfg`` reject the call.
    """
    call = _predict(classes=["person"], prompt="helmet")
    assert "classes" not in call
    assert "prompt" not in call
