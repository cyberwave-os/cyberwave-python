"""``predict(..., native_resolution=True)`` on the three depth runtimes.

Every depth runtime used to resample its output to the camera frame, which on
a 1080p camera turns ~0.27 M model samples into ~2 M interpolated ones that
every downstream encoder, publisher and viewer then pays for. The flag keeps
the model's own grid whenever it is the smaller one; ``DepthResult.h``/``w``
still name the camera frame so consumers can rescale per axis.

The runtimes are exercised with stand-in models, so these run without torch,
the vendored VDA library or a Hailo device.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any

import numpy as np
import pytest

from cyberwave.models.runtimes.depth_anything_v2_rt import (
    DepthAnythingV2Runtime,
    _Dav2Handle,
)
from cyberwave.models.runtimes.hailo_rt import _extract_depth_map
from cyberwave.models.runtimes.vda_rt import (
    VideoDepthAnythingRuntime,
    _resize_to_working_grid,
    _VdaHandle,
    _working_grid,
)


def _frame(width: int, height: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), np.uint8)


# --- VDA working grid -------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        # Short side lands on input_size, long side on the nearest multiple of 14.
        (1280, 720, (924, 518)),
        (1920, 1080, (924, 518)),
        (720, 1280, (518, 924)),
        (640, 480, (686, 518)),
        # Wider than 16:9: VDA shrinks input_size first (518 -> 392 here).
        (2560, 1080, (924, 392)),
    ],
)
def test_working_grid_matches_vda_transform(
    width: int, height: int, expected: tuple[int, int]
) -> None:
    """Pinned against VDA@4f5ae23's ``Resize`` (verified over 8k sizes)."""
    assert _working_grid(width, height, 518) == expected


def test_resize_to_working_grid_never_upsamples() -> None:
    """640x480 works on a 686x518 grid — larger than the camera, so the camera
    resolution is already the cheaper output and the frame is left alone."""
    assert _resize_to_working_grid(_frame(640, 480), 518) is None


# --- VDA predict ------------------------------------------------------------


class _FakeVdaModel:
    """Stands in for ``VideoDepthAnything``: returns depth on the input's grid,
    which is what the real one does after its own resize round-trip."""

    def __init__(self) -> None:
        self.frame_height: int | None = None
        self.frame_width: int | None = None
        self.transform: Any = None
        self.frame_id_list: list[int] = []
        self.frame_cache_list: list[Any] = []
        self.id = -1
        self.inputs: list[tuple[int, ...]] = []

    def infer_video_depth_one(self, frame: Any, **_: Any) -> np.ndarray:
        self.inputs.append(frame.shape)
        self.frame_height, self.frame_width = frame.shape[:2]
        self.transform = object()
        return np.ones(frame.shape[:2], dtype=np.float32)


def _vda_handle() -> _VdaHandle:
    return _VdaHandle(
        model=_FakeVdaModel(),
        torch=None,
        device="cpu",
        encoder="vits",
        metric=True,
        input_size=518,
        fp32=True,
    )


def test_vda_native_resolution_returns_the_working_grid() -> None:
    # The pre-resize onto VDA's grid is the one step here that is not a stand-in:
    # it runs cv2.INTER_CUBIC to match the library's own resize.
    pytest.importorskip("cv2", reason="OpenCV not installed (extras: camera)")

    handle = _vda_handle()
    result = VideoDepthAnythingRuntime().predict(
        handle, _frame(1280, 720), native_resolution=True
    )
    assert result.depth_map.shape == (518, 924)
    assert (result.h, result.w) == (720, 1280)
    assert result.metric is True
    assert handle.model.inputs == [(518, 924, 3)]
    # Without the flag the camera resolution is kept, as before.
    default = VideoDepthAnythingRuntime().predict(_vda_handle(), _frame(1280, 720))
    assert default.depth_map.shape == (720, 1280)


def test_vda_native_resolution_keeps_temporal_state_across_frames() -> None:
    """The reset guard compares against what VDA was fed. Comparing against the
    camera size would wipe the temporal window on every native frame."""
    pytest.importorskip("cv2", reason="OpenCV not installed (extras: camera)")

    handle = _vda_handle()
    runtime = VideoDepthAnythingRuntime()
    runtime.predict(handle, _frame(1280, 720), native_resolution=True)
    handle.model.frame_cache_list = ["cached"]
    runtime.predict(handle, _frame(1280, 720), native_resolution=True)
    assert handle.model.frame_cache_list == ["cached"]


# --- DA2 predict ------------------------------------------------------------


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.array, dim))

    def to(self, _device: str) -> _FakeTensor:
        return self

    def squeeze(self) -> _FakeTensor:
        return _FakeTensor(np.squeeze(self.array))

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return _FakeTensor(self.array.astype(np.float32))

    def numpy(self) -> np.ndarray:
        return self.array


def _da2_handle() -> _Dav2Handle:
    fake_torch = types.SimpleNamespace(
        from_numpy=_FakeTensor, no_grad=contextlib.nullcontext
    )
    # DA2's forward returns depth on the (square) input grid.
    model = lambda tensor: _FakeTensor(  # noqa: E731
        np.ones(tensor.array.shape[-2:], dtype=np.float32)
    )
    return _Dav2Handle(
        model=model, torch=fake_torch, device="cpu", encoder="vits", input_size=518
    )


def test_da2_native_resolution_returns_the_network_grid() -> None:
    result = DepthAnythingV2Runtime().predict(
        _da2_handle(), _frame(1280, 720), native_resolution=True
    )
    assert result.depth_map.shape == (518, 518)
    assert (result.h, result.w) == (720, 1280)
    assert result.metric is False
    default = DepthAnythingV2Runtime().predict(_da2_handle(), _frame(1280, 720))
    assert default.depth_map.shape == (720, 1280)


def test_da2_native_resolution_never_upsamples_past_the_camera() -> None:
    """A 320x240 camera has fewer pixels than the 518x518 grid; resampling to
    the camera is then the downsample and still happens."""
    result = DepthAnythingV2Runtime().predict(
        _da2_handle(), _frame(320, 240), native_resolution=True
    )
    assert result.depth_map.shape == (240, 320)


# --- Hailo ------------------------------------------------------------------


def test_hailo_keep_native_returns_the_hef_grid() -> None:
    outputs = {"depth": np.ones((1, 256, 320, 1), dtype=np.float32)}
    native = _extract_depth_map(outputs, orig_h=720, orig_w=1280, keep_native=True)
    assert native.shape == (256, 320)
    resized = _extract_depth_map(outputs, orig_h=720, orig_w=1280)
    assert resized.shape == (720, 1280)
    # A camera smaller than the HEF grid is still the cheaper output.
    small = _extract_depth_map(outputs, orig_h=120, orig_w=160, keep_native=True)
    assert small.shape == (120, 160)
