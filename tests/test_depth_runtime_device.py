"""Device and input-size resolution for the two Depth-Anything runtimes.

The device tests pin the regression that made GPU depth inference impossible:
``ModelManager._detect_device()`` returns ``"cuda:0"``, and both runtimes used
to validate against a ``{"cpu", "cuda"}`` whitelist and raise on it.

Both helpers are pure functions taking ``torch`` as an argument, so these run
without torch installed.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from cyberwave.models.runtimes.depth_anything_v2_rt import (
    _resolve_device as da2_resolve_device,
)
from cyberwave.models.runtimes.depth_anything_v2_rt import (
    _resolve_input_size as da2_resolve_input_size,
)
from cyberwave.models.runtimes.vda_rt import _resolve_device as vda_resolve_device
from cyberwave.models.runtimes.vda_rt import (
    _resolve_input_size as vda_resolve_input_size,
)

DEVICE_RESOLVERS = pytest.mark.parametrize(
    "resolve",
    [
        pytest.param(da2_resolve_device, id="da2"),
        pytest.param(vda_resolve_device, id="vda"),
    ],
)
INPUT_SIZE_RESOLVERS = pytest.mark.parametrize(
    "resolve",
    [
        pytest.param(da2_resolve_input_size, id="da2"),
        pytest.param(vda_resolve_input_size, id="vda"),
    ],
)


def _fake_torch(*, cuda_available: bool = True) -> Any:
    """Minimal ``torch`` stand-in: ``cuda.is_available`` + a validating ``device()``."""

    def device(spec: str) -> Any:
        head, _, index = str(spec).partition(":")
        if head not in {"cpu", "cuda", "mps", "xpu", "meta"}:
            raise RuntimeError(f"Expected one of cpu, cuda, ... device type: {head}")
        if index and not index.isdigit():
            raise RuntimeError(f"Invalid device string: '{spec}'")
        return types.SimpleNamespace(type=head, index=int(index) if index else None)

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        device=device,
    )


@DEVICE_RESOLVERS
def test_accepts_indexed_cuda_device(resolve: Any) -> None:
    """The regression: this is exactly what ``ModelManager`` passes on a GPU host."""
    assert resolve("cuda:0", _fake_torch()) == "cuda:0"
    assert resolve("cuda:1", _fake_torch()) == "cuda:1"


@DEVICE_RESOLVERS
def test_auto_prefers_cuda_when_available(resolve: Any) -> None:
    assert resolve("auto", _fake_torch(cuda_available=True)) == "cuda"
    assert resolve("auto", _fake_torch(cuda_available=False)) == "cpu"


@DEVICE_RESOLVERS
def test_none_is_treated_as_auto(resolve: Any) -> None:
    assert resolve(None, _fake_torch(cuda_available=False)) == "cpu"


@DEVICE_RESOLVERS
def test_plain_cpu_and_cuda_still_work(resolve: Any) -> None:
    """The two values the old whitelist allowed must keep resolving unchanged."""
    assert resolve("cpu", _fake_torch()) == "cpu"
    assert resolve("cuda", _fake_torch()) == "cuda"


@DEVICE_RESOLVERS
def test_accepts_non_cuda_accelerators(resolve: Any) -> None:
    """Apple-silicon hosts run the SDK too; ``torch.device`` is the authority."""
    assert resolve("mps", _fake_torch(cuda_available=False)) == "mps"


@DEVICE_RESOLVERS
def test_normalises_case_and_whitespace(resolve: Any) -> None:
    assert resolve("  CUDA:0 ", _fake_torch()) == "cuda:0"


@DEVICE_RESOLVERS
def test_rejects_garbage_device_with_actionable_message(resolve: Any) -> None:
    """Loose validation must still reject nonsense, and say what is accepted."""
    with pytest.raises(ValueError) as excinfo:
        resolve("gpu", _fake_torch())
    message = str(excinfo.value)
    assert "gpu" in message
    assert "cuda:0" in message


@INPUT_SIZE_RESOLVERS
def test_accepts_patch_size_multiples(resolve: Any) -> None:
    assert resolve(518) == 518
    assert resolve(322) == 322
    assert resolve(266) == 266


@INPUT_SIZE_RESOLVERS
@pytest.mark.parametrize("bad", [300, 519, 0, -14])
def test_rejects_non_patch_multiples(resolve: Any, bad: int) -> None:
    """Caught at load rather than deep in the DPT head on the first frame."""
    with pytest.raises(ValueError) as excinfo:
        resolve(bad)
    assert "14" in str(excinfo.value)
