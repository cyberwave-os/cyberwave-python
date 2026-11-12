"""Exercise the notebook's audio collector without a robot or API credentials."""
import ast
import asyncio
import json
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


def collector():
    notebook = Path(__file__).parents[1] / "examples/notebooks/g1_voice_astra_v2.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "async def collect_g1_audio" in "".join(c["source"]))
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "collect_g1_audio")
    scope = {"asyncio": asyncio, "av": av}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), scope)
    return scope["collect_g1_audio"]


class Track:
    def __init__(self, delay=0, stop_after=None):
        self.delay = delay
        self.stop_after = stop_after
        self.frames = 0
        self.cancelled = False

    async def recv(self):
        try:
            if self.stop_after is not None and self.frames >= self.stop_after:
                await asyncio.Future()
            await asyncio.sleep(self.delay)
            frame = av.AudioFrame.from_ndarray(np.full((1, 960), 1000, dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate = 48000
            self.frames += 1
            return frame
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_delayed_frames_produce_exact_complete_clip():
    # Delivery is slower than real time but remains within its explicit budget.
    track = Track(delay=0.04)
    pcm = asyncio.run(collector()(track, seconds=0.1, capture_timeout=1, stall_timeout=0.2))
    assert len(pcm) == 1600 * 2
    assert np.any(np.frombuffer(pcm, dtype="<i2"))
    assert not track.cancelled


def test_stall_discards_partial_audio_and_cancels_receive():
    track = Track(stop_after=2)
    with pytest.raises(TimeoutError, match="G1 audio incomplete:.*Partial audio was discarded"):
        asyncio.run(collector()(track, seconds=0.1, capture_timeout=1, stall_timeout=0.05))
    assert track.cancelled


def test_total_budget_expires_even_when_frames_keep_arriving():
    track = Track(delay=0.04)
    with pytest.raises(TimeoutError, match="of 0.1s"):
        asyncio.run(collector()(track, seconds=0.1, capture_timeout=0.12, stall_timeout=0.1))
    assert track.cancelled


def test_polling_does_not_cancel_a_slow_receive():
    track = Track(delay=1.1)
    pcm = asyncio.run(collector()(track, seconds=0.01, capture_timeout=2, stall_timeout=1.5))
    assert len(pcm) == 160 * 2
    assert not track.cancelled


def test_interruption_cleans_up_receive():
    async def run():
        track = Track(stop_after=0)
        task = asyncio.create_task(collector()(track, seconds=0.1, capture_timeout=1, stall_timeout=1))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert track.cancelled
    asyncio.run(run())
