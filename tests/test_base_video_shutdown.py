"""Camera synchronization must not outlive the stream or cross a reconnect."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("aiortc")

from cyberwave.sensor import base_video  # noqa: E402


class _Streamer(base_video.BaseVideoStreamer):
    def initialize_track(self):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_stop_drains_sync_waiter_before_reconnect(monkeypatch):
    streamer = _Streamer(
        client=Mock(), twin_uuid="camera-twin", enable_health_check=False
    )
    sleep = asyncio.sleep

    async def yield_once(_delay):
        await sleep(0)

    async def setup():
        streamer.streamer = SimpleNamespace(
            fps=15, sync_frame_pts=None, frame_count=0, close=Mock()
        )

    monkeypatch.setattr(base_video.asyncio, "sleep", yield_once)
    monkeypatch.setattr(
        base_video, "apply_h264_hw_patch", lambda: SimpleNamespace(codec_name="libx264")
    )
    monkeypatch.setattr(streamer, "_subscribe_to_answer", Mock())
    monkeypatch.setattr(streamer, "_setup_webrtc", setup)
    monkeypatch.setattr(streamer, "_perform_signaling", AsyncMock())
    publish = Mock()
    monkeypatch.setattr(streamer, "_publish_camera_sync_frame", publish)

    for _ in range(2):
        before = asyncio.all_tasks()
        await streamer.start()
        await sleep(0)
        waiters = asyncio.all_tasks() - before
        assert len(waiters) == 1
        assert not next(iter(waiters)).done()
        await streamer.stop()
        assert all(task.done() for task in waiters)
        assert streamer.streamer is None
    await streamer.stop()  # Already stopped is harmless.
    publish.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_fails", [False, True])
async def test_finished_sync_task_does_not_block_stop(publish_fails):
    streamer = _Streamer(
        client=Mock(), twin_uuid="camera-twin", enable_health_check=False
    )
    streamer.streamer = SimpleNamespace(
        fps=15,
        sync_frame_pts=90000,
        sync_frame_timestamp=123.0,
        sync_frame_timestamp_monotonic=100.0,
        sync_frame_time_base_num=1,
        sync_frame_time_base_den=90000,
        close=Mock(),
    )
    publish = Mock(
        side_effect=RuntimeError("MQTT unavailable") if publish_fails else None
    )
    streamer._publish_camera_sync_frame = publish
    task = asyncio.create_task(streamer._wait_and_publish_camera_sync_frame())
    streamer._sync_frame_task = task
    if publish_fails:
        with pytest.raises(RuntimeError, match="MQTT unavailable"):
            await task
    else:
        await task
    await streamer.stop()
    assert streamer.streamer is None
    assert streamer._sync_frame_task is None
    publish.assert_called_once_with(15, 90000, 1, 90000, 123.0, 100.0)
