"""Tests for the persistent offline queue."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cyberwave.zenoh_mqtt.queue import OfflineQueue, QueuedMessage


@pytest.fixture
def queue_dir(tmp_path):
    return str(tmp_path / "test_queue")


@pytest.fixture
def queue(queue_dir):
    q = OfflineQueue(queue_dir=queue_dir, max_bytes=10 * 1024 * 1024)
    yield q
    q.close()


def _msg(topic: str = "test/topic", payload: bytes = b"hello") -> QueuedMessage:
    return QueuedMessage(
        mqtt_topic=topic, payload=payload, qos=1, enqueued_at=time.time()
    )


class TestOfflineQueue:
    def test_enqueue_and_drain(self, queue):
        queue.enqueue(_msg(payload=b"msg1"))
        queue.enqueue(_msg(payload=b"msg2"))
        queue.enqueue(_msg(payload=b"msg3"))

        batch = queue.drain(batch_size=2)
        assert len(batch) == 2
        assert batch[0].payload == b"msg1"
        assert batch[1].payload == b"msg2"

        batch2 = queue.drain(batch_size=10)
        assert len(batch2) == 1
        assert batch2[0].payload == b"msg3"

    def test_empty_drain(self, queue):
        assert queue.drain() == []

    def test_is_empty(self, queue):
        assert queue.is_empty
        queue.enqueue(_msg())
        assert not queue.is_empty

    def test_preserves_topic_and_qos(self, queue):
        queue.enqueue(
            QueuedMessage(
                mqtt_topic="cyberwave/twin/abc/event",
                payload=b"data",
                qos=2,
                enqueued_at=123.456,
            )
        )
        batch = queue.drain(1)
        assert len(batch) == 1
        assert batch[0].mqtt_topic == "cyberwave/twin/abc/event"
        assert batch[0].qos == 2
        assert batch[0].enqueued_at == 123.456

    def test_fifo_order(self, queue):
        for i in range(10):
            queue.enqueue(_msg(payload=f"msg{i}".encode()))

        all_msgs = queue.drain(batch_size=100)
        assert [m.payload for m in all_msgs] == [
            f"msg{i}".encode() for i in range(10)
        ]

    def test_persistence_across_instances(self, queue_dir):
        q1 = OfflineQueue(queue_dir=queue_dir)
        q1.enqueue(_msg(payload=b"persistent"))
        q1.close()

        q2 = OfflineQueue(queue_dir=queue_dir)
        batch = q2.drain(1)
        assert len(batch) == 1
        assert batch[0].payload == b"persistent"
        q2.close()

    def test_max_bytes_eviction(self, queue_dir):
        max_bytes = 5000
        q = OfflineQueue(queue_dir=queue_dir, max_bytes=max_bytes)
        for i in range(200):
            q.enqueue(_msg(payload=b"x" * 50))
        # Eviction can only remove fully-written segments other than the
        # active one, so the actual size can exceed max_bytes by up to one
        # segment (~4 MiB ceiling, but here each message is small so segments
        # stay well under that).  The key invariant: the queue did not grow
        # unboundedly.
        assert q.size_bytes < max_bytes * 10
        q.close()

    def test_binary_payload(self, queue):
        payload = bytes(range(256))
        queue.enqueue(_msg(payload=payload))
        batch = queue.drain(1)
        assert batch[0].payload == payload

    def test_large_payload(self, queue):
        payload = b"A" * 100_000
        queue.enqueue(_msg(payload=payload))
        batch = queue.drain(1)
        assert batch[0].payload == payload

    def test_directory_created_automatically(self, tmp_path):
        new_dir = str(tmp_path / "nonexistent" / "subdir")
        q = OfflineQueue(queue_dir=new_dir)
        assert os.path.isdir(new_dir)
        q.enqueue(_msg(payload=b"test"))
        batch = q.drain(1)
        assert batch[0].payload == b"test"
        q.close()

    def test_drain_spanning_multiple_segments(self, queue_dir):
        """Drain that spans non-active and active segments must not corrupt the active segment.

        Regression: ``_drain_batch`` previously passed ``batch_size`` as the
        consumed count to ``_rewrite_active_segment`` instead of the number of
        records actually read from the active segment.  When prior non-active
        segments were drained first, too many records were skipped during the
        rewrite, silently destroying unread messages in the active segment.
        """
        _ts = [1_000_000.0]

        def _fake_time() -> float:
            return _ts[0]

        # Write 2 messages into a first segment.
        with patch("cyberwave.zenoh_mqtt.queue.time.time", side_effect=_fake_time):
            q1 = OfflineQueue(queue_dir=queue_dir, max_bytes=10 * 1024 * 1024)
            q1.enqueue(_msg(payload=b"seg1_msg1"))
            q1.enqueue(_msg(payload=b"seg1_msg2"))
            q1.close()

        # Advance the clock so the second OfflineQueue opens a new segment file.
        _ts[0] = 2_000_000.0

        with patch("cyberwave.zenoh_mqtt.queue.time.time", side_effect=_fake_time):
            # Reopen: creates a fresh active segment; the old one stays on disk.
            q2 = OfflineQueue(queue_dir=queue_dir, max_bytes=10 * 1024 * 1024)
            q2.enqueue(_msg(payload=b"seg2_msg1"))
            q2.enqueue(_msg(payload=b"seg2_msg2"))
            q2.enqueue(_msg(payload=b"seg2_msg3"))

            # Drain 4: should consume all 2 from seg1 and 2 of 3 from seg2.
            batch1 = q2.drain(4)
            assert [m.payload for m in batch1] == [
                b"seg1_msg1",
                b"seg1_msg2",
                b"seg2_msg1",
                b"seg2_msg2",
            ]

            # The fifth message must still be retrievable.
            batch2 = q2.drain(10)
            assert len(batch2) == 1
            assert batch2[0].payload == b"seg2_msg3"

            q2.close()
