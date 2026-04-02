"""Tests specific to the FilesystemBackend implementation."""

import json
import threading
import time
from pathlib import Path

import pytest

from cyberwave.data.backend import Sample
from cyberwave.data.filesystem_backend import FilesystemBackend, _safe_channel_name


@pytest.fixture
def backend(tmp_path):
    be = FilesystemBackend(
        base_dir=tmp_path / "fs_data",
        ring_buffer_size=5,
        poll_interval_s=0.03,
    )
    yield be
    be.close()


class TestRingBuffer:
    def test_prune_keeps_at_most_n_samples(self, backend: FilesystemBackend, tmp_path):
        for i in range(10):
            backend.publish("ring", f"v{i}".encode())
            time.sleep(0.002)

        channel_dir = tmp_path / "fs_data" / _safe_channel_name("ring")
        bins = [p for p in channel_dir.glob("[0-9]*.bin") if p.name != "latest.bin"]
        assert len(bins) <= 5

    def test_latest_survives_pruning(self, backend: FilesystemBackend):
        for i in range(10):
            backend.publish("prune", f"v{i}".encode())
            time.sleep(0.002)
        sample = backend.latest("prune")
        assert sample is not None
        assert sample.payload == b"v9"


class TestAtomicLatest:
    def test_concurrent_publish_does_not_corrupt(self, backend: FilesystemBackend):
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(20):
                    backend.publish("atomic", f"w{n}_{i}".encode())
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        sample = backend.latest("atomic")
        assert sample is not None
        assert len(sample.payload) > 0

    def test_subscriber_never_sees_half_written_sample(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        backend = FilesystemBackend(
            base_dir=tmp_path / "fs_data",
            poll_interval_s=0.005,
        )
        channel = "atomic/race"
        channel_dir = backend._base_dir / _safe_channel_name(channel)
        received: list[bytes] = []
        delivered = threading.Event()
        read_attempted = threading.Event()

        def is_sample_file(path: Path) -> bool:
            return (
                path.parent == channel_dir
                and path.suffix == ".bin"
                and path.name != "latest.bin"
                and not path.name.startswith(".")
            )

        original_write_bytes = Path.write_bytes
        original_read_bytes = Path.read_bytes

        def patched_write_bytes(path: Path, data: bytes) -> int:
            if is_sample_file(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb"):
                    pass
                assert read_attempted.wait(timeout=1.0)
            return original_write_bytes(path, data)

        def patched_read_bytes(path: Path) -> bytes:
            if is_sample_file(path):
                read_attempted.set()
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "write_bytes", patched_write_bytes)
        monkeypatch.setattr(Path, "read_bytes", patched_read_bytes)

        def cb(sample: Sample) -> None:
            received.append(sample.payload)
            delivered.set()

        sub = backend.subscribe(channel, cb, policy="fifo")
        time.sleep(0.05)
        backend.publish(channel, b"payload")

        assert delivered.wait(timeout=2.0)
        assert received == [b"payload"]

        sub.close()
        backend.close()


class TestChannelNameEscaping:
    def test_slashes_replaced(self):
        assert _safe_channel_name("frames/default") == "frames__default"
        assert _safe_channel_name("a/b/c") == "a__b__c"

    def test_no_slash_unchanged(self):
        assert _safe_channel_name("simple") == "simple"

    def test_dotdot_sanitized(self):
        result = _safe_channel_name("..")
        assert ".." not in result

    def test_traversal_path_sanitized(self):
        result = _safe_channel_name("../../etc/passwd")
        assert ".." not in result

    def test_empty_channel_safe(self):
        result = _safe_channel_name("")
        assert result and result not in ("", ".", "..")


class TestMetadata:
    def test_metadata_written_as_sidecar(self, backend: FilesystemBackend, tmp_path):
        backend.publish("meta", b"data", metadata={"key": "value"})
        channel_dir = tmp_path / "fs_data" / _safe_channel_name("meta")
        meta_files = list(channel_dir.glob("*.meta.json"))
        assert len(meta_files) == 1
        content = json.loads(meta_files[0].read_text())
        assert content == {"key": "value"}


class TestCustomBaseDir:
    def test_respects_base_dir(self, tmp_path):
        custom = tmp_path / "custom_dir"
        be = FilesystemBackend(base_dir=custom)
        be.publish("ch", b"data")
        assert (custom / _safe_channel_name("ch") / "latest.bin").exists()
        be.close()


class TestCleanup:
    def test_close_stops_watcher_threads(self, backend: FilesystemBackend):
        received: list[Sample] = []
        backend.subscribe("cleanup", lambda s: received.append(s))
        time.sleep(0.1)
        backend.close()
        time.sleep(0.1)

        initial_threads = threading.active_count()
        be2 = FilesystemBackend(
            base_dir=backend._base_dir,
            poll_interval_s=0.03,
        )
        be2.subscribe("cleanup2", lambda s: None)
        be2.close()
        time.sleep(0.15)
        assert threading.active_count() <= initial_threads + 1

    def test_close_idempotent(self, backend: FilesystemBackend):
        backend.close()
        backend.close()
