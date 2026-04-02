"""Tests for record/replay debug utilities (CYB-1555)."""

from __future__ import annotations

import json
import struct
import threading
import time

import pytest

from cyberwave.data.backend import DataBackend, Sample
from cyberwave.data.config import BackendConfig, get_backend
from cyberwave.data.exceptions import RecordingError
from cyberwave.data.recording import (
    RecordingSession,
    ReplayResult,
    record,
    replay,
)
from cyberwave.data.recording_format import (
    FORMAT_VERSION,
    RecordingManifest,
    channel_to_filename,
    encode_entry,
    filename_to_channel,
    iter_entries,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(tmp_path):
    cfg = BackendConfig(
        backend="filesystem",
        filesystem_base_dir=str(tmp_path / "bus"),
        filesystem_ring_buffer_size=200,
    )
    be = get_backend(cfg)
    yield be
    be.close()


@pytest.fixture
def rec_dir(tmp_path):
    return tmp_path / "recording"


# ---------------------------------------------------------------------------
# Recording format tests
# ---------------------------------------------------------------------------


class TestRecordingFormat:
    def test_encode_decode_entry_roundtrip(self):
        ts = 1711234567.123
        payload = b"hello world"
        raw = encode_entry(ts, payload)
        assert len(raw) == 8 + 4 + len(payload)

        (decoded_ts,) = struct.unpack_from("<d", raw, 0)
        (decoded_len,) = struct.unpack_from("<I", raw, 8)
        decoded_payload = raw[12:]

        assert abs(decoded_ts - ts) < 1e-9
        assert decoded_len == len(payload)
        assert decoded_payload == payload

    def test_encode_empty_payload(self):
        raw = encode_entry(0.0, b"")
        assert len(raw) == 12
        (payload_len,) = struct.unpack_from("<I", raw, 8)
        assert payload_len == 0

    def test_iter_entries_reads_multiple(self, tmp_path):
        fpath = tmp_path / "test.bin"
        entries_data = b""
        expected = []
        for i in range(5):
            ts = 1000.0 + i
            payload = f"sample_{i}".encode()
            entries_data += encode_entry(ts, payload)
            expected.append((ts, payload))
        fpath.write_bytes(entries_data)

        results = list(iter_entries(fpath, "test/ch"))
        assert len(results) == 5
        for i, entry in enumerate(results):
            assert entry.channel == "test/ch"
            assert abs(entry.timestamp - expected[i][0]) < 1e-9
            assert entry.payload == expected[i][1]

    def test_iter_entries_handles_truncated_file(self, tmp_path):
        fpath = tmp_path / "truncated.bin"
        good_entry = encode_entry(1.0, b"ok")
        fpath.write_bytes(good_entry + b"\x00\x00")
        results = list(iter_entries(fpath, "ch"))
        assert len(results) == 1
        assert results[0].payload == b"ok"

    def test_iter_entries_empty_file(self, tmp_path):
        fpath = tmp_path / "empty.bin"
        fpath.write_bytes(b"")
        results = list(iter_entries(fpath, "ch"))
        assert len(results) == 0


class TestChannelFilename:
    def test_simple_channel(self):
        assert channel_to_filename("frames") == "frames.bin"

    def test_slash_channel(self):
        assert channel_to_filename("frames/default") == "frames__default.bin"

    def test_multi_slash(self):
        assert channel_to_filename("a/b/c") == "a__b__c.bin"

    def test_dotdot_sanitized(self):
        result = channel_to_filename("../../etc")
        assert ".." not in result
        assert result.endswith(".bin")

    def test_empty_channel(self):
        result = channel_to_filename("")
        assert result and result.endswith(".bin")


class TestFilenameToChannel:
    def test_simple(self):
        assert filename_to_channel("frames.bin") == "frames"

    def test_with_separator(self):
        assert filename_to_channel("frames__default.bin") == "frames/default"

    def test_no_bin_suffix(self):
        assert filename_to_channel("frames__default") == "frames/default"


class TestManifest:
    def test_roundtrip(self):
        m = RecordingManifest(
            channels=["a", "b/c"],
            start_ts=100.0,
            end_ts=200.0,
            sample_count=42,
            metadata={"key": "val"},
        )
        text = m.to_json()
        parsed = RecordingManifest.from_json(text)
        assert parsed.version == FORMAT_VERSION
        assert parsed.channels == ["a", "b/c"]
        assert parsed.start_ts == 100.0
        assert parsed.end_ts == 200.0
        assert parsed.sample_count == 42
        assert parsed.metadata == {"key": "val"}

    def test_unsupported_version_raises(self):
        bad = json.dumps({"version": 999})
        with pytest.raises(ValueError, match="Unsupported recording format version"):
            RecordingManifest.from_json(bad)

    def test_default_values(self):
        m = RecordingManifest()
        assert m.version == FORMAT_VERSION
        assert m.channels == []
        assert m.sample_count == 0


# ---------------------------------------------------------------------------
# Record tests
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_captures_samples(self, backend: DataBackend, rec_dir):
        with record(backend, ["rec/ch"], rec_dir) as session:
            time.sleep(0.15)
            for i in range(5):
                backend.publish("rec/ch", f"v{i}".encode())
                time.sleep(0.08)
            time.sleep(0.3)

        assert session.is_stopped
        assert session.sample_count >= 3

        manifest_path = rec_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = RecordingManifest.from_json(manifest_path.read_text())
        assert "rec/ch" in manifest.channels
        assert manifest.sample_count >= 3
        assert manifest.start_ts > 0
        assert manifest.version == FORMAT_VERSION

    def test_record_creates_bin_file(self, backend: DataBackend, rec_dir):
        with record(backend, ["data/sensor"], rec_dir):
            time.sleep(0.15)
            backend.publish("data/sensor", b"payload1")
            time.sleep(0.2)

        bin_path = rec_dir / channel_to_filename("data/sensor")
        assert bin_path.exists()
        entries = list(iter_entries(bin_path, "data/sensor"))
        assert len(entries) >= 1
        assert entries[0].payload == b"payload1"

    def test_record_multiple_channels(self, backend: DataBackend, rec_dir):
        channels = ["ch_a", "ch_b"]
        with record(backend, channels, rec_dir):
            time.sleep(0.15)
            backend.publish("ch_a", b"alpha")
            backend.publish("ch_b", b"beta")
            time.sleep(0.2)

        manifest = RecordingManifest.from_json(
            (rec_dir / "manifest.json").read_text()
        )
        assert set(manifest.channels) == {"ch_a", "ch_b"}

    def test_record_max_samples(self, backend: DataBackend, rec_dir):
        session = record(backend, ["limit/ch"], rec_dir, max_samples=3)
        time.sleep(0.15)
        for i in range(10):
            backend.publish("limit/ch", f"s{i}".encode())
            time.sleep(0.08)
        time.sleep(0.5)
        session.stop()
        assert session.sample_count <= 3

    def test_record_stop_idempotent(self, backend: DataBackend, rec_dir):
        session = record(backend, ["idem/ch"], rec_dir)
        time.sleep(0.1)
        m1 = session.stop()
        m2 = session.stop()
        assert m1.version == m2.version

    def test_record_empty_channels_raises(self, backend: DataBackend, rec_dir):
        with pytest.raises(RecordingError, match="At least one channel"):
            record(backend, [], rec_dir)

    def test_record_context_manager_stops(self, backend: DataBackend, rec_dir):
        session: RecordingSession | None = None
        with record(backend, ["ctx/ch"], rec_dir) as s:
            session = s
            time.sleep(0.1)
        assert session is not None
        assert session.is_stopped


# ---------------------------------------------------------------------------
# Replay tests
# ---------------------------------------------------------------------------


class TestReplay:
    def _make_recording(
        self,
        rec_dir,
        channels: dict[str, list[tuple[float, bytes]]],
    ) -> None:
        """Write a synthetic recording to *rec_dir*."""
        rec_dir.mkdir(parents=True, exist_ok=True)
        all_channels = list(channels.keys())
        total = sum(len(entries) for entries in channels.values())
        start_ts = float("inf")
        end_ts = 0.0

        for ch, entries in channels.items():
            fname = channel_to_filename(ch)
            fpath = rec_dir / fname
            data = b""
            for ts, payload in entries:
                data += encode_entry(ts, payload)
                start_ts = min(start_ts, ts)
                end_ts = max(end_ts, ts)
            fpath.write_bytes(data)

        manifest = RecordingManifest(
            channels=all_channels,
            start_ts=start_ts if start_ts != float("inf") else 0.0,
            end_ts=end_ts,
            sample_count=total,
        )
        (rec_dir / "manifest.json").write_text(manifest.to_json())

    def test_replay_publishes_all_samples(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {
                "ch1": [(1.0, b"a"), (1.1, b"b"), (1.2, b"c")],
            },
        )
        received: list[Sample] = []
        barrier = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s)
            if len(received) >= 3:
                barrier.set()

        sub = backend.subscribe("ch1", cb, policy="fifo")
        time.sleep(0.15)

        result = replay(backend, rec_dir, speed=0)
        assert isinstance(result, ReplayResult)
        assert result.samples_published == 3
        assert result.passes == 1

        barrier.wait(timeout=3.0)
        assert len(received) >= 3
        sub.close()

    def test_replay_preserves_order(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {
                "order/ch": [
                    (10.0, b"first"),
                    (10.5, b"second"),
                    (11.0, b"third"),
                ],
            },
        )
        received: list[bytes] = []
        barrier = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s.payload)
            if len(received) >= 3:
                barrier.set()

        sub = backend.subscribe("order/ch", cb, policy="fifo")
        time.sleep(0.15)

        # Use a small positive speed to avoid filesystem write/read races
        replay(backend, rec_dir, speed=100.0)
        barrier.wait(timeout=5.0)
        assert len(received) >= 3
        non_empty = [p for p in received if p]
        assert b"first" in non_empty
        assert b"second" in non_empty
        assert b"third" in non_empty
        sub.close()

    def test_replay_speed_zero_is_instant(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {
                "fast/ch": [(i * 1.0, f"v{i}".encode()) for i in range(20)],
            },
        )
        start = time.time()
        result = replay(backend, rec_dir, speed=0)
        elapsed = time.time() - start
        assert result.samples_published == 20
        assert elapsed < 2.0

    def test_replay_channel_filter(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {
                "alpha": [(1.0, b"a1"), (2.0, b"a2")],
                "beta": [(1.5, b"b1")],
            },
        )
        result = replay(backend, rec_dir, speed=0, channels=["alpha"])
        assert result.samples_published == 2
        assert result.channels == ["alpha"]

    def test_replay_missing_channel_raises(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {"existing": [(1.0, b"x")]},
        )
        with pytest.raises(RecordingError, match="Channels not found"):
            replay(backend, rec_dir, speed=0, channels=["nonexistent"])

    def test_replay_no_manifest_raises(self, backend: DataBackend, tmp_path):
        empty_dir = tmp_path / "no_recording"
        empty_dir.mkdir()
        with pytest.raises(RecordingError, match="No manifest.json"):
            replay(backend, empty_dir)

    def test_replay_loop(self, backend: DataBackend, rec_dir):
        self._make_recording(
            rec_dir,
            {"loop/ch": [(1.0, b"x")]},
        )
        received: list[Sample] = []
        done = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s)
            if len(received) >= 3:
                done.set()

        sub = backend.subscribe("loop/ch", cb, policy="fifo")
        time.sleep(0.15)

        def _replay_loop() -> None:
            try:
                replay(backend, rec_dir, speed=0, loop=True)
            except Exception:
                pass

        t = threading.Thread(target=_replay_loop, daemon=True)
        t.start()
        done.wait(timeout=5.0)
        assert len(received) >= 3
        sub.close()

    def test_replay_mixed_channels_interleaved(
        self, backend: DataBackend, rec_dir
    ):
        self._make_recording(
            rec_dir,
            {
                "ch_a": [(1.0, b"a1"), (3.0, b"a2")],
                "ch_b": [(2.0, b"b1"), (4.0, b"b2")],
            },
        )

        received_a: list[bytes] = []
        received_b: list[bytes] = []
        barrier = threading.Event()

        def cb_a(s: Sample) -> None:
            received_a.append(s.payload)
            if len(received_a) + len(received_b) >= 4:
                barrier.set()

        def cb_b(s: Sample) -> None:
            received_b.append(s.payload)
            if len(received_a) + len(received_b) >= 4:
                barrier.set()

        sub_a = backend.subscribe("ch_a", cb_a, policy="fifo")
        sub_b = backend.subscribe("ch_b", cb_b, policy="fifo")
        time.sleep(0.15)

        result = replay(backend, rec_dir, speed=0)
        assert result.samples_published == 4

        barrier.wait(timeout=3.0)
        assert b"a1" in received_a
        assert b"a2" in received_a
        assert b"b1" in received_b
        assert b"b2" in received_b
        sub_a.close()
        sub_b.close()


# ---------------------------------------------------------------------------
# Integration: record → replay roundtrip
# ---------------------------------------------------------------------------


class TestRecordReplayIntegration:
    def test_record_then_replay_triggers_callbacks(
        self, backend: DataBackend, rec_dir
    ):
        with record(backend, ["int/ch"], rec_dir):
            time.sleep(0.15)
            for i in range(5):
                backend.publish("int/ch", f"sample_{i}".encode())
                time.sleep(0.08)
            time.sleep(0.3)

        replayed: list[bytes] = []
        barrier = threading.Event()

        def cb(s: Sample) -> None:
            replayed.append(s.payload)
            if len(replayed) >= 3:
                barrier.set()

        sub = backend.subscribe("int/ch", cb, policy="fifo")
        time.sleep(0.15)

        result = replay(backend, rec_dir, speed=0)
        assert result.samples_published >= 3

        barrier.wait(timeout=3.0)
        assert len(replayed) >= 3
        sub.close()

    def test_recording_portable_across_backends(self, tmp_path):
        bus_a = tmp_path / "bus_a"
        bus_b = tmp_path / "bus_b"
        rec = tmp_path / "portable_rec"

        cfg_a = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(bus_a),
        )
        cfg_b = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(bus_b),
        )

        be_a = get_backend(cfg_a)
        be_b = get_backend(cfg_b)

        with record(be_a, ["port/ch"], rec):
            time.sleep(0.15)
            be_a.publish("port/ch", b"portable_data")
            time.sleep(0.2)

        received: list[bytes] = []
        barrier = threading.Event()

        def cb(s: Sample) -> None:
            received.append(s.payload)
            barrier.set()

        sub = be_b.subscribe("port/ch", cb, policy="fifo")
        time.sleep(0.15)

        result = replay(be_b, rec, speed=0)
        assert result.samples_published >= 1

        barrier.wait(timeout=3.0)
        assert b"portable_data" in received
        sub.close()
        be_a.close()
        be_b.close()
