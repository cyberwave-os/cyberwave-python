"""
Tests for update_twin_position / update_twin_rotation rate-limiting and throughput.

The SDK enforces a 40 Hz cap per channel and silently drops duplicate payloads.
These tests verify that contract at three levels:

Unit tests (always run, ~8 s total):
    TestRateLimiting     – first call published, immediate second dropped, independent
                           channels, call-after-interval published.
    TestDeduplication    – identical payloads skipped even after the rate-limit window.
    TestRateLimitWarnings– a WARNING log is emitted for every dropped call.
    TestThroughput       – hammers both channels for 2 s and asserts ≤ 40 Hz published.
                           Prints actual published / discarded rates (run with -s).

Integration tests (skipped unless env vars are set, ~50 s total):
    TestLivePositionRotation – connects to a real MQTT broker and:
        · runs three 10-second throughput benchmarks (position, rotation, combined);
        · animates the twin along a circular orbit with a yaw spin so changes are
          visible in the Cyberwave frontend.

    Required env vars:
        CYBERWAVE_API_KEY    – API key (https://app.cyberwave.com/profile)
        CYBERWAVE_TWIN_UUID  – UUID of the twin to animate

    Run:
        export CYBERWAVE_API_KEY="<key>"
        export CYBERWAVE_TWIN_UUID="<uuid>"
        pytest tests/test_twin_position_rotation_perf.py -v -s
"""

import logging
import math
import os
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.mqtt import CyberwaveMQTTClient

TWIN_UUID = "test-twin-uuid"
MIN_UPDATE_INTERVAL = 0.025  # 40 Hz – must match mqtt/__init__.py
BENCHMARK_WINDOW = 2.0        # seconds per unit-test throughput benchmark
LIVE_BENCHMARK_WINDOW = 10.0  # seconds per live integration benchmark
MAX_HZ = int(1.0/MIN_UPDATE_INTERVAL)              # SDK rate-limit cap
MAX_EXPECTED = int(MAX_HZ * BENCHMARK_WINDOW) + 2  # +2 for timing jitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mqtt_client():
    """CyberwaveMQTTClient with the underlying paho connection mocked out."""
    with patch("cyberwave.mqtt.mqtt.Client"):
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="test-api-key",
            auto_connect=False,
            source_type="tele",
        )
    client.publish = MagicMock()
    client._handle_twin_update_with_telemetry = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _capture_rate_limit_warnings():
    """Capture all 'Rate limited' WARNING records emitted by the SDK logger.

    Yields a list that is populated with matching LogRecord objects.
    Propagation is suppressed so console output is silent during tests.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Rate limited" in record.getMessage():
                records.append(record)

    sdk_logger = logging.getLogger("cyberwave.mqtt")
    handler = _Collector(level=logging.WARNING)
    original_propagate = sdk_logger.propagate
    sdk_logger.propagate = False
    sdk_logger.addHandler(handler)
    try:
        yield records
    finally:
        sdk_logger.removeHandler(handler)
        sdk_logger.propagate = original_propagate


def _distinct_positions(count: int) -> list[dict]:
    """Return *count* distinct position dicts (avoids deduplication)."""
    return [{"x": math.cos(i * 0.01), "y": math.sin(i * 0.01), "z": 0.5} for i in range(count)]


def _distinct_rotations(count: int) -> list[dict]:
    """Return *count* distinct unit-quaternion dicts (avoids deduplication)."""
    result = []
    for i in range(count):
        half = (i * 0.01) / 2.0
        result.append({"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)})
    return result


# ---------------------------------------------------------------------------
# Rate-limit unit tests (no real sleeping – time is faked)
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_first_position_call_is_published(self, mqtt_client):
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 1.0, "y": 0.0, "z": 0.0})
        mqtt_client.publish.assert_called_once()

    def test_second_immediate_position_call_is_dropped(self, mqtt_client):
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 1.0, "y": 0.0, "z": 0.0})
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 2.0, "y": 0.0, "z": 0.0})
        # Only the first call goes through; the second is rate-limited.
        assert mqtt_client.publish.call_count == 1

    def test_position_call_after_interval_is_published(self, mqtt_client):
        rate_key = f"twin:{TWIN_UUID}:position"
        mqtt_client._last_update_times[rate_key] = time.time() - MIN_UPDATE_INTERVAL - 0.001

        mqtt_client.update_twin_position(TWIN_UUID, {"x": 3.0, "y": 0.0, "z": 0.0})
        mqtt_client.publish.assert_called_once()

    def test_first_rotation_call_is_published(self, mqtt_client):
        mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        mqtt_client.publish.assert_called_once()

    def test_second_immediate_rotation_call_is_dropped(self, mqtt_client):
        mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.1, "w": 0.99})
        assert mqtt_client.publish.call_count == 1

    def test_position_and_rotation_rate_limits_are_independent(self, mqtt_client):
        """Sending position then rotation without delay should publish both – different keys."""
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 1.0, "y": 0.0, "z": 0.0})
        mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        assert mqtt_client.publish.call_count == 2


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_identical_position_is_not_republished(self, mqtt_client):
        pos = {"x": 1.0, "y": 2.0, "z": 3.0}

        # First call: always published.
        mqtt_client.update_twin_position(TWIN_UUID, pos.copy())

        # Advance time past the rate-limit window.
        rate_key = f"twin:{TWIN_UUID}:position"
        mqtt_client._last_update_times[rate_key] = time.time() - MIN_UPDATE_INTERVAL - 0.001

        # Same payload: should be deduped.
        mqtt_client.update_twin_position(TWIN_UUID, pos.copy())
        assert mqtt_client.publish.call_count == 1

    def test_identical_rotation_is_not_republished(self, mqtt_client):
        rot = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

        mqtt_client.update_twin_rotation(TWIN_UUID, rot.copy())

        rate_key = f"twin:{TWIN_UUID}:rotation"
        mqtt_client._last_update_times[rate_key] = time.time() - MIN_UPDATE_INTERVAL - 0.001

        mqtt_client.update_twin_rotation(TWIN_UUID, rot.copy())
        assert mqtt_client.publish.call_count == 1

    def test_changed_position_is_republished_after_interval(self, mqtt_client):
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 1.0, "y": 0.0, "z": 0.0})

        rate_key = f"twin:{TWIN_UUID}:position"
        mqtt_client._last_update_times[rate_key] = time.time() - MIN_UPDATE_INTERVAL - 0.001

        mqtt_client.update_twin_position(TWIN_UUID, {"x": 2.0, "y": 0.0, "z": 0.0})
        assert mqtt_client.publish.call_count == 2


# ---------------------------------------------------------------------------
# Warning-log tests
# ---------------------------------------------------------------------------


class TestRateLimitWarnings:
    def test_rate_limited_position_emits_warning(self, mqtt_client):
        with _capture_rate_limit_warnings() as warnings:
            mqtt_client.update_twin_position(TWIN_UUID, {"x": 1.0, "y": 0.0, "z": 0.0})
            mqtt_client.update_twin_position(TWIN_UUID, {"x": 2.0, "y": 0.0, "z": 0.0})

        assert len(warnings) == 1

    def test_rate_limited_rotation_emits_warning(self, mqtt_client):
        with _capture_rate_limit_warnings() as warnings:
            mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
            mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.1, "w": 0.99})

        assert len(warnings) == 1

    def test_multiple_rate_limited_calls_emit_matching_warnings(self, mqtt_client):
        positions = _distinct_positions(5)

        with _capture_rate_limit_warnings() as warnings:
            for pos in positions:
                mqtt_client.update_twin_position(TWIN_UUID, pos)

        # First call goes through; remaining 4 are rate-limited.
        assert len(warnings) == 4


# ---------------------------------------------------------------------------
# Throughput benchmark tests
# ---------------------------------------------------------------------------


class TestThroughput:
    """Verify the 40 Hz cap holds over a real 2-second window."""

    def _spam_position(self, mqtt_client: CyberwaveMQTTClient, duration: float) -> tuple[int, int, float]:
        """Call update_twin_position as fast as possible for *duration* seconds.

        Returns (sdk_calls, published, elapsed_seconds).
        """
        mqtt_client.publish.reset_mock()
        positions = _distinct_positions(100_000)
        start = time.perf_counter()
        deadline = start + duration
        i = 0
        while time.perf_counter() < deadline:
            mqtt_client.update_twin_position(TWIN_UUID, positions[i % len(positions)])
            i += 1
        elapsed = time.perf_counter() - start
        return i, mqtt_client.publish.call_count, elapsed

    def _spam_rotation(self, mqtt_client: CyberwaveMQTTClient, duration: float) -> tuple[int, int, float]:
        """Call update_twin_rotation as fast as possible for *duration* seconds.

        Returns (sdk_calls, published, elapsed_seconds).
        """
        mqtt_client.publish.reset_mock()
        rotations = _distinct_rotations(100_000)
        start = time.perf_counter()
        deadline = start + duration
        i = 0
        while time.perf_counter() < deadline:
            mqtt_client.update_twin_rotation(TWIN_UUID, rotations[i % len(rotations)])
            i += 1
        elapsed = time.perf_counter() - start
        return i, mqtt_client.publish.call_count, elapsed

    def test_position_publish_rate_does_not_exceed_40hz(self, mqtt_client):
        calls, published, elapsed = self._spam_position(mqtt_client, BENCHMARK_WINDOW)
        discarded = calls - published
        print(
            f"\n  update_twin_position: {calls} SDK calls in {elapsed:.2f}s"
            f"  →  published: {published} ({published / elapsed:.1f} Hz),"
            f"  discarded: {discarded} ({discarded / elapsed:.1f} Hz)"
        )
        assert published <= MAX_EXPECTED, (
            f"Published {published} position updates in {elapsed:.2f}s "
            f"(max allowed: {MAX_EXPECTED} at 40 Hz)"
        )

    def test_rotation_publish_rate_does_not_exceed_40hz(self, mqtt_client):
        calls, published, elapsed = self._spam_rotation(mqtt_client, BENCHMARK_WINDOW)
        discarded = calls - published
        print(
            f"\n  update_twin_rotation: {calls} SDK calls in {elapsed:.2f}s"
            f"  →  published: {published} ({published / elapsed:.1f} Hz),"
            f"  discarded: {discarded} ({discarded / elapsed:.1f} Hz)"
        )
        assert published <= MAX_EXPECTED, (
            f"Published {published} rotation updates in {elapsed:.2f}s "
            f"(max allowed: {MAX_EXPECTED} at 40 Hz)"
        )

    def test_combined_position_and_rotation_each_capped_independently(self, mqtt_client):
        """Combined loop: both channels should each respect the 40 Hz cap."""
        mqtt_client.publish.reset_mock()
        positions = _distinct_positions(100_000)
        rotations = _distinct_rotations(100_000)
        start = time.perf_counter()
        deadline = start + BENCHMARK_WINDOW
        i = 0
        while time.perf_counter() < deadline:
            mqtt_client.update_twin_position(TWIN_UUID, positions[i % len(positions)])
            mqtt_client.update_twin_rotation(TWIN_UUID, rotations[i % len(rotations)])
            i += 1
        elapsed = time.perf_counter() - start
        total_published = mqtt_client.publish.call_count
        total_calls = i * 2
        discarded = total_calls - total_published
        print(
            f"\n  combined (pos + rot): {total_calls} SDK calls in {elapsed:.2f}s"
            f"  →  published: {total_published} ({total_published / elapsed:.1f} Hz),"
            f"  discarded: {discarded} ({discarded / elapsed:.1f} Hz)"
        )
        # publish() is called for both channels, so total is up to 2× the cap.
        assert total_published <= MAX_EXPECTED * 2, (
            f"Total publishes {total_published} exceeded "
            f"2× the 40 Hz cap over {elapsed:.2f}s"
        )

    def test_at_least_one_update_is_published_per_channel(self, mqtt_client):
        """Sanity check: the rate limiter should not block the very first call."""
        mqtt_client.update_twin_position(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0})
        mqtt_client.update_twin_rotation(TWIN_UUID, {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
        assert mqtt_client.publish.call_count == 2


# ---------------------------------------------------------------------------
# Live integration tests – skipped unless credentials are set
# ---------------------------------------------------------------------------

# Visual demo parameters (match the original example script).
_ORBIT_RADIUS = 2.0   # metres
_ORBIT_HEIGHT = 0.5   # metres
_DEMO_DURATION = 10   # seconds
_DEMO_HZ = 40         # target update rate


@pytest.fixture(scope="module")
def live_client():
    """Connect to a real Cyberwave MQTT broker.

    Skipped unless both env vars are present:
        CYBERWAVE_API_KEY   – API key (also needed for REST auth)
        CYBERWAVE_TWIN_UUID – UUID of the twin to animate
    """
    from cyberwave import Cyberwave

    api_key = os.getenv("CYBERWAVE_API_KEY")
    if not api_key:
        pytest.skip("CYBERWAVE_API_KEY not set")

    cw = Cyberwave(api_key=api_key)
    try:
        cw.mqtt.connect()
    except Exception as exc:
        pytest.skip(f"Could not connect to MQTT broker: {exc}")

    yield cw
    cw.disconnect()


@pytest.fixture(scope="module")
def live_twin_uuid():
    twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID")
    if not twin_uuid:
        pytest.skip("CYBERWAVE_TWIN_UUID not set")
    return twin_uuid


class TestLivePositionRotation:
    """Integration tests that publish real MQTT messages.

    Requires:
        export CYBERWAVE_API_KEY="<key>"
        export CYBERWAVE_TWIN_UUID="<uuid>"
        pytest tests/test_twin_position_rotation_perf.py -v -s -k live
    """

    def _live_benchmark(
        self,
        label: str,
        fn,
        duration: float,
        calls_per_iter: int = 1,
    ) -> None:
        """Hammer *fn(i)* for *duration* seconds and print publish stats."""
        with _capture_rate_limit_warnings() as warnings:
            attempts = 0
            start = time.perf_counter()
            deadline = start + duration
            while time.perf_counter() < deadline:
                fn(attempts)
                attempts += 1
            elapsed = time.perf_counter() - start

        rejected = len(warnings)
        total_calls = attempts * calls_per_iter
        published = total_calls - rejected
        print(
            f"\n  {label}: {total_calls} SDK calls in {elapsed:.2f}s"
            f"  →  published: {published} ({published / elapsed:.1f} Hz),"
            f"  discarded: {rejected} ({rejected / elapsed:.1f} Hz)"
        )

    def test_live_position_benchmark(self, live_client, live_twin_uuid):
        """Benchmark position-only updates against the live broker."""
        print("\n── Live benchmark: position-only ───────────────────────────")

        def pos_fn(i: int) -> None:
            angle = i * 0.01
            live_client.mqtt.update_twin_position(
                live_twin_uuid,
                {"x": math.cos(angle), "y": math.sin(angle), "z": 0.5},
            )

        self._live_benchmark("update_twin_position", pos_fn, LIVE_BENCHMARK_WINDOW)

    def test_live_rotation_benchmark(self, live_client, live_twin_uuid):
        """Benchmark rotation-only updates against the live broker."""
        print("\n── Live benchmark: rotation-only ───────────────────────────")

        def rot_fn(i: int) -> None:
            half = (i * 0.01) / 2.0
            live_client.mqtt.update_twin_rotation(
                live_twin_uuid,
                {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
            )

        self._live_benchmark("update_twin_rotation", rot_fn, LIVE_BENCHMARK_WINDOW)

    def test_live_combined_benchmark(self, live_client, live_twin_uuid):
        """Benchmark combined position + rotation updates against the live broker."""
        print("\n── Live benchmark: combined pos + rot ──────────────────────")

        def combined_fn(i: int) -> None:
            angle = i * 0.01
            half = angle / 2.0
            live_client.mqtt.update_twin_position(
                live_twin_uuid,
                {"x": math.cos(angle), "y": math.sin(angle), "z": 0.5},
            )
            live_client.mqtt.update_twin_rotation(
                live_twin_uuid,
                {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
            )

        self._live_benchmark("combined (pos + rot)", combined_fn, LIVE_BENCHMARK_WINDOW, calls_per_iter=2)

    def test_live_visual_demo(self, live_client, live_twin_uuid):
        """Animate the twin along a circular orbit with a continuous yaw spin.

        Open the Cyberwave frontend to watch the twin move.
        The twin completes one full orbit and two full yaw rotations over
        _DEMO_DURATION seconds, paced at _DEMO_HZ so every frame is published.
        """
        print(
            f"\n── Live visual demo ({_DEMO_DURATION}s at {_DEMO_HZ} Hz) ──────────────"
            f"\n   Orbit radius: {_ORBIT_RADIUS:.1f} m  |  Height: {_ORBIT_HEIGHT:.1f} m"
            f"\n   Open the Cyberwave frontend to watch the twin move."
        )

        interval = 1.0 / _DEMO_HZ
        start = time.perf_counter()
        frames = 0
        deadline = start + _DEMO_DURATION

        with _capture_rate_limit_warnings() as jitter_warnings:
            while time.perf_counter() < deadline:
                t = time.perf_counter() - start

                orbit_angle = (2 * math.pi * t) / _DEMO_DURATION
                x = _ORBIT_RADIUS * math.cos(orbit_angle)
                y = _ORBIT_RADIUS * math.sin(orbit_angle)

                yaw = (2 * math.pi * 2 * t) / _DEMO_DURATION
                half_yaw = yaw / 2.0

                live_client.mqtt.update_twin_position(
                    live_twin_uuid, {"x": x, "y": y, "z": _ORBIT_HEIGHT}
                )
                live_client.mqtt.update_twin_rotation(
                    live_twin_uuid,
                    {"x": 0.0, "y": 0.0, "z": math.sin(half_yaw), "w": math.cos(half_yaw)},
                )
                frames += 1

                next_tick = start + frames * interval
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        elapsed = time.perf_counter() - start
        dropped = len(jitter_warnings)
        print(
            f"   Sent {frames} frames in {elapsed:.2f}s ({frames / elapsed:.1f} Hz actual)"
            + (f",  {dropped} frame(s) dropped by jitter" if dropped else "")
        )
        # Pacing at the exact rate-limit boundary means OS scheduling jitter
        # will occasionally cause a frame to arrive slightly early and be
        # dropped by the SDK.  We only verify that the animation ran at the
        # expected rate – the drop count is informational.
        assert frames > 0, "No frames were sent"
        actual_hz = frames / elapsed
        assert abs(actual_hz - _DEMO_HZ) <= 5, (
            f"Actual framerate {actual_hz:.1f} Hz deviates too far from {_DEMO_HZ} Hz"
        )
