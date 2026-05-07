"""Tests for CYB-1713: multi-camera / multi-twin detection routing.

Covers:
  - ``LoadedModel.predict(twin_uuid=...)`` routing detections to the correct twin
  - Cross-twin ``@cw.on_synchronized(twin_channels=...)`` dispatch
  - Backward compatibility of single-twin ``@cw.on_synchronized``
  - Validation and duplicate-registration warnings
"""

from __future__ import annotations

import time
from typing import Any, Callable
from unittest.mock import MagicMock, PropertyMock

import pytest

from cyberwave.data.backend import Sample, Subscription
from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.types import BoundingBox, Detection, PredictionResult
from cyberwave.workers.context import HookContext
from cyberwave.workers.hooks import HookRegistry, SynchronizedGroup
from cyberwave.workers.runtime import WorkerRuntime

TWIN_A = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
TWIN_B = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"


# ── Fakes ────────────────────────────────────────────────────────────


class FakeSubscription(Subscription):
    def close(self) -> None:
        pass


class FakeBackend:
    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable[[Sample], None]]] = {}
        self._published: list[tuple[str, bytes]] = []

    def subscribe(
        self,
        key: str,
        callback: Callable[[Sample], None],
        **_kwargs: Any,
    ) -> FakeSubscription:
        self._callbacks.setdefault(key, []).append(callback)
        return FakeSubscription()

    def publish(self, key: str, payload: bytes, **_kwargs: Any) -> None:
        self._published.append((key, payload))

    def inject(self, key: str, sample: Sample) -> None:
        for cb in self._callbacks.get(key, []):
            cb(sample)

    def latest(self, key: str, timeout_s: float = 1.0) -> Sample | None:
        for pub_key, payload in reversed(self._published):
            if pub_key == key:
                return Sample(channel=key, payload=payload)
        return None


class FakeDataBus:
    KEY_PREFIX = "cw"

    def __init__(self) -> None:
        self._backend = FakeBackend()

    @property
    def backend(self) -> FakeBackend:
        return self._backend

    @property
    def key_prefix(self) -> str:
        return self.KEY_PREFIX

    def publish_raw(
        self,
        channel: str,
        payload: bytes,
        *,
        twin_uuid: str | None = None,
    ) -> None:
        from cyberwave.data.keys import build_key

        effective_twin = twin_uuid if twin_uuid is not None else TWIN_A
        key = build_key(effective_twin, channel, prefix=self.KEY_PREFIX)
        self._backend.publish(key, payload)

    def inject(self, key: str, sample: Sample) -> None:
        self._backend.inject(key, sample)


class FakeCyberwave:
    def __init__(self, data_bus: FakeDataBus) -> None:
        self._hook_registry = HookRegistry()
        self._data_bus = data_bus

    @property
    def data(self) -> FakeDataBus:
        return self._data_bus


def _make_sample(channel: str, payload: bytes, ts: float) -> Sample:
    return Sample(channel=channel, payload=payload, timestamp=ts)


# ── Multi-twin detection routing tests ───────────────────────────────


class TestMultiTwinDetectionRouting:
    """Verify that ``model.predict(twin_uuid=X)`` publishes to the correct
    twin's ``detections/{runtime}`` Zenoh key.
    """

    @staticmethod
    def _detection_result() -> PredictionResult:
        return PredictionResult(
            detections=[
                Detection(
                    label="person",
                    confidence=0.92,
                    bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
                )
            ],
        )

    @staticmethod
    def _make_model(bus: FakeDataBus) -> LoadedModel:
        runtime = MagicMock()
        type(runtime).name = PropertyMock(return_value="ultralytics")
        return LoadedModel(
            name="yolov8n",
            runtime=runtime,
            model_handle=MagicMock(),
            device="cpu",
            data_bus=bus,
        )

    @pytest.fixture()
    def bus(self) -> FakeDataBus:
        return FakeDataBus()

    def test_predict_routes_to_twin_a(self, bus: FakeDataBus) -> None:
        model = self._make_model(bus)
        model._runtime.predict.return_value = self._detection_result()

        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(frame, confidence=0.5, twin_uuid=TWIN_A)

        sample = bus.backend.latest(
            f"cw/{TWIN_A}/data/detections/ultralytics", timeout_s=0.5
        )
        assert sample is not None

        sample_b = bus.backend.latest(
            f"cw/{TWIN_B}/data/detections/ultralytics", timeout_s=0.5
        )
        assert sample_b is None

    def test_predict_routes_to_twin_b(self, bus: FakeDataBus) -> None:
        model = self._make_model(bus)
        model._runtime.predict.return_value = self._detection_result()

        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(frame, confidence=0.5, twin_uuid=TWIN_B)

        sample = bus.backend.latest(
            f"cw/{TWIN_B}/data/detections/ultralytics", timeout_s=0.5
        )
        assert sample is not None

        sample_a = bus.backend.latest(
            f"cw/{TWIN_A}/data/detections/ultralytics", timeout_s=0.5
        )
        assert sample_a is None

    def test_predict_without_twin_uuid_backward_compat(self, bus: FakeDataBus) -> None:
        """Existing code without twin_uuid continues to work."""
        model = self._make_model(bus)
        model._runtime.predict.return_value = self._detection_result()

        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.predict(frame, confidence=0.5)
        assert len(result.detections) == 1

        sample = bus.backend.latest(
            f"cw/{TWIN_A}/data/detections/ultralytics", timeout_s=0.5
        )
        assert sample is not None


# ── Cross-twin synchronized hook tests ───────────────────────────────


class TestCrossTwinSynchronized:
    @pytest.fixture()
    def bus(self) -> FakeDataBus:
        return FakeDataBus()

    @pytest.fixture()
    def cw(self, bus: FakeDataBus) -> FakeCyberwave:
        return FakeCyberwave(bus)

    @pytest.fixture()
    def registry(self, cw: FakeCyberwave) -> HookRegistry:
        return cw._hook_registry

    def test_cross_twin_subscribes_to_correct_keys(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        @registry.on_synchronized(
            twin_channels={
                "left": (TWIN_A, "frames/default"),
                "right": (TWIN_B, "frames/default"),
            },
            tolerance_ms=100,
        )
        def handler(samples: Any, ctx: Any) -> None:
            pass

        rt = WorkerRuntime(cw)
        rt.start()

        expected_keys = {
            f"cw/{TWIN_A}/data/frames/default",
            f"cw/{TWIN_B}/data/frames/default",
        }
        assert set(bus.backend._callbacks.keys()) == expected_keys

    def test_cross_twin_fires_when_aligned(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []
        contexts: list[HookContext] = []

        @registry.on_synchronized(
            twin_channels={
                "left": (TWIN_A, "frames/default"),
                "right": (TWIN_B, "frames/default"),
            },
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)
            contexts.append(ctx)

        rt = WorkerRuntime(cw)
        rt.start()

        now = time.time()
        bus.inject(
            f"cw/{TWIN_A}/data/frames/default",
            _make_sample("frames/default", b"frame-a", now),
        )
        bus.inject(
            f"cw/{TWIN_B}/data/frames/default",
            _make_sample("frames/default", b"frame-b", now + 0.01),
        )

        assert len(received) == 1
        assert set(received[0].keys()) == {"left", "right"}
        assert received[0]["left"].payload == b"frame-a"
        assert received[0]["right"].payload == b"frame-b"

        assert contexts[0].metadata["twin_uuids"] == sorted([TWIN_A, TWIN_B])

    def test_cross_twin_does_not_fire_when_stale(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            twin_channels={
                "left": (TWIN_A, "frames/default"),
                "right": (TWIN_B, "frames/default"),
            },
            tolerance_ms=50,
        )
        def handler(samples: Any, ctx: Any) -> None:
            received.append(samples)

        rt = WorkerRuntime(cw)
        rt.start()

        now = time.time()
        bus.inject(
            f"cw/{TWIN_A}/data/frames/default",
            _make_sample("frames/default", b"frame-a", now),
        )
        bus.inject(
            f"cw/{TWIN_B}/data/frames/default",
            _make_sample("frames/default", b"frame-b", now + 0.2),
        )

        assert len(received) == 0

    def test_single_twin_backward_compat(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_A,
            ["frames/front", "depth/default"],
            tolerance_ms=100,
        )
        def handler(samples: Any, ctx: Any) -> None:
            received.append(samples)

        rt = WorkerRuntime(cw)
        rt.start()

        expected_keys = {
            f"cw/{TWIN_A}/data/frames/front",
            f"cw/{TWIN_A}/data/depth/default",
        }
        assert set(bus.backend._callbacks.keys()) == expected_keys

        now = time.time()
        bus.inject(
            f"cw/{TWIN_A}/data/frames/front",
            _make_sample("frames/front", b"frame", now),
        )
        bus.inject(
            f"cw/{TWIN_A}/data/depth/default",
            _make_sample("depth/default", b"depth", now + 0.01),
        )

        assert len(received) == 1
        assert set(received[0].keys()) == {"frames/front", "depth/default"}


class TestCrossTwinWithDifferentChannels:
    """Cross-twin sync with distinct channel names per twin."""

    @pytest.fixture()
    def bus(self) -> FakeDataBus:
        return FakeDataBus()

    @pytest.fixture()
    def cw(self, bus: FakeDataBus) -> FakeCyberwave:
        return FakeCyberwave(bus)

    @pytest.fixture()
    def registry(self, cw: FakeCyberwave) -> HookRegistry:
        return cw._hook_registry

    def test_different_channels_different_twins(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            twin_channels={
                "rgb": (TWIN_A, "frames/front"),
                "depth": (TWIN_B, "depth/default"),
            },
            tolerance_ms=100,
        )
        def handler(samples: Any, ctx: Any) -> None:
            received.append(samples)

        rt = WorkerRuntime(cw)
        rt.start()

        expected_keys = {
            f"cw/{TWIN_A}/data/frames/front",
            f"cw/{TWIN_B}/data/depth/default",
        }
        assert set(bus.backend._callbacks.keys()) == expected_keys

        now = time.time()
        bus.inject(
            f"cw/{TWIN_A}/data/frames/front",
            _make_sample("frames/front", b"frame-a", now),
        )
        bus.inject(
            f"cw/{TWIN_B}/data/depth/default",
            _make_sample("depth/default", b"depth-b", now + 0.01),
        )

        assert len(received) == 1
        assert received[0]["rgb"].payload == b"frame-a"
        assert received[0]["depth"].payload == b"depth-b"


# ── Validation tests ─────────────────────────────────────────────────


class TestOnSynchronizedValidation:
    def test_raises_without_channels_or_twin_channels(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="requires either"):

            @registry.on_synchronized(TWIN_A, tolerance_ms=50)
            def handler(samples: Any, ctx: Any) -> None:
                pass

    def test_raises_with_both_channels_and_twin_channels(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="not both"):

            @registry.on_synchronized(
                TWIN_A,
                ["frames/front"],
                twin_channels={"left": (TWIN_A, "frames/default")},
            )
            def handler(samples: Any, ctx: Any) -> None:
                pass

    def test_duplicate_cross_twin_warning(self) -> None:
        registry = HookRegistry()

        def my_handler(samples: Any, ctx: Any) -> None:
            pass

        registry.on_synchronized(
            twin_channels={
                "left": (TWIN_A, "frames/default"),
                "right": (TWIN_B, "frames/default"),
            },
        )(my_handler)

        with pytest.warns(UserWarning, match="Duplicate"):
            registry.on_synchronized(
                twin_channels={
                    "left": (TWIN_A, "frames/default"),
                    "right": (TWIN_B, "frames/default"),
                },
            )(my_handler)


class TestSynchronizedGroupDataclass:
    def test_single_twin_channels_are_bare(self) -> None:
        group = SynchronizedGroup(
            channels=("frames/front", "depth/default"),
            twin_uuid=TWIN_A,
            callback=lambda s, c: None,
        )
        assert group.channels == ("frames/front", "depth/default")
        assert group.twin_channels == ()

    def test_cross_twin_channels_are_labels(self) -> None:
        group = SynchronizedGroup(
            channels=("left", "right"),
            twin_uuid="",
            callback=lambda s, c: None,
            twin_channels=(
                ("left", TWIN_A, "frames/default"),
                ("right", TWIN_B, "frames/default"),
            ),
        )
        assert group.channels == ("left", "right")
        assert len(group.twin_channels) == 2
        assert group.twin_channels[0] == ("left", TWIN_A, "frames/default")
        assert group.twin_channels[1] == ("right", TWIN_B, "frames/default")
