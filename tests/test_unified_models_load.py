"""Tests for the unified ``cw.models.load(...)`` surface.

The whole point of this refactor: edge and cloud models are loaded and
invoked through the same API. ``ModelManager.load()`` dispatches based
on the shape of the identifier — local catalog ids (``yolov8n``,
``yolov8n-pose-onnx``) resolve via the on-node weights cache, Cyberwave
slugs (``ws/models/name``) and UUIDs resolve through the Playground.

These tests stub out the cloud client so they exercise the dispatch
logic without needing a live backend.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from cyberwave.models import CloudLoadedModel, ModelManager
from cyberwave.models.cloud import _to_prediction_result
from cyberwave.models.types import PredictionResult
from cyberwave.rest.models.ml_model_run_queued_schema import MLModelRunQueuedSchema
from cyberwave.rest.models.ml_model_run_result_schema import MLModelRunResultSchema
from cyberwave.rest.models.ml_model_schema import MLModelSchema


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePlaygroundClient:
    """In-memory stand-in for :class:`cyberwave.models.playground.PlaygroundClient`.

    Records every ``get()`` and ``run()`` call so tests can assert on
    the slug / UUID / kwargs that flow through. Returns a configurable
    result so individual tests can simulate boxes / points / queued /
    plain-text outputs without re-implementing the HTTP layer.
    """

    def __init__(
        self,
        summary: MLModelSchema,
        run_result: MLModelRunResultSchema | MLModelRunQueuedSchema,
    ) -> None:
        self._summary = summary
        self._run_result = run_result
        self.get_calls: list[str] = []
        self.run_calls: list[dict] = []

    def get(self, model_ref: str) -> MLModelSchema:
        self.get_calls.append(model_ref)
        return self._summary

    def run(self, model, **kwargs) -> MLModelRunResultSchema | MLModelRunQueuedSchema:
        self.run_calls.append({"model": model, **kwargs})
        return self._run_result

    def fetch_weights_url(self, model) -> dict:
        return {}

    def __call__(self, model_ref):
        from cyberwave.models.playground import PlaygroundHandle
        handle = PlaygroundHandle(model_ref=model_ref, api=None)  # type: ignore[arg-type]
        handle._resolved = self._summary
        handle._client_override = self  # type: ignore[attr-defined]
        # Monkey-patch run so it delegates to this fake
        def _patched_run(**kwargs):
            self.run_calls.append({"model": self._summary, **kwargs})
            handle._last_result = self._run_result
            return self._run_result

        handle.run = _patched_run  # type: ignore[method-assign]
        return handle


def _summary(slug: str = "acme/models/sam-3.1", uuid_str: str | None = None) -> MLModelSchema:
    return MLModelSchema(
        uuid=uuid_str or str(uuid4()),
        slug=slug,
        name="SAM 3.1",
        description="",
        model_external_id="sam-3.1",
        model_provider_name="custom",
        output_format="json",
        deployment="cloud",
        workspace_uuid=str(uuid4()),
        metadata={},
        visibility="public",
        tags=["segmentation"],
        is_trainable=False,
        supported_level="stable",
        can_take_video_as_input=False,
        can_take_audio_as_input=False,
        can_take_image_as_input=True,
        can_take_text_as_input=True,
        can_take_action_as_input=False,
        is_edge_compatible=False,
        is_cloud_compatible=True,
        allowed_structured_tasks=["segment", "detect_boxes", "free"],
        execution_surfaces=["playground"],
        created_at="2024-01-01T00:00:00Z",  # type: ignore[arg-type]
        updated_at="2024-01-01T00:00:00Z",  # type: ignore[arg-type]
    )


def _completed_result(output_format: str, output) -> MLModelRunResultSchema:
    return MLModelRunResultSchema(
        status="completed",
        output_format=output_format,
        output=output,
        raw=json.dumps(output) if isinstance(output, (list, dict)) else None,
    )


def _queued_result(workload_uuid: str = "wl-1") -> MLModelRunQueuedSchema:
    return MLModelRunQueuedSchema(
        status="queued",
        workload_uuid=workload_uuid,
        poll_url=f"/cloud-node-workloads/{workload_uuid}",
    )


# ---------------------------------------------------------------------------
# ModelManager dispatch
# ---------------------------------------------------------------------------


class TestLooksLikeCloudRef:
    @pytest.mark.parametrize(
        "ref,expected",
        [
            ("acme/models/sam-3.1", True),
            ("the-robot-studio/models/openvla", True),
            ("yolov8n", False),
            ("yolov8n.pt", False),
            ("yolov8n-pose-onnx", False),
            ("weights/latest.pt", False),  # ambiguous but has no /models/
            ("", False),
        ],
    )
    def test_slug_heuristic(self, ref: str, expected: bool) -> None:
        assert ModelManager._looks_like_cloud_ref(ref) is expected

    def test_uuid_is_cloud(self) -> None:
        assert ModelManager._looks_like_cloud_ref(str(uuid4())) is True


class TestLoadCloudDispatch:
    def test_slug_routes_to_cloud_client_and_returns_cloud_loaded_model(self) -> None:
        summary = _summary()
        run_result = _completed_result(
            "boxes",
            [{"box_2d": [10, 20, 30, 40], "label": "cup"}],
        )
        fake_client = _FakePlaygroundClient(summary, run_result)
        mgr = ModelManager()
        mgr._mlmodels_client = fake_client

        model = mgr.load("acme/models/sam-3.1")

        assert isinstance(model, CloudLoadedModel)
        assert fake_client.get_calls == ["acme/models/sam-3.1"]
        # Same slug must be re-used from the cache on a second call.
        model2 = mgr.load("acme/models/sam-3.1")
        assert model2 is model
        assert fake_client.get_calls == ["acme/models/sam-3.1"]  # no second get

    def test_uuid_routes_to_cloud_client(self) -> None:
        uid = str(uuid4())
        summary = _summary(slug=None, uuid_str=uid)  # type: ignore[arg-type]
        fake_client = _FakePlaygroundClient(summary, _completed_result("text", "ok"))
        mgr = ModelManager()
        mgr._mlmodels_client = fake_client

        model = mgr.load(uid)

        assert isinstance(model, CloudLoadedModel)
        assert fake_client.get_calls == [uid]

    def test_local_catalog_id_does_not_touch_cloud_client(self) -> None:
        fake_client = _FakePlaygroundClient(
            _summary(), _completed_result("text", "unused")
        )
        mgr = ModelManager()
        mgr._mlmodels_client = fake_client

        # yolov8n isn't present in the fake model cache, so this will raise
        # from the local resolver (FileNotFoundError when the runtime is
        # installed, ImportError in CI without the optional extras). The
        # important assertion is that we never called the cloud client first.
        with pytest.raises((FileNotFoundError, ImportError, ValueError)):
            mgr.load("yolov8n")

        assert fake_client.get_calls == []

    def test_runtime_edge_forces_local_even_for_cloud_shaped_slug(self) -> None:
        fake_client = _FakePlaygroundClient(
            _summary(), _completed_result("text", "unused")
        )
        mgr = ModelManager()
        mgr._mlmodels_client = fake_client

        # Caller explicitly asked for an edge runtime — we should not
        # hijack to cloud even though "acme/models/sam-3.1" matches the
        # slug heuristic.
        with pytest.raises((FileNotFoundError, ImportError, ValueError)):
            mgr.load("acme/models/sam-3.1", runtime="edge")

        assert fake_client.get_calls == []

    def test_no_cloud_client_attached_falls_back_to_local(self) -> None:
        mgr = ModelManager()  # no mlmodels_client
        # Slug-shaped id; falls through to local which will raise
        # because the file isn't in the cache (or the runtime isn't installed).
        with pytest.raises((FileNotFoundError, ImportError, ValueError)):
            mgr.load("acme/models/sam-3.1")


# ---------------------------------------------------------------------------
# CloudLoadedModel.predict
# ---------------------------------------------------------------------------


class TestCloudLoadedModelPredict:
    def test_predict_forwards_image_prompt_and_structured_task(self) -> None:
        summary = _summary()
        fake_client = _FakePlaygroundClient(
            summary,
            _completed_result("boxes", [{"box_2d": [0, 0, 10, 10], "label": "x"}]),
        )
        model = CloudLoadedModel(summary=summary, client=fake_client)

        result = model.predict("scene.jpg", prompt="cup", structured_task="detect_boxes")

        assert isinstance(result, PredictionResult)
        assert len(fake_client.run_calls) == 1
        call = fake_client.run_calls[0]
        assert call["model"] is summary
        assert call["image"] == "scene.jpg"
        assert call["prompt"] == "cup"
        assert call["structured_task"] == "detect_boxes"

    def test_predict_stores_last_result_for_postamble_helpers(self) -> None:
        summary = _summary()
        run_result = _completed_result("masks", [{"box_2d": [0, 0, 5, 5], "mask": "b64"}])
        fake_client = _FakePlaygroundClient(summary, run_result)
        model = CloudLoadedModel(summary=summary, client=fake_client)

        model.predict("scene.jpg", prompt="cup", structured_task="segment")

        assert model.last_result is run_result

    def test_predict_warns_when_confidence_and_classes_are_ignored(self) -> None:
        """Parity with :class:`LoadedModel.predict` is preserved, but callers
        should be told these kwargs are not part of the cloud contract yet.
        """
        summary = _summary()
        fake_client = _FakePlaygroundClient(summary, _completed_result("text", "ok"))
        model = CloudLoadedModel(summary=summary, client=fake_client)

        with pytest.warns(RuntimeWarning) as record:
            model.predict("frame.jpg", confidence=0.25, classes=["cup", "mug"])

        messages = [str(w.message) for w in record.list]
        assert any("ignores confidence" in m for m in messages), messages
        assert any("ignores classes" in m for m in messages), messages

        # confidence/classes must NOT be forwarded to the cloud client
        assert "confidence" not in fake_client.run_calls[0]
        assert "classes" not in fake_client.run_calls[0]

    def test_predict_forwards_advanced_kwargs_verbatim(self) -> None:
        """``frames``, ``depth_base64``, ``camera_intrinsics``, ``camera_pose``,
        ``history`` and ``params`` should reach the cloud client unchanged.
        """
        summary = _summary()
        fake_client = _FakePlaygroundClient(summary, _completed_result("text", "ok"))
        model = CloudLoadedModel(summary=summary, client=fake_client)

        model.predict(
            "frame.jpg",
            prompt="describe",
            frames=[{"image_base64": "..."}],
            depth_base64="depth",
            camera_intrinsics={"fx": 600},
            camera_pose={"position": [0, 0, 0]},
            history=[{"role": "user", "content": "hi"}],
            params={"temperature": 0.2},
        )

        call = fake_client.run_calls[0]
        assert call["frames"] == [{"image_base64": "..."}]
        assert call["depth_base64"] == "depth"
        assert call["camera_intrinsics"] == {"fx": 600}
        assert call["camera_pose"] == {"position": [0, 0, 0]}
        assert call["history"] == [{"role": "user", "content": "hi"}]
        assert call["params"] == {"temperature": 0.2}


# ---------------------------------------------------------------------------
# Result translation
# ---------------------------------------------------------------------------


class TestResultTranslation:
    def test_boxes_become_detections(self) -> None:
        result = _completed_result(
            "boxes",
            [
                {"box_2d": [10, 20, 30, 40], "label": "cup", "score": 0.8},
                {"box_2d": [100, 100, 150, 200], "label": "mug"},
            ],
        )
        pr = _to_prediction_result(result)
        assert len(pr) == 2
        assert pr[0].label == "cup"
        assert pr[0].bbox.x1 == 20 and pr[0].bbox.x2 == 40
        assert 0.79 < pr[0].confidence < 0.81
        assert pr.metadata["output_format"] == "boxes"
        assert pr.metadata["mlmodel_run_result"] is result

    def test_points_become_zero_area_detections_with_raw_point_on_mask(self) -> None:
        result = _completed_result(
            "points", [{"point": [50, 100], "label": "cup"}]
        )
        pr = _to_prediction_result(result)
        assert len(pr) == 1
        det = pr[0]
        assert det.bbox.x1 == 100 and det.bbox.x2 == 100
        assert det.bbox.y1 == 50 and det.bbox.y2 == 50
        assert det.mask == [50, 100]

    def test_masks_attach_base64_payload_to_detection(self) -> None:
        result = _completed_result(
            "masks",
            [
                {"box_2d": [0, 0, 10, 10], "mask": "iVBORw0KGgo", "label": "cup"},
            ],
        )
        pr = _to_prediction_result(result)
        assert len(pr) == 1
        assert pr[0].mask == "iVBORw0KGgo"

    def test_text_output_leaves_detections_empty_but_preserves_raw(self) -> None:
        result = _completed_result("text", "A cup is on the table.")
        pr = _to_prediction_result(result)
        assert len(pr) == 0
        assert pr.raw == "A cup is on the table."
        assert pr.metadata["output_format"] == "text"

    def test_queued_result_carries_workload_uuid_on_metadata(self) -> None:
        result = _queued_result("wl-1")
        pr = _to_prediction_result(result)
        assert len(pr) == 0
        assert pr.metadata["workload_uuid"] == "wl-1"
        assert pr.metadata["poll_url"] == "/cloud-node-workloads/wl-1"


# ---------------------------------------------------------------------------
# Snippet-compat regression
# ---------------------------------------------------------------------------


def test_snippet_shape_works_end_to_end() -> None:
    """Execute the exact shape of the unified Python snippet shipped from
    the frontend. Ensures the three symbols we ask users to call actually
    resolve: ``cw.models.load(slug)``, ``model.predict(image, prompt=...,
    structured_task=...)``, and ``model.last_result``.
    """
    summary = _summary()
    run_result = _completed_result(
        "boxes", [{"box_2d": [10, 20, 30, 40], "label": "cup", "score": 0.9}]
    )
    fake_client = _FakePlaygroundClient(summary, run_result)
    mgr = ModelManager()
    mgr._mlmodels_client = fake_client

    model = mgr.load("acme/models/sam-3.1")
    result = model.predict("scene.jpg", prompt="cup", structured_task="detect_boxes")

    assert isinstance(result, PredictionResult)
    assert model.last_result is run_result

