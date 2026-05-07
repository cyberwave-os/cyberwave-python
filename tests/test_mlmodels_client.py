"""Tests for :mod:`cyberwave.mlmodels` — cloud Playground client.

The tests stub the auto-generated ``ApiClient`` so they run without network
access and without the full ``cyberwave.rest`` code-generated schemas.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from cyberwave.mlmodels import MLModelsClient, STRUCTURED_ACTIONS, get_action
from cyberwave.mlmodels.client import _looks_like_uuid
from cyberwave.mlmodels.types import MLModelRunResult, MLModelSummary


def _decode_body(body: dict | bytes | str) -> dict:
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


# ---------------------------------------------------------------------------
# Fake ApiClient
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, body: dict | list) -> None:
        self.status = status
        self.data = json.dumps(body).encode("utf-8")

    def read(self) -> None:
        # call_api clients expect a side-effecting ``.read()`` call so the
        # response body is materialised before deserialisation.
        pass


class _FakeApiClient:
    """Minimal stand-in for the auto-generated OpenAPI client."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._responses: list[_FakeResponse] = []

    def queue(self, status: int, body: dict | list) -> None:
        self._responses.append(_FakeResponse(status, body))

    def param_serialize(
        self,
        *,
        method: str,
        resource_path: str,
        query_params=None,
        header_params=None,
        body=None,
        auth_settings=None,
    ):
        return (
            method,
            resource_path,
            list(query_params or []),
            dict(header_params or {}),
            body,
            list(auth_settings or []),
        )

    def call_api(self, *args):
        self.calls.append(args)
        if not self._responses:
            raise RuntimeError("No queued response")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_api() -> _FakeApiClient:
    return _FakeApiClient()


@pytest.fixture
def client(fake_api: _FakeApiClient) -> MLModelsClient:
    return MLModelsClient(fake_api)


@pytest.fixture
def model_payload() -> dict:
    return {
        "uuid": str(uuid4()),
        "slug": "acme/models/gemini-robotics-er",
        "name": "Gemini Robotics ER",
        "model_external_id": "gemini-robotics-er-1.5-preview",
        "model_provider_name": "google",
        "output_format": "json",
        "deployment": "cloud",
        "can_take_image_as_input": True,
        "can_take_text_as_input": True,
        "playground_kind": "vlm-spatial-reasoner",
        "allowed_structured_tasks": ["detect_points", "caption", "free"],
        "execution_surfaces": ["playground"],
        "sdk_load_id": "gemini-er-sdk-id",
        "metadata": {"point_format": "[y, x] normalized to 0-1000"},
        "tags": ["robotics", "vision"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_looks_like_uuid(self) -> None:
        assert _looks_like_uuid(str(uuid4())) is True
        assert _looks_like_uuid("acme/models/foo") is False
        assert _looks_like_uuid("") is False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestStructuredActionsDiscovery:
    def test_list_structured_actions_matches_catalog(self) -> None:
        actions = MLModelsClient.list_structured_actions()
        assert [a.id for a in actions] == [a.id for a in STRUCTURED_ACTIONS]

    def test_get_structured_action_returns_entry(self) -> None:
        action = MLModelsClient.get_structured_action("detect_points")
        assert action is not None
        assert action.output_format == "points"

    def test_catalog_contains_expected_ids(self) -> None:
        ids = {a.id for a in STRUCTURED_ACTIONS}
        assert {"free", "caption", "detect_points", "detect_boxes", "segment"} <= ids

    def test_fetch_structured_actions_catalog_uses_live_endpoint(
        self, client: MLModelsClient, fake_api: _FakeApiClient
    ) -> None:
        fake_api.queue(
            200,
            {
                "version": 1,
                "actions": [
                    {
                        "id": "detect_points",
                        "label": "Detect points",
                        "description": "",
                        "output_format": "points",
                        "default_prompt_template": "t",
                        "requires_image": True,
                        "output_schema": {},
                        "per_model_prompt_templates": {},
                    }
                ],
            },
        )
        catalog = client.fetch_structured_actions_catalog()
        assert catalog["version"] == 1
        assert catalog["actions"][0]["id"] == "detect_points"
        # One HTTP call was made against the canonical endpoint.
        assert fake_api.calls[0][1] == "/api/v1/mlmodels/structured-actions"

    def test_fetch_structured_actions_catalog_falls_back_on_error(
        self, client: MLModelsClient, fake_api: _FakeApiClient
    ) -> None:
        fake_api.queue(500, {"detail": "boom"})
        catalog = client.fetch_structured_actions_catalog()
        # Local mirror is returned; must contain every known action.
        ids = {a["id"] for a in catalog["actions"]}
        assert {a.id for a in STRUCTURED_ACTIONS} == ids


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_by_slug_hits_by_slug_endpoint(
        self, client: MLModelsClient, fake_api: _FakeApiClient, model_payload: dict
    ) -> None:
        fake_api.queue(200, model_payload)
        summary = client.get("acme/models/gemini-robotics-er")
        method, path, query, _headers, _body, _auth = fake_api.calls[0]
        assert method == "GET"
        assert path == "/api/v1/mlmodels/by-slug"
        assert ("slug", "acme/models/gemini-robotics-er") in query
        assert isinstance(summary, MLModelSummary)
        assert summary.slug == model_payload["slug"]
        assert summary.model_external_id == "gemini-robotics-er-1.5-preview"
        assert summary.metadata["point_format"] == "[y, x] normalized to 0-1000"
        assert summary.playground_kind == "vlm-spatial-reasoner"
        assert summary.allowed_structured_tasks == ["detect_points", "caption", "free"]
        assert summary.sdk_load_id == "gemini-er-sdk-id"

    def test_by_uuid_hits_uuid_endpoint(
        self, client: MLModelsClient, fake_api: _FakeApiClient, model_payload: dict
    ) -> None:
        uuid = model_payload["uuid"]
        fake_api.queue(200, model_payload)
        summary = client.get(uuid)
        method, path, *_ = fake_api.calls[0]
        assert method == "GET"
        assert path == f"/api/v1/mlmodels/{uuid}"
        assert summary.uuid == uuid

    def test_rejects_empty_reference(self, client: MLModelsClient) -> None:
        with pytest.raises(ValueError):
            client.get("")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_encodes_image_and_posts_payload(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
        tmp_path,
    ) -> None:
        # First call resolves slug; second call executes /run.
        fake_api.queue(200, model_payload)
        fake_api.queue(
            200,
            {
                "status": "completed",
                "output_format": "points",
                "output": [{"point": [500, 500], "label": "cup"}],
                "raw": "[{...}]",
            },
        )

        img = tmp_path / "scene.png"
        # Minimal 1x1 PNG (IHDR + IDAT + IEND).
        img.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
                "890000000a49444154789c6300010000000500010d0a2db40000000049454e44"
                "ae426082"
            )
        )

        result = client.run(
            "acme/models/gemini-robotics-er",
            prompt="cups",
            image=img,
            structured_task="detect_points",
        )

        assert isinstance(result, MLModelRunResult)
        assert result.is_completed()
        assert result.output_format == "points"
        assert result.model_slug == "acme/models/gemini-robotics-er"
        assert result.model_uuid == model_payload["uuid"]
        assert result.structured_task == "detect_points"

        # Second call is the run POST.
        run_call = fake_api.calls[1]
        method, path, _query, headers, body, _auth = run_call
        assert method == "POST"
        assert path == f"/api/v1/mlmodels/{model_payload['uuid']}/run"
        assert headers["Content-Type"] == "application/json"
        decoded = _decode_body(body)
        assert decoded["prompt"] == "cups"
        assert decoded["structured_task"] == "detect_points"
        assert decoded["image_base64"].startswith("iVBOR")  # PNG header

    def test_run_passes_json_object_to_api_client(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        """The generated REST client serializes JSON bodies itself.

        Passing pre-serialized bytes here causes a second ``json.dumps`` in the
        REST layer and crashes with ``Object of type bytes is not JSON
        serializable``.
        """
        summary = MLModelSummary.from_api(model_payload)
        fake_api.queue(
            200,
            {"status": "completed", "output_format": "text", "output": "ok"},
        )

        client.run(summary, prompt="describe", image=b"fake-image")

        body = fake_api.calls[0][4]
        assert isinstance(body, dict)
        assert body["prompt"] == "describe"
        assert isinstance(body["image_base64"], str)

    def test_run_with_summary_skips_resolution(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        summary = MLModelSummary.from_api(model_payload)
        fake_api.queue(
            200,
            {"status": "completed", "output_format": "text", "output": "hi"},
        )
        result = client.run(summary, prompt="describe the scene")
        assert len(fake_api.calls) == 1  # No resolution call.
        assert result.output == "hi"

    def test_run_rejects_model_scoped_structured_task(
        self,
        client: MLModelsClient,
        model_payload: dict,
    ) -> None:
        summary = MLModelSummary.from_api(model_payload)
        with pytest.raises(ValueError, match="does not advertise structured_task"):
            client.run(summary, prompt="cups", structured_task="detect_boxes")

    def test_run_forwards_extended_envelope_when_set(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        """The extended perception envelope (frames / depth / intrinsics /
        pose / history) must be passed through verbatim. We only assert
        round-trip wire shape here — the backend-side behaviour is
        covered by ``test_mlmodels_run.py``."""
        summary = MLModelSummary.from_api(model_payload)
        fake_api.queue(
            200,
            {
                "status": "completed",
                "output_format": "points",
                "output": [{"point": [10, 20], "label": "cup"}],
            },
        )

        client.run(
            summary,
            prompt="cups",
            structured_task="detect_points",
            frames=[
                {"image_base64": "aGk=", "camera_id": "wrist"},
                {"image_base64": "aGk=", "camera_id": "overhead"},
            ],
            depth_base64="ZGVwdGg=",
            camera_intrinsics={"fx": 600, "fy": 600, "cx": 320, "cy": 240},
            camera_pose={
                "position": [0.0, 0.0, 0.5],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
            history=[{"role": "user", "content": "earlier: cup in scene"}],
        )

        method, path, _query, _headers, body, _auth = fake_api.calls[0]
        assert method == "POST"
        assert path.endswith("/run")
        decoded = _decode_body(body)
        assert decoded["frames"] and len(decoded["frames"]) == 2
        assert decoded["depth_base64"] == "ZGVwdGg="
        assert decoded["camera_intrinsics"]["fx"] == 600
        assert decoded["camera_pose"]["position"] == [0.0, 0.0, 0.5]
        assert decoded["history"][0]["content"].startswith("earlier")

    def test_run_accepts_frames_only_as_primary_input(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        """``frames`` alone must count as a primary input — the client
        must not raise ``Nothing to run`` when the caller only populates
        the multi-frame envelope. The backend auto-promotes frame 0 to
        ``image_base64``, so the wire payload stays valid without the
        SDK duplicating it."""
        summary = MLModelSummary.from_api(model_payload)
        fake_api.queue(
            200,
            {"status": "completed", "output_format": "points", "output": []},
        )

        client.run(
            summary,
            frames=[{"image_base64": "aGk=", "camera_id": "wrist"}],
            structured_task="detect_points",
        )

        decoded = _decode_body(fake_api.calls[0][4])
        assert "frames" in decoded
        assert "image_base64" not in decoded  # not duplicated client-side

    def test_run_rejects_multiple_image_inputs(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        with pytest.raises(ValueError, match="at most one of"):
            client.run(
                MLModelSummary.from_api(model_payload),
                prompt="x",
                image=b"fake",
                image_url="https://example.com/foo.jpg",
            )

    def test_run_requires_at_least_one_input(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        with pytest.raises(ValueError, match="Nothing to run"):
            client.run(MLModelSummary.from_api(model_payload))

    def test_run_surfaces_202_queued_workload(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        summary = MLModelSummary.from_api(model_payload)
        workload_uuid = str(uuid4())
        fake_api.queue(
            202,
            {
                "status": "queued",
                "workload_uuid": workload_uuid,
                "poll_url": f"/api/v1/cloud-node-workloads/{workload_uuid}",
            },
        )
        result = client.run(summary, prompt="make 3D", image_url="https://x/y.jpg")
        assert result.is_queued()
        assert result.is_completed() is False
        assert result.workload_uuid == workload_uuid
        assert result.poll_url.endswith(workload_uuid)

    def test_run_surfaces_http_error(
        self,
        client: MLModelsClient,
        fake_api: _FakeApiClient,
        model_payload: dict,
    ) -> None:
        from cyberwave.exceptions import CyberwaveAPIError

        summary = MLModelSummary.from_api(model_payload)
        fake_api.queue(400, {"detail": "bad image"})
        with pytest.raises(CyberwaveAPIError):
            client.run(summary, prompt="x", image_url="https://x/y.jpg")


class TestRunResult:
    def test_save_annotated_image_rejects_queued(self) -> None:
        result = MLModelRunResult(
            status="queued",
            output_format=None,
            output=None,
            workload_uuid="abc",
            poll_url="/poll/abc",
        )
        with pytest.raises(RuntimeError, match="queued"):
            result.save_annotated_image("nonexistent.png", "out.png")

    def test_save_annotated_image_rejects_text_output(self) -> None:
        result = MLModelRunResult(
            status="completed", output_format="text", output="hi"
        )
        with pytest.raises(RuntimeError, match="spatial output"):
            result.save_annotated_image("nonexistent.png", "out.png")
