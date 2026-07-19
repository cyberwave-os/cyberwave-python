"""Tests for the :class:`cyberwave.rl_tasks.RLTaskClient` REST wrapper.

The client is intentionally thin so the tests focus on:

1. URL/HTTP verb construction for the scene-entity CRUD endpoints.
2. Payload shape for the high-level convenience methods
   (``assign_articulation_entity``, ``assign_rigid_entity``).
3. Round-trip behaviour for ``export_task_spec_python`` and
   ``import_task_spec_python``.

We mock :class:`urllib3.PoolManager.request` directly so the tests can
run without a live backend; the goal is to lock the public contract, not
exercise the network stack.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberwave.rl_tasks import (
    RLTaskClient,
    TaskSpecExport,
    make_action_spec,
    make_action_term,
    make_camera_observation,
    make_custom_observation,
    make_default_ppo_rlmodule_config,
    make_joint_observation,
    make_observation_spec,
    make_observation_term,
    make_position_delta_action,
    make_previous_action_observation,
    make_rl_config_spec,
)


@dataclass
class _FakeConfig:
    base_url: str = "http://example.com"
    api_key: str = "test-key"


class _FakeClient:
    """Minimal stand-in for :class:`cyberwave.Cyberwave`.

    Only the bits the RL task client actually reads (``config.base_url``
    and ``config.api_key``) are stubbed out.
    """

    def __init__(self) -> None:
        self.config = _FakeConfig()


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def patched_requests(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch ``urllib3.PoolManager.request`` and fan it out per HTTP verb.

    The client funnels every call through a single ``PoolManager.request``
    method, but tests still want to assert on each verb independently. We
    dispatch on the ``method`` argument into separate per-verb mocks, each
    invoked as ``mock(url, headers=..., json=..., **kwargs)`` so existing
    assertions on ``call_args.args[0]`` (URL) and ``call_args.kwargs``
    keep working unchanged.
    """

    mocks = {
        "get": MagicMock(),
        "post": MagicMock(),
        "put": MagicMock(),
        "patch": MagicMock(),
        "delete": MagicMock(),
    }

    def _make_response(payload: Any, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.json.return_value = payload
        # urllib3 exposes the raw body via ``.data`` (bytes); encode the
        # payload so error paths (which decode ``.data``) see real content.
        resp.data = b"" if payload is None else _json.dumps(payload).encode()
        return resp

    for mock in mocks.values():
        mock.return_value = _make_response({})

    def dispatch(
        self: Any,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Any = None,
        **kwargs: Any,
    ) -> MagicMock:
        return mocks[method.lower()](url, headers=headers, json=json, **kwargs)

    monkeypatch.setattr("urllib3.PoolManager.request", dispatch)
    # Expose the response factory so tests can override individual return values.
    mocks["_make_response"] = _make_response  # type: ignore[assignment]
    return mocks


def test_list_scene_entities_hits_expected_url(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["get"].return_value = patched_requests["_make_response"](
        [{"name": "robot"}]
    )

    rows = rl.list_scene_entities("task-uuid")

    patched_requests["get"].assert_called_once()
    call_args = patched_requests["get"].call_args
    assert (
        call_args.args[0]
        == "http://example.com/api/v1/rl-tasks/task-uuid/scene-entities"
    )
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert rows == [{"name": "robot"}]


def test_assign_articulation_entity_builds_strict_payload(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"name": "robot", "twin_uuid": "robot-twin", "entity_kind": "articulation"}
    )

    rl.assign_articulation_entity(
        "task-uuid",
        twin_uuid="robot-twin",
        name="robot",
        actuators=[
            {
                "type": "xml_position",
                "target_names_expr": [".*__joint[1-7]$"],
                "effort_limit": 100.0,
            }
        ],
        soft_joint_pos_limit_factor=0.9,
        include_contacts=True,
    )

    assert patched_requests["post"].call_count == 1
    call_args = patched_requests["post"].call_args
    assert call_args.args[0].endswith("/scene-entities")
    payload = call_args.kwargs["json"]
    # Required strict-schema fields:
    assert payload["name"] == "robot"
    assert payload["twin_uuid"] == "robot-twin"
    assert payload["entity_kind"] == "articulation"
    assert payload["base_type"] == "fixed"
    assert payload["include_actuators"] is True
    assert payload["include_contacts"] is True
    assert payload["articulation"]["soft_joint_pos_limit_factor"] == 0.9
    assert payload["articulation"]["actuators"][0]["type"] == "xml_position"


def test_assign_articulation_entity_forwards_entity_cfg_reference(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """``entity_cfg`` must round-trip into the POST payload verbatim.

    Power users wire custom ``EntityCfg`` subclasses (e.g.
    ``OpenArmCfg``) by passing ``entity_cfg={"module": ..., "symbol":
    ...}``. The SDK should not silently drop the reference.
    """

    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"name": "robot"}
    )

    rl.assign_articulation_entity(
        "task-uuid",
        twin_uuid="robot-twin",
        name="robot",
        actuators=[{"type": "xml_position", "target_names_expr": [".*__joint[1-7]$"]}],
        entity_cfg={
            "module": "openarm_entity",
            "symbol": "OpenArmCfg",
            "kind": "class",
        },
    )

    payload = patched_requests["post"].call_args.kwargs["json"]
    assert payload["entity_cfg"] == {
        "module": "openarm_entity",
        "symbol": "OpenArmCfg",
        "kind": "class",
    }


def test_assign_rigid_entity_defaults_include_actuators_false(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"name": "cube"}
    )

    rl.assign_rigid_entity(
        "task-uuid",
        twin_uuid="cube-twin",
        name="cube",
        base_type="free",
    )

    payload = patched_requests["post"].call_args.kwargs["json"]
    assert payload["entity_kind"] == "rigid_object"
    assert payload["base_type"] == "free"
    assert payload["include_actuators"] is False
    # We must NOT silently inject an empty articulation block; the
    # backend would reject it for a rigid_object.
    assert "articulation" not in payload


def test_replace_scene_entities_wraps_entities_key(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"]([])

    rl.replace_scene_entities(
        "task-uuid",
        [
            {
                "name": "cube",
                "twin_uuid": "cube-twin",
                "entity_kind": "rigid_object",
                "base_type": "free",
            }
        ],
    )

    payload = patched_requests["put"].call_args.kwargs["json"]
    assert "entities" in payload
    assert payload["entities"][0]["name"] == "cube"


def test_export_task_spec_python_returns_dataclass(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["get"].return_value = patched_requests["_make_response"](
        {
            "rl_task_uuid": "t",
            "path": "cyberwave_task_spec.py",
            "content": "TASK_SPEC = {'schema_version': 1, 'entities': []}\n",
            "schema_version": 1,
            "entity_count": 0,
        }
    )

    spec = rl.export_task_spec_python("task-uuid")

    assert isinstance(spec, TaskSpecExport)
    assert spec.path == "cyberwave_task_spec.py"
    assert spec.entity_count == 0
    assert "TASK_SPEC" in spec.content


def test_import_task_spec_python_validate_only_uses_validate_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {
            "valid": True,
            "schema_version": 1,
            "entity_names": [],
            "errors": [],
            "warnings": [],
        }
    )

    result = rl.import_task_spec_python(
        "task-uuid",
        "TASK_SPEC = {'schema_version': 1, 'entities': []}\n",
        validate_only=True,
    )

    assert patched_requests["post"].call_count == 1
    url = patched_requests["post"].call_args.args[0]
    assert url.endswith("/task-spec/validate")
    # validate_only must NOT hit the PUT endpoint.
    patched_requests["put"].assert_not_called()
    assert result == {
        "valid": True,
        "schema_version": 1,
        "entity_names": [],
        "errors": [],
        "warnings": [],
    }


def test_import_task_spec_python_default_writes_via_put(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        [{"name": "robot"}]
    )

    rl.import_task_spec_python(
        "task-uuid",
        "TASK_SPEC = {'schema_version': 1, 'entities': []}\n",
    )

    assert patched_requests["put"].call_count == 1
    url = patched_requests["put"].call_args.args[0]
    assert url.endswith("/task-spec.py")
    payload = patched_requests["put"].call_args.kwargs["json"]
    assert payload["validate_only"] is False


def test_client_raises_on_http_error(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    failing = patched_requests["_make_response"]({"detail": "boom"}, status=400)
    patched_requests["get"].return_value = failing

    with pytest.raises(RuntimeError) as exc_info:
        rl.list_scene_entities("task-uuid")

    assert "400" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Runtime metadata + checkpoints + inference dispatch
# ---------------------------------------------------------------------------


def test_set_runtime_sends_only_provided_fields(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {"uuid": "task-uuid", "runtime_target": "cyberwave-rl-experimental"}
    )

    rl.set_runtime(
        "task-uuid",
        runtime_target="cyberwave-rl-experimental",
        runtime_accelerator="cpu",
        runtime_versions={"mjlab": "1.0.0", "skrl": "1.4.0"},
    )

    assert patched_requests["put"].call_count == 1
    payload = patched_requests["put"].call_args.kwargs["json"]
    assert payload == {
        "runtime_target": "cyberwave-rl-experimental",
        "runtime_accelerator": "cpu",
        "runtime_versions": {"mjlab": "1.0.0", "skrl": "1.4.0"},
    }
    # ``policy_interface`` was not passed → must not appear in payload.
    assert "policy_interface" not in payload


def test_set_runtime_no_args_returns_current_task(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["get"].return_value = patched_requests["_make_response"](
        {"uuid": "task-uuid", "runtime_target": "cyberwave-rl"}
    )

    result = rl.set_runtime("task-uuid")

    patched_requests["put"].assert_not_called()
    assert patched_requests["get"].call_count == 1
    url = patched_requests["get"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid")
    assert result["runtime_target"] == "cyberwave-rl"


def test_register_checkpoint_emits_clean_payload(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"uuid": "ckpt-uuid", "name": "ckpt-1"}
    )

    rl.register_checkpoint(
        "task-uuid",
        name="ckpt-1",
        weights_url="https://example.com/w.tar",
        runtime_target="cyberwave-rl-experimental",
        runtime_accelerator="gpu",
        runtime_versions={"mjlab": "1.0.0"},
        metadata={"steps": 1000},
    )

    url = patched_requests["post"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid/checkpoints")
    payload = patched_requests["post"].call_args.kwargs["json"]
    assert payload == {
        "name": "ckpt-1",
        "weights_url": "https://example.com/w.tar",
        "runtime_target": "cyberwave-rl-experimental",
        "runtime_accelerator": "gpu",
        "runtime_versions": {"mjlab": "1.0.0"},
        "metadata": {"steps": 1000},
    }


def test_upload_checkpoint_posts_multipart_to_upload_endpoint(
    fake_client: _FakeClient,
    patched_requests: dict[str, MagicMock],
    tmp_path: Any,
) -> None:
    """``upload_checkpoint`` must POST multipart-form data to the
    convenience upload endpoint with the checkpoint file attached."""

    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"uuid": "ckpt-uuid", "name": "best", "attachment_uuid": "att-uuid"}
    )

    src = tmp_path / "best_agent.pt"
    src.write_bytes(b"weights")

    result = rl.upload_checkpoint(
        "task-uuid",
        src,
        name="best",
        description="PPO @ 5M",
        metadata={"reward": 0.91},
    )

    call = patched_requests["post"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid/checkpoints/upload")
    # urllib3 carries multipart parts in a single ``fields`` mapping: form
    # fields plus a ``(filename, bytes, content_type)`` file tuple.
    assert "fields" in call.kwargs, "expected multipart fields= kwarg"
    fields = call.kwargs["fields"]
    assert "file" in fields
    file_tuple = fields["file"]
    assert file_tuple[0] == "best_agent.pt"
    assert file_tuple[1] == b"weights"
    assert fields["name"] == "best"
    assert fields["description"] == "PPO @ 5M"
    # metadata is forwarded as JSON-encoded string so the backend can
    # parse it from the multipart form payload.
    assert fields["metadata"] == '{"reward": 0.91}'
    # Auth header is preserved but no JSON Content-Type override (urllib3
    # injects the multipart boundary itself).
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "Content-Type" not in call.kwargs["headers"]
    assert result["uuid"] == "ckpt-uuid"


def test_upload_checkpoint_rejects_missing_file(
    fake_client: _FakeClient,
    patched_requests: dict[str, MagicMock],
    tmp_path: Any,
) -> None:
    rl = RLTaskClient(fake_client)
    with pytest.raises(FileNotFoundError):
        rl.upload_checkpoint("task-uuid", tmp_path / "no-such-file.pt")
    patched_requests["post"].assert_not_called()


def test_download_checkpoint_streams_to_destination(
    fake_client: _FakeClient,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``download_checkpoint`` resolves the attachment URL via the
    standard backend route and streams the response to disk."""

    rl = RLTaskClient(fake_client)
    checkpoint_payload = {
        "uuid": "ckpt-uuid",
        "name": "best",
        "attachment_uuid": "att-uuid",
        "weights_url": "",
        "metadata": {"original_filename": "best_agent.pt"},
    }

    class _StreamResp:
        def __init__(self) -> None:
            self.status = 200
            self.data = b""

        def stream(self, chunk_size: int) -> Any:  # noqa: ARG002
            yield b"chunk1"
            yield b"chunk2"

        def release_conn(self) -> None:
            return None

    # The download helper makes two HTTP GETs: the first reads the
    # checkpoint row (JSON response via ``_get``), the second streams the
    # attachment payload. Both go through ``PoolManager.request`` so we
    # dispatch on the URL.
    stream_calls: list[Any] = []

    def fake_request(self: Any, method: str, url: str, **kwargs: Any) -> Any:
        if url.endswith("/checkpoints/ckpt-uuid"):
            resp = MagicMock()
            resp.status = 200
            resp.json.return_value = checkpoint_payload
            resp.data = b"{...}"
            return resp
        stream_calls.append((url, kwargs))
        return _StreamResp()

    monkeypatch.setattr("urllib3.PoolManager.request", fake_request)

    dest = tmp_path / "subdir"
    dest.mkdir()
    out = rl.download_checkpoint(
        "task-uuid",
        "ckpt-uuid",
        destination=dest,
    )

    # Resolves the attachment via the backend download route, not the
    # raw weights_url (which is empty here).
    assert len(stream_calls) == 1
    stream_url, stream_kwargs = stream_calls[0]
    assert stream_url.endswith("/api/v1/attachments/att-uuid/download")
    assert stream_kwargs["preload_content"] is False
    # Bytes were assembled from the streaming response.
    assert out.read_bytes() == b"chunk1chunk2"
    # Destination dir was treated as a directory and filename was
    # picked up from the metadata.
    assert out.parent == dest
    assert out.name == "best_agent.pt"


def test_prepare_checkpoint_for_play_stages_into_expected_layout(
    tmp_path: Any,
) -> None:
    src = tmp_path / "raw" / "best_agent.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"weights")

    staged = RLTaskClient.prepare_checkpoint_for_play(
        task_id="my-task",
        checkpoint_path=src,
        logs_root=tmp_path / "logs",
        run_name="uploaded",
    )

    assert staged.exists()
    assert staged.read_bytes() == b"weights"
    # logs/<task>/<run>/checkpoints/best_agent.pt
    assert staged.relative_to(tmp_path) == Path(
        "logs/my-task/uploaded/checkpoints/best_agent.pt"
    )
    # Source file is preserved (copy, not move).
    assert src.exists()


def test_launch_inference_targets_inference_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["post"].return_value = patched_requests["_make_response"](
        {"uuid": "run-uuid", "rl_task_uuid": "task-uuid"}
    )

    rl.launch_inference(
        "task-uuid",
        checkpoint_uuid="ckpt-uuid",
        twin_uuid="twin-uuid",
        max_steps=200,
        control_rate_hz=20.0,
        mode="loop",
        runtime_accelerator="cpu",
    )

    url = patched_requests["post"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid/inference")
    payload = patched_requests["post"].call_args.kwargs["json"]
    assert payload == {
        "checkpoint_uuid": "ckpt-uuid",
        "twin_uuid": "twin-uuid",
        "max_steps": 200,
        "control_rate_hz": 20.0,
        "mode": "loop",
        "runtime_accelerator": "cpu",
    }


def test_clone_task_posts_to_clone_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """Cover both the explicit-payload and the no-arg default-empty-payload
    paths in one test; the underlying ``_post`` call shape is the same and
    we just want to lock in that ``clone_task`` always targets
    ``/rl-tasks/{uuid}/clone`` and never sends ``None`` fields.
    """
    rl = RLTaskClient(fake_client)
    response = patched_requests["_make_response"](
        {"uuid": "new-uuid", "name": "Copy of Task"}
    )
    patched_requests["post"].return_value = response

    result = rl.clone_task(
        "task-uuid",
        name="Renamed",
        workspace_uuid="ws-uuid",
    )

    url = patched_requests["post"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid/clone")
    assert patched_requests["post"].call_args.kwargs["json"] == {
        "name": "Renamed",
        "workspace_uuid": "ws-uuid",
    }
    assert result["uuid"] == "new-uuid"

    rl.clone_task("task-uuid")
    assert patched_requests["post"].call_args.kwargs["json"] == {}


# ---------------------------------------------------------------------------
# Orchestration tabs (Actions / Observations / RL Config)
# ---------------------------------------------------------------------------


def test_get_orchestration_hints_hits_expected_url(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["get"].return_value = patched_requests["_make_response"](
        {
            "articulation_entities": ["robot"],
            "entity_actuator_types": {"robot": ["xml_position"]},
            "entity_actuator_groups": {
                "robot": [
                    {"type": "xml_position", "target_names_expr": [".*_joint[1-7]$"]}
                ]
            },
            "entity_actuated_joint_names": {"robot": ["a", "b"]},
            "entity_passive_joint_names": {"robot": []},
            "available_sensors": [],
        }
    )

    hints = rl.get_orchestration_hints("task-uuid")

    url = patched_requests["get"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid/orchestration-hints")
    assert hints["articulation_entities"] == ["robot"]
    assert hints["entity_actuated_joint_names"]["robot"] == ["a", "b"]


def test_set_action_spec_wraps_payload(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """``set_action_spec`` must wrap the dict under the ``action_spec`` key.

    The backend schema is ``{"action_spec": {...}}`` (not the raw dict)
    so the test pins that envelope shape to avoid breaking the
    contract when the underlying wrapper changes.
    """

    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {"action_spec": {"schema_version": 1, "actions": []}}
    )

    spec = make_action_spec(
        [
            make_action_term(
                "joint_delta",
                action_type="position_delta",
                entity="robot",
                target_names_expr=[".*_joint[1-7]$"],
                scale=0.04,
                offset="none",
                use_default_offset=False,
            )
        ]
    )

    rl.set_action_spec("task-uuid", spec)

    assert patched_requests["put"].call_count == 1
    call = patched_requests["put"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid/actions")
    payload = call.kwargs["json"]
    assert "action_spec" in payload
    assert payload["action_spec"]["schema_version"] == 1
    assert payload["action_spec"]["actions"][0]["type"] == "position_delta"
    assert payload["action_spec"]["actions"][0]["scale"] == 0.04


def test_get_action_spec_targets_actions_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["get"].return_value = patched_requests["_make_response"](
        {"action_spec": {}}
    )

    result = rl.get_action_spec("task-uuid")

    url = patched_requests["get"].call_args.args[0]
    assert url.endswith("/rl-tasks/task-uuid/actions")
    assert result == {"action_spec": {}}


def test_set_observation_spec_wraps_payload(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {"observation_spec": {"schema_version": 1, "groups": {}}}
    )

    spec = make_observation_spec(
        {
            "policy": [
                make_observation_term(
                    "joint_pos",
                    term_type="joint_position",
                    entity="robot",
                    target_names_expr=[".*_joint[1-7]$"],
                ),
                make_observation_term(
                    "joint_vel",
                    term_type="joint_velocity",
                    entity="robot",
                    target_names_expr=[".*_joint[1-7]$"],
                ),
            ],
            "critic": [
                make_observation_term("last_action", term_type="previous_action"),
            ],
        },
    )

    rl.set_observation_spec("task-uuid", spec)

    call = patched_requests["put"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid/observations")
    payload = call.kwargs["json"]
    assert "observation_spec" in payload
    groups = payload["observation_spec"]["groups"]
    assert set(groups) == {"policy", "critic"}
    assert groups["policy"]["terms"][0]["type"] == "joint_position"
    assert groups["policy"]["terms"][0]["target_names_expr"] == [".*_joint[1-7]$"]
    assert groups["policy"]["concatenate_terms"] is True
    assert groups["policy"]["enable_corruption"] is False


def test_set_rl_config_spec_targets_rl_config_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {"rl_config_spec": {"schema_version": 1, "algorithm": "PPO"}}
    )

    spec = make_rl_config_spec(
        algorithm="ppo",
        network={
            "library": "rlmodule",
            "topology": "shared",
            "architecture": "mlp",
            "blocks": {
                "shared": {
                    "type": "mlp",
                    "layers": [
                        {"units": 256, "activation": "ELU", "layer_norm": False},
                        {"units": 256, "activation": "ELU", "layer_norm": False},
                    ],
                }
            },
        },
        algorithm_config={"learning_rate": 3e-4, "rollouts": 24},
    )

    rl.set_rl_config_spec("task-uuid", spec)

    call = patched_requests["put"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid/rl-config")
    payload = call.kwargs["json"]
    assert payload["rl_config_spec"]["algorithm"] == "ppo"
    assert payload["rl_config_spec"]["network"]["library"] == "rlmodule"
    assert payload["rl_config_spec"]["algorithm_config"]["rollouts"] == 24


def test_set_training_command_spec_targets_task_update_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """``set_training_command_spec`` has no dedicated sub-resource — it writes
    the training command spec through the RL task update endpoint, wrapped
    under ``training_command_spec``."""
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {
            "uuid": "task-uuid",
            "training_command_spec": {"schema_version": 1, "commands": []},
        }
    )

    spec = {
        "schema_version": 1,
        "commands": [
            {
                "name": "viewpoint",
                "type": "custom",
                "module": "env_cfg",
                "symbol": "ViewpointCommandCfg",
                "resampling_time_range": [1.0e9, 1.0e9],
                "kwargs": {"target_source": "sampled"},
            }
        ],
    }

    rl.set_training_command_spec("task-uuid", spec)

    call = patched_requests["put"].call_args
    # Hits the task update endpoint, NOT a /commands sub-resource.
    assert call.args[0].endswith("/rl-tasks/task-uuid")
    assert not call.args[0].endswith("/commands")
    payload = call.kwargs["json"]
    cmd = payload["training_command_spec"]["commands"][0]
    assert cmd["name"] == "viewpoint"
    assert cmd["symbol"] == "ViewpointCommandCfg"
    assert cmd["resampling_time_range"] == [1.0e9, 1.0e9]


def test_set_inference_command_spec_targets_task_update_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """``set_inference_command_spec`` writes the inference command spec through
    the RL task update endpoint, wrapped under ``inference_command_spec``."""
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {
            "uuid": "task-uuid",
            "inference_command_spec": {"schema_version": 1, "commands": []},
        }
    )

    spec = {
        "schema_version": 1,
        "commands": [
            {
                "name": "viewpoint",
                "type": "custom",
                "module": "env_cfg",
                "symbol": "ViewpointCommandCfg",
                "payload_schema": {"target_position": "vec3"},
                "source": {"kind": "mqtt"},
                "kwargs": {"target_source": "external_target"},
            }
        ],
    }

    rl.set_inference_command_spec("task-uuid", spec)

    call = patched_requests["put"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid")
    payload = call.kwargs["json"]
    cmd = payload["inference_command_spec"]["commands"][0]
    assert cmd["name"] == "viewpoint"
    assert cmd["payload_schema"]["target_position"] == "vec3"
    assert cmd["source"]["kind"] == "mqtt"
    # No explicit ``enabled`` → the toggle key is omitted (backend derives it).
    assert "inference_command_setup_enabled" not in payload


def test_set_command_spec_forwards_explicit_enabled_toggle(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """The explicit ``enabled`` switch (ADR 0004) is forwarded verbatim, so a
    caller can disable a lifecycle (clearing its spec) in one request."""
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {"uuid": "task-uuid", "training_command_spec": {}}
    )

    rl.set_training_command_spec("task-uuid", {}, enabled=False)

    payload = patched_requests["put"].call_args.kwargs["json"]
    assert payload["training_command_setup_enabled"] is False
    assert payload["training_command_spec"] == {}


def test_upsert_source_file_writes_to_source_files_endpoint(
    fake_client: _FakeClient, patched_requests: dict[str, MagicMock]
) -> None:
    """``upsert_source_file`` PUTs path + content to the source-files endpoint.

    This is how a code-backed command term's ``commands/<name>.py`` is uploaded
    before the term is registered via ``set_*_command_spec`` (module
    ``commands.<name>`` / symbol ``<CfgClass>``)."""
    rl = RLTaskClient(fake_client)
    patched_requests["put"].return_value = patched_requests["_make_response"](
        {
            "path": "commands/viewpoint.py",
            "content": "class ViewpointCommandCfg: pass",
        }
    )

    rl.upsert_source_file(
        "task-uuid",
        "commands/viewpoint.py",
        "class ViewpointCommandCfg: pass\n",
    )

    call = patched_requests["put"].call_args
    assert call.args[0].endswith("/rl-tasks/task-uuid/source-files")
    payload = call.kwargs["json"]
    assert payload["path"] == "commands/viewpoint.py"
    assert "ViewpointCommandCfg" in payload["content"]
    # Optional flag omitted unless requested (absent != False server-side).
    assert "is_entrypoint" not in payload


def test_make_action_term_only_sets_provided_optional_fields() -> None:
    """Optional knobs must not appear in the dict if the caller omitted them.

    The backend treats absent keys as "use server default". Sending an
    explicit ``None`` would override that, so the SDK helper has to
    elide unset optionals.
    """

    term = make_action_term(
        "joint_pos",
        action_type="position",
        entity="robot",
        target_names_expr=[".*"],
        scale=0.5,
    )
    assert term["type"] == "position"
    assert term["target_names_expr"] == [".*"]
    assert "offset" not in term
    assert "use_default_offset" not in term
    assert "baseline_delta" not in term


def test_make_action_term_accepts_per_joint_scale_and_offset() -> None:
    """Per-joint maps round-trip into the persisted shape so users can
    author per-joint values from setup scripts."""

    term = make_action_term(
        "joint_pos",
        action_type="position",
        entity="robot",
        target_names_expr=["arm_joint_[0-9]+"],
        scale={"arm_joint_0": 0.5, "arm_joint_1": 0.25},
        offset={"arm_joint_0": 0.1, "arm_joint_1": -0.1},
    )
    assert term["scale"] == {"arm_joint_0": 0.5, "arm_joint_1": 0.25}
    assert term["offset"] == {"arm_joint_0": 0.1, "arm_joint_1": -0.1}


def test_make_action_term_accepts_baseline_delta_on_position_delta() -> None:
    """``baseline_delta`` flows through as either scalar or map and is
    only legal on ``position_delta`` action terms."""

    term = make_action_term(
        "joint_delta",
        action_type="position_delta",
        entity="robot",
        target_names_expr=["arm_joint_[0-9]+"],
        scale=0.04,
        baseline_delta={"arm_joint_0": 0.01, "arm_joint_1": -0.005},
    )
    assert term["baseline_delta"] == {
        "arm_joint_0": 0.01,
        "arm_joint_1": -0.005,
    }

    with pytest.raises(ValueError):
        make_action_term(
            "joint_pos",
            action_type="position",
            entity="robot",
            target_names_expr=[".*"],
            baseline_delta=0.1,
        )


def test_make_action_term_rejects_invalid_scalar_or_map() -> None:
    """Bools and non-numeric mapping values must surface as
    ``ValueError`` instead of being silently coerced."""

    with pytest.raises(ValueError):
        make_action_term(
            "joint_pos",
            action_type="position",
            entity="robot",
            target_names_expr=[".*"],
            scale=True,
        )
    with pytest.raises(ValueError):
        make_action_term(
            "joint_pos",
            action_type="position",
            entity="robot",
            target_names_expr=[".*"],
            scale={"joint_a": "bad"},  # type: ignore[dict-item]
        )


def test_make_observation_term_only_sets_provided_optional_fields() -> None:
    """Observation helper must elide unset optionals so the persisted
    term stays minimal and forward-compatible with new fields."""

    term = make_observation_term(
        "joint_pos",
        term_type="joint_position",
        entity="robot",
        target_names_expr=[".*"],
    )
    assert term == {
        "name": "joint_pos",
        "type": "joint_position",
        "entity": "robot",
        "target_names_expr": [".*"],
    }

    # ``previous_action`` carries no extra fields.
    prev = make_observation_term("last_action", term_type="previous_action")
    assert prev == {"name": "last_action", "type": "previous_action"}


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------


def test_make_position_delta_action_matches_editor_preset() -> None:
    """The preset must produce the same dict shape ``make_action_term``
    would build for the editor's "Joint delta-position" choice."""

    term = make_position_delta_action(
        "joint_delta",
        entity="robot",
        target_names_expr=["arm_joint_[0-9]+"],
        scale=0.04,
    )
    assert term == {
        "name": "joint_delta",
        "type": "position_delta",
        "entity": "robot",
        "target_names_expr": ["arm_joint_[0-9]+"],
        "scale": 0.04,
        "offset": "none",
        "use_default_offset": False,
    }


@pytest.mark.parametrize(
    "kind,expected_type",
    [
        ("position", "joint_position"),
        ("velocity", "joint_velocity"),
        ("effort", "joint_effort"),
    ],
)
def test_make_joint_observation_kinds(kind: str, expected_type: str) -> None:
    term = make_joint_observation(
        "joint_pos",
        kind=kind,
        entity="robot",
        target_names_expr=[".*"],
    )
    assert term["type"] == expected_type
    assert term["entity"] == "robot"
    assert term["target_names_expr"] == [".*"]


def test_make_joint_observation_invalid_kind_errors() -> None:
    with pytest.raises(ValueError):
        make_joint_observation(
            "joint_pos",
            kind="acceleration",
            entity="robot",
            target_names_expr=[".*"],
        )


def test_make_previous_action_observation_defaults_to_last_action_name() -> None:
    assert make_previous_action_observation() == {
        "name": "last_action",
        "type": "previous_action",
    }


def test_make_previous_action_observation_accepts_action_name() -> None:
    term = make_previous_action_observation("actions", action_name="joint_delta")
    assert term == {
        "name": "actions",
        "type": "previous_action",
        "action_name": "joint_delta",
    }


def test_make_camera_observation_default_data_type() -> None:
    term = make_camera_observation("wrist_depth", sensor="robot__wrist_cam")
    assert term == {
        "name": "wrist_depth",
        "type": "camera",
        "sensor": "robot__wrist_cam",
        "data_type": "depth",
    }


def test_make_camera_observation_grayscale_rgb() -> None:
    term = make_camera_observation(
        "wrist_color", sensor="robot__wrist_cam", data_type="rgb", grayscale=True
    )
    assert term == {
        "name": "wrist_color",
        "type": "camera",
        "sensor": "robot__wrist_cam",
        "data_type": "rgb",
        "grayscale": True,
    }


def test_make_camera_observation_grayscale_defaults_off() -> None:
    term = make_camera_observation(
        "wrist_color", sensor="robot__wrist_cam", data_type="rgb"
    )
    assert "grayscale" not in term


def test_make_custom_observation_defaults_symbol_to_name() -> None:
    term = make_custom_observation("ee_to_cube", module="env_cfg")
    assert term == {
        "name": "ee_to_cube",
        "type": "custom",
        "module": "env_cfg",
        "symbol": "ee_to_cube",
        "kind": "function",
    }


def test_make_custom_observation_accepts_shape() -> None:
    term = make_custom_observation(
        "gripper_effort_left",
        module="env_cfg",
        shape=[2],
    )
    assert term["shape"] == [2]


def test_make_default_ppo_rlmodule_config_shape() -> None:
    spec = make_default_ppo_rlmodule_config(
        hidden_layers=[64, 64],
        write_interval=100,
        checkpoint_interval=1_000,
        experiment_directory="logs/test",
    )
    assert spec["schema_version"] == 1
    assert spec["algorithm"] == "ppo"
    assert spec["backend"] == "skrl"
    layers = spec["network"]["blocks"]["shared"]["layers"]
    assert [layer["units"] for layer in layers] == [64, 64]
    assert layers[0]["activation"] == "ELU"
    assert spec["network"]["output"]["initial_log_std"] == pytest.approx(-0.5)
    assert spec["algorithm_config"]["rollouts"] == 16
    assert spec["trainer"]["timesteps"] == 100_000
    assert spec["trainer"]["write_interval"] == 100
    assert spec["trainer"]["checkpoint_interval"] == 1_000
    assert spec["experiment"]["directory"] == "logs/test"


def test_make_default_ppo_rlmodule_config_auto_intervals_omit_fields() -> None:
    """Setting ``"auto"`` means: don't emit the trainer key (use editor
    default). The persisted payload then stays minimal."""

    spec = make_default_ppo_rlmodule_config()
    assert "write_interval" not in spec["trainer"]
    assert "checkpoint_interval" not in spec["trainer"]
    assert "experiment" not in spec
