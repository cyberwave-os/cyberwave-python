"""Tests for :class:`~cyberwave.resources.MLModelsResourceManager` and the
catalog surface it powers on :class:`~cyberwave.models.manager.ModelManager`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cyberwave.models.manager import ModelManager
from cyberwave.resources import MLModelsResourceManager
from cyberwave.rest import MLModelSchema


def _fake_schema(*, uuid: str = "m-1", slug: str | None = "ws/models/a") -> MLModelSchema:
    return MLModelSchema.model_validate(
        {
            "uuid": uuid,
            "name": "Demo",
            "description": "",
            "slug": slug,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "created_by": None,
            "updated_by": None,
            "workspace_uuid": "ws-1",
            "metadata": {},
            "visibility": "private",
            "tags": [],
            "model_external_id": "demo.pt",
            "model_provider_name": "ultralytics",
            "mapped_model_id": None,
            "output_format": None,
            "deployment": "edge",
            "is_trainable": False,
            "supported_level": "driver",
            "can_take_video_as_input": False,
            "can_take_audio_as_input": False,
            "can_take_image_as_input": True,
            "can_take_text_as_input": False,
            "can_take_action_as_input": False,
            "is_edge_compatible": True,
            "is_cloud_compatible": False,
            "playground_kind": None,
            "output_family": None,
            "allowed_structured_tasks": None,
            "execution_surfaces": None,
            "sdk_load_id": None,
            "edge_catalog_id": None,
            "edge_runtime": None,
        }
    )


def test_list_delegates_to_openapi_wrapper() -> None:
    fake = [_fake_schema()]
    api = MagicMock()
    api.src_app_api_mlmodels_list_mlmodels.return_value = fake

    mgr = MLModelsResourceManager(api)
    out = mgr.list(deployment="edge", edge_compatible=True)

    api.src_app_api_mlmodels_list_mlmodels.assert_called_once_with(
        deployment="edge",
        edge_compatible=True,
        model_external_id=None,
        supported_level=None,
        is_trainable=None,
        catalog_seed_id=None,
    )
    assert out is fake


def test_get_routes_slug_vs_uuid() -> None:
    slug_row = _fake_schema(uuid="uu-1", slug="ws/models/foo")
    uuid_row = _fake_schema(uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    api = MagicMock()

    mgr = MLModelsResourceManager(api)
    api.src_app_api_mlmodels_get_mlmodel_by_slug.return_value = slug_row
    assert mgr.get("ws/models/foo") == slug_row

    api.src_app_api_mlmodels_get_mlmodel.return_value = uuid_row
    assert mgr.get("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") == uuid_row


def test_delete_returns_success_dict() -> None:
    api = MagicMock()
    mgr = MLModelsResourceManager(api)
    assert mgr.delete("  uuid-here ") == {"success": True}
    api.src_app_api_mlmodels_delete_mlmodel.assert_called_once_with(
        uuid="uuid-here",
    )


def test_create_update_stubs_raise() -> None:
    mgr = MLModelsResourceManager(MagicMock())
    with pytest.raises(NotImplementedError):
        mgr.create()
    with pytest.raises(NotImplementedError):
        mgr.update(uuid="x")


# ---------------------------------------------------------------------------
# ModelManager catalog delegation (cw.models.list / .get / .delete)
# ---------------------------------------------------------------------------


def test_model_manager_without_api_client_raises_on_catalog() -> None:
    mgr = ModelManager()
    with pytest.raises(Exception, match="api_client"):
        mgr.list()


def test_model_manager_list_delegates_through_catalog() -> None:
    fake = [_fake_schema()]
    api = MagicMock()
    api.src_app_api_mlmodels_list_mlmodels.return_value = fake

    mgr = ModelManager(api_client=api)
    out = mgr.list(deployment="edge")

    api.src_app_api_mlmodels_list_mlmodels.assert_called_once()
    assert out is fake


def test_model_manager_get_delegates_through_catalog() -> None:
    row = _fake_schema()
    api = MagicMock()
    api.src_app_api_mlmodels_get_mlmodel_by_slug.return_value = row

    mgr = ModelManager(api_client=api)
    result = mgr.get("ws/models/demo")

    api.src_app_api_mlmodels_get_mlmodel_by_slug.assert_called_once_with(
        slug="ws/models/demo"
    )
    assert result is row


def test_model_manager_delete_delegates_through_catalog() -> None:
    api = MagicMock()
    mgr = ModelManager(api_client=api)
    assert mgr.delete("some-uuid") == {"success": True}
    api.src_app_api_mlmodels_delete_mlmodel.assert_called_once_with(uuid="some-uuid")


def test_model_manager_create_update_stubs_raise() -> None:
    mgr = ModelManager(api_client=MagicMock())
    with pytest.raises(NotImplementedError):
        mgr.create()
    with pytest.raises(NotImplementedError):
        mgr.update(uuid="x")
