from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.ml_model_lookup import (
    MLModelLookupError,
    MLModelMatch,
    resolve_ml_model_uuid,
    search_ml_models,
)


def _model(
    uuid: str,
    name: str,
    external_id: str,
    *,
    deployment: str = "cloud",
    edge: bool = False,
    cloud: bool = True,
):
    return SimpleNamespace(
        uuid=uuid,
        name=name,
        model_external_id=external_id,
        deployment=deployment,
        is_edge_compatible=edge,
        is_cloud_compatible=cloud,
    )


def _api_with_models(models):
    api = MagicMock()
    api.src_app_api_mlmodels_list_mlmodels.return_value = models
    return api


def test_search_ml_models_returns_partial_name_matches():
    api = _api_with_models(
        [
            _model("a", "Gemini Flash Preview", "gemini-3-flash-preview"),
            _model("b", "Vision QA", "vision-qa"),
        ]
    )
    out = search_ml_models(api, "gemini")
    assert len(out) == 1
    assert isinstance(out[0], MLModelMatch)
    assert out[0].uuid == "a"


def test_search_ml_models_matches_partial_external_id():
    api = _api_with_models([_model("a", "Other", "gemini-3-flash-preview")])
    out = search_ml_models(api, "flash")
    assert [m.uuid for m in out] == ["a"]


def test_search_ml_models_applies_limit():
    api = _api_with_models(
        [
            _model("a", "Gemini One", "g1"),
            _model("b", "Gemini Two", "g2"),
            _model("c", "Gemini Three", "g3"),
        ]
    )
    out = search_ml_models(api, "gemini", limit=2)
    assert [m.uuid for m in out] == ["a", "b"]


def test_search_ml_models_forwards_filters_and_timeout():
    api = _api_with_models([])
    search_ml_models(
        api,
        "x",
        deployment="cloud",
        edge_compatible=True,
        request_timeout=12.0,
    )
    api.src_app_api_mlmodels_list_mlmodels.assert_called_once_with(
        deployment="cloud",
        edge_compatible=True,
        _request_timeout=12.0,
    )


def test_resolve_ml_model_uuid_exact_external_id_first():
    api = _api_with_models(
        [
            _model("a", "Gemini Flash", "gemini-3-flash-preview"),
            _model("b", "Gemini Flash", "other-id"),
        ]
    )
    assert resolve_ml_model_uuid(api, "gemini-3-flash-preview") == "a"


def test_resolve_ml_model_uuid_exact_name_case_insensitive():
    api = _api_with_models([_model("a", "Gemini Flash", "id-a")])
    assert resolve_ml_model_uuid(api, "gemini flash") == "a"


def test_resolve_ml_model_uuid_single_partial_match():
    api = _api_with_models(
        [
            _model("a", "Gemini Flash Preview", "gemini-3-flash-preview"),
            _model("b", "Vision QA", "vision-qa"),
        ]
    )
    assert resolve_ml_model_uuid(api, "flash") == "a"


def test_resolve_ml_model_uuid_ambiguous_partial_raises():
    api = _api_with_models(
        [
            _model("a", "Gemini Flash Preview", "gemini-3-flash-preview"),
            _model("b", "Gemini Flash Fast", "gemini-3-flash-fast"),
        ]
    )
    with pytest.raises(MLModelLookupError, match="Ambiguous model query"):
        resolve_ml_model_uuid(api, "gemini")


def test_resolve_ml_model_uuid_no_match_raises():
    api = _api_with_models([_model("a", "Vision QA", "vision-qa")])
    with pytest.raises(MLModelLookupError, match="No model matched query"):
        resolve_ml_model_uuid(api, "gemini")


def test_resolve_ml_model_uuid_empty_query_raises():
    api = _api_with_models([])
    with pytest.raises(ValueError, match="non-empty"):
        resolve_ml_model_uuid(api, " ")

