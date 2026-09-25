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


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "2"])
def test_search_rejects_invalid_limit_before_request(limit):
    api = _api_with_models([])
    with pytest.raises(ValueError, match="positive integer"):
        search_ml_models(api, "model", limit=limit)
    api.src_app_api_mlmodels_list_mlmodels.assert_not_called()


def test_search_rejects_blank_query_before_request():
    api = _api_with_models([])
    with pytest.raises(ValueError, match="non-empty"):
        search_ml_models(api, " \t ")
    api.src_app_api_mlmodels_list_mlmodels.assert_not_called()


@pytest.mark.parametrize(
    "rows,query,message",
    [
        ([("a", "One", "same"), ("b", "Two", "same")], "same", "Ambiguous external id"),
        ([("a", "Same", "one"), ("b", "same", "two")], "SAME", "Ambiguous model name"),
    ],
)
def test_resolve_rejects_duplicate_exact_matches(rows, query, message):
    api = _api_with_models([_model(*row) for row in rows])
    with pytest.raises(MLModelLookupError, match=message):
        resolve_ml_model_uuid(api, query)


def test_external_id_wins_over_another_models_exact_name():
    api = _api_with_models([_model("a", "Other", "target"), _model("b", "target", "else")])
    assert resolve_ml_model_uuid(api, " target ") == "a"


def test_exact_name_wins_over_partial_match():
    api = _api_with_models([_model("a", "Target", "one"), _model("b", "Target extended", "two")])
    assert resolve_ml_model_uuid(api, " TARGET ") == "a"


def test_resolve_forwards_filters_and_timeout():
    api = _api_with_models([_model("a", "Target", "one")])
    assert resolve_ml_model_uuid(
        api, "Target", deployment="edge", edge_compatible=True, request_timeout=12.0
    ) == "a"
    api.src_app_api_mlmodels_list_mlmodels.assert_called_once_with(
        deployment="edge", edge_compatible=True, _request_timeout=12.0
    )


def test_search_preserves_disambiguation_metadata():
    model = _model("a", "Target", "one")
    model.model_provider_name = "provider"
    model.workspace_uuid = "workspace"
    model.visibility = "public"
    model.slug = "workspace/models/target"
    match = search_ml_models(_api_with_models([model]), "TARGET")[0]
    assert (match.model_provider_name, match.workspace_uuid, match.visibility, match.slug) == (
        "provider", "workspace", "public", "workspace/models/target"
    )


def test_model_manager_search_and_resolve_use_shared_lookup():
    from cyberwave.models.manager import ModelManager

    api = _api_with_models([_model("a", "Target", "one"), _model("b", "Target two", "two")])
    manager = ModelManager(api_client=api)
    matches = manager.search("target", deployment="cloud", request_timeout=9.0, limit=1)
    assert [match.uuid for match in matches] == ["a"]
    api.src_app_api_mlmodels_list_mlmodels.assert_called_once_with(
        deployment="cloud", edge_compatible=None, _request_timeout=9.0
    )
    api.reset_mock()
    assert manager.resolve_uuid("one", edge_compatible=True, request_timeout=7.0) == "a"
    api.src_app_api_mlmodels_list_mlmodels.assert_called_once_with(
        deployment=None, edge_compatible=True, _request_timeout=7.0
    )
    with pytest.raises(MLModelLookupError, match="Ambiguous"):
        manager.resolve_uuid("tar")


@pytest.mark.parametrize("method", ["search", "resolve_uuid"])
def test_model_manager_lookup_requires_api_connection(method):
    from cyberwave.exceptions import CyberwaveAPIError
    from cyberwave.models.manager import ModelManager

    with pytest.raises(CyberwaveAPIError, match="API connection"):
        getattr(ModelManager(), method)("target")


def test_lookup_helpers_are_public_exports():
    import cyberwave

    for name in ("MLModelLookupError", "MLModelMatch", "search_ml_models", "resolve_ml_model_uuid"):
        assert name in cyberwave.__all__
        assert getattr(cyberwave, name) is globals()[name]
