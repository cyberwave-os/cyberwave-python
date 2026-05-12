"""Helpers for searching and resolving ML models from the Cyberwave catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


class MLModelLookupError(RuntimeError):
    """Raised when model lookup is ambiguous or has no matches."""


@dataclass(frozen=True)
class MLModelMatch:
    """Compact model view suitable for UI selection and debug output."""

    uuid: str
    name: str
    model_external_id: str
    deployment: Optional[str]
    is_edge_compatible: Optional[bool]
    is_cloud_compatible: Optional[bool]


def _to_match(model: Any) -> MLModelMatch:
    return MLModelMatch(
        uuid=str(getattr(model, "uuid", "")),
        name=str(getattr(model, "name", "")),
        model_external_id=str(getattr(model, "model_external_id", "")),
        deployment=getattr(model, "deployment", None),
        is_edge_compatible=getattr(model, "is_edge_compatible", None),
        is_cloud_compatible=getattr(model, "is_cloud_compatible", None),
    )


def _name_norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _external_norm(value: Optional[str]) -> str:
    return (value or "").strip()


def search_ml_models(
    api: Any,
    query: str,
    *,
    deployment: Optional[str] = None,
    edge_compatible: Optional[bool] = None,
    request_timeout: float = 60.0,
    limit: int = 20,
) -> list[MLModelMatch]:
    """
    Search ML models by partial name/external-id match (case-insensitive).

    Returns ordered matches from ``list_mlmodels`` filtered locally.
    """
    needle = _name_norm(query)
    if not needle:
        raise ValueError("query must be non-empty")

    models = api.src_app_api_mlmodels_list_mlmodels(
        deployment=deployment,
        edge_compatible=edge_compatible,
        _request_timeout=request_timeout,
    )

    out: list[MLModelMatch] = []
    for model in models:
        name = _name_norm(getattr(model, "name", None))
        external_id = _name_norm(getattr(model, "model_external_id", None))
        if needle in name or needle in external_id:
            out.append(_to_match(model))
            if len(out) >= max(limit, 1):
                break
    return out


def resolve_ml_model_uuid(
    api: Any,
    query: str,
    *,
    deployment: Optional[str] = None,
    edge_compatible: Optional[bool] = None,
    request_timeout: float = 60.0,
) -> str:
    """
    Resolve one model UUID from external id/name query.

    Match order:
      1) exact external id
      2) exact name (case-insensitive)
      3) partial name/external-id (case-insensitive)
    """
    raw = query.strip()
    if not raw:
        raise ValueError("query must be non-empty")

    models: Sequence[Any] = api.src_app_api_mlmodels_list_mlmodels(
        deployment=deployment,
        edge_compatible=edge_compatible,
        _request_timeout=request_timeout,
    )

    ext_key = _external_norm(raw)
    exact_external = [
        model for model in models if _external_norm(getattr(model, "model_external_id", None)) == ext_key
    ]
    if len(exact_external) == 1:
        return str(getattr(exact_external[0], "uuid"))
    if len(exact_external) > 1:
        raise MLModelLookupError(
            f"Ambiguous external id {raw!r}: {[str(getattr(m, 'uuid', '')) for m in exact_external]}"
        )

    name_key = _name_norm(raw)
    exact_name = [model for model in models if _name_norm(getattr(model, "name", None)) == name_key]
    if len(exact_name) == 1:
        return str(getattr(exact_name[0], "uuid"))
    if len(exact_name) > 1:
        raise MLModelLookupError(
            f"Ambiguous model name {raw!r}: {[str(getattr(m, 'uuid', '')) for m in exact_name]}"
        )

    partial = []
    for model in models:
        n = _name_norm(getattr(model, "name", None))
        e = _name_norm(getattr(model, "model_external_id", None))
        if name_key in n or name_key in e:
            partial.append(model)

    if len(partial) == 1:
        return str(getattr(partial[0], "uuid"))
    if len(partial) > 1:
        candidates = [
            f"{getattr(m, 'name', '')} ({getattr(m, 'model_external_id', '')}) [{getattr(m, 'uuid', '')}]"
            for m in partial[:10]
        ]
        raise MLModelLookupError(
            f"Ambiguous model query {raw!r}; {len(partial)} matches: {candidates}"
        )

    raise MLModelLookupError(
        f"No model matched query {raw!r}. Use search_ml_models(...) to inspect candidates."
    )
