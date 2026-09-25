"""Twin introspection helpers used during driver startup.

Pure functions over a twin API handle — no driver state — so they are easy to
test and reuse. Twin objects store some fields (e.g. the attached controller) on
an internal ``_data`` blob rather than as top-level attributes, so callers must
go through these helpers instead of bare ``getattr``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_twin_attached_controller(twin: Any) -> tuple[str | None, str | None]:
    """Return ``(controller_policy_uuid, controller_type)`` from a twin API handle.

    Twin objects store ``controller_policy_uuid`` on ``_data``, not as a top-level
    attribute — callers must not use bare ``getattr(twin, "controller_policy_uuid")``.
    """
    policy_uuid: str | None = None
    if hasattr(twin, "_attached_controller_policy_uuid"):
        policy_uuid = twin._attached_controller_policy_uuid()
    if not policy_uuid and hasattr(twin, "_data_get"):
        raw = twin._data_get("controller_policy_uuid")
        policy_uuid = str(raw) if raw else None
    if not policy_uuid:
        raw = getattr(twin, "controller_policy_uuid", None)
        policy_uuid = str(raw) if raw else None
    if not policy_uuid:
        data = getattr(twin, "_data", None)
        if data is not None:
            if hasattr(data, "controller_policy_uuid"):
                raw = data.controller_policy_uuid
            elif isinstance(data, dict):
                raw = data.get("controller_policy_uuid")
            else:
                raw = None
            policy_uuid = str(raw) if raw else None

    from cyberwave.twin._helpers import _get_twin_metadata

    meta = (
        twin.metadata
        if hasattr(twin, "metadata") and isinstance(getattr(twin, "metadata"), dict)
        else _get_twin_metadata(getattr(twin, "_data", twin))
    )
    ctype = meta.get("controller_type") or meta.get("controller_policy_type")
    if ctype is not None:
        ctype = str(ctype)
    return policy_uuid, ctype


def refresh_driver_twin_from_api(driver: Any) -> Any:
    """Re-fetch the twin from the API so startup sees the latest controller assignment."""
    twin = getattr(driver, "_twin", None)
    if twin is None:
        return twin
    cw = getattr(driver, "_cw", None)
    twin_uuid = getattr(driver, "twin_uuid", None)
    if cw is not None and twin_uuid and hasattr(cw, "twins"):
        try:
            return cw.twins.get(twin_uuid)
        except Exception:
            logger.warning(
                "Failed to refresh twin %s for initial operation mode; using cached twin",
                twin_uuid,
                exc_info=True,
            )
    if hasattr(twin, "refresh"):
        try:
            twin.refresh()
        except Exception:
            logger.warning(
                "Failed to refresh cached twin for initial operation mode",
                exc_info=True,
            )
    return twin
