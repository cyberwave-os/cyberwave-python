"""Tests for :mod:`cyberwave.mlmodels.actions`.

The SDK now ships only a slim view of the backend catalog (ids + short
metadata for autocomplete). Prompt templates and JSON schemas live on the
backend and are fetched lazily through
:meth:`cyberwave.mlmodels.MLModelsClient.fetch_structured_actions_catalog`.

This module guards:

* the in-process shape (unique ids, declared output_format values stay in
  the SDK-supported set);
* drift between the SDK id list and the backend source of truth.

The drift check is opt-in — it only runs when the backend source is on the
Python path (e.g. during monorepo CI from the repo root). When the backend
is not importable the test xfails with an informative message so isolated
SDK installs aren't blocked.
"""

from __future__ import annotations

import pytest

from cyberwave.mlmodels.actions import (
    STRUCTURED_ACTIONS,
    get_action,
    list_actions,
)


class TestCatalogShape:
    def test_ids_are_unique(self) -> None:
        ids = [a.id for a in STRUCTURED_ACTIONS]
        assert len(ids) == len(set(ids))

    def test_list_actions_is_ordered(self) -> None:
        assert [a.id for a in list_actions()] == [a.id for a in STRUCTURED_ACTIONS]

    def test_get_action_lookup(self) -> None:
        assert get_action("detect_points") is not None
        assert get_action("NONEXISTENT") is None

    def test_output_formats_are_renderable_by_sdk(self) -> None:
        allowed = {
            "text",
            "points",
            "boxes",
            "masks",
            "trajectory",
            "plan_steps",
            "detections_3d",
            "grasps",
            "relations",
        }
        for a in STRUCTURED_ACTIONS:
            assert a.output_format in allowed, a

    def test_er_1_6_actions_are_mirrored(self) -> None:
        """Gemini Robotics-ER 1.6 adds two structured tasks; the SDK must
        surface them so consumers using ``cyberwave.mlmodels.actions`` get
        auto-complete without a network round-trip."""
        ids = {a.id for a in STRUCTURED_ACTIONS}
        assert "point_trajectory" in ids
        assert "plan_steps" in ids
        plan = get_action("plan_steps")
        assert plan is not None and plan.requires_image is False

    def test_perception_3d_actions_are_mirrored(self) -> None:
        """The CaP-X-inspired expansion (3D detections + grasps +
        scene graph relations) must be discoverable from the SDK.
        Without the mirror, SDK-first workflows would fail validation
        in ``MLModelsClient.run`` even though the backend accepts the
        task id."""
        ids = {a.id for a in STRUCTURED_ACTIONS}
        assert "detect_objects_3d" in ids
        assert "predict_grasps" in ids
        assert "detect_relations" in ids

    def test_frozen_dataclass(self) -> None:
        action = get_action("detect_points")
        assert action is not None
        with pytest.raises(Exception):
            action.id = "mutated"  # type: ignore[misc]


class TestBackendAlignment:
    """Catch drift between the SDK id list and the backend source of truth.

    The SDK no longer mirrors prompt templates or JSON schemas — only ids
    plus the ``output_format`` contract — so we compare those two fields.
    """

    def test_sdk_ids_match_backend_catalog(self) -> None:
        try:
            from src.lib.structured_actions import (  # type: ignore[import-untyped]
                STRUCTURED_ACTIONS as BACKEND_ACTIONS,
            )
        except ModuleNotFoundError:
            pytest.xfail(
                "Backend source (src.lib.structured_actions) not importable. "
                "Run pytest from the repo root to include this cross-package check."
            )

        sdk_ids = [a.id for a in STRUCTURED_ACTIONS]
        backend_ids = [a.id for a in BACKEND_ACTIONS]
        assert sdk_ids == backend_ids, (
            "SDK id list drifted from backend catalog. "
            f"SDK={sdk_ids}  backend={backend_ids}"
        )

        backend_by_id = {a.id: a for a in BACKEND_ACTIONS}
        for sdk_a in STRUCTURED_ACTIONS:
            backend_a = backend_by_id[sdk_a.id]
            assert sdk_a.output_format == backend_a.output_format, sdk_a.id
