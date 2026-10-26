"""Hand-written operating-range helpers (replay-feed README §16.4): twin
overrides in ``metadata.operating_ranges`` and the backend's ``sensor_status``."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyberwave.resources import (
    OPERATING_RANGE_OFF,
    TwinManager,
    _normalize_alert_action,
    _normalize_operating_ranges,
)

TWIN = "0f3c6a2e-5b1d-4c7e-9a8f-1b2c3d4e5f60"


def twin_manager(metadata, stored_metadata=None):
    api = Mock()
    api.src_app_api_twins_get_twin.return_value = SimpleNamespace(
        uuid=TWIN, metadata=metadata
    )
    api.src_app_api_twins_update_twin.return_value = SimpleNamespace(
        uuid=TWIN, metadata=stored_metadata if stored_metadata is not None else metadata
    )
    return TwinManager(api), api


def written_metadata(api):
    twin_id, schema = api.src_app_api_twins_update_twin.call_args.args
    assert twin_id == TWIN
    return schema.metadata


def test_operating_ranges_reads_the_override_map_and_drops_junk():
    manager, _ = twin_manager(
        {
            "operating_ranges": {
                "inlet_temperature": {"min": 15, "max": 35},
                "total_power": {"min": None, "max": 2500.0},
                "fan_speed": {"min": 1000},
                "both_null": {"min": None, "max": None},
                "text": {"min": "15", "max": 35},
                "inverted": {"min": 35, "max": 15},
                "junk": "15-35",
                "": {"min": 1, "max": 2},
            }
        }
    )

    assert manager.operating_ranges(TWIN) == {
        "inlet_temperature": {"min": 15.0, "max": 35.0},
        "total_power": {"min": None, "max": 2500.0},
        "fan_speed": {"min": 1000.0, "max": None},
    }


def test_operating_ranges_is_empty_without_overrides():
    manager, _ = twin_manager({})
    assert manager.operating_ranges(TWIN) == {}
    assert _normalize_operating_ranges("nope") == {}


def test_set_operating_range_writes_the_whole_map():
    existing = {"total_power": {"min": 0, "max": 2500}}
    expected = {
        "total_power": {"min": 0.0, "max": 2500.0},
        "inlet_temperature": {"min": 15.0, "max": 20.0},
    }
    manager, api = twin_manager(
        {"operating_ranges": existing, "data_sources": {}},
        stored_metadata={"operating_ranges": expected},
    )

    result = manager.set_operating_range(TWIN, "inlet_temperature", min=15, max=20)

    assert result == expected
    # Only the top-level key is sent: PUT /twins merges top-level metadata keys.
    assert written_metadata(api) == {"operating_ranges": expected}


def test_set_operating_range_with_one_open_side():
    manager, api = twin_manager({})

    manager.set_operating_range(TWIN, "inlet_temperature", max=20)

    assert written_metadata(api) == {
        "operating_ranges": {"inlet_temperature": {"min": None, "max": 20.0}}
    }


def test_set_operating_range_returns_what_the_backend_stored():
    stored = {"inlet_temperature": {"min": 10, "max": 20}}
    manager, _ = twin_manager({}, stored_metadata={"operating_ranges": stored})

    result = manager.set_operating_range(TWIN, "inlet_temperature", min=15, max=20)

    assert result == {"inlet_temperature": {"min": 10.0, "max": 20.0}}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "at least one bound"),
        ({"min": None, "max": None}, "at least one bound"),
        ({"min": 35, "max": 15}, "must not exceed"),
        ({"min": "15", "max": 35}, "must be a number"),
        ({"min": True}, "must be a number"),
        ({"max": float("nan")}, "must be finite"),
    ],
)
def test_set_operating_range_refuses_bad_bounds_before_sending(kwargs, match):
    manager, api = twin_manager({})

    with pytest.raises(ValueError, match=match):
        manager.set_operating_range(TWIN, "inlet_temperature", **kwargs)

    api.src_app_api_twins_update_twin.assert_not_called()


def test_clear_operating_range_omits_the_key_and_skips_unknown_slots():
    ranges = {
        "inlet_temperature": {"min": 15, "max": 35},
        "total_power": {"min": 0, "max": 2500},
    }
    manager, api = twin_manager({"operating_ranges": ranges})

    manager.clear_operating_range(TWIN, "total_power")
    assert written_metadata(api) == {
        "operating_ranges": {"inlet_temperature": {"min": 15.0, "max": 35.0}}
    }

    api.src_app_api_twins_update_twin.reset_mock()
    assert manager.clear_operating_range(TWIN, "missing") == {
        "inlet_temperature": {"min": 15.0, "max": 35.0},
        "total_power": {"min": 0.0, "max": 2500.0},
    }
    api.src_app_api_twins_update_twin.assert_not_called()


def test_sensor_status_reads_the_backend_verdict():
    status = {
        "out_of_range": ["inlet_temperature"],
        "checked_at": "2026-10-19T10:00:00Z",
    }
    manager, _ = twin_manager({"sensor_status": status})
    assert manager.sensor_status(TWIN) == status


@pytest.mark.parametrize("metadata", [{}, {"sensor_status": None}, {"sensor_status": 1}])
def test_sensor_status_is_none_until_the_backend_judged(metadata):
    manager, _ = twin_manager(metadata)
    assert manager.sensor_status(TWIN) is None


def test_disable_operating_range_stores_the_off_switch_and_clear_removes_it():
    manager, api = twin_manager(
        {"operating_ranges": {"inlet_temperature": {"min": 15, "max": 35}}},
        stored_metadata={"operating_ranges": {"inlet_temperature": {"enabled": False}}},
    )
    assert manager.disable_operating_range(TWIN, "inlet_temperature") == {
        "inlet_temperature": {"enabled": False}
    }
    assert written_metadata(api) == {
        "operating_ranges": {"inlet_temperature": {"enabled": False}}
    }

    manager, api = twin_manager({"operating_ranges": {"fan_speed": OPERATING_RANGE_OFF}})
    assert _normalize_operating_ranges({"fan_speed": {"enabled": False}}) == {
        "fan_speed": {"enabled": False}
    }
    # Already off: nothing is written.
    manager.disable_operating_range(TWIN, "fan_speed")
    api.src_app_api_twins_update_twin.assert_not_called()
    assert manager.operating_ranges(TWIN) == {"fan_speed": {"enabled": False}}
    manager.clear_operating_range(TWIN, "fan_speed")
    assert written_metadata(api) == {"operating_ranges": {}}


def test_alert_action_reads_the_key_and_defaults_to_none():
    manager, _ = twin_manager({"alert_action": {"kind": "color", "color": "#EF4444"}})
    assert manager.alert_action(TWIN) == {"kind": "color", "color": "#ef4444"}

    manager, _ = twin_manager({})
    assert manager.alert_action(TWIN) == {"kind": "none"}
    assert _normalize_alert_action({"kind": "blink"}) == {"kind": "none"}
    assert _normalize_alert_action({"kind": "marker", "color": "red"}) == {
        "kind": "marker"
    }


def test_set_alert_action_writes_only_the_top_level_key():
    manager, api = twin_manager(
        {"data_sources": {}},
        stored_metadata={"alert_action": {"kind": "color", "color": "#ef4444"}},
    )

    result = manager.set_alert_action(TWIN, "color", "#EF4444")

    assert result == {"kind": "color", "color": "#ef4444"}
    assert written_metadata(api) == {
        "alert_action": {"kind": "color", "color": "#ef4444"}
    }

    api.src_app_api_twins_update_twin.reset_mock()
    manager.set_alert_action(TWIN, "marker")
    assert written_metadata(api) == {"alert_action": {"kind": "marker"}}


@pytest.mark.parametrize(
    ("kind", "color", "match"),
    [
        ("blink", None, "kind must be one of"),
        ("color", None, "needs a color"),
        ("color", "red", "must be '#rrggbb'"),
        ("color", "#fff", "must be '#rrggbb'"),
        ("marker", "#12345g", "must be '#rrggbb'"),
    ],
)
def test_set_alert_action_refuses_bad_input_before_sending(kind, color, match):
    manager, api = twin_manager({})

    with pytest.raises(ValueError, match=match):
        manager.set_alert_action(TWIN, kind, color)

    api.src_app_api_twins_update_twin.assert_not_called()
