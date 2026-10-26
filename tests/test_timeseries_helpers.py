"""Hand-written time-series helpers (replay-feed README §13): dataset channels and
rows, twin bindings and thing_id, environment sensor sources and auto-bind."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyberwave.exceptions import (
    ConflictException,
    CyberwaveAPIError,
    ServiceException,
    UnprocessableEntityException,
)
from cyberwave.resources import DatasetManager, EnvironmentManager, TwinManager

DATASET = "a1985343-53c2-45b2-b366-09a0606dc649"
ENVIRONMENT = "ba7e2341-b3a9-4fdf-97e9-808a234443a2"
TWIN = "0f3c6a2e-5b1d-4c7e-9a8f-1b2c3d4e5f60"
AUTH = ["CustomTokenAuthentication"]


def transport(result):
    """A fake generated ApiClient: records param_serialize kwargs, answers ``result``."""
    api_client = Mock()
    api_client.param_serialize.return_value = ("serialized",)
    api_client.response_deserialize.return_value = SimpleNamespace(data=result)
    return api_client


def request_of(api_client):
    return api_client.param_serialize.call_args.kwargs


def dataset_manager(result):
    api_client = transport(result)
    return DatasetManager(SimpleNamespace(api_client=api_client)), api_client


def environment_manager(result):
    api_client = transport(result)
    return EnvironmentManager(SimpleNamespace(api_client=api_client)), api_client


# --- datasets --------------------------------------------------------------


def test_channels_sends_filters_through_the_session():
    page = {
        "items": [{"channel_id": "ipmi/ambient/node=960"}],
        "total": 1,
        "limit": 20,
        "offset": 40,
    }
    manager, api_client = dataset_manager(page)

    result = manager.channels(
        DATASET, q="ambient node", type="temperature", limit=20, offset=40
    )

    assert result is page
    request = request_of(api_client)
    assert request["method"] == "GET"
    assert request["resource_path"] == "/api/v1/datasets/{uuid}/timeseries/channels"
    assert request["path_params"] == {"uuid": DATASET}
    assert request["auth_settings"] == AUTH
    assert request["header_params"]["Accept"] == "application/json"
    query = dict(request["query_params"])
    assert query == {
        "include_hidden": False,
        "limit": 20,
        "offset": 40,
        "q": "ambient node",
        "type": "temperature",
    }
    api_client.call_api.assert_called_once_with("serialized")
    assert (
        api_client.response_deserialize.call_args.kwargs["response_types_map"]["200"]
        == "object"
    )


def test_channels_defaults_match_the_contract():
    manager, api_client = dataset_manager({"items": []})

    manager.channels(
        DATASET, signal="ipmi/ambient", thing="node=960", include_hidden=True
    )

    query = dict(request_of(api_client)["query_params"])
    assert query == {
        "include_hidden": True,
        "limit": 50,
        "offset": 0,
        "signal": "ipmi/ambient",
        "thing": "node=960",
    }


def test_rows_repeats_channel_and_formats_times():
    rows = {"rows": [], "count": 0, "truncated": False}
    manager, api_client = dataset_manager(rows)
    channels = ["ipmi/ambient/node=960", "ipmi/total_power/node=960"]

    result = manager.rows(
        DATASET,
        channels,
        datetime(2022, 2, 8, 5, 0),
        datetime(
            2022, 2, 8, 7, 0, tzinfo=timezone(__import__("datetime").timedelta(hours=1))
        ),
        limit=5000,
    )

    assert result is rows
    request = request_of(api_client)
    assert request["method"] == "GET"
    assert request["resource_path"] == "/api/v1/datasets/{uuid}/timeseries/rows"
    assert request["path_params"] == {"uuid": DATASET}
    assert request["query_params"] == [
        ("channel", channels[0]),
        ("channel", channels[1]),
        ("start", "2022-02-08T05:00:00Z"),
        ("end", "2022-02-08T06:00:00Z"),
        ("limit", 5000),
    ]


def test_rows_accepts_one_channel_string_and_iso_text():
    manager, api_client = dataset_manager({"rows": []})

    manager.rows(
        DATASET, "weather/temp", "2022-02-01T00:00:00Z", "2022-02-02T00:00:00Z"
    )

    query = request_of(api_client)["query_params"]
    assert query[:3] == [
        ("channel", "weather/temp"),
        ("start", "2022-02-01T00:00:00Z"),
        ("end", "2022-02-02T00:00:00Z"),
    ]
    assert ("limit", 100000) in query


def test_rows_refuses_more_than_200_channels_before_sending():
    manager, api_client = dataset_manager({"rows": []})
    with pytest.raises(ValueError, match="1 to 200"):
        manager.rows(
            DATASET,
            [f"c{i}" for i in range(201)],
            "2022-02-01T00:00:00Z",
            "2022-02-02T00:00:00Z",
        )
    with pytest.raises(ValueError, match="1 to 200"):
        manager.rows(DATASET, [], "2022-02-01T00:00:00Z", "2022-02-02T00:00:00Z")
    api_client.call_api.assert_not_called()


def test_errors_carry_the_backend_detail():
    manager, api_client = dataset_manager(None)
    api_client.response_deserialize.side_effect = ConflictException(
        status=409,
        reason="Conflict",
        body=json.dumps({"detail": "Dataset has no time-series data"}),
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.channels(DATASET)

    error = excinfo.value
    assert error.status_code == 409
    assert error.response_data == {"detail": "Dataset has no time-series data"}
    assert error.operation == f"list channels of dataset {DATASET}"
    assert (
        str(error)
        == f"Failed to list channels of dataset {DATASET}: Dataset has no time-series data (HTTP 409)"
    )


def test_validation_error_lists_are_rendered():
    manager, api_client = dataset_manager(None)
    detail = [{"loc": ["query", "end"], "msg": "field required"}]
    api_client.response_deserialize.side_effect = UnprocessableEntityException(
        status=422, reason="Unprocessable Entity", body=json.dumps({"detail": detail})
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.rows(DATASET, ["c"], "2022-02-01T00:00:00Z", "2022-02-02T00:00:00Z")

    assert excinfo.value.status_code == 422
    assert "field required" in str(excinfo.value)


def test_error_without_json_body_still_raises_api_error():
    manager, api_client = dataset_manager(None)
    api_client.response_deserialize.side_effect = ServiceException(
        status=502, reason="Bad Gateway", body="<html>bad gateway</html>"
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.channels(DATASET)

    assert excinfo.value.status_code == 502
    assert str(excinfo.value).startswith(
        f"Failed to list channels of dataset {DATASET}: "
    )


# --- twins -----------------------------------------------------------------


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


def test_data_sources_returns_the_v2_map_with_legacy_entries_converted():
    manager, api = twin_manager(
        {
            "thing_id": "node=960",
            "data_sources": {
                "inlet_temperature": {
                    "dataset_uuid": DATASET,
                    "channel_id": "ipmi/ambient/node=960",
                    "use_recorded": True,
                    "label": "Inlet air temperature",
                    "unit": "°C",
                    "type": "temperature",
                    "allow_type_mismatch": True,
                },
                "power": {
                    "dataset_uuid": DATASET,
                    "channel_id": "ipmi/total_power/node=960",
                    "source": "replay",
                },
                "fan_speed": {"source": "environment"},
                "": {"dataset_uuid": DATASET, "channel_id": "x"},
                "broken": "not a dict",
            },
        }
    )

    sources = manager.data_sources(TWIN)

    assert sources == {
        "inlet_temperature": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/ambient/node=960",
            "use_recorded": True,
            "label": "Inlet air temperature",
            "unit": "°C",
            "type": "temperature",
            "allow_type_mismatch": True,
        },
        "power": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/total_power/node=960",
            "use_recorded": True,
        },
    }
    api.src_app_api_twins_get_twin.assert_called_once_with(TWIN)


def test_data_sources_is_empty_without_bindings():
    manager, _ = twin_manager({})
    assert manager.data_sources(TWIN) == {}
    manager, _ = twin_manager(None)
    assert manager.data_sources(TWIN) == {}


def test_bind_sensor_writes_the_whole_map_in_v2_shape():
    existing = {
        "power": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/total_power/node=960",
            "source": "replay",
        }
    }
    manager, api = twin_manager({"data_sources": existing})

    result = manager.bind_sensor(
        TWIN,
        "inlet_temperature",
        DATASET,
        "ipmi/ambient/node=960",
        label="Inlet air temperature",
        unit="°C",
        type="temperature",
    )

    expected = {
        "power": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/total_power/node=960",
            "use_recorded": True,
        },
        "inlet_temperature": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/ambient/node=960",
            "use_recorded": True,
            "label": "Inlet air temperature",
            "unit": "°C",
            "type": "temperature",
        },
    }
    assert written_metadata(api) == {"data_sources": expected}
    # The returned map is the one the backend stored (here the fake echoes the twin).
    assert result == {"power": expected["power"]}


def test_bind_sensor_options_and_returned_map_from_response():
    stored = {
        "fan_speed": {
            "dataset_uuid": DATASET,
            "channel_id": "vertiv/Fan_Speed",
            "use_recorded": False,
            "type": "fan_speed",
            "allow_type_mismatch": True,
        }
    }
    manager, api = twin_manager(
        {"data_sources": {}}, stored_metadata={"data_sources": stored}
    )

    result = manager.bind_sensor(
        TWIN,
        "fan_speed",
        DATASET,
        "vertiv/Fan_Speed",
        use_recorded=False,
        allow_type_mismatch=True,
    )

    assert written_metadata(api) == {
        "data_sources": {
            "fan_speed": {
                "dataset_uuid": DATASET,
                "channel_id": "vertiv/Fan_Speed",
                "use_recorded": False,
                "allow_type_mismatch": True,
            }
        }
    }
    assert result == stored


def test_bind_sensor_surfaces_the_type_gate_refusal():
    manager, api = twin_manager({"data_sources": {}})
    api.src_app_api_twins_update_twin.side_effect = UnprocessableEntityException(
        status=422,
        reason="Unprocessable Entity",
        body=json.dumps({"detail": "unknown channel"}),
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.bind_sensor(TWIN, "inlet_temperature", DATASET, "ipmi/nope/node=960")

    assert excinfo.value.status_code == 422
    assert "unknown channel" in str(excinfo.value)


def test_unbind_sensor_omits_the_key_and_skips_unknown_slots():
    sources = {
        "inlet_temperature": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/ambient/node=960",
            "use_recorded": True,
        },
        "power": {
            "dataset_uuid": DATASET,
            "channel_id": "ipmi/total_power/node=960",
            "use_recorded": True,
        },
    }
    manager, api = twin_manager({"data_sources": sources})

    manager.unbind_sensor(TWIN, "power")
    assert written_metadata(api) == {
        "data_sources": {"inlet_temperature": sources["inlet_temperature"]}
    }

    api.src_app_api_twins_update_twin.reset_mock()
    assert manager.unbind_sensor(TWIN, "missing") == sources
    api.src_app_api_twins_update_twin.assert_not_called()


def test_set_thing_id_merges_one_metadata_key():
    manager, api = twin_manager({"data_sources": {"x": {}}})

    manager.set_thing_id(TWIN, "node=960")

    assert written_metadata(api) == {"thing_id": "node=960"}


# --- environments ----------------------------------------------------------


def test_sensor_sources_reads_the_environment_route():
    payload = {"twins": [], "datasets": []}
    manager, api_client = environment_manager(payload)

    assert manager.sensor_sources(ENVIRONMENT) is payload
    request = request_of(api_client)
    assert request["method"] == "GET"
    assert request["resource_path"] == "/api/v1/environments/{uuid}/sensor-sources"
    assert request["path_params"] == {"uuid": ENVIRONMENT}
    assert request["auth_settings"] == AUTH
    assert request["body"] is None


def test_auto_bind_posts_the_contract_body():
    report = {"twins": [], "bound": 0, "skipped": 0}
    manager, api_client = environment_manager(report)
    template = {"inlet_temperature": "ipmi/ambient"}

    result = manager.auto_bind(
        ENVIRONMENT,
        DATASET,
        twin_ids=[SimpleNamespace(uuid=TWIN), "second"],
        include_docked_children=True,
        template=template,
        use_recorded=False,
        overwrite=True,
    )

    assert result is report
    request = request_of(api_client)
    assert request["method"] == "POST"
    assert (
        request["resource_path"] == "/api/v1/environments/{uuid}/sensor-bindings/auto"
    )
    assert request["path_params"] == {"uuid": ENVIRONMENT}
    assert request["body"] == {
        "dataset_uuid": DATASET,
        "twin_uuids": [TWIN, "second"],
        "include_docked_children": True,
        "template": template,
        "use_recorded": False,
        "overwrite": True,
    }
    assert request["body"]["template"] is not template


def test_auto_bind_omits_twins_and_template_when_not_given():
    manager, api_client = environment_manager({})

    manager.auto_bind(ENVIRONMENT, DATASET)

    assert request_of(api_client)["body"] == {
        "dataset_uuid": DATASET,
        "include_docked_children": False,
        "use_recorded": True,
        "overwrite": False,
    }


def test_auto_bind_422_names_the_reason():
    manager, api_client = environment_manager(None)
    api_client.response_deserialize.side_effect = UnprocessableEntityException(
        status=422,
        reason="Unprocessable Entity",
        body=json.dumps({"detail": "dataset has no slot template"}),
    )

    with pytest.raises(CyberwaveAPIError, match="dataset has no slot template"):
        manager.auto_bind(ENVIRONMENT, DATASET)
