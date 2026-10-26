"""ReplayFeedManager (replay-feed README §3.2 routes through §13 helpers) and its wiring."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyberwave import Cyberwave
from cyberwave.exceptions import (
    ConflictException,
    CyberwaveAPIError,
    NotFoundException,
    UnprocessableEntityException,
)
from cyberwave.resources import AssetKitManager, ReplayFeedManager

ENVIRONMENT = "ba7e2341-b3a9-4fdf-97e9-808a234443a2"
FEED = "6a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
FEEDS = "/api/v1/environments/{uuid}/replay-feeds"
FEED_PATH = "/api/v1/environments/{uuid}/replay-feeds/{feed_uuid}"


def manager_with_transport(result):
    api_client = Mock()
    api_client.param_serialize.return_value = ("serialized",)
    api_client.response_deserialize.return_value = SimpleNamespace(data=result)
    return ReplayFeedManager(SimpleNamespace(api_client=api_client)), api_client


def request_of(api_client):
    return api_client.param_serialize.call_args.kwargs


def test_client_exposes_the_new_managers():
    client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
    assert isinstance(client.replay_feeds, ReplayFeedManager)
    assert client.replay_feeds.api is client.api
    assert isinstance(client.asset_kits, AssetKitManager)
    assert client.asset_kits._assets is client.assets


def test_start_posts_the_contract_body():
    feed = {"uuid": FEED, "status": "queued"}
    manager, api_client = manager_with_transport(feed)

    result = manager.start(
        ENVIRONMENT,
        datetime(2022, 2, 8, 5, 0, tzinfo=timezone.utc),
        end="2022-02-08T06:00:00Z",
        speed=60,
        loop=True,
        run_id="demo-01",
        source_type="sim",
        max_duration_s=600,
    )

    assert result is feed
    request = request_of(api_client)
    assert request["method"] == "POST"
    assert request["resource_path"] == FEEDS
    assert request["path_params"] == {"uuid": ENVIRONMENT}
    assert request["auth_settings"] == ["CustomTokenAuthentication"]
    assert request["header_params"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    assert request["body"] == {
        "start": "2022-02-08T05:00:00Z",
        "end": "2022-02-08T06:00:00Z",
        "speed": 60.0,
        "loop": True,
        "run_id": "demo-01",
        "source_type": "sim",
        "max_duration_s": 600,
    }
    types = api_client.response_deserialize.call_args.kwargs["response_types_map"]
    assert types["201"] == "object"


def test_start_defaults_leave_optional_fields_out():
    manager, api_client = manager_with_transport({})

    manager.start(ENVIRONMENT, "2022-02-08T05:00:00Z")

    assert request_of(api_client)["body"] == {
        "start": "2022-02-08T05:00:00Z",
        "speed": 1.0,
        "loop": False,
        "source_type": "auto",
    }


def test_start_422_carries_the_backend_reason():
    manager, api_client = manager_with_transport(None)
    detail = "Recorded data selected on r48-s19-top.fan_speed but its dataset is missing; clear it or attach another channel"
    api_client.response_deserialize.side_effect = UnprocessableEntityException(
        status=422, reason="Unprocessable Entity", body=json.dumps({"detail": detail})
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.start(ENVIRONMENT, "2022-02-08T05:00:00Z")

    assert excinfo.value.status_code == 422
    assert (
        str(excinfo.value)
        == f"Failed to start replay feed in environment {ENVIRONMENT}: {detail} (HTTP 422)"
    )
    assert excinfo.value.response_data == {"detail": detail}


def test_current_and_get_use_the_read_routes():
    manager, api_client = manager_with_transport({"uuid": FEED})

    assert manager.current(ENVIRONMENT) == {"uuid": FEED}
    request = request_of(api_client)
    assert (request["method"], request["resource_path"]) == ("GET", FEEDS + "/current")
    assert request["path_params"] == {"uuid": ENVIRONMENT}
    assert request["body"] is None

    manager.get(ENVIRONMENT, FEED)
    request = request_of(api_client)
    assert (request["method"], request["resource_path"]) == ("GET", FEED_PATH)
    assert request["path_params"] == {"uuid": ENVIRONMENT, "feed_uuid": FEED}


def test_current_404_raises_api_error():
    manager, api_client = manager_with_transport(None)
    api_client.response_deserialize.side_effect = NotFoundException(
        status=404,
        reason="Not Found",
        body=json.dumps({"detail": "No replay feed in this environment"}),
    )
    with pytest.raises(CyberwaveAPIError, match="No replay feed in this environment"):
        manager.current(ENVIRONMENT)


@pytest.mark.parametrize("action", ["pause", "resume", "stop"])
def test_pause_resume_stop_post_to_the_feed(action):
    manager, api_client = manager_with_transport({"uuid": FEED, "status": "paused"})

    result = getattr(manager, action)(ENVIRONMENT, SimpleNamespace(uuid=FEED))

    assert result["uuid"] == FEED
    request = request_of(api_client)
    assert request["method"] == "POST"
    assert request["resource_path"] == FEED_PATH + "/" + action
    assert request["path_params"] == {"uuid": ENVIRONMENT, "feed_uuid": FEED}
    assert request["body"] is None


def test_feed_dicts_are_accepted_as_feed_id():
    manager, api_client = manager_with_transport({"uuid": FEED, "status": "stopped"})

    manager.stop(ENVIRONMENT, {"uuid": FEED, "status": "playing"})

    assert request_of(api_client)["path_params"] == {
        "uuid": ENVIRONMENT,
        "feed_uuid": FEED,
    }


def test_pause_409_is_an_api_error_with_status():
    manager, api_client = manager_with_transport(None)
    api_client.response_deserialize.side_effect = ConflictException(
        status=409,
        reason="Conflict",
        body=json.dumps({"detail": "Feed is not playing"}),
    )
    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager.pause(ENVIRONMENT, FEED)
    assert excinfo.value.status_code == 409
    assert "Feed is not playing" in str(excinfo.value)
