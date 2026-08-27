"""Tests for cyberwave.managers.recordings."""

from __future__ import annotations

import logging

import pytest

from cyberwave.exceptions import CyberwaveError, RecordingPayloadTooLargeError
from cyberwave.managers.recordings import (
    RecordingList,
    RecordingListItem,
    RecordingType,
)
from cyberwave.rest.exceptions import ServiceException


def _item(uuid: str, twin: str | None, metadata: dict) -> RecordingListItem:
    return RecordingListItem(
        uuid=uuid, twin_uuid=twin, environment_uuid="env-1", metadata=metadata
    )


def test_classify_camera_from_recording_type() -> None:
    item = _item("r1", "t1", {"recording_type": "camera"})
    assert item.types == frozenset({RecordingType.CAMERA})


def test_classify_camera_from_mp4_path() -> None:
    item = _item("r1", "t1", {"mp4_path": "mp4/x.mp4"})
    assert RecordingType.CAMERA in item.types


def test_classify_robot() -> None:
    item = _item("r1", "t1", {"metadata_type": "TwinRecordingMetadata"})
    assert item.types == frozenset({RecordingType.ROBOT})


def test_classify_depth_camera_is_camera_and_depth() -> None:
    # The ``pointcloud`` key carries raw depth maps -> DEPTH (not POINTCLOUD).
    item = _item("r1", "t1", {"recording_type": "camera", "pointcloud": {"t1": "pc1"}})
    assert item.types == frozenset({RecordingType.CAMERA, RecordingType.DEPTH})


def test_classify_colored_is_pointcloud() -> None:
    item = _item("r1", "t1", {"colored_pointcloud": {"t1": "cpc1"}})
    assert item.types == frozenset({RecordingType.POINTCLOUD})


def test_classify_depth_camera_with_colored_is_camera_depth_pointcloud() -> None:
    item = _item(
        "r1",
        "t1",
        {
            "recording_type": "camera",
            "pointcloud": {"t1": "pc1"},
            "colored_pointcloud": {"t1": "cpc1"},
        },
    )
    assert item.types == frozenset(
        {RecordingType.CAMERA, RecordingType.DEPTH, RecordingType.POINTCLOUD}
    )


def test_classify_audio() -> None:
    item = _item("r1", "t1", {"audio_parts": [{"chunk_index": 0}]})
    assert item.types == frozenset({RecordingType.AUDIO})


def test_classify_active_manifest_camera_from_video_parts() -> None:
    # In-progress recordings arrive as a "recording_type": "active" manifest that
    # carries video_parts instead of mp4_path; they must still classify as CAMERA
    # so filtering does not drop recordings that are still being written.
    item = _item(
        "r1",
        "t1",
        {"recording_type": "active", "video_parts": [{"chunk_index": 0}]},
    )
    assert RecordingType.CAMERA in item.types


def test_classify_active_manifest_robot_from_robot_parts() -> None:
    item = _item(
        "r1",
        "t1",
        {"recording_type": "active", "robot_parts": [{"chunk_index": 0}]},
    )
    assert RecordingType.ROBOT in item.types


def test_classify_active_manifest_empty_parts_is_untyped() -> None:
    # Empty part lists (a bare active shell) must not spuriously match any type.
    item = _item(
        "r1",
        "t1",
        {"recording_type": "active", "video_parts": [], "robot_parts": []},
    )
    assert item.types == frozenset()


def test_recording_list_filter_single_type() -> None:
    items = RecordingList([
        _item("cam", "t1", {"recording_type": "camera"}),
        _item("rob", "t1", {"metadata_type": "TwinRecordingMetadata"}),
    ])
    filtered = items.filter(RecordingType.CAMERA)
    assert isinstance(filtered, RecordingList)
    assert [i.uuid for i in filtered] == ["cam"]


def test_recording_list_filter_array_is_or_combined() -> None:
    items = RecordingList([
        _item("cam", "t1", {"recording_type": "camera"}),
        _item("rob", "t1", {"metadata_type": "TwinRecordingMetadata"}),
        _item("aud", "t1", {"audio_parts": [{}]}),
    ])
    filtered = items.filter([RecordingType.CAMERA, RecordingType.ROBOT])
    assert {i.uuid for i in filtered} == {"cam", "rob"}


def test_filter_accepts_string_values() -> None:
    items = RecordingList([_item("cam", "t1", {"recording_type": "camera"})])
    assert [i.uuid for i in items.filter("camera")] == ["cam"]


from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.managers.recordings import RecordingManager, TwinRecordingsHandle


def _rest_item(uuid: str, twin: str | None, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid, twin_uuid=twin, environment_uuid="env-1", metadata=metadata
    )


def _stub_availability(
    api: MagicMock, last_date: "str | None" = "2026-07-05"
) -> MagicMock:
    """Stub the availability call the way the REST contract shapes it.

    ``first_date``/``last_date`` are ISO **strings** on the wire, not ``date``
    objects — stubbing them as dates would hide the normalization the manager
    has to do before handing them to the catalog endpoint.
    """
    api.src_app_api_environments_recordings_get_environment_recordings_availability.return_value = SimpleNamespace(
        first_date=last_date,
        last_date=last_date,
        days=[],
        total_count=0,
        facets={},
        timezone="UTC",
    )
    return api


def _api_listing(*items: SimpleNamespace) -> MagicMock:
    """A single-page API stub whose environment has recordings on 2026-07-05."""
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(items=list(items), next_cursor=None, has_more=False)
    )
    return api


_DEFAULT_LAST_DATE = object()


def _bind_to_generated_client(operation: str, uuid: str, kwargs: dict) -> dict:
    """Reject a call the generated ``DefaultApi`` would not accept.

    A ``**kwargs`` stub swallows a parameter that was renamed or dropped from
    the OpenAPI schema, so the pagination tests below would keep passing
    against a client that can no longer send the window, the cursor, or the
    twin filter. Binding to the real signature makes that a failure here rather
    than a silently unbounded request in production.

    ``cyberwave/rest`` is generated from the backend's OpenAPI schema and is not
    committed, so a stale checkout is a normal state to be in. This helper is
    reached from every transport-stubbed test in the file, so a missing
    operation has to name itself: a bare ``AttributeError`` from ``getattr``
    would surface ~20 times with no hint about regeneration, which is exactly
    the confusion ``test_generated_client_exposes_the_bounded_catalog_contract``
    exists to prevent.
    """
    import inspect

    from cyberwave.rest import DefaultApi

    method = getattr(DefaultApi, operation, None)
    if method is None:
        raise AssertionError(
            f"Generated REST client has no operation {operation!r}. "
            "cyberwave/rest is generated from the backend OpenAPI schema and is "
            "not committed — regenerate it before running these tests "
            "(see the generate step in "
            ".github/workflows/sdk-python-test-and-release.yml)."
        )

    try:
        inspect.signature(method).bind(None, uuid, **kwargs)  # `None` is `self`
    except TypeError as exc:
        raise AssertionError(
            f"{operation} does not accept {sorted(kwargs)} — the generated "
            f"client is out of step with what RecordingManager sends ({exc}). "
            "Regenerate cyberwave/rest, then re-check the manager if the "
            "parameter was genuinely renamed or dropped upstream."
        ) from exc
    return {"uuid": uuid, **kwargs}


class _FakeCatalogApi:
    """A REST stub that pages exactly like the catalog endpoint.

    ``cursor`` is the offset of the next unread row, which is enough to model
    the server's ``(effective_start_us, uuid)`` keyset walk for tests. Pass
    ``last_date=None`` for an environment that has no recordings at all;
    otherwise it is an ISO string, as the availability contract defines it.

    Both operations bind their arguments to the generated ``DefaultApi``
    signature before answering, so these stubs stay honest about what the real
    client accepts.
    """

    def __init__(
        self,
        rows: "list[SimpleNamespace]",
        *,
        last_date: object = _DEFAULT_LAST_DATE,
    ) -> None:
        self._rows = list(rows)
        self._last_date = (
            "2026-07-05" if last_date is _DEFAULT_LAST_DATE else last_date
        )
        self.availability_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def src_app_api_environments_recordings_get_environment_recordings_availability(
        self, uuid: str, **kwargs
    ) -> SimpleNamespace:
        self.availability_calls.append(
            _bind_to_generated_client(
                "src_app_api_environments_recordings_get_environment_recordings_availability",
                uuid,
                kwargs,
            )
        )
        return SimpleNamespace(
            first_date=self._last_date,
            last_date=self._last_date,
            days=[],
            total_count=len(self._rows),
            facets={},
            timezone="UTC",
        )

    def src_app_api_environments_recordings_get_environment_recordings(
        self, uuid: str, **kwargs
    ) -> SimpleNamespace:
        self.list_calls.append(
            _bind_to_generated_client(
                "src_app_api_environments_recordings_get_environment_recordings",
                uuid,
                kwargs,
            )
        )
        offset = int(kwargs["cursor"]) if kwargs.get("cursor") else 0
        window = self._rows[offset : offset + kwargs["limit"]]
        consumed = offset + len(window)
        has_more = consumed < len(self._rows)
        return SimpleNamespace(
            items=window,
            next_cursor=str(consumed) if has_more else None,
            has_more=has_more,
            page_size=len(window),
            facets={},
        )


def _rest_items(count: int, twin: str = "t1") -> "list[SimpleNamespace]":
    return [
        _rest_item(f"rec-{index:04d}", twin, {"recording_type": "camera"})
        for index in range(count)
    ]


def test_manager_list_without_dates_scopes_to_the_latest_available_day() -> None:
    """No date filter means "the most recent day that has recordings" — the same
    availability-then-window flow the replay calendar picker uses. The ISO
    string the availability contract returns must reach the catalog call as a
    real ``date``."""
    api = _FakeCatalogApi(_rest_items(3), last_date="2026-07-05")

    items = RecordingManager(api).list("env-1")

    assert [item.uuid for item in items] == ["rec-0000", "rec-0001", "rec-0002"]
    assert api.availability_calls[0]["uuid"] == "env-1"
    assert api.list_calls[0]["start_date"] == date(2026, 7, 5)
    assert api.list_calls[0]["end_date"] == date(2026, 7, 5)


def test_manager_list_defaults_to_ready_recordings_like_the_replay_picker() -> None:
    api = _FakeCatalogApi(_rest_items(1))

    RecordingManager(api).list("env-1")

    assert api.availability_calls[0]["include_unready"] is False
    assert api.list_calls[0]["include_unready"] is False


def test_manager_list_returns_empty_without_requesting_a_catalog_page() -> None:
    api = _FakeCatalogApi([], last_date=None)

    items = RecordingManager(api).list("env-1")

    assert list(items) == []
    assert api.list_calls == []


def test_manager_list_default_limit_reads_four_safe_pages() -> None:
    api = _FakeCatalogApi(_rest_items(500))

    items = RecordingManager(api).list("env-1")

    assert len(items) == 200
    assert [call["limit"] for call in api.list_calls] == [50, 50, 50, 50]


def test_manager_list_limit_zero_reads_every_page() -> None:
    api = _FakeCatalogApi(_rest_items(230))

    items = RecordingManager(api).list("env-1", limit=0)

    assert len(items) == 230
    assert [item.uuid for item in items][:2] == ["rec-0000", "rec-0001"]
    assert [call["limit"] for call in api.list_calls] == [50, 50, 50, 50, 50]


def test_manager_list_requests_only_the_remainder_on_the_final_page() -> None:
    api = _FakeCatalogApi(_rest_items(500))

    items = RecordingManager(api).list("env-1", limit=150)

    assert len(items) == 150
    assert [call["limit"] for call in api.list_calls] == [50, 50, 50]


def test_manager_list_follows_the_server_cursor_between_pages() -> None:
    api = _FakeCatalogApi(_rest_items(150))

    RecordingManager(api).list("env-1", limit=0)

    assert [call.get("cursor") for call in api.list_calls] == [None, "50", "100"]


def test_manager_list_stops_when_the_server_reports_no_further_pages() -> None:
    api = _FakeCatalogApi(_rest_items(30))

    items = RecordingManager(api).list("env-1", limit=0)

    assert len(items) == 30
    assert len(api.list_calls) == 1


def test_manager_list_stops_when_a_page_advertises_more_without_a_cursor() -> None:
    """A server that claims ``has_more`` but omits ``next_cursor`` must end the
    walk rather than replay the same page forever."""
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(items=_rest_items(2), next_cursor=None, has_more=True)
    )

    items = RecordingManager(api).list("env-1", limit=0)

    assert len(items) == 2
    assert (
        api.src_app_api_environments_recordings_get_environment_recordings.call_count
        == 1
    )


def test_manager_list_warns_that_only_the_latest_day_was_listed(caplog) -> None:
    """The implicit day window is a silent narrowing otherwise — a caller who
    expects the whole history has to be told which day they actually got."""
    api = _FakeCatalogApi(_rest_items(3), last_date="2026-07-05")

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        RecordingManager(api).list("env-1")

    assert "2026-07-05" in caplog.text
    assert "start=" in caplog.text and "end=" in caplog.text


def test_manager_list_does_not_warn_about_the_window_when_dates_are_given(
    caplog,
) -> None:
    api = _FakeCatalogApi(_rest_items(3))

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        RecordingManager(api).list("env-1", start="2026-07-01", end="2026-07-05")

    assert caplog.text == ""


def test_manager_list_warns_when_the_limit_left_recordings_unread(caplog) -> None:
    api = _FakeCatalogApi(_rest_items(500))

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        items = RecordingManager(api).list("env-1", start="2026-07-01", end="2026-07-05")

    assert len(items) == 200
    assert "limit=200" in caplog.text
    assert "limit=0" in caplog.text  # tells the caller how to get the rest


def test_manager_list_does_not_warn_about_the_limit_when_nothing_was_left(
    caplog,
) -> None:
    api = _FakeCatalogApi(_rest_items(30))

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        RecordingManager(api).list("env-1", start="2026-07-01", end="2026-07-05")

    assert caplog.text == ""


def test_manager_list_limit_zero_never_warns_about_truncation(caplog) -> None:
    api = _FakeCatalogApi(_rest_items(230))

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        RecordingManager(api).list(
            "env-1", start="2026-07-01", end="2026-07-05", limit=0
        )

    assert caplog.text == ""


def test_manager_list_warns_when_the_server_cannot_hand_back_a_next_cursor(
    caplog,
) -> None:
    """Stopping on an unusable cursor drops recordings the server says exist —
    that must not look like a complete result."""
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(items=_rest_items(2), next_cursor=None, has_more=True)
    )

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        RecordingManager(api).list("env-1", start="2026-07-01", end="2026-07-05")

    assert "cursor" in caplog.text


def test_manager_list_stops_when_the_cursor_stops_advancing() -> None:
    """A server that keeps handing back the cursor we just used must end the
    walk. Without this the ``limit=0`` loop re-requests one page forever."""
    calls = {"count": 0}

    def stuck_page(*_args, **_kwargs) -> SimpleNamespace:
        calls["count"] += 1
        if calls["count"] > 5:
            raise AssertionError("cursor walk never terminated")
        return SimpleNamespace(items=_rest_items(2), next_cursor="stuck", has_more=True)

    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.side_effect = (
        stuck_page
    )

    items = RecordingManager(api).list("env-1", limit=0)

    assert calls["count"] == 2  # the first page, then the one that proves no progress
    assert len(items) == 4


def test_manager_list_negative_limit_raises() -> None:
    with pytest.raises(CyberwaveError, match="(?i)limit"):
        RecordingManager(_FakeCatalogApi([])).list("env-1", limit=-1)


def test_manager_list_reports_probable_gateway_payload_limit() -> None:
    api = _FakeCatalogApi(_rest_items(1))
    error = ServiceException(status=500, reason="Internal Server Error", body="")
    error.headers = {
        "Server": "Google Frontend",
        "Content-Length": "0",
        "Content-Type": "text/html",
        "X-Cloud-Trace-Context": "trace-id/123;o=1",
    }
    api.src_app_api_environments_recordings_get_environment_recordings = MagicMock(
        side_effect=error
    )

    with pytest.raises(RecordingPayloadTooLargeError) as raised:
        RecordingManager(api).list(
            "env-1", start="2026-08-04", end="2026-08-11"
        )

    message = str(raised.value)
    assert "payload is too large" in message
    assert "shorter start/end interval" in message
    assert "limit" in message
    assert "trace-id/123" in message


def test_manager_list_does_not_relabel_an_ordinary_server_error() -> None:
    api = _FakeCatalogApi(_rest_items(1))
    error = ServiceException(status=500, reason="Internal Server Error", body="boom")
    error.headers = {"Server": "Google Frontend", "Content-Type": "text/html"}
    api.src_app_api_environments_recordings_get_environment_recordings = MagicMock(
        side_effect=error
    )

    with pytest.raises(CyberwaveError) as raised:
        RecordingManager(api).list(
            "env-1", start="2026-08-04", end="2026-08-11"
        )

    assert not isinstance(raised.value, RecordingPayloadTooLargeError)
    assert "payload is too large" not in str(raised.value)


def test_manager_list_with_explicit_dates_skips_the_availability_request() -> None:
    api = _FakeCatalogApi(_rest_items(1))

    RecordingManager(api).list("env-1", start="2026-07-01", end="2026-07-05")

    assert api.availability_calls == []
    assert api.list_calls[0]["start_date"] == date(2026, 7, 1)
    assert api.list_calls[0]["end_date"] == date(2026, 7, 5)


def test_manager_list_forwards_include_unready_to_availability() -> None:
    api = _FakeCatalogApi(_rest_items(1))

    RecordingManager(api).list("env-1", include_unready=True)

    assert api.availability_calls[0]["include_unready"] is True
    assert api.list_calls[0]["include_unready"] is True


def test_generated_client_exposes_the_bounded_catalog_contract() -> None:
    """Fail loudly when ``cyberwave/rest`` predates the bounded catalog.

    ``cyberwave/rest`` is generated from the backend's OpenAPI schema and is
    not committed, so a stale checkout is a normal state to be in. Every other
    test here stubs the transport, which means a stale client would otherwise
    surface as a confusing ``AttributeError``/``TypeError`` deep inside
    ``list()``. Assert the operations and models the manager depends on exist,
    with the parameters it actually sends.
    """
    import inspect

    from cyberwave.rest import DefaultApi
    from cyberwave.rest.models.recording_availability_day import (
        RecordingAvailabilityDay,
    )
    from cyberwave.rest.models.recording_availability_response import (
        RecordingAvailabilityResponse,
    )
    from cyberwave.rest.models.recording_list_response import RecordingListResponse

    catalog = inspect.signature(
        DefaultApi.src_app_api_environments_recordings_get_environment_recordings
    ).parameters
    assert {
        "start_date",
        "end_date",
        "include_unready",
        "limit",
        "cursor",
        "twin_uuid",
    } <= set(catalog)
    # The legacy aliases stay in the contract, so pre-0.7.0 direct REST callers
    # keep working against the same deployment.
    assert {"start_timestamp", "end_timestamp"} <= set(catalog)

    availability = inspect.signature(
        DefaultApi.src_app_api_environments_recordings_get_environment_recordings_availability
    ).parameters
    assert {"start_date", "end_date", "include_unready", "twin_uuid"} <= set(
        availability
    )
    # Availability is a whole-history question, so it is never paged — sending
    # limit/cursor would be a sign the manager is driving the wrong endpoint.
    assert "limit" not in availability and "cursor" not in availability

    # ``_latest_available_date`` reads ``last_date``; the pagination walk reads
    # ``next_cursor``/``has_more``.
    assert "last_date" in RecordingAvailabilityResponse.model_fields
    assert {"next_cursor", "has_more"} <= set(RecordingListResponse.model_fields)
    # The import above is the real assertion for this model; assert on a field so the
    # check cannot pass against a stub that merely has the right name.
    #
    # Match the WIRE name, not the attribute name. ``date`` is a reserved word for the
    # Python generator, so the property is emitted as ``var_date`` with ``alias="date"``
    # — the same rename ``RecordingGenerationRequestSchemaByDate`` already carries.
    # ``model_fields`` is keyed by the attribute, so asserting ``"date"`` against it
    # fails for a perfectly correct client.
    assert "date" in {
        field.alias or name
        for name, field in RecordingAvailabilityDay.model_fields.items()
    }


def test_manager_list_paginates_through_the_real_generated_client(monkeypatch) -> None:
    """Drive the actual generated REST client with only the transport stubbed.

    The hand-written stubs above accept whatever we send; this one runs the
    real ``@validate_call`` signatures and query serializer, so a renamed or
    retyped parameter (``limit``, ``cursor``, ``start_date``, ``twin_uuid``)
    fails here instead of silently dropping off the wire.
    """
    from cyberwave.rest import ApiClient, Configuration, DefaultApi
    from cyberwave.rest.models.recording_availability_response import (
        RecordingAvailabilityResponse,
    )
    from cyberwave.rest.models.recording_list_item import RecordingListItem as RestItem
    from cyberwave.rest.models.recording_list_response import RecordingListResponse

    api_client = ApiClient(Configuration(host="https://example.invalid"))
    requested_urls: list[str] = []
    rows = [
        RestItem(uuid=f"rec-{index:04d}", twin_uuid="twin-1", environment_uuid="env-1",
                 metadata={"recording_type": "camera"})
        for index in range(150)
    ]

    def fake_call_api(method, url, *_args, **_kwargs):
        requested_urls.append(url)
        return SimpleNamespace(read=lambda: None)

    def fake_deserialize(*, response_data, response_types_map):
        if "availability" in requested_urls[-1]:
            return SimpleNamespace(
                data=RecordingAvailabilityResponse(
                    first_date="2026-07-05", last_date="2026-07-05", days=[],
                    total_count=len(rows), facets={}, timezone="UTC",
                )
            )
        from urllib.parse import parse_qs, urlparse

        offset = int(parse_qs(urlparse(requested_urls[-1]).query).get("cursor", [0])[0])
        window = rows[offset : offset + 50]
        has_more = offset + len(window) < len(rows)
        return SimpleNamespace(
            data=RecordingListResponse(
                items=window,
                next_cursor=str(offset + len(window)) if has_more else None,
                has_more=has_more,
                page_size=len(window),
                facets={},
            )
        )

    monkeypatch.setattr(api_client, "call_api", fake_call_api)
    monkeypatch.setattr(api_client, "response_deserialize", fake_deserialize)

    items = RecordingManager(DefaultApi(api_client)).list(
        "env-1", limit=0, twin_uuids=["twin-1"]
    )

    assert len(items) == 150
    assert "recordings/availability" in requested_urls[0]
    assert "twin_uuid=twin-1" in requested_urls[0]
    assert "include_unready=false" in requested_urls[0]
    # The availability window reaches the catalog call as start_date/end_date —
    # not as the legacy start_timestamp/end_timestamp aliases.
    assert "start_timestamp" not in requested_urls[1]
    assert "start_date=2026-07-05" in requested_urls[1]
    assert "end_date=2026-07-05" in requested_urls[1]
    assert "include_unready=false" in requested_urls[1]
    assert "limit=50" in requested_urls[1]
    assert "twin_uuid=twin-1" in requested_urls[1]
    assert "cursor=" not in requested_urls[1]
    assert "cursor=50" in requested_urls[2]
    # Every narrowing survives onto page 2 as well: a cursor walk that dropped
    # the window or the readiness filter would silently widen mid-listing.
    assert "start_date=2026-07-05" in requested_urls[2]
    assert "end_date=2026-07-05" in requested_urls[2]
    assert "include_unready=false" in requested_urls[2]
    assert "twin_uuid=twin-1" in requested_urls[2]
    assert "limit=50" in requested_urls[2]


def test_twin_handle_list_narrows_both_requests_server_side() -> None:
    """Pagination makes client-side twin narrowing lossy: a page of 100
    environment-wide rows can contain none of this twin's. Push the filter down
    so both the availability window and the pages are already twin-scoped."""
    api = _FakeCatalogApi(_rest_items(2, twin="twin-1"))
    twin = SimpleNamespace(
        uuid="twin-1",
        environment_id="env-1",
        client=SimpleNamespace(
            environments=SimpleNamespace(recordings=RecordingManager(api))
        ),
    )

    TwinRecordingsHandle(twin).list()

    assert api.availability_calls[0]["twin_uuid"] == ["twin-1"]
    assert api.list_calls[0]["twin_uuid"] == ["twin-1"]


def test_manager_list_returns_wrapped_items() -> None:
    api = _api_listing(
        _rest_item("cam", "t1", {"recording_type": "camera"}),
        _rest_item("rob", "t2", {"metadata_type": "TwinRecordingMetadata"}),
    )
    items = RecordingManager(api).list("env-1")
    assert isinstance(items, RecordingList)
    assert [i.uuid for i in items] == ["cam", "rob"]
    call = api.src_app_api_environments_recordings_get_environment_recordings.call_args
    assert call.args[0] == "env-1"


def test_manager_list_applies_filter() -> None:
    api = _api_listing(
        _rest_item("cam", "t1", {"recording_type": "camera"}),
        _rest_item("rob", "t2", {"metadata_type": "TwinRecordingMetadata"}),
    )
    items = RecordingManager(api).list("env-1", filter=RecordingType.ROBOT)
    assert [i.uuid for i in items] == ["rob"]


def test_manager_types_attribute_exposes_enum() -> None:
    assert RecordingManager(MagicMock()).types.CAMERA == RecordingType.CAMERA


def test_twin_handle_list_narrows_to_twin_uuid() -> None:
    api = _api_listing(
        _rest_item("cam", "twin-1", {"recording_type": "camera"}),
        _rest_item("other", "twin-2", {"recording_type": "camera"}),
    )
    twin = SimpleNamespace(
        uuid="twin-1",
        environment_id="env-1",
        client=SimpleNamespace(
            environments=SimpleNamespace(recordings=RecordingManager(api))
        ),
    )
    items = TwinRecordingsHandle(twin).list()
    assert [i.uuid for i in items] == ["cam"]


def test_twin_handle_list_filter_after_narrowing() -> None:
    api = _api_listing(
        _rest_item("cam", "twin-1", {"recording_type": "camera"}),
        _rest_item("rob", "twin-1", {"metadata_type": "TwinRecordingMetadata"}),
    )
    twin = SimpleNamespace(
        uuid="twin-1",
        environment_id="env-1",
        client=SimpleNamespace(
            environments=SimpleNamespace(recordings=RecordingManager(api))
        ),
    )
    items = TwinRecordingsHandle(twin).list(filter=[RecordingType.ROBOT])
    assert [i.uuid for i in items] == ["rob"]


from pathlib import Path

from cyberwave.managers.recordings import Recording


def _envelope(twin_data: dict) -> SimpleNamespace:
    return SimpleNamespace(items=SimpleNamespace(twin_data=twin_data))


def test_get_downloads_all_sources_to_temp(monkeypatch) -> None:
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {
                "camera": {"signed_url": "https://x/a.mp4", "videos": [
                    {"signed_url": "https://x/a.mp4"}]},
                "actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]},
                "pointcloud": {"signed_urls": ["https://x/pc_000.fb"],
                               "timestamps": [1, 2]},
            }
        }
    )
    mgr = RecordingManager(api)

    downloaded: list[tuple[str, Path]] = []

    def fake_download(url: str, dest: Path, **_) -> None:
        dest.write_bytes(b"data")
        downloaded.append((url, dest))

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))

    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera", "pointcloud": {"twin-1": "pc"}},
    )
    rec = mgr.get(item)

    assert isinstance(rec, Recording)
    assert set(rec.local_paths) == {"camera", "actuation", "pointcloud"}
    assert all(p.exists() for paths in rec.local_paths.values() for p in paths)
    assert RecordingType.CAMERA in rec.types
    # ``pointcloud`` source (raw depth maps) classifies as DEPTH, not POINTCLOUD.
    assert RecordingType.DEPTH in rec.types
    call = api.src_app_api_environments_recordings_get_recording_data.call_args
    assert call.args == ("env-1", "rec-1")
    rec.close()
    assert not any(p.exists() for paths in rec.local_paths.values() for p in paths)


def test_get_does_not_download_camera_parquet(monkeypatch) -> None:
    """We intentionally do NOT surface a camera recording's dataset parquet as a
    downloadable artifact — only the mp4 video is collected for a camera entry."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {
                "camera": {
                    "videos": [{"signed_url": "https://x/a.mp4"}],
                    "parquet": {
                        "signed_url": "https://x/cam.parquet",
                        "format": "parquet",
                    },
                },
            }
        }
    )
    mgr = RecordingManager(api)

    def fake_download(url: str, dest: Path, **_) -> None:
        dest.write_bytes(b"data")

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera"},
    )
    rec = mgr.get(item)
    assert "camera_parquet" not in rec.local_paths
    assert set(rec.local_paths) == {"camera"}
    rec.close()


def test_manager_list_one_sided_date_filter_raises() -> None:
    api = _api_listing()
    with pytest.raises(CyberwaveError, match="(?i)both"):
        RecordingManager(api).list("env-1", start="2026-07-01")
    with pytest.raises(CyberwaveError, match="(?i)both"):
        RecordingManager(api).list("env-1", end="2026-07-05")
    # Neither bound is fine (no filtering).
    RecordingManager(api).list("env-1")
    api.src_app_api_environments_recordings_get_environment_recordings.assert_called()


def test_get_path_filter_restricts_download(monkeypatch) -> None:
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {
            "camera": {"videos": [{"signed_url": "https://x/a.mp4"}]},
            "actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]},
        }}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera"},
    )
    rec = RecordingManager(api).get(item, path="j.parquet")
    assert set(rec.local_paths) == {"actuation"}
    rec.close()


def test_get_by_uuid_requires_environment_id() -> None:
    with pytest.raises(Exception):
        RecordingManager(MagicMock()).get("rec-1")


def test_get_cleans_up_tempdir_when_a_download_fails(monkeypatch) -> None:
    """A mid-loop download failure must not orphan the temp dir: no Recording is
    returned to ``close()`` it, so ``get`` reaps it before re-raising."""
    import tempfile
    from pathlib import Path

    from cyberwave.exceptions import CyberwaveError

    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {
                "camera": {"videos": [
                    {"signed_url": "https://x/a.mp4"},
                    {"signed_url": "https://x/b.mp4"},
                ]},
            }
        }
    )

    before = {
        p for p in Path(tempfile.gettempdir()).glob("cw-recording-*")
    }

    def flaky_download(url: str, dest: Path, **_) -> None:
        if url.endswith("a.mp4"):
            dest.write_bytes(b"partial")  # first file lands...
            return
        raise CyberwaveError("boom")  # ...second fails mid-loop

    monkeypatch.setattr(
        RecordingManager, "_download", staticmethod(flaky_download)
    )

    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera"},
    )
    with pytest.raises(CyberwaveError):
        RecordingManager(api).get(item)

    after = {p for p in Path(tempfile.gettempdir()).glob("cw-recording-*")}
    assert after == before  # no new cw-recording-* dir left behind


def _camera_parts(*names: str) -> MagicMock:
    """An API whose envelope carries one camera stream of ``names`` video parts."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"camera": {"videos": [{"signed_url": f"https://x/{n}"} for n in names]}}}
    )
    return api


def _camera_item() -> RecordingListItem:
    return RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera"},
    )


def test_get_downloads_parts_concurrently(monkeypatch) -> None:
    """A segmented recording is fetched in parallel: every part must be in flight
    at once, otherwise each pays its own connect/first-byte round trip serially."""
    import threading

    api = _camera_parts("a.mp4", "b.mp4", "c.mp4", "d.mp4")
    # A barrier proves concurrency exactly rather than inferring it from timing:
    # if the downloads were serialized, the first worker would never be released
    # and this raises BrokenBarrierError instead of passing on a lucky schedule.
    barrier = threading.Barrier(4, timeout=10)

    def fake_download(url: str, dest: Path, **_) -> None:
        barrier.wait()
        dest.write_bytes(b"data")

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    rec = RecordingManager(api).get(_camera_item())
    assert len(rec.local_paths["camera"]) == 4
    rec.close()


def test_get_max_workers_one_downloads_serially(monkeypatch) -> None:
    """``max_workers=1`` is the escape hatch back to the pre-parallel behavior."""
    import threading
    import time

    api = _camera_parts("a.mp4", "b.mp4", "c.mp4")
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def fake_download(url: str, dest: Path, **_) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.02)
        dest.write_bytes(b"data")
        with lock:
            state["active"] -= 1

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    rec = RecordingManager(api).get(_camera_item(), max_workers=1)
    assert state["peak"] == 1
    assert len(rec.local_paths["camera"]) == 3
    rec.close()


def test_get_local_paths_order_is_independent_of_completion_order(monkeypatch) -> None:
    """``local_paths`` follows envelope order, not whichever download finished
    first. ``_paths_with_ext`` does not sort, so completion order would otherwise
    decide which part ``show_video()`` plays."""
    import time

    names = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    api = _camera_parts(*names)

    def fake_download(url: str, dest: Path, **_) -> None:
        # Finish in reverse: the last part lands first.
        time.sleep(0.02 * (len(names) - names.index(url.rsplit("/", 1)[-1])))
        dest.write_bytes(b"data")

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    rec = RecordingManager(api).get(_camera_item())
    assert [p.name for p in rec.local_paths["camera"]] == [
        "000_a.mp4", "001_b.mp4", "002_c.mp4", "003_d.mp4"
    ]
    rec.close()


def test_get_cancels_queued_downloads_when_one_fails(monkeypatch) -> None:
    """One part failing abandons the whole fetch, so parts still queued behind it
    must be dropped rather than downloaded into a temp dir about to be reaped.

    It also pins the ordering of cleanup: ``get()`` returns only once the running
    workers have settled, so the rmtree never races a download mid-write.
    """
    import tempfile
    import threading

    from cyberwave.exceptions import CyberwaveError

    parts = [f"p{i:02d}.mp4" for i in range(20)]
    api = _camera_parts(*parts)
    release = threading.Event()
    attempted: list[str] = []
    finished: list[str] = []
    lock = threading.Lock()

    def fake_download(url: str, dest: Path, **_) -> None:
        with lock:
            attempted.append(url)
        if url.endswith("p00.mp4"):
            raise CyberwaveError("boom")
        release.wait(timeout=1.0)  # still in flight when the failure surfaces
        with lock:
            finished.append(url)
        if dest.parent.exists():
            dest.write_bytes(b"data")

    before = {p for p in Path(tempfile.gettempdir()).glob("cw-recording-*")}
    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    with pytest.raises(CyberwaveError, match="boom"):
        RecordingManager(api).get(_camera_item(), max_workers=2)

    # With 2 workers and 20 parts, the failure surfaces while 18 are still queued;
    # those must never be requested. Serial code would have attempted only 1, so
    # the lower bound also proves the parallel path ran.
    assert 2 <= len(attempted) <= 5, attempted
    # Everything that did start has finished by the time get() returns.
    assert len(finished) == len(attempted) - 1
    assert {p for p in Path(tempfile.gettempdir()).glob("cw-recording-*")} == before


def test_get_grows_http_pool_to_match_requested_concurrency(monkeypatch) -> None:
    """Raising ``max_workers`` past the default must grow the shared pool. Left at
    the default size, urllib3 opens the surplus connections and discards them on
    return, so the artifacts past the pool size re-pay a handshake each and the
    extra workers buy far less than they should."""
    from cyberwave.managers import recordings as rec_mod

    sizes: list[int | None] = []

    def fake_pool_manager(**kwargs) -> MagicMock:
        sizes.append(kwargs.get("maxsize"))
        return MagicMock()

    monkeypatch.setattr(rec_mod, "_http_pool", None, raising=False)
    monkeypatch.setattr(rec_mod, "_http_pool_maxsize", 0, raising=False)
    monkeypatch.setattr(rec_mod.urllib3, "PoolManager", fake_pool_manager)
    # _download is stubbed, so the pool can only be sized by get() itself.
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )

    api = _camera_parts(*[f"p{i:02d}.mp4" for i in range(20)])
    requested = rec_mod.DEFAULT_DOWNLOAD_WORKERS * 2
    rec = RecordingManager(api).get(_camera_item(), max_workers=requested)
    assert sizes, "get() never sized the shared pool"
    assert max(s or 0 for s in sizes) >= requested
    rec.close()


def test_video_parts_arrive_in_timeline_order_not_envelope_order(monkeypatch) -> None:
    """Envelope order is not a contract: the finalized server path sorts video
    chunks by ``chunk_index``, but the progressive path emits the manifest's
    ``video_parts`` in stored order and a gap-fill can append a part belonging
    earlier. The receiving end must still see segments in timeline order —
    ``_paths_with_ext`` does not sort, so ``show_video()`` plays local_paths[0]."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"camera": {"videos": [
            # Deliberately shuffled, as a gap-fill append would leave them.
            {"signed_url": "https://x/c.mp4", "first_timestamp_us": 3_000_000,
             "chunk_index": 2},
            {"signed_url": "https://x/a.mp4", "first_timestamp_us": 1_000_000,
             "chunk_index": 0},
            {"signed_url": "https://x/d.mp4", "first_timestamp_us": 4_000_000,
             "chunk_index": 3},
            {"signed_url": "https://x/b.mp4", "first_timestamp_us": 2_000_000,
             "chunk_index": 1},
        ]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get(_camera_item())
    assert [p.name for p in rec.local_paths["camera"]] == [
        "000_a.mp4", "001_b.mp4", "002_c.mp4", "003_d.mp4"
    ]
    assert rec._paths_with_ext(".mp4")[0].name == "000_a.mp4"  # show_video() plays this
    rec.close()


def test_zero_filled_timestamps_fall_through_to_the_index(monkeypatch) -> None:
    """A timestamp tier that answers for every part but orders none of them must
    not claim the sort.

    The server zero-fills a missing window twice over — once when it records the
    part (``int(start_timestamp_us or 0)``) and again when it builds the envelope
    (``.get("start_timestamp_us", 0)``) — so "every part answered" is satisfied by
    a column of zeros. Accepting that tier makes ``chunk_index``/``part_index``
    unreachable on exactly the envelope this ordering exists to defend, and leaves
    the parts in whatever order they arrived.
    """
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"camera": {"videos": [
            {"signed_url": "https://x/c.mp4", "first_timestamp_us": 0, "chunk_index": 2},
            {"signed_url": "https://x/a.mp4", "first_timestamp_us": 0, "chunk_index": 0},
            {"signed_url": "https://x/b.mp4", "first_timestamp_us": 0, "chunk_index": 1},
        ]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get(_camera_item())
    assert [p.name for p in rec.local_paths["camera"]] == [
        "000_a.mp4", "001_b.mp4", "002_c.mp4"
    ]
    rec.close()


def test_robot_parts_ordered_by_part_index_when_no_timestamps(monkeypatch) -> None:
    """Actuation parts label their window ``start_timestamp_us`` (not
    ``first_timestamp_us``); with neither present, ``part_index`` orders them.
    read_robot() concatenates in filename order with no timestamp re-sort, so a
    wrong order here silently interleaves rows against time."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"actuation": {"parts": [
            {"signed_url": "https://x/p2.parquet", "part_index": 2},
            {"signed_url": "https://x/p0.parquet", "part_index": 0},
            {"signed_url": "https://x/p1.parquet", "part_index": 1},
        ]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get("rec-1", environment_id="env-1")
    assert [p.name for p in rec.local_paths["actuation"]] == [
        "000_p0.parquet", "001_p1.parquet", "002_p2.parquet"
    ]
    rec.close()


def test_parts_keep_arrival_order_when_ordering_keys_are_incomplete(monkeypatch) -> None:
    """Partial ordering information must not reorder anything: interleaving parts
    that carry a timestamp with parts that do not would be worse than trusting the
    order the server sent."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"camera": {"videos": [
            {"signed_url": "https://x/first.mp4"},                          # no keys
            {"signed_url": "https://x/second.mp4", "chunk_index": 0},       # keyed
        ]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get(_camera_item())
    assert [p.name for p in rec.local_paths["camera"]] == [
        "000_first.mp4", "001_second.mp4"
    ]
    rec.close()


def test_filename_index_pad_keeps_lexical_order_past_999_parts(monkeypatch) -> None:
    """read_robot()/read_depth() sort parquet parts by FILENAME, so the index pad
    has to be wide enough for the count: at three digits, part 1000 sorts before
    part 999 and the concatenated rows jump backwards in time."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"actuation": {"parts": [
            {"signed_url": f"https://x/p{i:04d}.parquet", "part_index": i}
            for i in range(1001)
        ]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get("rec-1", environment_id="env-1", max_workers=4)
    paths = rec.local_paths["actuation"]
    assert len(paths) == 1001
    # What the readers actually do: sort by name. It must equal fetch order.
    assert sorted(str(p) for p in paths) == [str(p) for p in paths]
    assert paths[0].name.startswith("0000_") and paths[-1].name.startswith("1000_")
    rec.close()


def test_growing_the_http_pool_closes_the_one_it_replaces(monkeypatch) -> None:
    """Growing the pool must close the pool it supersedes.

    The replaced PoolManager still holds established keep-alive sockets. Dropping
    the reference without closing leaks them until GC — the exact leak the shared
    pool exists to prevent — and costs the next fetch a handshake for every
    connection lost.
    """
    from unittest.mock import MagicMock

    from cyberwave.managers import recordings as rec_mod

    created: list[MagicMock] = []

    def fake_pool_manager(**_kwargs) -> MagicMock:
        pool = MagicMock()
        created.append(pool)
        return pool

    monkeypatch.setattr(rec_mod, "_http_pool", None, raising=False)
    monkeypatch.setattr(rec_mod, "_http_pool_maxsize", 0, raising=False)
    monkeypatch.setattr(rec_mod.urllib3, "PoolManager", fake_pool_manager)

    first = rec_mod._get_http_pool(8)
    first.clear.assert_not_called()  # nothing to replace yet
    second = rec_mod._get_http_pool(16)

    assert second is not first, "a larger request must grow the pool"
    first.clear.assert_called_once()
    # The pool that is still in use is never closed, and a request that fits in
    # the current size does not rebuild at all.
    second.clear.assert_not_called()
    assert rec_mod._get_http_pool(4) is second
    assert len(created) == 2


def test_get_http_pool_builds_one_pool_under_concurrent_callers(monkeypatch) -> None:
    """Concurrent callers must not each build a pool. The factory sleeps so the
    race is certain rather than lucky: unguarded, every thread passes the resize
    check and the losers' pools are orphaned along with their connections."""
    import threading
    import time
    from unittest.mock import MagicMock

    from cyberwave.managers import recordings as rec_mod

    created: list[MagicMock] = []

    def slow_pool_manager(**_kwargs) -> MagicMock:
        time.sleep(0.05)  # widen the window between the check and the assignment
        pool = MagicMock()
        created.append(pool)
        return pool

    monkeypatch.setattr(rec_mod, "_http_pool", None, raising=False)
    monkeypatch.setattr(rec_mod, "_http_pool_maxsize", 0, raising=False)
    monkeypatch.setattr(rec_mod.urllib3, "PoolManager", slow_pool_manager)

    got: list[object] = []
    threads = [
        threading.Thread(target=lambda: got.append(rec_mod._get_http_pool(8)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == 1, f"{len(created)} pools built for 8 concurrent callers"
    assert {id(p) for p in got} == {id(created[0])}


def test_get_clamps_max_workers_to_the_ceiling(monkeypatch) -> None:
    """``len(sources)`` is no bound — a long recording holds hundreds of parts, so
    an unclamped request would spawn a thread and a socket for every one."""
    import threading

    from cyberwave.managers import recordings as rec_mod

    api = _camera_parts(*[f"p{i:03d}.mp4" for i in range(120)])
    lock = threading.Lock()
    # Count this executor's own workers by name, not threading.active_count():
    # a global count picks up every other thread the test session is running.
    # ThreadPoolExecutor names and reuses its threads, so distinct names is the
    # number of workers it actually created.
    workers_seen: set[str] = set()

    def fake_download(url: str, dest: Path, **_) -> None:
        with lock:
            workers_seen.add(threading.current_thread().name)
        dest.write_bytes(b"d")

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))
    rec = RecordingManager(api).get(_camera_item(), max_workers=120)
    assert workers_seen, "no worker threads recorded"
    assert all(n.startswith("cw-artifact") for n in workers_seen), workers_seen
    assert len(workers_seen) <= rec_mod.MAX_DOWNLOAD_WORKERS, len(workers_seen)
    assert len(rec.local_paths["camera"]) == 120  # all parts still fetched
    rec.close()


def test_download_abandons_stream_when_cancelled(monkeypatch, tmp_path) -> None:
    """A failure elsewhere in the fetch must stop sibling transfers at the next
    chunk. urllib3's read timeout cannot do this: it measures inactivity and
    resets on every chunk, so a healthy transfer would run to completion into a
    temp dir that is about to be deleted."""
    import threading
    from unittest.mock import MagicMock

    from cyberwave.exceptions import CyberwaveError
    from cyberwave.managers import recordings as rec_mod

    cancel = threading.Event()
    chunks_read = {"n": 0}

    def chunks(_size):
        # Bounded on purpose: an unbounded stream would hang forever if the
        # cancel check regressed, instead of failing the assertion below.
        for _ in range(50):
            chunks_read["n"] += 1
            if chunks_read["n"] == 3:
                cancel.set()  # a sibling download fails on the third chunk
            yield b"x" * 16

    pool = MagicMock()
    response = MagicMock()
    response.status = 200
    response.stream = chunks
    pool.request.return_value = response
    monkeypatch.setattr(rec_mod, "_http_pool", pool, raising=False)
    monkeypatch.setattr(rec_mod, "_http_pool_maxsize", 99, raising=False)

    with pytest.raises(CyberwaveError, match="cancelled mid-stream"):
        RecordingManager._download(
            "https://x/big.parquet", tmp_path / "big.parquet", cancel=cancel
        )
    assert chunks_read["n"] == 3  # stopped at the boundary, did not drain
    response.release_conn.assert_called_once()


def test_download_skips_entirely_when_already_cancelled(monkeypatch, tmp_path) -> None:
    """A worker that has not started yet must not open a connection at all."""
    from unittest.mock import MagicMock

    import threading

    from cyberwave.exceptions import CyberwaveError
    from cyberwave.managers import recordings as rec_mod

    pool = MagicMock()
    monkeypatch.setattr(rec_mod, "_http_pool", pool, raising=False)
    monkeypatch.setattr(rec_mod, "_http_pool_maxsize", 99, raising=False)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(CyberwaveError, match="abandoned before starting"):
        RecordingManager._download("https://x/a.mp4", tmp_path / "a.mp4", cancel=cancel)
    pool.request.assert_not_called()
    assert not (tmp_path / "a.mp4").exists()


def test_get_rejects_non_positive_max_workers() -> None:
    """``0`` reads as "no concurrency" but would fall through to the default and
    start the full worker set, so it is rejected rather than coerced."""
    from cyberwave.exceptions import CyberwaveError

    api = _camera_parts("a.mp4")
    for bad in (0, -1):
        with pytest.raises(CyberwaveError, match="max_workers"):
            RecordingManager(api).get(_camera_item(), max_workers=bad)


def test_get_path_filter_keeps_unfiltered_source_indexes(monkeypatch) -> None:
    """Filenames carry the index over ALL sources, so a filtered fetch names its
    artifacts identically to an unfiltered one (and caches keyed on those names
    stay valid across filters)."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {
            "camera": {"videos": [{"signed_url": "https://x/a.mp4"}]},
            "actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]},
        }}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    rec = RecordingManager(api).get(_camera_item(), path=".parquet")
    assert [p.name for p in rec.local_paths["actuation"]] == ["001_j.parquet"]
    rec.close()


def test_twin_handle_get_uses_twin_environment(monkeypatch) -> None:
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    mgr = RecordingManager(api)
    twin = SimpleNamespace(
        uuid="twin-1", environment_id="env-9",
        client=SimpleNamespace(environments=SimpleNamespace(recordings=mgr)),
    )
    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"metadata_type": "TwinRecordingMetadata"},
    )
    TwinRecordingsHandle(twin).get(item).close()
    assert api.src_app_api_environments_recordings_get_recording_data.call_args.args == (
        "env-9", "rec-1",
    )


def test_twin_handle_get_excludes_other_twins_artifacts(monkeypatch) -> None:
    """A multi-twin recording (e.g. a shared multi-robot parquet) must not leak
    another twin's files into a twin-scoped ``get()``."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {"actuation": {"parts": [{"signed_url": "https://x/a.parquet"}]}},
            "twin-2": {"actuation": {"parts": [{"signed_url": "https://x/b.parquet"}]}},
        }
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        RecordingManager,
        "_download",
        staticmethod(
            lambda url, dest, **_: (downloaded.append(url), dest.write_bytes(b"d"))
        ),
    )
    mgr = RecordingManager(api)
    twin = SimpleNamespace(
        uuid="twin-1", environment_id="env-9",
        client=SimpleNamespace(environments=SimpleNamespace(recordings=mgr)),
    )
    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"metadata_type": "TwinRecordingMetadata"},
    )
    rec = TwinRecordingsHandle(twin).get(item)
    assert downloaded == ["https://x/a.parquet"]
    rec.close()


def test_manager_get_without_twin_uuid_includes_all_twins(monkeypatch) -> None:
    """Environment-scoped fetches (no twin_uuid) still see every twin's artifacts."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {"actuation": {"parts": [{"signed_url": "https://x/a.parquet"}]}},
            "twin-2": {"actuation": {"parts": [{"signed_url": "https://x/b.parquet"}]}},
        }
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        RecordingManager,
        "_download",
        staticmethod(
            lambda url, dest, **_: (downloaded.append(url), dest.write_bytes(b"d"))
        ),
    )
    rec = RecordingManager(api).get("rec-1", environment_id="env-1")
    assert set(downloaded) == {"https://x/a.parquet", "https://x/b.parquet"}
    rec.close()


import importlib


def _recording(local_paths: dict, signed_urls=None, types=frozenset()) -> Recording:
    import tempfile
    return Recording(
        uuid="r", twin_uuid="t", environment_uuid="e",
        types=types, signed_urls=signed_urls,
        local_paths=local_paths, tempdir=tempfile.mkdtemp(prefix="cw-test-"),
    )


def _robot_envelope() -> SimpleNamespace:
    return SimpleNamespace(
        items=SimpleNamespace(twin_data={"t": {"actuation": {"parts": []}}})
    )


def test_read_robot_missing_dep_raises_install_hint(monkeypatch) -> None:
    real_import = importlib.import_module

    def fake_import(name, *a, **k):
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    rec = _recording(
        {"actuation": [Path("/tmp/x.parquet")]}, signed_urls=_robot_envelope()
    )
    with pytest.raises(CyberwaveError, match="cyberwave\\[data\\]"):
        rec.read_robot()
    rec.close()


def test_read_robot_reads_actuation_table() -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rec = _recording({}, signed_urls=_robot_envelope())
    parquet_path = Path(rec._tempdir) / "j.parquet"
    pq.write_table(pa.table({"action": [1, 2, 3]}), parquet_path)
    rec.local_paths = {"actuation": [parquet_path]}

    table = rec.read_robot()
    assert table.num_rows == 3
    assert "action" in table.column_names
    rec.close()


def test_read_robot_absent_without_actuation_stream() -> None:
    """read_robot is a contextual accessor: a camera-only recording must not
    expose it, and never touches point-cloud parquets."""
    rec = _recording({"camera": [Path("/tmp/a.mp4")]}, signed_urls=_envelope({"t": {}}))
    assert not hasattr(rec, "read_robot")
    with pytest.raises(AttributeError):
        rec.read_robot()
    rec.close()


def test_read_robot_concatenates_all_parts() -> None:
    """A segmented recording downloads several actuation parquet parts; read_robot
    joins them all (schema shared -> joint column order preserved) instead of
    silently returning only the first part."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rec = _recording({}, signed_urls=_robot_envelope())
    p0 = Path(rec._tempdir) / "part_000.parquet"
    p1 = Path(rec._tempdir) / "part_001.parquet"
    pq.write_table(pa.table({"action": [1, 2]}), p0)
    pq.write_table(pa.table({"action": [3, 4, 5]}), p1)
    rec.local_paths = {"actuation": [p0, p1]}

    table = rec.read_robot()
    assert table.num_rows == 5  # 2 + 3, not just the first part
    assert table.column("action").to_pylist() == [1, 2, 3, 4, 5]
    rec.close()


def test_read_robot_ignores_pointcloud_parquet() -> None:
    """read_robot reads ONLY the actuation parquet, never the (incompatible)
    point-cloud parquet."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rec = _recording({}, signed_urls=_robot_envelope())
    act = Path(rec._tempdir) / "actuation.parquet"
    pc = Path(rec._tempdir) / "pointcloud.parquet"
    pq.write_table(pa.table({"action": [1, 2, 3]}), act)
    pq.write_table(pa.table({"timestamp_us": [10], "rows": [1]}), pc)
    rec.local_paths = {"actuation": [act], "pointcloud": [pc]}

    table = rec.read_robot()
    assert table.column_names == ["action"]
    rec.close()


def test_info_summarizes_recording_and_accessors() -> None:
    envelope = SimpleNamespace(
        items=SimpleNamespace(
            twin_data={"t": {"actuation": {"parts": []}, "camera": {"videos": []}}}
        )
    )
    rec = _recording(
        {"camera": [Path("/tmp/a.mp4")]},
        signed_urls=envelope,
        types=frozenset({RecordingType.ROBOT, RecordingType.CAMERA}),
    )
    info = rec.info()
    assert info["uuid"] == "r"
    assert set(info["types"]) == {"robot", "camera"}
    assert "read_robot" in info["accessors"]
    assert "show_video" in info["accessors"]
    rec.close()


def test_show_video_dispatches(monkeypatch) -> None:
    called = {}
    monkeypatch.setattr(
        Recording, "_show_video",
        lambda self, path: called.setdefault("video", path),
    )
    rec = _recording(
        {"camera": [Path("/tmp/a.mp4")]}, types=frozenset({RecordingType.CAMERA})
    )
    rec.show_video()
    assert called["video"].name == "a.mp4"
    rec.close()


from datetime import date, datetime, timedelta, timezone

from cyberwave.managers.recordings import _parse_date_filter


def test_parse_date_filter_none_stays_none() -> None:
    assert _parse_date_filter(None) is None


def test_parse_date_filter_passes_date_through() -> None:
    d = date(2026, 7, 1)
    assert _parse_date_filter(d) == d


def test_parse_date_filter_datetime_drops_time_of_day() -> None:
    dt = datetime(2026, 7, 1, 10, 30, 0)
    assert _parse_date_filter(dt) == date(2026, 7, 1)


def test_parse_date_filter_iso_date_string() -> None:
    assert _parse_date_filter("2026-07-01") == date(2026, 7, 1)


def test_parse_date_filter_iso_datetime_string() -> None:
    assert _parse_date_filter("2026-07-01T10:30:00") == date(2026, 7, 1)


def test_parse_date_filter_iso_datetime_string_with_z_suffix() -> None:
    assert _parse_date_filter("2026-07-01T10:30:00Z") == date(2026, 7, 1)


def test_parse_date_filter_iso_datetime_string_with_offset() -> None:
    # The backend filters by UTC calendar day, so an offset datetime must be
    # normalized to UTC before dropping the time-of-day. 02:00+05:00 is
    # 21:00Z on the PREVIOUS day.
    assert _parse_date_filter("2026-07-01T02:00:00+05:00") == date(2026, 6, 30)


def test_parse_date_filter_aware_datetime_normalizes_to_utc_day() -> None:
    dt = datetime(2026, 7, 1, 2, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    assert _parse_date_filter(dt) == date(2026, 6, 30)


def test_parse_date_filter_invalid_string_raises() -> None:
    with pytest.raises(CyberwaveError, match="Invalid ISO"):
        _parse_date_filter("not-a-date")


def test_parse_date_filter_unsupported_type_raises() -> None:
    with pytest.raises(CyberwaveError, match="Unsupported type"):
        _parse_date_filter(12345)  # type: ignore[arg-type]


def test_manager_list_forwards_parsed_dates_to_rest_call() -> None:
    api = _api_listing(_rest_item("cam", "t1", {"recording_type": "camera"}))
    RecordingManager(api).list(
        "env-1", start="2026-07-01", end=datetime(2026, 7, 5, 12, 0, 0)
    )
    call = api.src_app_api_environments_recordings_get_environment_recordings.call_args
    assert call.args == ("env-1",)
    assert call.kwargs["start_date"] == date(2026, 7, 1)
    assert call.kwargs["end_date"] == date(2026, 7, 5)


def test_twin_handle_list_forwards_parsed_dates() -> None:
    api = _api_listing(_rest_item("cam", "twin-1", {"recording_type": "camera"}))
    twin = SimpleNamespace(
        uuid="twin-1",
        environment_id="env-1",
        client=SimpleNamespace(
            environments=SimpleNamespace(recordings=RecordingManager(api))
        ),
    )
    TwinRecordingsHandle(twin).list(start="2026-07-01T00:00:00Z", end="2026-07-05")
    call = api.src_app_api_environments_recordings_get_environment_recordings.call_args
    assert call.args == ("env-1",)
    assert call.kwargs["start_date"] == date(2026, 7, 1)
    assert call.kwargs["end_date"] == date(2026, 7, 5)
    assert call.kwargs["twin_uuid"] == ["twin-1"]


def test_recording_list_item_get_uses_attached_manager(monkeypatch) -> None:
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(
            items=[_rest_item("cam", "t1", {"recording_type": "camera"})]
        )
    )
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"t1": {"actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    items = RecordingManager(api).list("env-1")
    rec = items[0].get()
    assert isinstance(rec, Recording)
    call = api.src_app_api_environments_recordings_get_recording_data.call_args
    assert call.args == ("env-1", "cam")
    rec.close()


def test_recording_list_item_get_without_manager_raises() -> None:
    item = _item("r1", "t1", {"recording_type": "camera"})
    with pytest.raises(CyberwaveError, match="not attached to a manager"):
        item.get()


def test_recording_list_item_get_reads_robot_and_cleans_up(monkeypatch) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(
            items=[_rest_item("rob", "t1", {"metadata_type": "TwinRecordingMetadata"})]
        )
    )
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"t1": {"actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]}}}
    )

    written_dirs: list[Path] = []

    def fake_download(url: str, dest: Path, **_) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"action": [1, 2]}), dest)
        written_dirs.append(dest.parent)

    monkeypatch.setattr(RecordingManager, "_download", staticmethod(fake_download))

    items = RecordingManager(api).list("env-1")
    with items[0].get() as rec:
        table = rec.read_robot()
        assert table.num_rows == 2
    assert not written_dirs[0].exists()  # temp dir reaped on context exit


def test_recording_list_item_get_shows_video_and_cleans_up(monkeypatch) -> None:
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(
            items=[_rest_item("cam", "t1", {"recording_type": "camera"})]
        )
    )
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"t1": {"camera": {"videos": [{"signed_url": "https://x/a.mp4"}]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    called = {}
    monkeypatch.setattr(
        Recording, "_show_video",
        lambda self, path: called.setdefault("path", path),
    )

    items = RecordingManager(api).list("env-1")
    with items[0].get() as rec:
        rec.show_video()
    assert called["path"].name.endswith("a.mp4")
    assert not called["path"].parent.exists()


def test_twin_handle_list_items_get_use_twin_environment(monkeypatch) -> None:
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(
            items=[_rest_item("rob", "twin-1", {"metadata_type": "TwinRecordingMetadata"})]
        )
    )
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {"twin-1": {"actuation": {"parts": [{"signed_url": "https://x/j.parquet"}]}}}
    )
    monkeypatch.setattr(
        RecordingManager, "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    mgr = RecordingManager(api)
    twin = SimpleNamespace(
        uuid="twin-1", environment_id="env-9",
        client=SimpleNamespace(environments=SimpleNamespace(recordings=mgr)),
    )
    items = TwinRecordingsHandle(twin).list()
    items[0].get().close()
    call = api.src_app_api_environments_recordings_get_recording_data.call_args
    assert call.args == ("env-9", "rob")


def test_recording_tempdir_reaped_when_reference_dropped() -> None:
    """A one-liner like ``twin.recordings.get(item).read_robot()`` never calls
    close(); the weakref.finalize must still reap the temp dir on gc."""
    import gc
    from pathlib import Path

    rec = _recording({})
    tempdir = rec._tempdir
    assert Path(tempdir).exists()
    del rec
    gc.collect()
    assert not Path(tempdir).exists()


def test_recording_close_is_idempotent() -> None:
    from pathlib import Path

    rec = _recording({})
    tempdir = rec._tempdir
    rec.close()
    assert not Path(tempdir).exists()
    rec.close()  # second call must not raise


def test_recording_list_item_get_forwards_twin_uuid(monkeypatch) -> None:
    """An item from twin.recordings.list() carries its twin_uuid; item.get()
    must forward it so a shared multi-twin recording only downloads THIS twin's
    artifacts."""
    api = _stub_availability(MagicMock())
    api.src_app_api_environments_recordings_get_environment_recordings.return_value = (
        SimpleNamespace(
            items=[_rest_item("rec", "twin-1", {"metadata_type": "TwinRecordingMetadata"})]
        )
    )
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {"actuation": {"parts": [{"signed_url": "https://x/a.parquet"}]}},
            "twin-2": {"actuation": {"parts": [{"signed_url": "https://x/b.parquet"}]}},
        }
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        RecordingManager,
        "_download",
        staticmethod(
            lambda url, dest, **_: (downloaded.append(url), dest.write_bytes(b"d"))
        ),
    )
    items = RecordingManager(api).list("env-1")
    rec = items[0].get()
    assert downloaded == ["https://x/a.parquet"]
    rec.close()


def test_twin_handle_get_hides_other_twin_accessor(monkeypatch) -> None:
    """A multi-twin envelope where only twin-2 has actuation must not surface
    read_robot on a twin-1 handle fetch (predicates and downloads must agree)."""
    api = MagicMock()
    api.src_app_api_environments_recordings_get_recording_data.return_value = _envelope(
        {
            "twin-1": {"camera": {"videos": [{"signed_url": "https://x/a.mp4"}]}},
            "twin-2": {"actuation": {"parts": [{"signed_url": "https://x/b.parquet"}]}},
        }
    )
    monkeypatch.setattr(
        RecordingManager,
        "_download",
        staticmethod(lambda url, dest, **_: dest.write_bytes(b"d")),
    )
    mgr = RecordingManager(api)
    twin = SimpleNamespace(
        uuid="twin-1",
        environment_id="env-9",
        client=SimpleNamespace(environments=SimpleNamespace(recordings=mgr)),
    )
    item = RecordingListItem(
        uuid="rec-1", twin_uuid="twin-1", environment_uuid="env-1",
        metadata={"recording_type": "camera"},
    )
    rec = TwinRecordingsHandle(twin).get(item)
    assert not hasattr(rec, "read_robot")  # twin-2's stream never surfaces here
    assert hasattr(rec, "show_video")  # twin-1's own camera stream does
    rec.close()


def test_download_reuses_shared_http_pool(monkeypatch, tmp_path) -> None:
    """#6: _download must reuse one process-wide PoolManager rather than creating
    (and leaking) a fresh keep-alive pool per artifact, and must release the
    connection back to the pool after streaming."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from cyberwave.managers import recordings as rec_mod
    from cyberwave.managers.recordings import RecordingManager

    created: list[MagicMock] = []
    responses: list[MagicMock] = []
    pool_kwargs: list[dict] = []

    def fake_pool_factory(**kwargs) -> MagicMock:
        pool_kwargs.append(kwargs)
        pool = MagicMock(name=f"pool-{len(created)}")

        def request(*_a, **_k):
            resp = MagicMock()
            resp.status = 200
            resp.stream = lambda _n: iter([b"chunk"])
            responses.append(resp)
            return resp

        pool.request.side_effect = request
        created.append(pool)
        return pool

    # Reset the lazily-created module-level pool so this test controls it.
    monkeypatch.setattr(rec_mod, "_http_pool", None, raising=False)
    monkeypatch.setattr(rec_mod.urllib3, "PoolManager", fake_pool_factory)

    for i in range(3):
        RecordingManager._download(f"https://x/{i}.bin", Path(tmp_path) / f"{i}.bin")

    # One pool created for all three downloads (no per-artifact pool leak) ...
    assert len(created) == 1
    assert created[0].request.call_count == 3
    # ... sized to what a bare _download actually needs, which is one connection
    # it reuses. Sizing the pool is get()'s job because only get() knows the
    # fetch's concurrency; asking for the default here would grow the pool back to
    # 8 on the first chunk of any smaller fetch, throwing away the pool get() had
    # just built for it.
    assert pool_kwargs[0].get("maxsize") == 1
    # ... and every response released its connection back to the shared pool.
    assert len(responses) == 3
    for resp in responses:
        resp.release_conn.assert_called_once()


def test_is_final_true_for_finalized_row_without_flag() -> None:
    assert _item("r1", "t1", {"recording_type": "camera"}).is_final is True


def test_is_final_false_for_active_manifest() -> None:
    item = _item(
        "r1", "t1", {"recording_type": "active", "video_parts": [{"chunk_index": 0}]}
    )
    assert item.is_final is False


def test_is_final_false_when_flag_explicitly_false() -> None:
    item = _item("r1", "t1", {"recording_type": "camera", "is_final": False})
    assert item.is_final is False


def test_processing_status_prefers_server_field() -> None:
    item = _item(
        "r1", "t1", {"recording_type": "camera", "processing_status": "processing"}
    )
    assert item.processing_status == "processing"


def test_processing_status_derived_when_field_absent() -> None:
    assert _item("r1", "t1", {"recording_type": "camera"}).processing_status == "ready"
    assert (
        _item("r1", "t1", {"recording_type": "active"}).processing_status == "processing"
    )


def test_get_warns_when_camera_stream_has_no_downloadable_video(caplog) -> None:
    """A CAMERA-classified item whose envelope yields no camera file (segment
    MP4s still converting) must log a warning instead of failing silently."""
    import logging
    from types import SimpleNamespace

    from cyberwave.managers.recordings import RecordingManager

    envelope = SimpleNamespace(
        items=SimpleNamespace(
            twin_data={"t1": {}},  # camera pending: no camera entry at all
        )
    )
    api = SimpleNamespace(
        src_app_api_environments_recordings_get_recording_data=(
            lambda env, rec, return_flatbuffers: envelope
        )
    )
    manager = RecordingManager(api)
    item = _item(
        "r1", "t1", {"recording_type": "active", "video_parts": [{"chunk_index": 0}]}
    )

    with caplog.at_level(logging.WARNING, logger="cyberwave.managers.recordings"):
        recording = manager.get(item, environment_id="env-1")
    recording.close()

    assert any("no downloadable video yet" in m for m in caplog.messages)
