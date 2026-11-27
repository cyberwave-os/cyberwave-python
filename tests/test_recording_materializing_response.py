"""``get()`` must surface a real 202 materializing body, not swallow it.

The endpoint answers 200 with a playback envelope or 202 with a materializing
body, and only the 202 body has a ``playback_readiness`` field — so the branch
has to be a type check. Duck-typing it also read ``retry_after_seconds`` from
inside ``playback_readiness`` only, silently dropping the top-level field that
the 202 schema actually declares.
"""

from __future__ import annotations

import pytest

import cyberwave.managers.recordings as recordings_module
from cyberwave.exceptions import CyberwaveError
from cyberwave.managers.recordings import (
    DEFAULT_MATERIALIZING_RETRY_SECONDS,
    RecordingListItem,
    RecordingManager,
)
from cyberwave.rest.models.recording_materializing_schema import (
    RecordingMaterializingSchema,
)
from cyberwave.rest.models.recording_sources_envelope_schema import (
    RecordingSourcesEnvelopeSchema,
)
from cyberwave.rest.models.recording_sources_schema import RecordingSourcesSchema


# Verbatim body from GET .../recordings/{uuid}/data when the recording exists
# but its playback assets are still being built.
MATERIALIZING_BODY = {
    "detail": "Recording assets are still materializing",
    "playback_readiness": {
        "state": "materializing",
        "reason": "camera parquet is still generating",
        "retry_after_seconds": 30,
        "streams": [
            {
                "type": "camera",
                "state": "materializing",
                "required": True,
                "reason": "camera parquet is still generating",
                "ready_until_us": None,
            }
        ],
    },
    "retry_after_seconds": 30,
}


class _StubApi:
    """Answers with ``results`` in order, repeating the last one."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def src_app_api_environments_recordings_get_recording_data(
        self, environment_uuid: str, recording_uuid: str, return_flatbuffers: bool = True
    ) -> object:
        self.calls.append((environment_uuid, recording_uuid))
        index = min(len(self.calls), len(self._results)) - 1
        return self._results[index]


def _manager(*results: object) -> RecordingManager:
    return RecordingManager(_StubApi(*results))


def test_get_raises_with_the_202_reason_and_retry_interval() -> None:
    body = RecordingMaterializingSchema.from_dict(MATERIALIZING_BODY)
    assert body is not None
    manager = _manager(body)

    with pytest.raises(CyberwaveError) as excinfo:
        manager.get("rec-1", environment_id="env-1")

    message = str(excinfo.value)
    assert "camera parquet is still generating" in message
    # The retry interval is a top-level field on the 202 body; losing it would
    # leave a caller with no idea when to poll again.
    assert "~30s" in message
    assert body.retry_after_seconds == 30


def test_a_top_level_retry_after_seconds_is_not_dropped() -> None:
    """``retry_after_seconds`` lives on the body, not inside the readiness dict.

    Reading only ``playback_readiness["retry_after_seconds"]`` reported the
    hardcoded default here, telling callers to poll far sooner than the server
    asked.
    """
    body = RecordingMaterializingSchema.from_dict(
        {
            "detail": "Recording assets are still materializing",
            "playback_readiness": {"state": "materializing", "reason": "robot parquet"},
            "retry_after_seconds": 45,
        }
    )
    assert body is not None
    assert "retry_after_seconds" not in body.playback_readiness

    with pytest.raises(CyberwaveError) as excinfo:
        _manager(body).get("rec-1", environment_id="env-1")

    message = str(excinfo.value)
    assert "robot parquet" in message
    assert "~45s" in message
    assert f"~{DEFAULT_MATERIALIZING_RETRY_SECONDS}s" not in message


def test_get_falls_back_to_a_default_retry_when_the_202_omits_one() -> None:
    body = RecordingMaterializingSchema.from_dict(
        {
            "detail": "Recording assets are still materializing",
            "playback_readiness": {"state": "materializing", "reason": "robot parquet"},
            "retry_after_seconds": None,
        }
    )
    assert body is not None

    with pytest.raises(CyberwaveError) as excinfo:
        _manager(body).get("rec-1", environment_id="env-1")

    assert "robot parquet" in str(excinfo.value)
    assert f"~{DEFAULT_MATERIALIZING_RETRY_SECONDS}s" in str(excinfo.value)


def test_a_200_envelope_is_never_mistaken_for_a_materializing_body() -> None:
    """The 200 schema has no ``playback_readiness`` at all — the old check's bug."""
    envelope = RecordingSourcesEnvelopeSchema(
        items=RecordingSourcesSchema(
            is_combined=False,
            twin_uuids=[],
            twin_data={},
            start_timestamp_us=0,
            end_timestamp_us=0,
        ),
    )
    assert not hasattr(envelope, "playback_readiness")

    recording = _manager(envelope).get("rec-1", environment_id="env-1")

    assert recording.uuid == "rec-1"
    assert recording.environment_uuid == "env-1"
    recording.close()


def test_a_missing_body_still_reports_a_retryable_condition() -> None:
    with pytest.raises(CyberwaveError, match="materializing"):
        _manager(None).get("rec-1", environment_id="env-1")


def test_environment_id_is_required_for_a_uuid_lookup() -> None:
    with pytest.raises(CyberwaveError, match="environment_id is required"):
        _manager(None).get("rec-1")


def _empty_envelope() -> RecordingSourcesEnvelopeSchema:
    return RecordingSourcesEnvelopeSchema(
        items=RecordingSourcesSchema(
            is_combined=False,
            twin_uuids=[],
            twin_data={},
            start_timestamp_us=0,
            end_timestamp_us=0,
        ),
    )


class _FakeClock:
    """``time`` stand-in for the polling loop: ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(recordings_module, "time", fake)
    return fake


def test_wait_polls_through_202_until_the_envelope_arrives(clock: _FakeClock) -> None:
    """``wait`` honours the 202's retry interval and returns once assets exist."""
    body = RecordingMaterializingSchema.from_dict(MATERIALIZING_BODY)
    manager = _manager(body, body, _empty_envelope())

    recording = manager.get("rec-1", environment_id="env-1", wait=120)

    assert len(manager.api.calls) == 3
    assert clock.sleeps == [30, 30]
    recording.close()


def test_wait_never_sleeps_past_its_deadline(clock: _FakeClock) -> None:
    """A 30 s retry hint on a 5 s wait costs the caller 5 s, not 30."""
    body = RecordingMaterializingSchema.from_dict(MATERIALIZING_BODY)
    manager = _manager(body)

    with pytest.raises(CyberwaveError, match="~30s"):
        manager.get("rec-1", environment_id="env-1", wait=5)

    assert clock.sleeps == [5]
    assert len(manager.api.calls) == 2


def _failed_body(**readiness: object) -> RecordingMaterializingSchema:
    body = RecordingMaterializingSchema.from_dict(
        {
            "detail": "Progressive recording assets are still materializing",
            "playback_readiness": {"state": "failed", **readiness},
            "retry_after_seconds": 5,
        }
    )
    assert body is not None
    return body


def test_a_failed_readiness_is_polled_until_the_recording_recovers(
    clock: _FakeClock,
) -> None:
    """A live recording reports ``failed`` while its first part is retried.

    ``missing_source`` counts as terminal on the server, but the camera MP4 task
    retries it ~10 s later, and one ready part turns the rollup ``ready``. The
    wait must ride that out rather than give up on the first poll.
    """
    failed = _failed_body(reason="camera_parts_failed")
    manager = _manager(failed, failed, _empty_envelope())

    recording = manager.get("rec-1", environment_id="env-1", wait=60)

    assert len(manager.api.calls) == 3
    assert clock.sleeps == [5, 5]
    recording.close()


def test_a_failed_readiness_without_wait_raises_without_promising_finality() -> None:
    manager = _manager(_failed_body(reason="camera_parts_failed"))

    with pytest.raises(CyberwaveError) as excinfo:
        manager.get("rec-1", environment_id="env-1")

    message = str(excinfo.value)
    assert "camera_parts_failed" in message
    assert "~5s" in message
    assert "will not help" not in message


def test_a_failed_message_names_the_failed_streams_reason() -> None:
    """The server can overwrite the top-level reason with a pending one."""
    body = _failed_body(
        reason="camera_mp4_pending",
        streams=[
            {
                "type": "robot",
                "state": "ready",
                "required": True,
                "reason": None,
                "ready_until_us": None,
            },
            {
                "type": "camera",
                "state": "failed",
                "required": True,
                "reason": "camera_sync_failed",
                "ready_until_us": None,
            },
        ],
    )

    with pytest.raises(CyberwaveError) as excinfo:
        _manager(body).get("rec-1", environment_id="env-1")

    message = str(excinfo.value)
    assert "camera_sync_failed" in message
    assert "camera_mp4_pending" not in message


def test_list_item_reports_only_a_finalized_failed_recording_as_never_playable() -> (
    None
):
    def item(state: str | None, **metadata: object) -> RecordingListItem:
        return RecordingListItem(
            uuid="rec-1",
            twin_uuid=None,
            environment_uuid="env-1",
            metadata=dict(metadata),
            readiness=None if state is None else {"state": state},
        )

    assert item("failed").will_never_be_playable
    assert not item("materializing").will_never_be_playable
    assert not item("ready").will_never_be_playable
    assert not item(None).will_never_be_playable
    # A recording still being captured can recover from ``failed``.
    assert not item("failed", is_final=False).will_never_be_playable
    assert not item("failed", recording_type="active").will_never_be_playable
