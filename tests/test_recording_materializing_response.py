"""``get()`` must surface a real 202 materializing body, not swallow it.

The endpoint answers 200 with a playback envelope or 202 with a materializing
body, and only the 202 body has a ``playback_readiness`` field — so the branch
has to be a type check. Duck-typing it also read ``retry_after_seconds`` from
inside ``playback_readiness`` only, silently dropping the top-level field that
the 202 schema actually declares.
"""

from __future__ import annotations

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.managers.recordings import (
    DEFAULT_MATERIALIZING_RETRY_SECONDS,
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
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, str]] = []

    def src_app_api_environments_recordings_get_recording_data(
        self, environment_uuid: str, recording_uuid: str, return_flatbuffers: bool = True
    ) -> object:
        self.calls.append((environment_uuid, recording_uuid))
        return self._result


def _manager(result: object) -> RecordingManager:
    return RecordingManager(_StubApi(result))


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
