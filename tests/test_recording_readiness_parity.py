"""Keep the SDK's legacy fallback aligned with server stream detection."""

from __future__ import annotations

import pytest

from cyberwave.managers.recordings import RecordingListItem, RecordingType


CASES = [
    (
        {"metadata_type": "TwinRecordingMetadata", "path": "r.parquet"},
        {RecordingType.ROBOT},
    ),
    ({"recording_type": "robot", "path": "r.parquet"}, {RecordingType.ROBOT}),
    ({"twin_type": "robot", "path": "r.parquet"}, {RecordingType.ROBOT}),
    (
        {"joint_names": ["j1"], "path": "r.parquet"},
        {RecordingType.ROBOT},
    ),
    (
        {"metadata_type": "CameraRecordingMetadata", "mp4_path": "v.mp4"},
        {RecordingType.CAMERA},
    ),
    (
        {"recording_type": "camera_wrist", "mp4_path": "v.mp4"},
        {RecordingType.CAMERA},
    ),
    ({"mp4_path": "v.mp4"}, {RecordingType.CAMERA}),
    (
        {
            "metadata_type": "CameraRecordingMetadata",
            "recording_type": "camera",
            "mp4_path": "v.mp4",
            "path": "cam.parquet",
            "num_rows": 900,
        },
        {RecordingType.CAMERA},
    ),
    (
        {
            "metadata_type": "CameraRecordingMetadata",
            "mp4_path": "v.mp4",
            "pointcloud": "a",
        },
        {RecordingType.CAMERA, RecordingType.DEPTH},
    ),
    (
        {
            "metadata_type": "CameraRecordingMetadata",
            "mp4_path": "v.mp4",
            "colored_pointcloud": "b",
        },
        {RecordingType.CAMERA, RecordingType.POINTCLOUD},
    ),
    ({"session_id": "s1"}, set()),
]


@pytest.mark.parametrize("metadata,expected", CASES)
def test_local_classifier_matches_expected_taxonomy(
    metadata: dict[str, object], expected: set[RecordingType]
) -> None:
    item = RecordingListItem(
        uuid="u", twin_uuid="t", environment_uuid="e", metadata=metadata
    )
    assert item.types == frozenset(expected)


def test_server_stream_list_wins_over_local_classification() -> None:
    item = RecordingListItem(
        uuid="u",
        twin_uuid="t",
        environment_uuid="e",
        metadata={"mp4_path": "v.mp4"},
        readiness={
            "state": "ready",
            "reason": None,
            "retry_after_seconds": None,
            "streams": [
                {
                    "type": "robot",
                    "state": "ready",
                    "required": True,
                    "reason": None,
                    "ready_until_us": None,
                }
            ],
        },
    )
    assert item.types == frozenset({RecordingType.ROBOT})
    assert item.is_playback_ready


def test_falls_back_to_local_classification_against_an_old_server() -> None:
    item = RecordingListItem(
        uuid="u",
        twin_uuid="t",
        environment_uuid="e",
        metadata={"mp4_path": "v.mp4"},
        readiness=None,
    )
    assert item.types == frozenset({RecordingType.CAMERA})
    assert item.is_playback_ready
