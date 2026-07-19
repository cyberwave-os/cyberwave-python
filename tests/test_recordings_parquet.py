import types

import pytest

from cyberwave.managers.recordings import RecordingManager, RecordingType


class _Env:
    def __init__(self, twin_data):
        self.items = type("I", (), {"twin_data": twin_data})()


def _make_recording(twin_data, local_paths=None):
    from cyberwave.managers.recordings import Recording

    return Recording(
        uuid="r1",
        twin_uuid="t1",
        environment_uuid="env-1",
        types=frozenset({RecordingType.POINTCLOUD}),
        signed_urls=_Env(twin_data),
        local_paths=local_paths or {},
        tempdir="",
    )


def test_contextual_methods_depth_only():
    rec = _make_recording({"t1": {"pointcloud": {"format": "parquet"}}})
    assert hasattr(rec, "read_depth")
    assert "read_depth" in dir(rec)
    assert not hasattr(rec, "read_pointcloud")
    assert "read_pointcloud" not in dir(rec)
    assert not hasattr(rec, "show_video")
    assert "show_video" not in dir(rec)


def test_contextual_methods_colored_only():
    rec = _make_recording({"t1": {"colored_pointcloud": {"format": "parquet"}}})
    assert hasattr(rec, "read_pointcloud")
    assert "read_pointcloud" in dir(rec)
    assert not hasattr(rec, "read_depth")


def test_contextual_methods_video_present():
    rec = _make_recording({"t1": {"camera": {"signed_url": "https://x/v.mp4"}}})
    assert hasattr(rec, "show_video")
    assert "show_video" in dir(rec)
    assert not hasattr(rec, "read_depth")
    assert not hasattr(rec, "read_pointcloud")


def test_contextual_depth_camera_has_video_and_depth():
    rec = _make_recording(
        {
            "t1": {
                "camera": {"signed_url": "https://x/v.mp4"},
                "pointcloud": {"format": "parquet"},
            }
        }
    )
    assert hasattr(rec, "show_video")
    assert hasattr(rec, "read_depth")
    assert not hasattr(rec, "read_pointcloud")


def test_read_depth_reconstructs_frames(tmp_path):
    np = pytest.importorskip("numpy")
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    frame = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
    table = pa.table(
        {
            "timestamp_us": [10],
            "rows": [4],
            "cols": [5],
            "dtype": ["uint16"],
            "data": [frame.tobytes()],
        }
    )
    path = tmp_path / "depth.parquet"
    pq.write_table(table, str(path))

    rec = _make_recording(
        {"t1": {"pointcloud": {"format": "parquet"}}},
        local_paths={"pointcloud": [path]},
    )
    frames = rec.read_depth()
    assert len(frames) == 1
    assert frames[0]["timestamp_us"] == 10
    assert np.array_equal(frames[0]["frame"], frame)


def test_read_depth_joins_multiple_parts_in_timestamp_order(tmp_path):
    """Several downloaded depth parquet parts are joined and returned in
    timestamp order, even when the parts arrive out of order on disk."""
    np = pytest.importorskip("numpy")
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    def _write(name, ts):
        frame = np.full((2, 2), ts, dtype=np.uint16)
        table = pa.table(
            {
                "timestamp_us": [ts],
                "rows": [2],
                "cols": [2],
                "dtype": ["uint16"],
                "data": [frame.tobytes()],
            }
        )
        path = tmp_path / name
        pq.write_table(table, str(path))
        return path

    # Part filenames sort so the later-timestamp part comes first lexically;
    # the reader must still emit frames in timestamp order.
    p_late = _write("a_part.parquet", 30)
    p_early = _write("b_part.parquet", 10)

    rec = _make_recording(
        {"t1": {"pointcloud": {"format": "parquet"}}},
        local_paths={"pointcloud": [p_late, p_early]},
    )
    frames = rec.read_depth()
    assert [f["timestamp_us"] for f in frames] == [10, 30]


def test_missing_contextual_method_raises_attribute_error():
    rec = _make_recording({"t1": {"pointcloud": {"format": "parquet"}}})
    with pytest.raises(AttributeError):
        rec.read_pointcloud  # noqa: B018 — attribute access should raise
    _ = types  # import used by other helpers


def test_collect_sources_ready_parquet():
    env = _Env(
        {
            "twin-1": {
                "colored_pointcloud": {
                    "format": "parquet",
                    "status": "ready",
                    "signed_url": "https://x/y.parquet",
                    "timestamps": [1, 2],
                }
            }
        }
    )
    sources = RecordingManager._collect_sources(env)
    assert ("colored_pointcloud", "https://x/y.parquet", "y.parquet") in [
        (s, u, n) for (s, u, n) in sources
    ]


def test_collect_sources_generating_is_skipped():
    env = _Env(
        {
            "twin-1": {
                "pointcloud": {
                    "format": "parquet",
                    "status": "generating",
                    "signed_url": None,
                    "message": "Parquet is being generated, retry in a few minutes.",
                }
            }
        }
    )
    sources = RecordingManager._collect_sources(env)
    assert sources == []


def test_collect_sources_flatbuffer_unchanged():
    env = _Env(
        {
            "twin-1": {
                "pointcloud": {
                    "format": "flatbuffer",
                    "signed_urls": ["https://x/a.fb", "https://x/b.fb"],
                }
            }
        }
    )
    sources = RecordingManager._collect_sources(env)
    assert len(sources) == 2
    assert all(n.endswith(".fb") for (_s, _u, n) in sources)
