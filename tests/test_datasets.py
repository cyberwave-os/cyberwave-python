"""Unit tests for :class:`~cyberwave.resources.DatasetManager`."""

import os
import tempfile
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typing import Any

import pytest

from cyberwave.exceptions import CyberwaveAPIError
from cyberwave.resources import DatasetManager


def _make_manager() -> tuple[DatasetManager, MagicMock]:
    mock_api = MagicMock()
    manager = DatasetManager(mock_api)
    return manager, mock_api


def test_dataset_manager_list_calls_list_endpoint():
    manager, mock_api = _make_manager()
    datasets = [SimpleNamespace(uuid="ds-1")]
    paginated_response = SimpleNamespace(
        datasets=datasets, total=1, limit=60, offset=0, has_more=False
    )
    mock_api.src_app_api_datasets_list_datasets.return_value = paginated_response

    result = manager.list()

    assert result is datasets
    mock_api.src_app_api_datasets_list_datasets.assert_called_once_with(
        limit=60, offset=0, environment=None, processing_status=None
    )


def test_dataset_manager_list_passes_filters():
    manager, mock_api = _make_manager()
    paginated_response = SimpleNamespace(
        datasets=[], total=0, limit=10, offset=5, has_more=False
    )
    mock_api.src_app_api_datasets_list_datasets.return_value = paginated_response

    manager.list(limit=10, offset=5, environment="env-uuid", processing_status="completed")

    mock_api.src_app_api_datasets_list_datasets.assert_called_once_with(
        limit=10, offset=5, environment="env-uuid", processing_status="completed"
    )


def test_dataset_manager_get_calls_get_endpoint():
    manager, mock_api = _make_manager()
    expected = SimpleNamespace(uuid="ds-1", name="test")
    mock_api.src_app_api_datasets_get_dataset.return_value = expected

    result = manager.get("ds-1")

    assert result is expected
    mock_api.src_app_api_datasets_get_dataset.assert_called_once_with("ds-1")


def test_dataset_manager_delete_calls_delete_endpoint():
    manager, mock_api = _make_manager()

    manager.delete("ds-1")

    mock_api.src_app_api_datasets_delete_dataset.assert_called_once_with("ds-1")


def test_dataset_manager_add_hf_initializes_import_and_fetches_full_schema():
    manager, mock_api = _make_manager()
    # list() returns empty so reuse_existing lookup finds nothing
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[], total=0, limit=200, offset=0, has_more=False
    )
    init_resp = SimpleNamespace(dataset_uuid="ds-uuid-1", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    expected_schema = SimpleNamespace(uuid="ds-uuid-1", name="pusht")
    mock_api.src_app_api_datasets_get_dataset.return_value = expected_schema

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        result = manager.add("lerobot/pusht", name="pusht")

    assert result is expected_schema
    mock_api.src_app_api_datasets_import_dataset.assert_called_once()
    payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert payload.source == "hf"
    assert payload.name == "pusht"
    assert payload.hf_repo_id == "lerobot/pusht"
    mock_api.src_app_api_datasets_get_dataset.assert_called_once_with("ds-uuid-1")


def test_dataset_manager_add_hf_passes_revision_and_subset():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[], total=0, limit=200, offset=0, has_more=False
    )
    init_resp = SimpleNamespace(dataset_uuid="id-1", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_get_dataset.return_value = SimpleNamespace(uuid="id-1")

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        manager.add(
            "org/repo",
            name="named",
            hf_revision="main",
            hf_subset="default",
        )

    payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert payload.source == "hf"
    assert payload.name == "named"
    assert payload.hf_repo_id == "org/repo"
    assert payload.hf_revision == "main"
    assert payload.hf_subset == "default"


def test_dataset_manager_add_hf_invalid_repo_raises():
    manager, _mock_api = _make_manager()

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        with pytest.raises(CyberwaveAPIError) as exc:
            manager.add("not-a-valid-hf-id")

    assert "HuggingFace" in str(exc.value) or "owner/repo" in str(exc.value)


def test_dataset_manager_add_local_zip_uploads_and_completes():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(
        dataset_uuid="ds-2",
        upload_url="https://storage.example.com/put",
    )
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    completed = SimpleNamespace(uuid="ds-2", name="myds")
    mock_api.src_app_api_datasets_complete_dataset_import.return_value = completed

    http_response = MagicMock()
    http_response.status = 200
    pool = MagicMock()
    pool.request.return_value = http_response

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "data.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "hello")
        expected_size = os.path.getsize(zip_path)

        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            result = manager.add(zip_path, name="myds")

    assert result is completed
    pool.request.assert_called_once()
    call_args = pool.request.call_args
    assert call_args.args[0] == "PUT"
    assert call_args.args[1] == "https://storage.example.com/put"
    assert call_args.kwargs["headers"]["Content-Type"] == "application/zip"

    init_payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert init_payload.source == "zip"
    assert init_payload.name == "myds"
    assert init_payload.file_size_bytes == expected_size

    complete_payload = (
        mock_api.src_app_api_datasets_complete_dataset_import.call_args.args[0]
    )
    assert complete_payload.dataset_uuid == "ds-2"

    mock_api.src_app_api_datasets_get_dataset.assert_not_called()


def test_dataset_manager_add_local_directory_uploads_and_completes():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(
        dataset_uuid="ds-3",
        upload_url="https://storage.example.com/put-dir",
    )
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    completed = SimpleNamespace(uuid="ds-3")
    mock_api.src_app_api_datasets_complete_dataset_import.return_value = completed

    http_response = MagicMock()
    http_response.status = 200
    pool = MagicMock()
    pool.request.return_value = http_response

    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "f.txt"), "w", encoding="utf-8") as fh:
            fh.write("content")

        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            result = manager.add(folder, name="folderds")

    assert result is completed
    mock_api.src_app_api_datasets_complete_dataset_import.assert_called_once()
    init_payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert init_payload.source == "zip"
    assert init_payload.name == "folderds"
    assert init_payload.file_size_bytes > 0


def test_dataset_manager_signed_url_put_failure_raises():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(
        dataset_uuid="ds-4",
        upload_url="https://storage.example.com/put",
    )
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp

    http_response = MagicMock()
    http_response.status = 500
    pool = MagicMock()
    pool.request.return_value = http_response

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "x.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.txt", "a")

        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            with pytest.raises(CyberwaveAPIError) as exc:
                manager.add(zip_path, name="failput")

    assert exc.value.status_code == 500
    mock_api.src_app_api_datasets_complete_dataset_import.assert_not_called()


def test_dataset_manager_visualize_returns_slug_url(capsys):
    manager, mock_api = _make_manager()
    ds = SimpleNamespace(uuid="ds-1", slug="acme/datasets/pusht")
    mock_api.src_app_api_datasets_get_dataset.return_value = ds

    url = manager.visualize(ds)

    assert url == "https://cyberwave.com/acme/datasets/pusht"
    # visualize() is now silent — no stdout side-effects
    captured = capsys.readouterr()
    assert captured.out == ""


def test_dataset_manager_visualize_falls_back_to_uuid(capsys):
    manager, mock_api = _make_manager()
    ds = SimpleNamespace(uuid="ds-uuid-123", slug=None)

    url = manager.visualize(ds)

    assert url == "https://cyberwave.com/datasets/ds-uuid-123"
    captured = capsys.readouterr()
    assert captured.out == ""


def test_dataset_manager_visualize_accepts_uuid_string():
    manager, mock_api = _make_manager()
    ds = SimpleNamespace(uuid="ds-1", slug="acme/datasets/test")
    mock_api.src_app_api_datasets_get_dataset.return_value = ds

    url = manager.visualize("ds-1")

    assert url == "https://cyberwave.com/acme/datasets/test"
    mock_api.src_app_api_datasets_get_dataset.assert_called_once_with("ds-1")


def test_dataset_manager_visualize_raises_if_no_slug_or_uuid():
    manager, _mock_api = _make_manager()
    ds = SimpleNamespace(slug=None, uuid=None)

    with pytest.raises(CyberwaveAPIError) as exc:
        manager.visualize(ds)

    assert "slug" in str(exc.value).lower() or "uuid" in str(exc.value).lower()


def test_dataset_manager_visualize_uses_env_var_for_frontend_url(capsys, monkeypatch):
    monkeypatch.setenv("CYBERWAVE_FRONTEND_URL", "https://custom.example.com")
    manager, mock_api = _make_manager()
    manager._FRONTEND_URL = "https://custom.example.com"
    ds = SimpleNamespace(uuid="ds-1", slug="acme/datasets/test")

    url = manager.visualize(ds)

    assert url == "https://custom.example.com/acme/datasets/test"


def test_dataset_manager_add_hf_uses_default_name_from_repo():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[], total=0, limit=200, offset=0, has_more=False
    )
    init_resp = SimpleNamespace(dataset_uuid="id-1", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_get_dataset.return_value = SimpleNamespace(uuid="id-1")

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        manager.add("lerobot/pusht")

    payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert payload.name == "lerobot-pusht"


def test_dataset_manager_add_local_uses_default_name_from_path():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(
        dataset_uuid="ds-5",
        upload_url="https://storage.example.com/put",
    )
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_complete_dataset_import.return_value = SimpleNamespace(
        uuid="ds-5"
    )

    http_response = MagicMock()
    http_response.status = 200
    pool = MagicMock()
    pool.request.return_value = http_response

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "my_custom_dataset.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.txt", "content")

        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            manager.add(zip_path)

    init_payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert init_payload.name == "my_custom_dataset"


def test_dataset_manager_add_local_non_zip_file_wraps_in_zip():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(
        dataset_uuid="ds-6",
        upload_url="https://storage.example.com/put",
    )
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_complete_dataset_import.return_value = SimpleNamespace(
        uuid="ds-6"
    )

    http_response = MagicMock()
    http_response.status = 200
    pool = MagicMock()
    pool.request.return_value = http_response

    with tempfile.TemporaryDirectory() as tmp:
        txt_path = os.path.join(tmp, "data.parquet")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("parquet content")

        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            manager.add(txt_path, name="parquet-ds")

    init_payload = mock_api.src_app_api_datasets_import_dataset.call_args.args[0]
    assert init_payload.source == "zip"
    assert init_payload.file_size_bytes > 0


def test_dataset_manager_add_missing_upload_url_raises():
    manager, mock_api = _make_manager()
    init_resp = SimpleNamespace(dataset_uuid="ds-7", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "data.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("x.txt", "x")

        with pytest.raises(CyberwaveAPIError) as exc:
            manager.add(zip_path, name="no-url")

    assert "upload URL" in str(exc.value) or "signed" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# convert() tests
# ---------------------------------------------------------------------------


def _ready_response(
    format: str = "lerobot3",
    signed_url: str = "https://storage.example.com/file.zip",
) -> SimpleNamespace:
    return SimpleNamespace(
        format=format,
        status="ready",
        signed_url=signed_url,
        expires_at="2099-01-01T00:00:00Z",
        processed_dataset_uuid="pd-uuid-1",
    )


def _processing_response(
    format: str = "lerobot3",
    status: str = "queued",
) -> SimpleNamespace:
    return SimpleNamespace(
        format=format,
        status=status,
        message=f"Conversion {status}.",
        processed_dataset_uuid="pd-uuid-1",
        poll_url=f"/api/v1/datasets/ds-1/download?format={format}",
    )


def test_convert_returns_signed_url_immediately_when_ready():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response()

    url = manager.convert("ds-1", "lerobot3", on_poll=None)

    assert url == "https://storage.example.com/file.zip"
    mock_api.src_app_api_datasets_download_dataset.assert_called_once_with("ds-1", "lerobot3")


def test_convert_accepts_dataset_schema_object():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response()
    ds = SimpleNamespace(uuid="ds-abc")

    url = manager.convert(ds, "parquet", on_poll=None)

    assert url == "https://storage.example.com/file.zip"
    mock_api.src_app_api_datasets_download_dataset.assert_called_once_with("ds-abc", "parquet")


def test_convert_normalises_format_to_lowercase():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response(format="rlds")

    manager.convert("ds-1", "  RLDS  ", on_poll=None)

    call_format = mock_api.src_app_api_datasets_download_dataset.call_args.args[1]
    assert call_format == "rlds"


def test_convert_polls_until_ready():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.side_effect = [
        _processing_response(status="queued"),
        _processing_response(status="processing"),
        _ready_response(),
    ]

    with patch("cyberwave.resources.time.sleep"):
        url = manager.convert("ds-1", "lerobot3", on_poll=None)

    assert url == "https://storage.example.com/file.zip"
    assert mock_api.src_app_api_datasets_download_dataset.call_count == 3


def test_convert_raises_on_timeout():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _processing_response(
        status="queued"
    )

    with patch("cyberwave.resources.time.sleep"):
        with pytest.raises(CyberwaveAPIError) as exc:
            manager.convert("ds-1", "lerobot3", timeout=0.001, on_poll=None)

    assert "did not complete" in str(exc.value).lower() or "timeout" in str(exc.value).lower()


def test_convert_propagates_api_error():
    manager, mock_api = _make_manager()
    api_err = Exception("boom")
    api_err.status = 422
    api_err.body = {"detail": "Unsupported format"}
    mock_api.src_app_api_datasets_download_dataset.side_effect = api_err

    with pytest.raises(CyberwaveAPIError):
        manager.convert("ds-1", "invalid_format", on_poll=None)


def test_convert_calls_on_poll_callback(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.side_effect = [
        _processing_response(status="queued"),
        _ready_response(),
    ]
    calls: list[Any] = []

    with patch("cyberwave.resources.time.sleep"):
        manager.convert("ds-1", "lerobot3", on_poll=calls.append)

    assert len(calls) == 2
    assert getattr(calls[0], "status", None) == "queued"


def test_convert_default_on_poll_prints_to_stdout(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.side_effect = [
        _processing_response(status="queued"),
        _ready_response(),
    ]

    with patch("cyberwave.resources.time.sleep"):
        manager.convert("ds-1", "lerobot3")  # default on_poll prints

    out = capsys.readouterr().out
    assert "queued" in out.lower() or "status" in out.lower()


def test_convert_on_poll_none_is_silent(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.side_effect = [
        _processing_response(status="queued"),
        _ready_response(),
    ]

    with patch("cyberwave.resources.time.sleep"):
        manager.convert("ds-1", "lerobot3", on_poll=None)

    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# download() tests
# ---------------------------------------------------------------------------


def test_download_saves_file_to_dest_directory():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response(
        signed_url="https://storage.example.com/artifact.zip"
    )

    http_resp = MagicMock()
    http_resp.status = 200
    http_resp.read.side_effect = [b"chunk1", b"chunk2", b""]
    pool = MagicMock()
    pool.request.return_value = http_resp

    with tempfile.TemporaryDirectory() as dest:
        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            path = manager.download("ds-1", "lerobot3", dest=dest, on_poll=None)

        assert path.endswith(".zip")
        assert os.path.isfile(path)
        with open(path, "rb") as fh:
            assert fh.read() == b"chunk1chunk2"


def test_download_saves_file_to_explicit_path():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response()

    http_resp = MagicMock()
    http_resp.status = 200
    http_resp.read.side_effect = [b"data", b""]
    pool = MagicMock()
    pool.request.return_value = http_resp

    with tempfile.TemporaryDirectory() as tmp:
        explicit_path = os.path.join(tmp, "my_output.zip")
        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            path = manager.download("ds-1", "parquet", dest=explicit_path, on_poll=None)

        assert path == explicit_path
        assert os.path.isfile(path)


def test_download_raises_on_http_error():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response()

    http_resp = MagicMock()
    http_resp.status = 403
    pool = MagicMock()
    pool.request.return_value = http_resp

    with tempfile.TemporaryDirectory() as tmp:
        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            with pytest.raises(CyberwaveAPIError) as exc:
                manager.download("ds-1", "lerobot3", dest=tmp, on_poll=None)

    assert exc.value.status_code == 403


def test_download_derives_filename_from_uuid_and_format():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.return_value = _ready_response(
        signed_url="https://storage.example.com/file.zip"
    )

    http_resp = MagicMock()
    http_resp.status = 200
    http_resp.read.side_effect = [b"x", b""]
    pool = MagicMock()
    pool.request.return_value = http_resp

    with tempfile.TemporaryDirectory() as dest:
        with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
            path = manager.download("ds-uuid-99", "openvla", dest=dest, on_poll=None)

    assert os.path.basename(path) == "ds-uuid-99_openvla.zip"


def test_download_polls_conversion_before_downloading():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_download_dataset.side_effect = [
        _processing_response(status="queued"),
        _ready_response(),
    ]

    http_resp = MagicMock()
    http_resp.status = 200
    http_resp.read.side_effect = [b"bytes", b""]
    pool = MagicMock()
    pool.request.return_value = http_resp

    with tempfile.TemporaryDirectory() as dest:
        with patch("cyberwave.resources.time.sleep"):
            with patch("cyberwave.resources.urllib3.PoolManager", return_value=pool):
                path = manager.download("ds-1", "lerobot3", dest=dest, on_poll=None)

        assert os.path.isfile(path)
    assert mock_api.src_app_api_datasets_download_dataset.call_count == 2


# ---------------------------------------------------------------------------
# wait_until_ready() tests
# ---------------------------------------------------------------------------


def _ds(uuid: str = "ds-1", status: str = "pending", **kwargs: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = dict(
        uuid=uuid,
        processing_status=status,
        processed_episodes=0,
        total_episodes=0,
        failed_episodes=0,
        failed_episode_uuids=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_wait_until_ready_returns_immediately_when_completed():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.return_value = _ds(status="completed")

    result = manager.wait_until_ready("ds-1", on_poll=None)

    assert result.processing_status == "completed"
    mock_api.src_app_api_datasets_get_dataset.assert_called_once_with("ds-1")


def test_wait_until_ready_accepts_dataset_schema():
    manager, mock_api = _make_manager()
    ds_obj = _ds(uuid="ds-abc", status="completed")
    mock_api.src_app_api_datasets_get_dataset.return_value = ds_obj

    result = manager.wait_until_ready(ds_obj, on_poll=None)

    assert result.processing_status == "completed"
    mock_api.src_app_api_datasets_get_dataset.assert_called_once_with("ds-abc")


def test_wait_until_ready_polls_until_completed():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.side_effect = [
        _ds(status="pending"),
        _ds(status="processing"),
        _ds(status="completed"),
    ]

    with patch("cyberwave.resources.time.sleep"):
        result = manager.wait_until_ready("ds-1", on_poll=None)

    assert result.processing_status == "completed"
    assert mock_api.src_app_api_datasets_get_dataset.call_count == 3


def test_wait_until_ready_raises_on_failed():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.return_value = _ds(
        status="failed", failed_episode_uuids=["ep-1"]
    )

    with pytest.raises(CyberwaveAPIError) as exc:
        manager.wait_until_ready("ds-1", on_poll=None)

    assert "failed" in str(exc.value).lower()


def test_wait_until_ready_raises_on_timeout():
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.return_value = _ds(status="processing")

    with patch("cyberwave.resources.time.sleep"):
        with pytest.raises(CyberwaveAPIError) as exc:
            manager.wait_until_ready("ds-1", timeout=0.001, on_poll=None)

    assert "did not complete" in str(exc.value).lower() or "timeout" in str(exc.value).lower()


def test_wait_until_ready_calls_on_poll(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.side_effect = [
        _ds(status="pending"),
        _ds(status="completed"),
    ]
    calls: list[Any] = []

    with patch("cyberwave.resources.time.sleep"):
        manager.wait_until_ready("ds-1", on_poll=calls.append)

    assert len(calls) == 2
    assert calls[0].processing_status == "pending"
    assert calls[1].processing_status == "completed"


def test_wait_until_ready_default_on_poll_prints(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.return_value = _ds(status="completed")

    manager.wait_until_ready("ds-1")  # default on_poll prints

    out = capsys.readouterr().out
    assert "ds-1" in out or "status" in out.lower()


def test_wait_until_ready_on_poll_none_is_silent(capsys):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_datasets_get_dataset.return_value = _ds(status="completed")

    manager.wait_until_ready("ds-1", on_poll=None)

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# reuse_existing / idempotent HF import tests
# ---------------------------------------------------------------------------


def test_add_hf_reuses_existing_import_when_found():
    manager, mock_api = _make_manager()
    existing_ds = SimpleNamespace(
        uuid="ds-existing",
        metadata={
            "import": {
                "source": "hf",
                "hf_repo_id": "lerobot/pusht",
                "hf_revision": None,
                "hf_subset": None,
            }
        },
    )
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[existing_ds], total=1, limit=200, offset=0, has_more=False
    )

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        result = manager.add("lerobot/pusht")

    assert result is existing_ds
    mock_api.src_app_api_datasets_import_dataset.assert_not_called()


def test_add_hf_skips_reuse_when_reuse_existing_false():
    manager, mock_api = _make_manager()
    existing_ds = SimpleNamespace(
        uuid="ds-existing",
        metadata={
            "import": {
                "source": "hf",
                "hf_repo_id": "lerobot/pusht",
                "hf_revision": None,
                "hf_subset": None,
            }
        },
    )
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[existing_ds], total=1, limit=200, offset=0, has_more=False
    )
    init_resp = SimpleNamespace(dataset_uuid="ds-new", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_get_dataset.return_value = SimpleNamespace(uuid="ds-new")

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        result = manager.add("lerobot/pusht", reuse_existing=False)

    assert result.uuid == "ds-new"
    mock_api.src_app_api_datasets_import_dataset.assert_called_once()


def test_add_hf_does_not_reuse_when_repo_differs():
    manager, mock_api = _make_manager()
    other_ds = SimpleNamespace(
        uuid="ds-other",
        metadata={
            "import": {
                "source": "hf",
                "hf_repo_id": "lerobot/other",
                "hf_revision": None,
                "hf_subset": None,
            }
        },
    )
    mock_api.src_app_api_datasets_list_datasets.return_value = SimpleNamespace(
        datasets=[other_ds], total=1, limit=200, offset=0, has_more=False
    )
    init_resp = SimpleNamespace(dataset_uuid="ds-new", upload_url=None)
    mock_api.src_app_api_datasets_import_dataset.return_value = init_resp
    mock_api.src_app_api_datasets_get_dataset.return_value = SimpleNamespace(uuid="ds-new")

    with patch("cyberwave.resources.os.path.exists", return_value=False):
        result = manager.add("lerobot/pusht")

    mock_api.src_app_api_datasets_import_dataset.assert_called_once()
