"""Unit tests for :class:`~cyberwave.resources.DatasetManager`."""

import os
import tempfile
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_dataset_manager_visualize_prints_slug_url(capsys):
    manager, mock_api = _make_manager()
    ds = SimpleNamespace(uuid="ds-1", slug="acme/datasets/pusht")
    mock_api.src_app_api_datasets_get_dataset.return_value = ds

    url = manager.visualize(ds)

    assert url == "https://cyberwave.com/acme/datasets/pusht"
    captured = capsys.readouterr()
    assert "https://cyberwave.com/acme/datasets/pusht" in captured.out


def test_dataset_manager_visualize_falls_back_to_uuid(capsys):
    manager, mock_api = _make_manager()
    ds = SimpleNamespace(uuid="ds-uuid-123", slug=None)

    url = manager.visualize(ds)

    assert url == "https://cyberwave.com/datasets/ds-uuid-123"
    captured = capsys.readouterr()
    assert "https://cyberwave.com/datasets/ds-uuid-123" in captured.out


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
