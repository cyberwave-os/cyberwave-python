from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.exceptions import CyberwaveAPIError
from cyberwave.resources import AssetManager


class PayloadTooLargeError(Exception):
    status = 413
    body = "Payload Too Large"


def _make_manager():
    mock_api = MagicMock()
    manager = AssetManager(mock_api)
    return manager, mock_api


def test_upload_glb_uses_attachment_flow_for_large_files(tmp_path):
    manager, _ = _make_manager()
    file_path = tmp_path / "large.glb"
    file_path.write_bytes(b"x")

    with (
        patch("os.path.getsize", return_value=AssetManager.MAX_STANDARD_UPLOAD_BYTES + 1),
        patch.object(manager, "_upload_glb_via_attachment", return_value="ok") as via_flow,
        patch.object(manager, "_upload_glb_direct") as direct_flow,
    ):
        result = manager.upload_glb("asset-1", str(file_path))

    assert result == "ok"
    via_flow.assert_called_once_with("asset-1", str(file_path))
    direct_flow.assert_not_called()


def test_upload_glb_uses_direct_flow_for_small_files(tmp_path):
    manager, _ = _make_manager()
    file_path = tmp_path / "small.glb"
    file_path.write_bytes(b"x")

    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(manager, "_upload_glb_direct", return_value="ok") as direct_flow,
        patch.object(manager, "_upload_glb_via_attachment") as via_flow,
    ):
        result = manager.upload_glb("asset-1", str(file_path))

    assert result == "ok"
    direct_flow.assert_called_once_with("asset-1", str(file_path))
    via_flow.assert_not_called()


def test_upload_glb_falls_back_to_attachment_on_payload_too_large(tmp_path):
    manager, _ = _make_manager()
    file_path = tmp_path / "small.glb"
    file_path.write_bytes(b"x")

    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(
            manager,
            "_upload_glb_direct",
            side_effect=PayloadTooLargeError(),
        ) as direct_flow,
        patch.object(manager, "_upload_glb_via_attachment", return_value="ok") as via_flow,
    ):
        result = manager.upload_glb("asset-1", str(file_path))

    assert result == "ok"
    direct_flow.assert_called_once_with("asset-1", str(file_path))
    via_flow.assert_called_once_with("asset-1", str(file_path))


def test_upload_glb_via_attachment_raises_when_signed_url_missing(tmp_path):
    manager, mock_api = _make_manager()
    file_path = tmp_path / "missing-url.glb"
    file_path.write_bytes(b"glb")

    mock_api.src_app_api_attachments_create_attachment.return_value = SimpleNamespace(
        uuid="attachment-1"
    )
    mock_api.src_app_api_attachments_initiate_large_attachment_upload.return_value = (
        SimpleNamespace(upload_url=None, storage_key=None)
    )

    with pytest.raises(CyberwaveAPIError):
        manager._upload_glb_via_attachment("asset-1", str(file_path))


def test_upload_glb_via_attachment_happy_path(tmp_path):
    manager, mock_api = _make_manager()
    file_path = tmp_path / "model.glb"
    file_path.write_bytes(b"glb-content")

    mock_api.src_app_api_attachments_create_attachment.return_value = SimpleNamespace(
        uuid="attachment-1"
    )
    mock_api.src_app_api_attachments_initiate_large_attachment_upload.return_value = (
        SimpleNamespace(
            upload_url="https://example.com/signed-upload",
            storage_key="attachments/large/attachment-1/model.glb",
        )
    )
    mock_api.src_app_api_assets_set_glb_from_attachment.return_value = SimpleNamespace(
        uuid="asset-1"
    )

    response = MagicMock()
    response.status = 200
    pool_manager = MagicMock()
    pool_manager.request.return_value = response

    with patch("urllib3.PoolManager", return_value=pool_manager):
        result = manager._upload_glb_via_attachment("asset-1", str(file_path))

    assert result.uuid == "asset-1"
    mock_api.src_app_api_attachments_complete_large_attachment_upload.assert_called_once()
    mock_api.src_app_api_assets_set_glb_from_attachment.assert_called_once()
