from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cyberwave.resources import AttachmentManager
from cyberwave.rest import AttachmentCreateSchema


def _make_manager() -> tuple[AttachmentManager, MagicMock]:
    mock_api = MagicMock()
    manager = AttachmentManager(mock_api)
    return manager, mock_api


def test_attachment_manager_get_calls_attachment_endpoint():
    manager, mock_api = _make_manager()
    expected = SimpleNamespace(uuid="attachment-1")
    mock_api.src_app_api_attachments_get_attachment.return_value = expected

    result = manager.get("attachment-1")

    assert result is expected
    mock_api.src_app_api_attachments_get_attachment.assert_called_once_with(
        "attachment-1"
    )


def test_attachment_manager_update_posts_schema_payload():
    manager, mock_api = _make_manager()
    expected = SimpleNamespace(uuid="attachment-1")
    mock_api.src_app_api_attachments_update_attachment.return_value = expected

    result = manager.update(
        "attachment-1",
        metadata={"analysis": {"status": "completed"}},
        twin_uuid="twin-1",
    )

    assert result is expected
    attachment_uuid, payload = (
        mock_api.src_app_api_attachments_update_attachment.call_args.args
    )
    assert attachment_uuid == "attachment-1"
    assert isinstance(payload, AttachmentCreateSchema)
    assert payload.to_dict() == {
        "asset_uuid": None,
        "twin_uuid": "twin-1",
        "metadata": {"analysis": {"status": "completed"}},
    }


def test_attachment_manager_upload_image_from_data_url_uses_upload_endpoint():
    manager, mock_api = _make_manager()
    expected = SimpleNamespace(uuid="attachment-1")
    mock_api.src_app_api_attachments_upload_attachment.return_value = expected

    result = manager.upload_image_from_url(
        "attachment-1",
        "data:image/png;base64,aGVsbG8=",
    )

    assert result is expected
    attachment_uuid, file_payload = (
        mock_api.src_app_api_attachments_upload_attachment.call_args.args
    )
    assert attachment_uuid == "attachment-1"
    assert file_payload == ("attachment-1_annotated.png", b"hello")


def test_attachment_manager_upload_image_from_remote_url_fetches_bytes():
    manager, mock_api = _make_manager()
    expected = SimpleNamespace(uuid="attachment-1")
    mock_api.src_app_api_attachments_upload_attachment.return_value = expected

    response = MagicMock()
    response.status = 200
    response.data = b"jpeg-bytes"
    response.headers = {"Content-Type": "image/jpeg; charset=binary"}
    pool_manager = MagicMock()
    pool_manager.request.return_value = response

    with patch("urllib3.PoolManager", return_value=pool_manager):
        result = manager.upload_image_from_url(
            "attachment-1",
            "https://example.com/annotated.jpg",
        )

    assert result is expected
    pool_manager.request.assert_called_once()
    attachment_uuid, file_payload = (
        mock_api.src_app_api_attachments_upload_attachment.call_args.args
    )
    assert attachment_uuid == "attachment-1"
    assert file_payload == ("attachment-1_annotated.jpg", b"jpeg-bytes")
