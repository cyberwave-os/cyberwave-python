from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cyberwave import Cyberwave


def test_client_twin_preserves_list_errors_instead_of_masking_them():
    client = Cyberwave(base_url="http://localhost:8000", api_key="test_key")
    client.config.environment_id = "env-uuid"
    client.assets.get_by_registry_id = lambda _asset_key: SimpleNamespace(
        uuid="asset-uuid",
        registry_id="the-robot-studio/so101",
    )
    expected_error = RuntimeError("list twins failed")
    client.twins.list = lambda environment_id=None: (_ for _ in ()).throw(expected_error)

    with patch("cyberwave.client.create_twin") as mock_create_twin:
        with pytest.raises(RuntimeError, match="list twins failed"):
            client.twin("the-robot-studio/so101")

    mock_create_twin.assert_not_called()
