"""Offline coverage for the public Cyberwave context-manager behavior."""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave import Cyberwave


def _client_with_mocked_resources():
    client = Cyberwave.__new__(Cyberwave)
    client._mqtt_client = MagicMock()
    client._data_bus = MagicMock()
    client._data_backend = None
    return client


def test_context_manager_returns_client_and_disconnects_on_normal_exit():
    client = _client_with_mocked_resources()

    with patch.object(client, "disconnect", wraps=client.disconnect) as disconnect:
        with client as entered:
            assert entered is client

    disconnect.assert_called_once_with()
    client._mqtt_client.disconnect.assert_called_once_with()
    assert client._data_bus is None


def test_context_manager_disconnects_and_propagates_block_error():
    client = _client_with_mocked_resources()
    error = RuntimeError("block failed")

    with patch.object(client, "disconnect", wraps=client.disconnect) as disconnect:
        with pytest.raises(RuntimeError) as caught:
            with client:
                raise error

    assert caught.value is error
    disconnect.assert_called_once_with()
    client._mqtt_client.disconnect.assert_called_once_with()
    assert client._data_bus is None
