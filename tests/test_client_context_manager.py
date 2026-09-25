"""Offline coverage for the public Cyberwave context-manager behavior."""

from unittest.mock import patch

import pytest

from cyberwave import Cyberwave


def test_context_manager_returns_client_and_calls_disconnect_on_normal_exit():
    client = Cyberwave.__new__(Cyberwave)

    with patch.object(client, "disconnect") as disconnect:
        with client as entered:
            assert entered is client

    disconnect.assert_called_once_with()


def test_context_manager_calls_disconnect_and_propagates_block_error():
    client = Cyberwave.__new__(Cyberwave)
    error = RuntimeError("block failed")

    with patch.object(client, "disconnect") as disconnect:
        with pytest.raises(RuntimeError) as caught:
            with client:
                raise error

    assert caught.value is error
    disconnect.assert_called_once_with()
