"""Tests for the SDK identity headers attached to outbound REST requests.

These headers let the backend attribute API traffic to an SDK version cohort
(``backend_api_activity``) without any per-call payload changes.
"""

from cyberwave._version import get_version
from cyberwave.client import (
    _SDK_USER_AGENT,
    _SDK_VERSION,
    _SDK_VERSION_HEADER,
    _apply_sdk_identity_headers,
)


def test_version_constants_match_package_version() -> None:
    assert _SDK_VERSION == get_version()
    assert _SDK_USER_AGENT == f"cyberwave-python/{get_version()}"


def test_applies_identity_headers_to_empty_dict() -> None:
    headers = _apply_sdk_identity_headers({})
    assert headers["User-Agent"] == _SDK_USER_AGENT
    assert headers[_SDK_VERSION_HEADER] == _SDK_VERSION


def test_preserves_caller_user_agent() -> None:
    headers = _apply_sdk_identity_headers({"User-Agent": "cyberwave-cli/1.2.3"})
    # A caller-provided User-Agent (e.g. the CLI) must not be clobbered.
    assert headers["User-Agent"] == "cyberwave-cli/1.2.3"
    # The dedicated version header is still added for reliable machine parsing.
    assert headers[_SDK_VERSION_HEADER] == _SDK_VERSION


def test_preserves_caller_user_agent_case_insensitively() -> None:
    headers = _apply_sdk_identity_headers({"user-agent": "custom/9.9"})
    assert headers["user-agent"] == "custom/9.9"
    assert "User-Agent" not in headers


def test_preserves_existing_sdk_version_header() -> None:
    headers = _apply_sdk_identity_headers({_SDK_VERSION_HEADER: "override"})
    assert headers[_SDK_VERSION_HEADER] == "override"
