"""``cyberwave.exceptions`` and ``cyberwave.rest.exceptions`` must name the same
classes, not two unrelated copies of them.

``python-sdk-gen.sh`` rewrites the generated module into a re-export to make that
true. ``cyberwave/rest/`` is gitignored and regenerated from a live backend, so
these tests are what catch a regen that drops the rewrite step.
"""

import json
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import (
    CyberwaveAPIError,
    CyberwaveInsufficientCreditsError,
)
from cyberwave.resources import BaseResourceManager

from _rest_probe import rest_module_is_real

DECOY_KEY = "cw_decoy_key_000000000000000000"

# Every name the generated client imports from ``cyberwave.rest.exceptions``,
# plus the Cyberwave-only additions the re-export carries over.
SHARED_NAMES = [
    "OpenApiException",
    "ApiTypeError",
    "ApiValueError",
    "ApiAttributeError",
    "ApiKeyError",
    "ApiException",
    "BadRequestException",
    "UnauthorizedException",
    "ForbiddenException",
    "NotFoundException",
    "PaymentRequiredException",
    "ConflictException",
    "UnprocessableEntityException",
    "ServiceException",
]


# Presence, not importability, and asked in exactly one place — see
# ``_rest_probe``. Importing here would turn "the package is there but broken" —
# a shim referencing a name ``cyberwave/exceptions.py`` no longer defines, which
# is the failure these tests exist to report — into a silent skip.
requires_rest = pytest.mark.skipif(
    not rest_module_is_real(), reason="generated cyberwave.rest package not available"
)


# --------------------------------------------------------------------------
# One class per name
# --------------------------------------------------------------------------


@requires_rest
@pytest.mark.parametrize("name", SHARED_NAMES)
def test_rest_exceptions_are_the_same_objects(name: str) -> None:
    import cyberwave.exceptions as sdk_exceptions
    import cyberwave.rest.exceptions as rest_exceptions

    # Default so a name the generated module never had (PaymentRequiredException)
    # reports this message rather than an AttributeError.
    assert getattr(rest_exceptions, name, None) is getattr(sdk_exceptions, name), (
        f"{name} is missing or a second, unrelated class in "
        "cyberwave.rest.exceptions — python-sdk-gen.sh did not rewrite the "
        "generated module"
    )


@requires_rest
def test_rest_package_reexports_resolve_to_the_same_objects() -> None:
    """``cyberwave.rest.__init__`` re-exports a subset by name; those are what
    a user reaching for ``from cyberwave.rest import ApiException`` gets."""
    import cyberwave.exceptions as sdk_exceptions
    import cyberwave.rest as rest

    for name in (
        "OpenApiException",
        "ApiTypeError",
        "ApiValueError",
        "ApiKeyError",
        "ApiAttributeError",
        "ApiException",
    ):
        assert getattr(rest, name) is getattr(sdk_exceptions, name), name


@requires_rest
def test_transport_raises_the_class_client_py_catches() -> None:
    """The identity check that matters: what ``from_response`` raises against
    what ``client.py``'s ``except`` was compiled to match."""
    from cyberwave.exceptions import UnauthorizedException
    from cyberwave.rest.exceptions import ApiException

    response = _fake_response(401, {"detail": "Invalid token."})

    with pytest.raises(UnauthorizedException):
        ApiException.from_response(http_resp=response, body=None, data=None)


# --------------------------------------------------------------------------
# The friendly 401 handler, now that it is reachable
# --------------------------------------------------------------------------


def _fake_response(status: int, body: dict) -> MagicMock:
    payload = json.dumps(body).encode()
    response = MagicMock()
    response.status = status
    response.data = payload
    response.reason = "error"
    response.headers = {"content-type": "application/json"}
    response.getheaders.return_value = response.headers
    return response


class _RestOnlyClient:
    """The REST-wiring slice of :class:`~cyberwave.client.Cyberwave`.

    Building a full client pulls in the manager import chain; these tests only
    need the closures in ``_setup_rest_client`` and the 401 wrapper they feed.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        from cyberwave.client import Cyberwave
        from cyberwave.config import CyberwaveConfig

        self._setup_rest_client = Cyberwave._setup_rest_client.__get__(self)
        self._wrap_api_methods = Cyberwave._wrap_api_methods.__get__(self)
        self._create_wrapped_method = Cyberwave._create_wrapped_method.__get__(self)
        self.config = CyberwaveConfig(
            api_key=api_key, base_url="https://api.example.invalid"
        )
        self._setup_rest_client()

    def respond_with(self, status: int, body: dict) -> None:
        self._api_client.rest_client.request = MagicMock(
            return_value=_fake_response(status, body)
        )


@requires_rest
def test_rejected_key_reaches_the_caller_as_guidance_not_a_bare_401() -> None:
    """End to end through the transport — before the fix this raised
    ``rest.UnauthorizedException`` and printed only ``(401)``."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(401, {"detail": "Invalid token."})

    with pytest.raises(CyberwaveAPIError) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    message = str(excinfo.value)
    assert "Authentication failed" in message
    assert "https://cyberwave.com/profile" in message
    assert "export CYBERWAVE_API_KEY" in message
    assert excinfo.value.status_code == 401
    # Still masked now that the handler is live.
    assert DECOY_KEY not in message
    assert "Authorization header: Bearer cw_decoy…" in message


@requires_rest
def test_rejected_key_without_credentials_says_so() -> None:
    client = _RestOnlyClient(None)
    client.respond_with(401, {"detail": "Authentication required."})

    with pytest.raises(CyberwaveAPIError) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    assert "No authentication credentials were provided" in str(excinfo.value)


@requires_rest
def test_the_transport_exception_survives_as_the_cause() -> None:
    """Kept off the friendly message, but still reachable for debugging."""
    from cyberwave.exceptions import UnauthorizedException

    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(401, {"detail": "Invalid token."})

    with pytest.raises(CyberwaveAPIError) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    cause = excinfo.value.__cause__
    assert isinstance(cause, UnauthorizedException)
    assert cause.request_headers["Authorization"] == "Bearer cw_decoy…"
    assert DECOY_KEY not in str(cause)
    assert DECOY_KEY not in repr(vars(cause))


# --------------------------------------------------------------------------
# Manager error handling must not undo the translation
# --------------------------------------------------------------------------


@requires_rest
def test_manager_preserves_the_401_translation() -> None:
    """``_handle_error`` keys off ``status``, which a translated error lacks, so
    re-wrapping used to reset ``status_code`` to None."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(401, {"detail": "Invalid token."})
    manager = BaseResourceManager(client.api)

    with pytest.raises(CyberwaveAPIError) as raised:
        client.api.src_app_api_alerts_list_alerts()

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager._handle_error(raised.value, "list alerts")

    assert excinfo.value.status_code == 401
    assert "Authentication failed" in str(excinfo.value)


@requires_rest
def test_manager_preserves_the_insufficient_credits_subclass() -> None:
    """Same path: a generic re-wrap discards the balance fields and the subclass
    callers catch on."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(402, {"detail": "Payment required balance=-5.0"})
    manager = BaseResourceManager(client.api)

    with pytest.raises(CyberwaveInsufficientCreditsError) as raised:
        client.api.src_app_api_alerts_list_alerts()

    with pytest.raises(CyberwaveInsufficientCreditsError) as excinfo:
        manager._handle_error(raised.value, "list alerts")

    assert excinfo.value.balance == -5.0
    assert excinfo.value.status_code == 402


@requires_rest
def test_manager_names_the_operation_on_a_translated_error() -> None:
    """Re-raising preserves the subclass, but the ``Failed to ...`` context that
    re-wrapping used to add still has to reach the user — otherwise a 402 from a
    bulk helper says only ``Insufficient credits`` with no idea which call."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(402, {"detail": "Payment required balance=-5.0"})
    manager = BaseResourceManager(client.api)

    with pytest.raises(CyberwaveInsufficientCreditsError) as raised:
        client.api.src_app_api_alerts_list_alerts()

    with pytest.raises(CyberwaveInsufficientCreditsError) as excinfo:
        manager._handle_error(raised.value, "get environment")

    error = excinfo.value
    assert str(error).startswith("Failed to get environment: Insufficient credits")
    assert error.operation == "get environment"
    # The re-raise still has to hold on to everything it was fixed to preserve.
    assert error.balance == -5.0
    assert error.status_code == 402
    assert DECOY_KEY not in str(error)


@requires_rest
def test_nested_handle_error_names_the_call_the_user_made() -> None:
    """``get_device`` calls ``list_devices``, so one failure passes through
    ``_handle_error`` twice on the same object. The outer operation is the one
    the caller invoked; prefixing blindly would stack them."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(402, {"detail": "Payment required balance=-5.0"})
    manager = BaseResourceManager(client.api)

    with pytest.raises(CyberwaveInsufficientCreditsError) as raised:
        client.api.src_app_api_alerts_list_alerts()

    error = raised.value
    for operation in ("list devices for twin abc", "get device xyz for twin abc"):
        with pytest.raises(CyberwaveInsufficientCreditsError) as excinfo:
            manager._handle_error(error, operation)
        error = excinfo.value

    message = str(error)
    assert message.count("Failed to ") == 1
    assert message.startswith("Failed to get device xyz for twin abc: ")
    assert "list devices" not in message
    assert error.balance == -5.0


@requires_rest
def test_nested_handle_error_does_not_stack_on_untranslated_errors() -> None:
    """The 402 above starts as a ``CyberwaveAPIError`` and survives as one
    object, so the stash is already there. A 404 does not — the inner hop builds
    a new one, which is the path every status except 401/402 actually takes."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(404, {"detail": "Not found."})
    manager = BaseResourceManager(client.api)

    with pytest.raises(Exception) as raised:
        client.api.src_app_api_alerts_list_alerts()

    error: Exception = raised.value
    for operation in ("list devices for twin abc", "get device xyz for twin abc"):
        with pytest.raises(CyberwaveAPIError) as excinfo:
            manager._handle_error(error, operation)
        error = excinfo.value

    message = str(error)
    assert message.count("Failed to ") == 1
    assert message.startswith("Failed to get device xyz for twin abc: ")
    assert "list devices" not in message
    assert DECOY_KEY not in message


@requires_rest
@pytest.mark.parametrize("status", [401, 402, 404, 500])
def test_operation_is_stamped_on_every_path(status: int) -> None:
    """``.operation`` is public, so it cannot exist only for the statuses the
    transport happens to translate."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(status, {"detail": "nope"})
    manager = BaseResourceManager(client.api)

    with pytest.raises(Exception) as raised:
        client.api.src_app_api_alerts_list_alerts()

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager._handle_error(raised.value, "get environment")

    assert excinfo.value.operation == "get environment"


def test_operation_defaults_to_none_outside_a_manager_call() -> None:
    """Raised directly, with no manager in the path — the attribute still exists."""
    assert CyberwaveAPIError("boom").operation is None


# --------------------------------------------------------------------------
# Rendering differences between the two former copies
# --------------------------------------------------------------------------


def test_str_leaves_request_headers_to_cyberwave_api_error() -> None:
    """Keeping this copy's rendering would have doubled it — ``_handle_error``
    interpolates ``str(inner)`` into a ``CyberwaveAPIError`` that renders the
    same dict. One renderer; the value stays on the attribute."""
    from cyberwave.exceptions import UnauthorizedException

    error = UnauthorizedException(status=401, reason="Unauthorized")
    error.request_headers = {"Authorization": "Bearer cw_decoy…"}

    assert "Request headers" not in str(error)
    assert error.request_headers == {"Authorization": "Bearer cw_decoy…"}


@requires_rest
def test_the_401_message_names_the_key_exactly_once() -> None:
    """The manager path was pinned at 404; this is the 401 branch, which builds
    its own message. It already names the Authorization header, so forwarding
    the dict too made ``__str__`` append it again — the same doubling, on the
    one path this whole change exists to make reachable."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(401, {"detail": "Invalid token."})

    with pytest.raises(CyberwaveAPIError) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    message = str(excinfo.value)
    assert message.count("Authorization header: Bearer cw_decoy…") == 1
    assert "Request headers" not in message
    assert DECOY_KEY not in message
    # Still reachable for anyone who wants the full masked dict.
    cause = excinfo.value.__cause__
    assert cause is not None
    assert cause.request_headers["Authorization"] == "Bearer cw_decoy…"


@requires_rest
def test_manager_errors_render_request_headers_exactly_once() -> None:
    """Not just noise: doubling pushed a plain 404 to ~490 characters, past the
    truncation on ``Alert.metadata.technical_detail``."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(404, {"detail": "Not found."})
    manager = BaseResourceManager(client.api)

    with pytest.raises(Exception) as raised:
        client.api.src_app_api_alerts_list_alerts()

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager._handle_error(raised.value, "get environment")

    message = str(excinfo.value)
    assert message.count("Request headers") == 1
    assert "Not found." in message
    assert DECOY_KEY not in message


def test_from_response_maps_402_to_payment_required() -> None:
    """The generated copy mapped 402 to a bare ``ApiException``."""
    from cyberwave.exceptions import ApiException, PaymentRequiredException

    with pytest.raises(PaymentRequiredException):
        ApiException.from_response(
            http_resp=_fake_response(402, {"detail": "no credits"}),
            body=None,
            data=None,
        )


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, "BadRequestException"),
        (401, "UnauthorizedException"),
        (403, "ForbiddenException"),
        (404, "NotFoundException"),
        (409, "ConflictException"),
        (422, "UnprocessableEntityException"),
        (500, "ServiceException"),
    ],
)
def test_from_response_status_mapping(status: int, expected: str) -> None:
    import cyberwave.exceptions as sdk_exceptions

    exc_class: Any = getattr(sdk_exceptions, expected)
    with pytest.raises(exc_class):
        sdk_exceptions.ApiException.from_response(
            http_resp=_fake_response(status, {"detail": "nope"}),
            body=None,
            data=None,
        )
