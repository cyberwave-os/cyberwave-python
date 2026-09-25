"""The SDK must never write the caller's own API key into error text, logs,
exception attributes or alert metadata (CYB-3374).

The SDK attaches a copy of its own outgoing request headers to failed requests,
so the fix — and these tests — target the moment of attachment: mask the value
on ``exc.__dict__`` and every downstream consumer is covered at once. Section
headers below map to the site numbers in the ticket.
"""

import json
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from cyberwave._error_metadata import (
    format_error_metadata,
    mask_header_value,
    mask_sensitive_headers,
    redact,
)
from cyberwave.config import CyberwaveConfig
from cyberwave.exceptions import CyberwaveAPIError, CyberwaveInsufficientCreditsError
from cyberwave.resources import BaseResourceManager

from _rest_probe import rest_module_is_real

# Never a live credential. Long enough that a masked prefix is a strict subset.
DECOY_KEY = "cw_decoy_key_000000000000000000"


requires_rest = pytest.mark.skipif(
    not rest_module_is_real(), reason="generated cyberwave.rest package not available"
)


# --------------------------------------------------------------------------
# Masking primitives
# --------------------------------------------------------------------------


def test_mask_sensitive_headers_masks_authorization() -> None:
    masked = mask_sensitive_headers(
        {
            "Accept": "application/json",
            "User-Agent": "cyberwave-python/0.6.4",
            "Authorization": f"Bearer {DECOY_KEY}",
            "X-Cyberwave-SDK-Version": "0.6.4",
        }
    )

    assert DECOY_KEY not in str(masked)
    assert masked["Authorization"] == "Bearer cw_decoy…"
    # Non-credential headers stay intact — they are why this dict is useful.
    assert masked["Accept"] == "application/json"
    assert masked["User-Agent"] == "cyberwave-python/0.6.4"


@pytest.mark.parametrize(
    "name", ["Authorization", "authorization", "X-API-Key", "Cookie", "X-Auth-Token"]
)
def test_mask_sensitive_headers_covers_credential_headers(name: str) -> None:
    masked = mask_sensitive_headers({name: DECOY_KEY})
    assert DECOY_KEY not in str(masked)


def test_mask_header_value_keeps_scheme_and_prefix() -> None:
    assert mask_header_value(f"Bearer {DECOY_KEY}") == "Bearer cw_decoy…"


def test_mask_header_value_hides_short_credentials_entirely() -> None:
    """A prefix of a short secret gives away most of it, so show none of it."""
    assert mask_header_value("Bearer abc") == "Bearer …"
    assert "abc" not in mask_header_value("Bearer abc")


def test_mask_header_value_hides_a_cookie_jar_entirely() -> None:
    """A cookie jar leads with a value, not a scheme. Splitting on the first
    space would re-emit the whole first cookie as if it were the scheme."""
    jar = f"sessionid={DECOY_SECRET}; csrftoken=ABCDEF"
    assert mask_header_value(jar, header="Cookie") == "…"
    assert DECOY_SECRET not in mask_header_value(jar, header="Cookie")


def test_mask_header_value_hides_structured_scheme_payloads() -> None:
    """8 characters of base64 decode to 6 real bytes of ``user:pass``."""
    masked = mask_header_value("Basic dXNlcjpwYXNzd29yZA==")
    # The scheme still answers "what did I send?"; none of the payload does.
    assert masked == "Basic …"
    assert "dXNlcjpw" not in masked


def test_mask_header_value_hides_a_credential_that_merely_contains_a_space() -> None:
    """The first token is echoed only when it is a real scheme. Anything else
    before a space is part of the credential, and echoing it emitted the secret
    verbatim — a fail-open branch inside a function whose contract is to fail
    closed. ``_headers`` is a public parameter on every generated method, so the
    value is caller-controlled."""
    value = f"{DECOY_SECRET} trailing"
    assert mask_header_value(value, header="X-API-Key") == "…"
    assert DECOY_SECRET not in mask_header_value(value, header="X-API-Key")


def test_mask_sensitive_headers_hides_a_spaced_credential() -> None:
    """Reachable through the real entry point, not just a direct call."""
    masked = mask_sensitive_headers({"X-API-Key": f"{DECOY_SECRET} trailing"})
    assert DECOY_SECRET not in repr(masked)


@pytest.mark.parametrize("scheme", ["Basic", "Digest", "Negotiate", "NTLM"])
def test_mask_header_value_still_names_recognised_schemes(scheme: str) -> None:
    """Naming the scheme answers "what did I send?" and leaks nothing, so the
    fail-closed rule above must not swallow it."""
    assert mask_header_value(f"{scheme} {DECOY_SECRET}") == f"{scheme} …"


def test_mask_sensitive_headers_fails_closed_on_non_string_values() -> None:
    """The type check must gate emission, not masking: a bytes credential is
    still a credential, and repr() of it lands in the same places."""
    for value in (DECOY_SECRET.encode(), bytearray(DECOY_SECRET.encode()), 12345):
        masked = mask_sensitive_headers({"X-API-Key": value})
        assert DECOY_SECRET not in repr(masked), f"leaked for {type(value).__name__}"


def test_mask_sensitive_headers_matches_byte_string_keys() -> None:
    """``str(b"Authorization")`` is ``"b'Authorization'"``, which matches no
    entry in SENSITIVE_HEADERS."""
    masked = mask_sensitive_headers({b"Authorization": f"Bearer {DECOY_KEY}"})
    assert DECOY_KEY not in repr(masked)


def test_mask_sensitive_headers_returns_independent_copy() -> None:
    source = {"Authorization": f"Bearer {DECOY_KEY}"}
    masked = mask_sensitive_headers(source)
    source.clear()
    assert masked["Authorization"] == "Bearer cw_decoy…"


def test_mask_sensitive_headers_handles_empty_input() -> None:
    assert mask_sensitive_headers(None) == {}
    assert mask_sensitive_headers({}) == {}


# --------------------------------------------------------------------------
# Site 3: config repr
# --------------------------------------------------------------------------


def test_config_repr_hides_credentials() -> None:
    config = CyberwaveConfig(
        api_key=DECOY_KEY, token="tok_decoy_secret", mqtt_password="pw_decoy_secret"
    )

    rendered = repr(config)

    assert DECOY_KEY not in rendered
    assert "tok_decoy_secret" not in rendered
    assert "pw_decoy_secret" not in rendered
    # Still readable for the fields that make a repr worth having.
    assert "base_url=" in rendered


def test_config_credentials_remain_accessible() -> None:
    """repr=False hides the field from __repr__ only; the SDK still needs it."""
    config = CyberwaveConfig(api_key=DECOY_KEY, mqtt_password="pw_decoy_secret")
    assert config.api_key == DECOY_KEY
    assert config.mqtt_password == "pw_decoy_secret"


# The same auto-generated-repr leak in the SDK's other credential-bearing
# configs. A plain @dataclass prints every field, and so does a pydantic model.
DECOY_SECRET = "s3cret_decoy_value"


def _sibling_configs() -> list[tuple[str, object, list[str]]]:
    from cyberwave.edge.amr import AdapterConfig
    from cyberwave.edge.config import EdgeNodeConfig
    from cyberwave.manifest.schema import ManifestSchema
    from cyberwave.sensor.config import EdgeCameraConfig

    return [
        (
            "EdgeCameraConfig",
            EdgeCameraConfig(
                source="rtsp://cam.invalid/stream",
                username="admin",
                password=DECOY_SECRET,
            ),
            ["password"],
        ),
        (
            "EdgeNodeConfig",
            EdgeNodeConfig(cyberwave_api_key=DECOY_SECRET),
            ["cyberwave_api_key"],
        ),
        (
            "AdapterConfig",
            AdapterConfig(password=DECOY_SECRET, api_key=DECOY_SECRET),
            ["password", "api_key"],
        ),
        (
            "ManifestSchema",
            ManifestSchema(version="1", name="probe", mqtt_password=DECOY_SECRET),
            ["mqtt_password"],
        ),
    ]


@pytest.mark.parametrize(
    "name,config,fields",
    _sibling_configs(),
    ids=[name for name, _, _ in _sibling_configs()],
)
def test_sibling_config_reprs_hide_credentials(
    name: str, config: Any, fields: list[str]
) -> None:
    assert DECOY_SECRET not in repr(config), f"{name}.__repr__ leaks a credential"
    # Pydantic renders __str__ separately from __repr__.
    assert DECOY_SECRET not in str(config), f"{name}.__str__ leaks a credential"
    # The value is hidden from rendering only — the SDK still has to use it.
    for field_name in fields:
        assert getattr(config, field_name) == DECOY_SECRET


def test_manifest_serialization_still_carries_the_password() -> None:
    """repr=False must not turn into a silent data-loss bug: a manifest round
    trip still has to hand the broker password to the driver."""
    from cyberwave.manifest.schema import ManifestSchema

    manifest = ManifestSchema(version="1", name="probe", mqtt_password=DECOY_SECRET)
    assert manifest.model_dump()["mqtt_password"] == DECOY_SECRET


# --------------------------------------------------------------------------
# Sites 1 and 2: exception text, and the SDK logging its own swallowed errors
# --------------------------------------------------------------------------


def test_api_error_str_does_not_leak_masked_headers() -> None:
    error = CyberwaveAPIError(
        "Failed to list environments",
        status_code=401,
        request_headers=mask_sensitive_headers(
            {"Authorization": f"Bearer {DECOY_KEY}"}
        ),
    )
    assert DECOY_KEY not in str(error)


def test_handle_error_propagates_only_masked_headers() -> None:
    """``BaseResourceManager._handle_error`` lifts ``request_headers`` off the
    inner exception; it must not be able to pick up an unmasked value."""
    manager = BaseResourceManager(MagicMock())
    inner = MagicMock()
    inner.status = 401
    inner.body = None
    inner.request_headers = mask_sensitive_headers(
        {"Authorization": f"Bearer {DECOY_KEY}"}
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        manager._handle_error(inner, "list environments")

    assert DECOY_KEY not in str(excinfo.value)
    assert DECOY_KEY not in repr(vars(excinfo.value))


def test_insufficient_credits_error_does_not_leak() -> None:
    error = CyberwaveInsufficientCreditsError(
        "Insufficient credits (balance: -3.0)",
        request_headers=mask_sensitive_headers(
            {"Authorization": f"Bearer {DECOY_KEY}"}
        ),
        balance=-3.0,
    )
    assert DECOY_KEY not in str(error)
    assert DECOY_KEY not in repr(vars(error))


def test_logging_a_swallowed_api_error_does_not_leak(caplog: Any) -> None:
    """``scene.py`` and ``twin.py`` warn-and-continue on API failures, so the
    token would land in the log without ever surfacing to the user."""
    import logging

    error = CyberwaveAPIError(
        "Failed to list twins",
        status_code=500,
        request_headers=mask_sensitive_headers(
            {"Authorization": f"Bearer {DECOY_KEY}"}
        ),
    )

    logger = logging.getLogger("cyberwave.test")
    with caplog.at_level(logging.WARNING):
        logger.warning(f"Failed to load twins: {error}")
        logger.warning("metadata sync failed (non-fatal): %s", error)

    assert DECOY_KEY not in caplog.text


# --------------------------------------------------------------------------
# Site 4: alert metadata stored server-side
# --------------------------------------------------------------------------


def test_redact_strips_bearer_tokens_from_text() -> None:
    text = (
        f"boom {{'Authorization': 'Bearer {DECOY_KEY}', 'Accept': 'application/json'}}"
    )
    cleaned = redact(text)
    assert DECOY_KEY not in cleaned
    assert "Bearer <redacted>" in cleaned
    # The match stops at the closing quote of the dict repr.
    assert "'Accept': 'application/json'" in cleaned


@pytest.mark.parametrize(
    "rendered",
    [
        "{'X-API-Key': 'DECOY', 'Accept': 'application/json'}",
        '{"Authorization": "DECOY"}',
        "{'Cookie': 'sessionid=DECOY; csrftoken=ABCDEF'}",
        "{'X-Auth-Token': 'DECOY'}",
    ],
)
def test_redact_strips_non_bearer_header_dicts(rendered: str) -> None:
    """``format_error_metadata`` runs on exceptions from code the SDK does not
    own, which renders header dicts in shapes the Bearer rule cannot see."""
    cleaned = redact(rendered.replace("DECOY", DECOY_KEY))
    assert DECOY_KEY not in cleaned
    assert "<redacted>" in cleaned


def test_redact_preserves_an_already_masked_preview() -> None:
    """``mask_header_value`` keeps 8 characters on purpose so the user can tell
    which key was used. A 401 message carrying one passes through ``redact`` on
    the alert path, and an unbounded Bearer rule rewrote it to ``<redacted>``,
    undoing the very affordance the masking preserves."""
    masked = mask_header_value(f"Bearer {DECOY_KEY}")
    assert redact(masked) == masked


@pytest.mark.parametrize(
    "prose",
    [
        "the bearer of this token is unknown",
        "Bearer token missing from request",
    ],
)
def test_redact_leaves_prose_alone(prose: str) -> None:
    """``redact`` also runs on arbitrary driver text via edge-core's alert
    ``technical_detail``, so a case-insensitive rule ate the word after any
    occurrence of "bearer"."""
    assert redact(prose) == prose


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_redact_still_strips_a_real_bearer_token_after_bounding(scheme: str) -> None:
    """The bound is length, not case. Dropping ``IGNORECASE`` would also have
    silenced the prose case, but a bare lowercase ``bearer <token>`` in free text
    reaches no other rule — the header-dict rule only sees dict shapes — so that
    would have traded a cosmetic bug for a leak."""
    assert DECOY_KEY not in redact(f"{scheme} {DECOY_KEY}")


def test_redact_leaves_non_credential_headers_readable() -> None:
    assert redact("{'Accept': 'application/json'}") == "{'Accept': 'application/json'}"


def test_error_metadata_does_not_depend_on_truncation() -> None:
    """A terse backend ``detail`` puts the token inside the character limit, so
    truncation cannot be what protects this path."""
    error = CyberwaveAPIError(
        "Failed: 401",
        status_code=401,
        request_headers={"Authorization": f"Bearer {DECOY_KEY}"},
    )
    _, technical_detail = format_error_metadata(error, limit=400)
    assert DECOY_KEY not in technical_detail


# --------------------------------------------------------------------------
# Site 6 / root supply: what the transport actually stamps onto exceptions
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
    need the two closures in ``_setup_rest_client`` that record and stamp
    request headers, plus the 401 wrapper they feed.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        from cyberwave.client import Cyberwave

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
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 500])
def test_transport_stamps_only_masked_headers(status: int) -> None:
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(status, {"detail": "nope"})

    with pytest.raises(Exception) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    error = excinfo.value
    # 401 is translated by the friendly handler in client.py, which deliberately
    # keeps the header dict off its four-step message; the stamped transport
    # exception it chains is where the dict lives.
    carrier = error if getattr(error, "request_headers", None) else error.__cause__
    headers = getattr(carrier, "request_headers", None)
    assert headers, "transport should still attach request headers for debugging"
    assert headers["Authorization"] == "Bearer cw_decoy…"
    # Covers consumers that never call __str__ (vars(exc) capture, structured
    # logging of exc.__dict__) as well as those that do.
    for exc in (error, carrier):
        assert DECOY_KEY not in repr(vars(exc))
        assert DECOY_KEY not in str(exc)


@requires_rest
def test_transport_stamps_headers_when_attribute_preexists_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tripwire for the exception-hierarchy unification.

    ``cyberwave.exceptions.ApiException.__init__`` already sets
    ``request_headers = None``. Once the transport starts raising that class,
    a ``not hasattr`` guard would see the attribute, skip attachment, and leave
    every transport error without the debugging headers — silently, because
    every other assertion here is of the form *the key is not in here*, and an
    empty dict passes all of them.
    """
    from cyberwave.exceptions import ApiException
    from cyberwave.rest import ApiClient

    def _raise_with_preset_attribute(self, response_data=None, response_types_map=None):
        raise ApiException(status=500, reason="nope")

    monkeypatch.setattr(ApiClient, "response_deserialize", _raise_with_preset_attribute)

    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(500, {"detail": "nope"})

    with pytest.raises(ApiException) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    error = excinfo.value
    assert error.request_headers, "preset None must not suppress attachment"
    assert error.request_headers["Authorization"] == "Bearer cw_decoy…"
    assert DECOY_KEY not in repr(vars(error))


@requires_rest
@pytest.mark.parametrize(
    "caller_headers",
    [
        {"Cookie": f"sessionid={DECOY_SECRET}; csrftoken=ABCDEF"},
        {"X-API-Key": DECOY_SECRET.encode()},
        {"Proxy-Authorization": f"Basic {DECOY_SECRET}"},
    ],
    ids=["cookie-jar", "bytes-value", "basic-scheme"],
)
def test_transport_masks_caller_supplied_credential_headers(
    caller_headers: dict,
) -> None:
    """``_headers`` is a public parameter on every generated method, so a
    credential the SDK did not build off ``config.api_key`` reaches the same
    attachment point — in shapes an ``Authorization: Bearer`` rule cannot see."""
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(500, {"detail": "nope"})

    with pytest.raises(Exception) as excinfo:
        client.api.src_app_api_alerts_list_alerts(_headers=caller_headers)

    error = excinfo.value
    assert DECOY_SECRET not in repr(vars(error))
    # And through the manager path that renders it into a message.
    manager = BaseResourceManager(client.api)
    with pytest.raises(CyberwaveAPIError) as wrapped:
        manager._handle_error(error, "list alerts")
    assert DECOY_SECRET not in str(wrapped.value)
    assert DECOY_SECRET not in format_error_metadata(wrapped.value, limit=4000)[1]


@requires_rest
def test_transport_402_path_does_not_leak() -> None:
    client = _RestOnlyClient(DECOY_KEY)
    client.respond_with(402, {"detail": "Payment required balance=-5.0"})

    with pytest.raises(CyberwaveInsufficientCreditsError) as excinfo:
        client.api.src_app_api_alerts_list_alerts()

    error = excinfo.value
    assert error.request_headers["Authorization"] == "Bearer cw_decoy…"
    assert DECOY_KEY not in str(error)
    assert DECOY_KEY not in repr(vars(error))


# --------------------------------------------------------------------------
# The friendly 401 handler — the one consumer that wants part of the key.
# These drive the wrapper directly to isolate its masking; the end-to-end path
# through the transport is covered in test_exception_hierarchy.py.
# --------------------------------------------------------------------------


def _raise_unauthorized(request_headers: Optional[dict]) -> Any:
    from cyberwave.exceptions import UnauthorizedException

    error = UnauthorizedException(status=401, reason="Unauthorized")
    error.request_headers = request_headers
    raise error


@requires_rest
def test_friendly_401_shows_a_prefix_but_not_the_key() -> None:
    client = _RestOnlyClient(DECOY_KEY)
    masked = mask_sensitive_headers({"Authorization": f"Bearer {DECOY_KEY}"})
    wrapped = client._create_wrapped_method(lambda: _raise_unauthorized(masked))

    with pytest.raises(CyberwaveAPIError) as excinfo:
        wrapped()

    message = str(excinfo.value)
    assert DECOY_KEY not in message
    # The prefix is deliberately kept: it answers "which key did I send?".
    assert "Authorization header: Bearer cw_decoy…" in message


@requires_rest
def test_friendly_401_without_api_key_reports_absent_header() -> None:
    client = _RestOnlyClient(None)
    wrapped = client._create_wrapped_method(lambda: _raise_unauthorized({}))

    with pytest.raises(CyberwaveAPIError) as excinfo:
        wrapped()

    assert "No authentication credentials were provided" in str(excinfo.value)


@requires_rest
def test_friendly_401_reports_absent_header_when_authorization_missing() -> None:
    client = _RestOnlyClient(DECOY_KEY)
    wrapped = client._create_wrapped_method(
        lambda: _raise_unauthorized({"Accept": "application/json"})
    )

    with pytest.raises(CyberwaveAPIError) as excinfo:
        wrapped()

    assert "Authorization header: Not present" in str(excinfo.value)
