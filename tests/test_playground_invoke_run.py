"""Tests for :func:`cyberwave.models.playground.invoke_mlmodel_run`.

Regression coverage for the transport-receiver bug: ``param_serialize`` /
``call_api`` / ``response_deserialize`` live on the generated ``ApiClient``,
but ``invoke_mlmodel_run`` was handed the ``DefaultApi`` operations facade that
merely *holds* one. Every playground and product run raised
``AttributeError: 'DefaultApi' object has no attribute 'param_serialize'``
before the first byte went over the wire.

The tests stub the transport so they run without network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cyberwave.models.playground import invoke_mlmodel_run

_UUID = "11111111-1111-1111-1111-111111111111"


class _FakeResponse:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self) -> None:
        self.read_calls += 1


class _FakeTransport:
    """Stands in for ``cyberwave.rest.api_client.ApiClient``."""

    def __init__(self) -> None:
        self.serialize_kwargs: dict[str, Any] = {}
        self.deserialize_kwargs: dict[str, Any] = {}
        self.response = _FakeResponse()

    def param_serialize(self, **kwargs: Any) -> tuple[str, ...]:
        self.serialize_kwargs = kwargs
        return ("POST", "/url", {}, None)

    def call_api(self, *args: Any) -> _FakeResponse:
        self.called_with = args
        return self.response

    def response_deserialize(self, **kwargs: Any) -> SimpleNamespace:
        self.deserialize_kwargs = kwargs
        return SimpleNamespace(data={"status": "completed"})


class _FakeDefaultApi:
    """Stands in for ``DefaultApi`` — holds a transport, does not *be* one.

    Deliberately exposes none of the transport methods, mirroring the real
    generated class, so passing this straight through would raise.
    """

    def __init__(self, transport: _FakeTransport) -> None:
        self.api_client = transport


class _Schema:
    def to_dict(self) -> dict[str, Any]:
        return {"prompt": "hello"}


def test_accepts_default_api_and_hops_to_api_client() -> None:
    """The DefaultApi facade must be unwrapped to its transport."""
    transport = _FakeTransport()

    result = invoke_mlmodel_run(
        _FakeDefaultApi(transport),
        uuid=_UUID,
        schema=_Schema(),
    )

    assert result == {"status": "completed"}
    assert transport.serialize_kwargs["method"] == "POST"
    assert transport.serialize_kwargs["path_params"] == {"uuid": _UUID}
    assert transport.serialize_kwargs["body"] == {"prompt": "hello"}
    assert transport.response.read_calls == 1
    assert transport.deserialize_kwargs["response_types_map"] == {
        "200": "MLModelRunResultSchema",
        "202": "MLModelRunQueuedSchema",
    }


def test_accepts_a_bare_api_client() -> None:
    """A caller holding the transport directly must still work.

    ``getattr(api, "api_client", api)`` keeps this working across
    generated-client versions that shuffle the layering.
    """
    transport = _FakeTransport()

    result = invoke_mlmodel_run(transport, uuid=_UUID, schema=_Schema())

    assert result == {"status": "completed"}
    assert transport.serialize_kwargs["path_params"] == {"uuid": _UUID}


def test_always_routes_to_the_product_run_endpoint() -> None:
    """The SDK must never reach ``/playground/run``.

    That endpoint is unauthenticated, metered against playground quota rather
    than credits, and browser-only server-side (it rejects callers sending no
    Origin). ``/run`` is the authenticated, credit-gated route for SDK,
    workflow, and automation callers.
    """
    transport = _FakeTransport()

    invoke_mlmodel_run(_FakeDefaultApi(transport), uuid=_UUID, schema=_Schema())

    assert transport.serialize_kwargs["resource_path"] == "/api/v1/mlmodels/{uuid}/run"


def test_offers_no_switch_back_to_the_playground_endpoint() -> None:
    """There must be no parameter that re-routes the SDK to the playground.

    Guards against reintroducing the ``playground`` / ``product`` flag whose
    default sent every SDK run to the browser-only endpoint.
    """
    import inspect

    params = inspect.signature(invoke_mlmodel_run).parameters
    assert "playground" not in params
    assert "product" not in params


def test_serializes_a_schema_without_to_dict() -> None:
    """Pydantic-style schemas fall back to ``model_dump(exclude_none=True)``."""
    transport = _FakeTransport()

    class _Pydanticish:
        def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
            assert exclude_none is True
            return {"prompt": "hi"}

    invoke_mlmodel_run(
        _FakeDefaultApi(transport),
        uuid=_UUID,
        schema=_Pydanticish(),
    )

    assert transport.serialize_kwargs["body"] == {"prompt": "hi"}
