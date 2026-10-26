"""One hand-rolled authenticated REST call over the generated ``ApiClient``.

For endpoints the generated ``DefaultApi`` does not cover (or covers with
models the caller does not want): ``param_serialize`` + ``call_api``, then the
HTTP status is checked here -- ``call_api`` only raises for connection-level
failures, the generated client checks the status in ``response_deserialize``,
which this path skips.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .exceptions import ApiException, CyberwaveAPIError, CyberwaveError

AUTH = ["CustomTokenAuthentication"]

Query = Union[Mapping[str, Any], Sequence[Tuple[str, Any]]]


def decode_json(raw: Any) -> Any:
    """Response body -> JSON value; ``None`` when empty, the text when not JSON."""
    if raw is None or raw == b"" or raw == "":
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def request(
    api_client: Any,
    method: str,
    resource_path: str,
    *,
    path_params: Optional[Dict[str, Any]] = None,
    query: Optional[Query] = None,
    body: Any = None,
    allow_statuses: Iterable[int] = (),
    result_shape: Iterable[str] = (),
) -> Tuple[int, Any]:
    """One authenticated REST call; returns ``(http_status, decoded_json)``.

    *api_client* is the generated ``ApiClient`` (``client.api.api_client``).
    2xx is success. A status in ``allow_statuses`` is returned rather than
    raised -- when ``result_shape`` is given, only if the body carries those
    keys (e.g. the command gateway's 400/403/503 result bodies, as opposed to
    a framework error with the same status). Anything else raises
    :class:`CyberwaveAPIError`; a transport failure raises
    :class:`CyberwaveError`.
    """
    operation = f"{method} {resource_path}"
    if isinstance(query, Mapping):
        query_params = list(query.items())
    else:
        query_params = list(query or [])
    try:
        params = api_client.param_serialize(
            method=method,
            resource_path=resource_path,
            path_params=path_params or {},
            query_params=query_params,
            header_params=(
                {"Content-Type": "application/json"} if body is not None else None
            ),
            body=body,
            auth_settings=AUTH,
        )
        response = api_client.call_api(*params)
        response.read()
    except ApiException as exc:
        raise api_error(operation, exc.status, exc.body or exc.data) from exc
    except CyberwaveError:
        raise
    except Exception as exc:
        raise CyberwaveError(f"{operation} failed: {exc}") from exc

    status = int(getattr(response, "status", 0) or 0)
    data = decode_json(getattr(response, "data", None))
    if 200 <= status < 300:
        return status, data
    if status in set(allow_statuses):
        shape = tuple(result_shape)
        if not shape or (isinstance(data, dict) and all(k in data for k in shape)):
            return status, data
    raise api_error(operation, status, data)


def api_error(operation: str, status: Any, data: Any) -> CyberwaveAPIError:
    """``CyberwaveAPIError`` for a failed *operation*, carrying the body's detail."""
    detail = data.get("detail") if isinstance(data, dict) else data
    message = f"{operation} failed with HTTP {status}"
    if detail:
        message = f"{message}: {detail}"
    return CyberwaveAPIError(
        message,
        status_code=status if isinstance(status, int) else None,
        response_data=data,
    )
