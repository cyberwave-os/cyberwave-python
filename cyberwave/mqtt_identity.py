"""Derivation of the MQTT username from the API token that authenticates it.

The broker sends the password only on CONNECT; every later authorization check
carries the username and client id alone. A username derived from the token
lets the backend identify the session from the check itself, instead of
remembering it in a slot shared by every client that connects as ``mqttcyb``.

The username is echoed into broker logs and authorization callbacks, so it is
the hash, never the token: it identifies, the password still authenticates.

Mirrored in ``cyberwave-backend/src/lib/mqtt_identity.py``. The two must agree
exactly -- a username that does not hash to its token is refused at CONNECT.

``is_api_token`` separates the two kinds of password a client may hold: a
Cyberwave API token, which the username is derived from, and a static broker
account password, which leaves the account name standing.
"""

import hashlib
from typing import Optional

MQTT_TOKEN_USERNAME_PREFIX = "cwh_"

#: Prefix every Cyberwave API token carries (backend ``APIToken.generate_token``).
API_TOKEN_PREFIX = "cw_"


def is_api_token(credential: Optional[str]) -> bool:
    """Whether ``credential`` is an API token rather than a broker password."""
    return bool(credential) and credential.startswith(API_TOKEN_PREFIX)


def mqtt_username_for_token(token: str) -> str:
    """The MQTT username a client holding ``token`` should connect under."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{MQTT_TOKEN_USERNAME_PREFIX}{digest}"
