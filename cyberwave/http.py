from __future__ import annotations

import requests


class HttpClient:
    """Minimal synchronous HTTP client used by the SDK facade.

    Wraps requests to keep higher-level modules clean and easily testable.
    """

    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self.h = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    def get(self, path: str):
        r = requests.get(f"{self.base}{path}", headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, json: dict | None = None):
        r = requests.post(f"{self.base}{path}", headers=self.h, json=json or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def post_bytes(self, path: str, data: bytes, content_type: str = "application/octet-stream"):
        headers = dict(self.h)
        headers["Content-Type"] = content_type
        r = requests.post(f"{self.base}{path}", headers=headers, data=data, timeout=30)
        r.raise_for_status()
        return r.json()

    def put(self, path: str, json: dict | None = None):
        r = requests.put(f"{self.base}{path}", headers=self.h, json=json or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def patch(self, path: str, json: dict | None = None):
        r = requests.patch(f"{self.base}{path}", headers=self.h, json=json or {}, timeout=30)
        r.raise_for_status()
        return r.json()


