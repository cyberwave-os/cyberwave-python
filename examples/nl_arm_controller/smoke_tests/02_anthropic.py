"""Smoke test 2/4 — Anthropic Claude.

Verifies the API key, model name, and JSON-output prompting we'll lean on in
Phase 4. Should print a tiny JSON object and exit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

try:
    from anthropic import Anthropic
except ImportError as exc:
    print(f"❌ anthropic import failed: {exc}")
    sys.exit(1)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        sys.exit(1)

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    print(f"→ Calling {model} (text-only, max 80 tokens)...")

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=80,
        system='Reply with ONLY a JSON object of the form: {"reply": "<one word>"}. No prose, no markdown.',
        messages=[{"role": "user", "content": "Say pong."}],
    )

    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    print(f"  raw: {text}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"❌ Claude did not return valid JSON: {exc}")
        sys.exit(1)

    if "reply" not in data:
        print(f"❌ JSON missing 'reply' key: {data}")
        sys.exit(1)

    print(f"✅ Anthropic OK — model: {model}, reply: {data['reply']!r}")


if __name__ == "__main__":
    main()
