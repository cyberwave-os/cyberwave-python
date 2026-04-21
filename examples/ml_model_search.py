#!/usr/bin/env python3
"""
Example: fuzzy ML model search + single UUID resolution.

Configuration:
    Required: CYBERWAVE_API_KEY (same as other examples; optional ``.env`` at repo root)

Edit the string constants below to try different queries against your workspace catalog.

Run:
  poetry run python examples/ml_model_search.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent
    _env_file = _repo_root / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from cyberwave import Cyberwave, MLModelLookupError, resolve_ml_model_uuid, search_ml_models

# --- edit these strings to explore your catalog --------------------------------

# Broad substring search (name or model_external_id, case-insensitive).
SEARCH_QUERY = "gemini"
SEARCH_LIMIT = 10

# Pick a string that resolves to exactly one model in your workspace.
# External ids are usually the safest single hit.
RESOLVE_QUERY = "gemini-3-flash-preview"


def main() -> int:
    if not (os.getenv("CYBERWAVE_API_KEY") or "").strip():
        print("ERROR: CYBERWAVE_API_KEY is required", file=sys.stderr)
        return 1

    cw = Cyberwave()
    try:
        print(f"fuzzy query: {SEARCH_QUERY!r} (limit={SEARCH_LIMIT})")
        matches = search_ml_models(cw.api, SEARCH_QUERY, limit=SEARCH_LIMIT)
        if not matches:
            print("search_ml_models: no matches")
        else:
            print(f"search_ml_models: {len(matches)} match(es)")
            for i, model in enumerate(matches, start=1):
                print(
                    f"  {i}. name={model.name!r} external_id={model.model_external_id!r} "
                    f"uuid={model.uuid} deployment={model.deployment!r} edge={model.is_edge_compatible}"
                )

        print(f"\nresolve query: {RESOLVE_QUERY!r}")
        try:
            model_uuid = resolve_ml_model_uuid(cw.api, RESOLVE_QUERY)
            print("resolve_ml_model_uuid:", model_uuid)
        except MLModelLookupError as exc:
            print("resolve_ml_model_uuid error:", exc)
    finally:
        cw.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
