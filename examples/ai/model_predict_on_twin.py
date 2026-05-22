"""Twin latest frame → ML model predict.  ``pip install "cyberwave[ml]"``

Authenticate with **``cyberwave login --token YOUR_TOKEN``** or set ``CYBERWAVE_API_KEY``.

The twin must expose an RGB camera capability and Edge Core's camera driver must be
feeding frames — see docs.cyberwave.com/hardware/camera/get-started.

The ``--model`` flag accepts three forms:

  1. Literal SDK load ID:     ``yolo26n.pt``        (local weight filename)
  2. Catalog slug:            ``acme/models/yolo26n`` (cloud or cached edge)
  3. Filter expression:       ``name=yolo26n``       (resolved from catalog)

Filter expression syntax: ``key=value`` pairs separated by ``,``
  Supported keys: ``name`` (substring), ``deployment``, ``tag``, ``sdk_load_id``
  Examples:
    ``name=nano``                     first model with "nano" in its name
    ``deployment=edge,name=yolo``     first edge model with "yolo" in its name
    ``tag=detection,deployment=edge`` first edge model tagged "detection"

Examples:

  python model_predict_on_twin.py YOUR_UUID
  python model_predict_on_twin.py YOUR_UUID -m yolo26n.pt
  python model_predict_on_twin.py YOUR_UUID -m "name=nano,deployment=edge"
  python model_predict_on_twin.py YOUR_UUID -m "tag=detection"
  python model_predict_on_twin.py workspace-slug/twins/my-cam --list-models
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image

from cyberwave import Cyberwave
from cyberwave.exceptions import CyberwaveAPIError

if TYPE_CHECKING:
    from cyberwave.rest import MLModelSchema

_DEFAULT_MODEL = "yolo26n.pt"
_PREDICT_CONF = 0.25


def _parse_filter_expr(expr: str) -> dict[str, str]:
    """Parse ``"key=value,key=value"`` into a dict.

    Raises ``ValueError`` for malformed pairs.
    """
    filters: dict[str, str] = {}
    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid filter token {token!r}. "
                "Expected key=value pairs separated by commas."
            )
        key, _, value = token.partition("=")
        filters[key.strip()] = value.strip()
    return filters


def _resolve_from_catalog(
    cw: Cyberwave,
    filters: dict[str, str],
) -> MLModelSchema:
    """Resolve catalog filters to a single entry.

    Supported filter keys: ``name`` (substring), ``deployment``,
    ``tag``, ``sdk_load_id``.
    """
    deployment = filters.get("deployment")
    models = cw.models.list(deployment=deployment)

    name_substr = filters.get("name", "").lower()
    tag_filter = filters.get("tag", "").lower()
    sdk_id_filter = filters.get("sdk_load_id", "")

    for m in models:
        if name_substr and name_substr not in m.name.lower():
            continue
        if tag_filter and tag_filter not in [t.lower() for t in (m.tags or [])]:
            continue
        if sdk_id_filter and (m.sdk_load_id or "") != sdk_id_filter:
            continue
        return m

    filter_desc = ", ".join(f"{k}={v!r}" for k, v in filters.items())
    raise SystemExit(
        f"No catalog entry matched [{filter_desc}]. "
        "Run with --list-models to see available entries."
    )


def _is_filter_expr(value: str) -> bool:
    """Return True when value looks like ``key=value`` rather than a plain ID."""
    # A plain ID is either a filename ("yolo26n.pt"), a slug ("ws/models/x"),
    # or a UUID. Filter expressions contain an "=" that isn't in a slug segment.
    return "=" in value and "/" not in value.split("=")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "twin_id",
        type=str,
        help="Twin UUID or full slug",
    )
    ap.add_argument(
        "-m", "--model",
        default=os.environ.get("CYBERWAVE_YOLO_MODEL", _DEFAULT_MODEL),
        metavar="ID_OR_FILTER",
        help=(
            f"SDK load ID, catalog slug, or filter expression "
            f"(default: {_DEFAULT_MODEL} or $CYBERWAVE_YOLO_MODEL)"
        ),
    )
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="List all catalog models and exit (useful for picking a --model filter)",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=_PREDICT_CONF,
        metavar="FLOAT",
        help=f"Detection confidence threshold (default: {_PREDICT_CONF})",
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print debug info (base URL, twin ID, resolved model)",
    )
    args = ap.parse_args()
    twin_identifier = "".join(args.twin_id.split())  # strip stray whitespace from paste

    cw = Cyberwave()

    if args.verbose:
        print(
            "debug:",
            os.environ.get("CYBERWAVE_BASE_URL") or cw.config.base_url,
            repr(twin_identifier),
            repr(args.model),
            flush=True,
        )

    # --list-models: print catalog and exit
    if args.list_models:
        models = cw.models.list()
        print(f"{'Name':<35} {'Deployment':<10} {'sdk_load_id':<25} Tags")
        print("-" * 90)
        for m in models:
            print(f"{m.name:<35} {m.deployment:<10} {(m.sdk_load_id or '—'):<25} {m.tags}")
        return

    # Resolve the model — either filter expression, catalog slug/UUID, or local filename
    if _is_filter_expr(args.model):
        try:
            filters = _parse_filter_expr(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        entry = _resolve_from_catalog(cw, filters)
        if args.verbose:
            print(
                f"debug: resolved filter {args.model!r} → "
                f"{entry.name!r} sdk_load_id={entry.sdk_load_id!r}",
                flush=True,
            )
        model_ref: str | MLModelSchema = entry
    else:
        model_ref = args.model

    # Fetch twin
    try:
        twin = cw.twins.get(twin_identifier)
    except CyberwaveAPIError as e:
        if getattr(e, "status_code", None) == 404:
            hint = (
                "Twin not found (HTTP 404). Common causes:\n"
                "  • Typo / extra whitespace in UUID or slug.\n"
                "  • CYBERWAVE_BASE_URL / API key pointing at a different env.\n"
                "  • Slug needs all three segments: ws/twins/name."
            )
            raise SystemExit(f"{hint}\n--- underlying error ---\n{e}") from e
        raise

    jpeg_bytes = twin.get_latest_frame()
    image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")

    model = cw.models.load(model_ref)  # accepts str or MLModelSchema entry
    pred = model.predict(image, confidence=args.conf)

    if args.verbose:
        loaded_name = getattr(model_ref, "name", args.model)
        print(f"debug: {loaded_name!r}  {len(pred)} detection(s)", flush=True)

    print(pred.describe())


if __name__ == "__main__":
    main()
