"""
Models catalog example: filter catalog entries by name or properties, then load and predict.

``cw.models`` is the unified surface for both catalog CRUD and runtime inference.

``cw.models.load()`` accepts a catalog entry (``MLModelSchema``) directly, so you
can filter ``cw.models.list()`` with any Python expression and pass the result
straight to ``load()`` — no need to inspect ``sdk_load_id`` or ``slug`` manually.

Required:
    CYBERWAVE_API_KEY       Knox API key (or: cyberwave login --token TOKEN)

Optional:
    CYBERWAVE_BASE_URL      Backend URL  (default: https://api.cyberwave.com)

Usage:
    python models.py
    python models.py --image path/to/frame.jpg
    python models.py --image path/to/frame.jpg --name yolo26n
    python models.py --image path/to/frame.jpg --deployment edge --tag detection
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from cyberwave import Cyberwave

if TYPE_CHECKING:
    from cyberwave.rest import MLModelSchema

cw = Cyberwave()


# ---------------------------------------------------------------------------
# Helper: filter a model list
# ---------------------------------------------------------------------------

def find_model(
    models: list[MLModelSchema],
    *,
    name: str | None = None,
    deployment: str | None = None,
    tag: str | None = None,
    sdk_load_id: str | None = None,
) -> MLModelSchema | None:
    """Return the first entry that matches all supplied filters (case-insensitive).

    Each filter is optional — omitting it skips that check.
    """
    for m in models:
        if name and name.lower() not in m.name.lower():
            continue
        if deployment and m.deployment != deployment:
            continue
        if tag and tag.lower() not in [t.lower() for t in (m.tags or [])]:
            continue
        if sdk_load_id and (m.sdk_load_id or "") != sdk_load_id:
            continue
        return m
    return None


# ---------------------------------------------------------------------------
# 1. List all ML model records visible in your workspace (+ public ones).
# ---------------------------------------------------------------------------
all_models = cw.models.list()

print(f"All visible models ({len(all_models)}):")
for m in all_models:
    flags = "/".join(f for f in ("edge" if m.is_edge_compatible else "",
                                  "cloud" if m.is_cloud_compatible else "") if f)
    print(f"  {m.slug or m.uuid:55s}  {m.model_external_id:25s}  [{flags}]  tags={m.tags}")


# ---------------------------------------------------------------------------
# 2. Filter by deployment.
# ---------------------------------------------------------------------------
edge_models = cw.models.list(deployment="edge")
print(f"\nEdge-compatible models ({len(edge_models)}):")
for m in edge_models:
    print(f"  {m.name:30s}  sdk_load_id={m.sdk_load_id}")

cloud_models = cw.models.list(deployment="cloud")
print(f"\nCloud models ({len(cloud_models)}):")
for m in cloud_models:
    print(f"  {m.name:30s}  playground_kind={m.playground_kind}")

print("\nPublic models (no workspace membership needed):")
for m in cw.models.list_public():
    print(f"  {m.slug or m.uuid}")


# ---------------------------------------------------------------------------
# 3. Filter locally by name substring, tag, or sdk_load_id.
# ---------------------------------------------------------------------------
# By name substring (case-insensitive)
yolo_models = [m for m in all_models if "yolo" in m.name.lower()]
print(f"\nModels whose name contains 'yolo' ({len(yolo_models)}):")
for m in yolo_models:
    print(f"  {m.name}  sdk_load_id={m.sdk_load_id}")

# By tag
detection_models = [m for m in all_models if "detection" in (m.tags or [])]
print(f"\nModels tagged 'detection' ({len(detection_models)}):")
for m in detection_models:
    print(f"  {m.name}  tags={m.tags}")

# By exact sdk_load_id
nano = next((m for m in all_models if m.sdk_load_id == "yolo26n.pt"), None)
if nano:
    print(f"\nFound by sdk_load_id 'yolo26n.pt': {nano.name}  slug={nano.slug}")

# By property: edge + trainable
trainable_edge = [m for m in all_models if m.is_edge_compatible and m.is_trainable]
print(f"\nEdge models that support fine-tuning ({len(trainable_edge)}):")
for m in trainable_edge:
    print(f"  {m.name}")

# Using the helper above: first edge model whose name contains "nano"
nano_entry = find_model(all_models, deployment="edge", name="nano")
if nano_entry:
    print(f"\nFirst edge-nano model: {nano_entry.name}  sdk_load_id={nano_entry.sdk_load_id}")


# ---------------------------------------------------------------------------
# 4. Get a single record by slug or UUID.
# ---------------------------------------------------------------------------
# m = cw.models.get("your-workspace/models/my-yolo-model")
# print(f"\nFetched: {m.name!r}  deployment={m.deployment}  sdk_load_id={m.sdk_load_id}")


# ---------------------------------------------------------------------------
# 5. Load a catalog entry and run inference.
#
#    cw.models.load() accepts an MLModelSchema entry directly.
#    The SDK picks the right load key: sdk_load_id → slug → uuid.
#    Pass --image to activate this section.
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, add_help=False)
ap.add_argument("--image", default=None, metavar="PATH")
ap.add_argument("--name", default="yolo26n", metavar="SUBSTR",
                help="Case-insensitive substring to filter model names")
ap.add_argument("--deployment", default="edge", choices=["edge", "cloud", "hybrid"],
                help="Deployment filter (default: edge)")
ap.add_argument("--tag", default=None, metavar="TAG",
                help="Filter to models with this tag")
args, _ = ap.parse_known_args()

if args.image:
    from PIL import Image

    img = Image.open(args.image).convert("RGB")

    # Resolve entry from catalog using the CLI filters
    candidates = cw.models.list(deployment=args.deployment)
    entry = find_model(
        candidates,
        name=args.name,
        tag=args.tag,
    )

    if entry is None:
        filter_desc = " | ".join(
            f"{k}={v!r}" for k, v in [
                ("deployment", args.deployment),
                ("name", args.name),
                ("tag", args.tag),
            ] if v is not None
        )
        raise SystemExit(
            f"No model matched filters [{filter_desc}]. "
            f"Run without --image to list all available entries."
        )

    print(f"\n[catalog] '{entry.name}'  deployment={entry.deployment}"
          f"  sdk_load_id={entry.sdk_load_id}")
    model = cw.models.load(entry)   # pass entry directly — SDK resolves load key

    pred = model.predict(img, confidence=0.25)
    print(f"[inference] {args.image}  →  {len(pred)} detection(s)")
    print(pred.describe())
else:
    print("\n(pass --image path/to/frame.jpg to also run inference)")
    print("Optional filters: --name SUBSTR  --deployment edge|cloud|hybrid  --tag TAG")


# ---------------------------------------------------------------------------
# 6. Delete a model record (uncomment to try).
#    This removes the catalog entry; local weight files are unaffected.
# ---------------------------------------------------------------------------
# result = cw.models.delete("some-uuid-here")
# print(f"deleted: {result}")
