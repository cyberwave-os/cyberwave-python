"""
ML Models Catalog — list, filter, and run inference with catalog models.

Usage:
    python models.py
    python models.py --image path/to/frame.jpg

Requirements:
    pip install cyberwave Pillow
"""

from __future__ import annotations

import argparse

from cyberwave import Cyberwave

cw = Cyberwave()

# List all models
all_models = cw.models.list()
print(f"All visible models ({len(all_models)}):")
for m in all_models:
    print(f"  {m.name:30s}  sdk_load_id={m.sdk_load_id}")

# Filter by deployment
edge_models = cw.models.list(deployment="edge")
print(f"\nEdge models ({len(edge_models)}):")
for m in edge_models:
    print(f"  {m.name}")

# Filter locally by name
yolo_models = [m for m in all_models if "yolo" in m.name.lower()]
print(f"\nYOLO models ({len(yolo_models)}):")
for m in yolo_models:
    print(f"  {m.name}  sdk_load_id={m.sdk_load_id}")

# Load and predict (optional)
ap = argparse.ArgumentParser(description=__doc__, add_help=False)
ap.add_argument("--image", default=None, metavar="PATH")
args, _ = ap.parse_known_args()

if args.image:
    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    entry = next(
        (m for m in all_models if m.sdk_load_id and "yolo" in m.sdk_load_id.lower()),
        None,
    )
    if entry:
        model = cw.models.load(entry)
        pred = model.predict(img, confidence=0.25)
        print(f"\n[inference] {args.image} → {len(pred)} detection(s)")
        print(pred.describe())
    else:
        print("\nNo YOLO model found in catalog for inference.")
else:
    print("\n(pass --image path/to/frame.jpg to run inference)")
