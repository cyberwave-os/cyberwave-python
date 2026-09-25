"""
Model Predict on Twin — grab a frame from a twin's camera and run ML inference.

Usage:
    python model_predict_on_twin.py YOUR_TWIN_UUID
    python model_predict_on_twin.py YOUR_TWIN_UUID -m yolo26n.pt

Requirements:
    pip install cyberwave[ml] Pillow
"""

from __future__ import annotations

import argparse
from io import BytesIO

from PIL import Image

from cyberwave import Cyberwave

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("twin_id", help="Twin UUID or slug")
ap.add_argument(
    "-m", "--model", default="yolo26n.pt", help="Model load ID (default: yolo26n.pt)"
)
ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
args = ap.parse_args()

cw = Cyberwave()

twin = cw.twins.get(args.twin_id)
jpeg_bytes = twin.get_frame("bytes")
if jpeg_bytes is None:
    raise RuntimeError("No frame available from twin camera")
image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")

model = cw.models.load(args.model)
pred = model.predict(image, confidence=args.conf)

print(pred.describe())
