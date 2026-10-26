"""
Time-series upload zip — pack a data folder and its recipe for Datasets > Upload zip.

The recipe lands at the zip root as cyberwave-timeseries.json and the folder is
stored under its own name next to it (contract section 2). Parquet files are
stored without recompression.

Usage:
    python make_zip.py <data-folder> <recipe.json> <out.zip>

Requirements:
    standard library only
"""

import argparse
import json
import zipfile
from pathlib import Path

RECIPE_NAME = "cyberwave-timeseries.json"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("data_folder", help="folder with the data files (stored under its own name)")
parser.add_argument("recipe", help="recipe JSON, copied to the zip root as cyberwave-timeseries.json")
parser.add_argument("out", help="zip file to write")
args = parser.parse_args()

folder = Path(args.data_folder)
if not folder.is_dir():
    raise SystemExit(f"{folder}: not a directory")
json.loads(Path(args.recipe).read_text())  # fail early on an unreadable recipe
# Skip dotfiles and anything inside a dot directory (.DS_Store, .git, ...).
files = sorted(
    p
    for p in folder.rglob("*")
    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)
)

with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    zf.write(args.recipe, RECIPE_NAME)
    for path in files:
        method = zipfile.ZIP_STORED if path.suffix == ".parquet" else zipfile.ZIP_DEFLATED
        zf.write(path, f"{folder.name}/{path.relative_to(folder).as_posix()}", compress_type=method)

print(f"{args.out}: {RECIPE_NAME} + {len(files)} files, {Path(args.out).stat().st_size / 1e6:.1f} MB")
