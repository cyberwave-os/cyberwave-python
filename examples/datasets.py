"""
Datasets — import from HuggingFace, list, convert to RLDS, and download.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# Import a dataset from HuggingFace (idempotent)
ds = cw.datasets.add("lerobot/pusht", name="pusht")
print(f"Dataset: {ds.uuid}  status={ds.processing_status}")

# Wait for import to finish
ds = cw.datasets.wait_until_ready(ds, poll_interval=5.0, timeout=1800)
print(f"Ready: {ds.is_ready}")

# List datasets in the workspace
for d in cw.datasets.list(limit=10):
    print(f"  {d.name:30s}  episodes={d.total_episodes}")

# View in browser
print(f"Visualize: {cw.datasets.visualize(ds)}")

# Convert to RLDS and download
url = cw.datasets.convert(ds, "rlds", poll_interval=5.0, timeout=3600)
print(f"RLDS ready: {url}")

path = cw.datasets.download(ds, "rlds")
print(f"Downloaded: {path}")
