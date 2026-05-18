"""
Datasets example: import lerobot/pusht from HuggingFace, list datasets,
visualize in the browser, and convert to RLDS.

wait_until_ready() and download() print one-line status on every poll by
default. Pass on_poll=None to any of them to silence progress output in
your own applications.

Required env vars:
    CYBERWAVE_API_KEY       Knox API key

Optional env vars:
    CYBERWAVE_BASE_URL      Backend URL  (default: https://api.cyberwave.com)
    CYBERWAVE_FRONTEND_URL  Frontend URL (default: https://cyberwave.com)
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# 1. Import from HuggingFace.
#    Idempotent: if a dataset for "lerobot/pusht" already exists in the
#    workspace it is returned immediately instead of queuing a new import.
ds = cw.datasets.add("lerobot/pusht", name="pusht")
print(f"dataset uuid={ds.uuid}  status={ds.processing_status}  ready={ds.is_ready}")

# 2. Wait for the async HF metadata import to finish.
#    The default on_poll prints one line per poll — pass on_poll=None to silence.
ds = cw.datasets.wait_until_ready(ds, poll_interval=5.0, timeout=1800)
print(f"import done: ready={ds.is_ready}  status={ds.processing_status}")

# 3. List the most recent datasets visible in this workspace.
print("\nDatasets in workspace:")
for d in cw.datasets.list(limit=20):
    print(f"  {d.uuid}  {d.name!r:30s}  {d.processing_status:12s}  episodes={d.total_episodes}")

# 4. Get the frontend URL to view this dataset in the browser.
print(f"\nvisualize: {cw.datasets.visualize(ds)}")

# 5. Convert to RLDS on the backend (idempotent; reuses artifact if it exists).
#    The default on_poll prints convert status each poll.
print()
url = cw.datasets.convert(ds, "rlds", poll_interval=5.0, timeout=3600)
print(f"rlds ready: {url}")

# 6. Download the artifact to the current directory.
#    Pass dest="./data" to save into a specific folder instead.
path = cw.datasets.download(ds, "rlds", on_poll=None)
print(f"rlds saved: {path}")

# 7. (Optional) Delete the dataset entry from Cyberwave.
#    Uncomment to clean up after the example run.
# result = cw.datasets.delete(ds.uuid)
# print(f"deleted: {result}")
