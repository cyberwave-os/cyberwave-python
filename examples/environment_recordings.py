"""
Environment recordings — list and download recordings across every twin in an
environment.

Recordings are environment-scoped, so this example only needs an environment
id (no twin object required). It runs safely against a fresh environment with
no recorded sessions yet — it just reports zero recordings in that case.

Each recording carries a processing status: "ready" means all derived
artifacts are available; "processing" means the recording is still active or
its derivatives are being built — downloading then yields the unprocessed
segments available so far.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave
from cyberwave.exceptions import CyberwaveError

cw = Cyberwave()

# Creating a twin ensures the environment exists; recordings themselves are
# environment-scoped, not twin-scoped.
robot = cw.twin("the-robot-studio/so101")
environment_id = robot.environment_id

# With no start/end, list() looks up which days have recordings and returns the
# most recent one (paged, 50 per request). Pass start/end for an older window,
# or limit=0 to follow every page.
recordings = cw.environments.recordings.list(environment_id=environment_id)
print(
    f"Found {len(recordings)} recording(s) on the latest recorded day "
    f"in environment {environment_id}"
)

for item in recordings:
    types = sorted(t.value for t in item.types)
    print(
        f"  {item.uuid} twin={item.twin_uuid} types={types} "
        f"status={item.processing_status}"
    )

# Download each recording's artifacts to a local temp dir. The context manager
# deletes the files on exit — copy them elsewhere to keep them. Works for ready
# recordings and for ones still processing (segment-stored data).
for item in recordings:
    try:
        with item.get() as recording:
            files = [p for paths in recording.local_paths.values() for p in paths]
            print(f"  downloaded {item.uuid}: {len(files)} file(s)")
            for f in files:
                print(f"    {f}")
    except CyberwaveError as exc:
        # Artifacts not servable yet (e.g. video segments still converting) —
        # retry in a few minutes.
        print(f"  skipped {item.uuid}: {exc}")
