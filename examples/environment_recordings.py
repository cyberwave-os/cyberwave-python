"""
Environment recordings — list recordings across every twin in an environment.

Recordings are environment-scoped, so this example only needs an environment
id (no twin object required). It runs safely against a fresh environment with
no recorded sessions yet — it just reports zero recordings in that case.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

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
    print(f"  {item.uuid} twin={item.twin_uuid} types={sorted(t.value for t in item.types)}")
