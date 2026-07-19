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

recordings = cw.environments.recordings.list(environment_id=environment_id)
print(f"Found {len(recordings)} recording(s) in environment {environment_id}")

for item in recordings:
    print(f"  {item.uuid} twin={item.twin_uuid} types={sorted(t.value for t in item.types)}")
