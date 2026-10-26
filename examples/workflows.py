"""
Workflows — list, trigger, and monitor workflow runs.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave
from cyberwave.exceptions import CyberwaveAPIError

cw = Cyberwave()

# List available workflows
workflows = cw.workflows.list()
for wf in workflows:
    print(f"{wf.name} ({wf.uuid}) — {wf.status}")

# Trigger the first active workflow. Listing returns everything you can read,
# which may include workflows only their owner is allowed to run, so skip those.
active = [wf for wf in workflows if wf.is_active]
if not active:
    print("No active workflows found.")
    exit()

run = None
for workflow in active:
    print(f"\nTriggering '{workflow.name}'…")
    try:
        run = workflow.trigger(
            inputs={"target_position": [1.0, 2.0, 0.0], "speed": 0.5}
        )
        break
    except CyberwaveAPIError as exc:
        if exc.status_code != 403:
            raise
        print(f"No permission to run '{workflow.name}'.")

if run is None:
    print("No workflow you can run.")
    exit()

print(f"Run: {run.uuid}  status={run.status}")

# Wait for completion
run.wait(timeout=120, poll_interval=3)
print(f"Final: {run.status}  duration={run.duration}s")

if run.result:
    print(f"Result: {run.result}")

cw.disconnect()
