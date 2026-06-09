"""
Workflows — list, trigger, and monitor workflow runs.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# List available workflows
workflows = cw.workflows.list()
for wf in workflows:
    print(f"{wf.name} ({wf.uuid}) — {wf.status}")

# Trigger the first active workflow
active = [wf for wf in workflows if wf.is_active]
if not active:
    print("No active workflows found.")
    exit()

workflow = active[0]
print(f"\nTriggering '{workflow.name}'…")

run = workflow.trigger(inputs={"target_position": [1.0, 2.0, 0.0], "speed": 0.5})
print(f"Run: {run.uuid}  status={run.status}")

# Wait for completion
run.wait(timeout=120, poll_interval=3)
print(f"Final: {run.status}  duration={run.duration}s")

if run.result:
    print(f"Result: {run.result}")

cw.disconnect()
