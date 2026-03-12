"""
Workflow Automation Example

List, trigger, monitor, and cancel workflows from the SDK.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()

# ── List available workflows ──────────────────────────────────────────

workflows = cw.workflows.list()
for wf in workflows:
    print(f"{wf.name} ({wf.uuid}) — {wf.status}")

if not workflows:
    print("No workflows found. Create one in the Cyberwave dashboard first.")
    exit()

# ── Trigger the first active workflow ─────────────────────────────────

active = [wf for wf in workflows if wf.is_active]
if not active:
    print("No active workflows. Activate one in the dashboard.")
    exit()

workflow = active[0]
print(f"\nTriggering '{workflow.name}' ...")

run = workflow.trigger(inputs={"target_position": [1.0, 2.0, 0.0], "speed": 0.5})
print(f"Run started: {run.uuid}  (status: {run.status})")

# ── Wait for the run to finish ────────────────────────────────────────

run.wait(timeout=120, poll_interval=3)
print(f"\nFinal status : {run.status}")
print(f"Duration     : {run.duration}s")

if run.result:
    print(f"Result       : {run.result}")
if run.error:
    print(f"Error        : {run.error}")

# ── Browse past runs ──────────────────────────────────────────────────

past_runs = workflow.runs(status="success")
print(f"\n{len(past_runs)} successful past runs for '{workflow.name}'")

cw.disconnect()
