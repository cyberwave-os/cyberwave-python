"""
Alerts — create, list, and resolve alerts on a twin.

Requirements:
    pip install cyberwave
"""

import time

from cyberwave import Cyberwave

cw = Cyberwave()

robot = cw.twin("the-robot-studio/so101")

# Create an alert
alert = robot.alerts.create(name="Calibration Needed", description="Needs calibration")
print("Created:", alert.uuid)

# List all alerts
for a in robot.alerts.list():
    print(f"  {a.name} — {a.uuid}")

# Resolve after a delay
time.sleep(5)
alert.resolve()
print("Resolved:", alert.uuid)
