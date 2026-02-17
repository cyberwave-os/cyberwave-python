"""
Example of using Alerts with the Cyberwave SDK
"""

from cyberwave import Cyberwave
import time

cw = Cyberwave()

# Create a digital twin from an asset
robot = cw.twin("the-robot-studio/so101")

# Create an alert for the twin
alert = robot.alerts.create(name="Calibration Needed", description="Needs calibration")

# List all alerts for the twin
alerts = robot.alerts.list()

print(alerts)

# Get an alert by its UUID
alert = robot.alerts.get(alert.uuid)

# Wait 10 seconds
time.sleep(10)

# Resolve an alert
alert.resolve()
