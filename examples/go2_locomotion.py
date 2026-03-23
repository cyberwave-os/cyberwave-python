from cyberwave import Cyberwave

# Configure the SDK
cw = Cyberwave()

# Declare whether commands should affect the simulation or the real robot.
# Use "simulation" to move the digital twin in Cyberwave.
# Use "real-world" to move the physical robot.
cw.affect("real-world")

# Create a digital twin from an asset
robot = cw.twin("unitree/go2")

# Move the real robot forward (uses "real-world" set above)
robot.move_forward()
