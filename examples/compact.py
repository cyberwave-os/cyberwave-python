from cyberwave import Cyberwave

# Configure using environment variables or explicitly
# Option 1: Set CYBERWAVE_API_KEY or CYBERWAVE_TOKEN environment variable
# Option 2: Configure explicitly:
# cw.configure(api_key="your-token-here", base_url="http://localhost:8000")

# Create a digital twin from an asset
cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")

# Get current joint positions
print(robot.joints.get_all())
