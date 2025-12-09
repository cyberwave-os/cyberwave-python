import os
import time
import logging
from cyberwave import Cyberwave


# Configure logging to see SDK logs
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def actuate_arm():
    # Configure the SDK
    cw = Cyberwave(api_key=os.getenv("CYBERWAVE_API_KEY"))

    # Create a digital twin from an asset
    robot = cw.twin(
        "the-robot-studio/so101", environment_id=os.getenv("CYBERWAVE_ENVIRONMENT_ID")
    )

    i = 0
    while i < 70:
        robot.joints.set("3", i)
        print(f"Joint angles - 3: {i}")
        i += 5
        time.sleep(0.1)


if __name__ == "__main__":
    actuate_arm()
