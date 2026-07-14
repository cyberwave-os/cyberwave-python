"""
Demo: drive an SO-101 arm through a scripted pick-and-place motion in simulation.

A short, narratable sequence for showing an audience how the Cyberwave SDK talks
to a digital twin: connect → place in the scene → move joints → grip → move → release.

Run:
    export CYBERWAVE_API_KEY=...      # from Profile → API Tokens
    python examples/demo_pick_and_place.py
"""

import time

from cyberwave import Cyberwave

# --- Connect ---------------------------------------------------------------
cw = Cyberwave()

# Target the simulator, not the physical robot.
cw.affect("simulation")

# Connect to the SO-101 twin living in a specific environment.
so_101 = cw.twin(
    "the-robot-studio/so101",
    twin_id="009c037a-8186-43bd-8373-4ade1d55ab6f",
    environment_id="51223661-2dd4-41a3-8f90-5d5baab7cef6",
)

# Discover the joint names for this arm (order matches set()).
print("Joints:", so_101.joints.list())


def settle(seconds: float = 0.6) -> None:
    """Give the simulation a beat so the motion is visible on screen."""
    time.sleep(seconds)


# --- Place the arm in the scene -------------------------------------------
so_101.edit_position(x=1.0, y=0.0, z=0.5)
so_101.edit_rotation(yaw=90)  # degrees
settle()


# --- Scripted pick-and-place motion ---------------------------------------
def move_to_home() -> None:
    """Neutral, upright pose."""
    so_101.release()
    so_101.joints.set("1", 0, degrees=True)
    so_101.joints.set("2", 0, degrees=True)
    so_101.joints.set("3", 0, degrees=True)
    settle()


def reach_down_and_grab() -> None:
    """Rotate toward the object, dip the arm, and close the gripper."""
    so_101.release()
    so_101.joints.set("1", 45, degrees=True)   # rotate base toward object
    settle()
    so_101.joints.set("2", 60, degrees=True)   # lower shoulder
    so_101.joints.set("3", 40, degrees=True)   # bend elbow down to the object
    settle()
    so_101.grip(force=0.8)                     # close on the object
    settle()


def lift_and_place() -> None:
    """Lift the object, swing across, and release it at the drop point."""
    so_101.joints.set("3", 10, degrees=True)   # lift elbow (object in hand)
    so_101.joints.set("2", 20, degrees=True)
    settle()
    so_101.joints.set("1", -45, degrees=True)  # swing base to the drop zone
    settle()
    so_101.joints.set("2", 55, degrees=True)   # lower to place
    so_101.joints.set("3", 35, degrees=True)
    settle()
    so_101.release()                           # let go
    settle()


if __name__ == "__main__":
    print("Homing…")
    move_to_home()

    print("Picking…")
    reach_down_and_grab()

    print("Placing…")
    lift_and_place()

    print("Returning home…")
    move_to_home()

    cw.disconnect()
