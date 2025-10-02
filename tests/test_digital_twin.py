import unittest
from cyberwave.digital_twin import (
    AbstractAsset,
    StaticAsset,
    RobotAsset,
    PhysicalDevice,
    DigitalTwin,
)


class DummyDevice(PhysicalDevice):
    def __init__(self, device_id: str, device_type: str):
        super().__init__(device_id, device_type)
        self.commands = []
        self.telemetry = {"status": "ok"}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def send_command(self, command: str, **kwargs):
        self.commands.append((command, kwargs))

    def get_telemetry(self):
        return self.telemetry


class TestDigitalTwin(unittest.TestCase):
    def test_robot_asset_attach_detach(self):
        robot = RobotAsset("TestBot")
        device = DummyDevice("dev1", "dummy")
        robot.attach_device(device)
        self.assertIs(robot.device, device)
        robot.detach_device()
        self.assertIsNone(robot.device)

    def test_digital_twin_updates_state(self):
        robot = RobotAsset("TestBot")
        device = DummyDevice("dev1", "dummy")
        twin = DigitalTwin(robot, device)
        twin.update_from_device()
        self.assertEqual(twin.live_state.get("status"), "ok")
        twin.send_command("move", x=1)
        self.assertEqual(device.commands[0][0], "move")
        self.assertEqual(device.commands[0][1]["x"], 1)


if __name__ == "__main__":
    unittest.main()
