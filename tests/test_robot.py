import unittest
from unittest import mock

import cyberwave.robot as robot_module
from cyberwave.robot import Robot

class DummyDriver:
    def __init__(self, robot_type):
        self.robot_type = robot_type
        self.connected = False
        self.disconnect_called = 0

    def connect(self, ip=None):
        self.connected = True
        self.ip = ip

    def disconnect(self):
        self.disconnect_called += 1
        self.connected = False

class TestRobotDisconnect(unittest.TestCase):
    def test_disconnect_without_driver(self):
        bot = Robot("custom")
        bot.connected = True
        bot.ip_address = "10.0.0.1"
        bot.is_flying = True
        bot.sensors = ["camera"]
        bot.disconnect()
        self.assertFalse(bot.connected)
        self.assertIsNone(bot.ip_address)
        self.assertFalse(bot.is_flying)
        self.assertEqual(bot.sensors, [])

    def test_disconnect_calls_driver(self):
        with mock.patch.object(robot_module, 'RobotDriver', DummyDriver):
            bot = Robot("dji/tello")
            bot.connect("10.0.0.2")
            self.assertTrue(bot.connected)
            bot.disconnect()
            self.assertFalse(bot.connected)
            self.assertEqual(bot._driver.disconnect_called, 1)
            self.assertFalse(bot._driver.connected)

if __name__ == '__main__':
    unittest.main()
