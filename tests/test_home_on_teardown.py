"""home_if_teardown_enabled() homes only when HOME_ON_TEARDOWN is set, swallowing errors."""
from cyberwave.driver.control.joint_teleop import JointCommandBufferMixin


def _driver(enabled: bool, raise_on_home: bool = False):
    class _D(JointCommandBufferMixin):
        HOME_ON_TEARDOWN = enabled
        def __init__(self):
            self._init_joint_command_buffer()
            self.home_calls = 0
        def _return_to_home(self):
            self.home_calls += 1
            if raise_on_home:
                raise RuntimeError("boom")
        def _publish_joint_command(self, positions): ...
    return _D()


def test_no_home_when_flag_off():
    d = _driver(enabled=False)
    d.home_if_teardown_enabled()
    assert d.home_calls == 0


def test_homes_when_flag_on():
    d = _driver(enabled=True)
    d.home_if_teardown_enabled()
    assert d.home_calls == 1


def test_home_errors_are_swallowed():
    d = _driver(enabled=True, raise_on_home=True)
    d.home_if_teardown_enabled()  # must not raise
    assert d.home_calls == 1
