"""
Catalog commands — asset MQTT ``commands.supported`` bound on ``twin.commands``.

Locomotion names delegate to ``twin.locomotion`` (burst). Others publish once.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")

dog = cw.twin("unitree/go2")
supported = dog.commands.get_supported_commands()
print(supported[:5], "...")
# dog.commands.get_schema()  # full MQTT catalog (topics, commands.specs, …)

if "move_forward" in supported:
    dog.commands.move_forward(linear_x=0.3, duration=0.5, rate_hz=10)
else:
    # Reused twins may predate asset driver-config seed; locomotion still works.
    dog.locomotion.move_forward(0.3, duration=0.5, rate_hz=10)
# dog.commands.sit_down()  # catalog-only when listed in supported

cw.disconnect()
