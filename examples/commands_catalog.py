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
print(dog.commands.get_schema()["commands"]["supported"][:5], "...")

dog.commands.move_forward(linear_x=0.3, duration=0.5, rate_hz=10)
# dog.commands.sit_down()  # catalog-only, single MQTT publish

cw.disconnect()
