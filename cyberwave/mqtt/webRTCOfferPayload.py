from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class WebRTCOfferPayload(Entity):

    class Target(str, Enum):
        backend = 'backend'
        frontend = 'frontend'
        edge = 'edge'

    class Sender(str, Enum):
        backend = 'backend'
        frontend = 'frontend'
        edge = 'edge'

    def __init__(
            self,
            type: str,
            sdp: str,
            target: Target,
            sender: Sender,
            frontendType: str,
            colorTrackId: str,
            depthTrackId: str,
            timestamp: float):
        self.type = type
        self.sdp = sdp
        self.target = target
        self.sender = sender
        self.frontendType = frontendType
        self.colorTrackId = colorTrackId
        self.depthTrackId = depthTrackId
        self.timestamp = timestamp


