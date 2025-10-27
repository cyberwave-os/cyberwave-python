#!/usr/bin/env python3
import configparser
import logging
import time

import messaging

from webRTCOfferPayload import WebRTCOfferPayload
from jointStatePayload import JointStatePayload
from jointState import JointState
from pingPayload import PingPayload
from pongPayload import PongPayload
from twinPositionPayload import TwinPositionPayload
from vector3 import Vector3
from twinRotationPayload import TwinRotationPayload
from quaternion import Quaternion
from twinScalePayload import TwinScalePayload



# Config has the connection properties.
def getConfig():
    configParser = configparser.ConfigParser()
    configParser.read('config.ini')
    config = configParser['DEFAULT']
    return config





def cyberwavePongResourceUuidResponse(client, userdata, msg):
    jsonString = msg.payload.decode('utf-8')
    logging.info('Received json: ' + jsonString)
    pongPayload = PongPayload.from_json(jsonString)
    logging.info('Received message: ' + str(pongPayload))






def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Start of main.')
    config = getConfig()

    cyberwavePongResourceUuidResponseMessenger = messaging.Messaging(config, 'cyberwave.pong.*.response', cyberwavePongResourceUuidResponse)
    cyberwavePongResourceUuidResponseMessenger.loop_start()

    # Example of how to publish a message. You will have to add arguments to the constructor on the next line:
    payload = WebRTCOfferPayload()
    payloadJson = payload.to_json()

    while (True):
        cyberwaveTwinTwinUuidWebrtcOfferMessenger.publish('cyberwave.twin.{twin_uuid}.webrtc-offer', payloadJson)
        time.sleep(1)

if __name__ == '__main__':
    main()

