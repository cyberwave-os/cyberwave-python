#!/usr/bin/env python3
import configparser
import logging
import time

import messaging

from cyberwave.mqtt.webRTCOfferPayload import WebRTCOfferPayload
from cyberwave.mqtt.jointStatePayload import JointStatePayload
from cyberwave.mqtt.jointState import JointState
from cyberwave.mqtt.pingPayload import PingPayload
from cyberwave.mqtt.pongPayload import PongPayload
from cyberwave.mqtt.twinPositionPayload import TwinPositionPayload
from cyberwave.mqtt.vector3 import Vector3
from cyberwave.mqtt.twinRotationPayload import TwinRotationPayload
from cyberwave.mqtt.quaternion import Quaternion
from cyberwave.mqtt.twinScalePayload import TwinScalePayload



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

