import paho.mqtt.client as mqtt
import configparser
import logging
import json
import paho.mqtt.client as mqtt

class Messaging:
    """ This is a wrapper for the mqtt client. """
    
    def __init__(self, config, subscription = None, on_message = None, clientId = None):
        global on_connect
        self.config = config
        defaultHost = 'host.docker.internal:1883'

        if (clientId):
            self.client = mqtt.Client(clientId)
        else:
            self.client = mqtt.Client()

        self.client.enable_logger()
        self.client.on_connect = on_connect

        if (subscription):
            self.client.user_data_set(subscription)

        if (on_message):
            self.client.on_message = on_message

        username = config.get('username', None)
        password = config.get('password', None)

        if username is not None:
            self.client.username_pw_set(username, password)

        port = config.get('port')
        port = int(port) if port else 1833
        host = config.get('host', defaultHost)
        print("Host: ", host, "port: ", port)

        if host is None:
            raise Exception("Host must be defined in the config file or in the servers section.")

        self.client.connect(host, port)

    def publish(self, topic, payload, qos = 0, retain = False):
        self.client.publish(topic, payload, qos, retain)

    def subscribe(self, topic):
        self.client.subscribe(topic)

    def loop_forever(self):
        self.client.loop_forever()

    def loop_start(self):
        self.client.loop_start()

def on_connect(client, userdata, flags, rc):
    if (userdata):
        client.subscribe(userdata)

