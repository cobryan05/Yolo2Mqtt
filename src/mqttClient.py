import paho.mqtt.client as mqtt
import time
import sys
import logging

from typing import Callable
from dataclasses import dataclass

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("MqttClient")


class MqttClient:
    @dataclass
    class SubMapData:
        callback: Callable[[mqtt.MQTTMessage], None]

    KEY_MQTT_TOPIC = "mqtt-topic"
    KEY_MQTT_RETAIN = "mqtt-retain"

    KEEPALIVE_TIME = 60
    CONNECTION_RETRIES = 10
    CONNECTION_RETRY_DELAY = 5

    def __init__(self, broker_address: str, prefix: str = "myhome/", broker_port: int = 1883):
        self._prefix: str = prefix.rstrip('/')
        self._mqtt: mqtt.Client = mqtt.Client()
        self._mqtt.on_connect = self.mqtt_connected_callback
        self._mqtt.on_message = self.mqtt_message_callback
        self._mqtt.on_disconnect = self.mqtt_disconnect_callback
        self._subMap: dict[str, MqttClient.SubMapData] = {}

        remaining_tries = MqttClient.CONNECTION_RETRIES
        while True:
            try:
                self._mqtt.connect(broker_address, broker_port, keepalive=MqttClient.KEEPALIVE_TIME)
                break
            except Exception as e:
                remaining_tries -= 1
                logger.warning("Failed to connect to broker: {}.  {} retries remaining.".format(str(e), remaining_tries))
                if remaining_tries == 0:
                    raise
                time.sleep(MqttClient.CONNECTION_RETRY_DELAY)
        self._mqtt.loop_start()

    def __del__(self):
        self.disconnect()
        logger.debug("Deleting mqtt client from MQTT broker")

    def disconnect(self):
        self._mqtt.disconnect()

    def publish(self, topic: str, value: str, retain: bool = False, absoluteTopic=False):
        if absoluteTopic:
            publish_topic = topic
        else:
            publish_topic = "{}/{}".format(self._prefix, topic)
        logger.debug("Publishing {} value of {}".format(publish_topic, value))
        self._mqtt.publish(publish_topic, value, retain=retain)

    def subscribe(self, topic: str, callback: Callable[[str], None]):
        subscribe_topic = "{}/{}".format(self._prefix, topic)
        assert(topic not in self._subMap)

        self._subMap[topic] = MqttClient.SubMapData(callback=callback)
        self._mqtt.subscribe(subscribe_topic)

    def unsubscribe(self, topic: str):
        unsubscribe_topic = "{}/{}}".format(self._prefix, topic)
        self._mqtt.unsubscribe(unsubscribe_topic)
        self._subMap.pop(topic)

    def mqtt_connected_callback(self, client: mqtt.Client, userdata, flags, rc):
        logger.info("MQTT server connect: {}".format(rc))
        # Reconnect subscriptions
        for subTopic, subData in self._subMap.items():
            self._mqtt.subscribe(subTopic)

    def mqtt_disconnect_callback(self, client: mqtt.Client, userdata, rc):
        logger.info("Mqtt disconnected: {}".format(rc))
        if rc != 0:
            remaining_tries = MqttClient.CONNECTION_RETRIES
            while True:
                try:
                    logger.info("Attempting reconnect...")
                    self._mqtt.reconnect()
                    break
                except Exception as e:
                    remaining_tries -= 1
                    logger.warning("Failed to connect to broker: {}.  {} retries remaining.".format(
                        str(e), remaining_tries))
                    if remaining_tries == 0:
                        raise
                    time.sleep(MqttClient.CONNECTION_RETRY_DELAY)
        else:
            logger.debug("Stopping Mqtt loop")
            self._mqtt.loop_stop()

    def mqtt_message_callback(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        #print(f"Received topic|message: {msg.topic}|{msg.payload.decode()}")
        for topic, data in self._subMap.items():
            topic = f"{self._prefix}/{topic.rstrip('#')}"
            if msg.topic.startswith(topic) and callable(data.callback):
                data.callback(msg)
