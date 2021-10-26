import paho.mqtt.client as mqtt
import time


class MqttClient:
    KEY_MQTT_TOPIC = "mqtt-topic"
    KEY_MQTT_RETAIN = "mqtt-retain"

    KEEPALIVE_TIME = 60
    CONNECTION_RETRIES = 10
    CONNECTION_RETRY_DELAY = 5

    def __init__(self, broker_address, prefix="myhome/", broker_port=1883):
        self._prefix = prefix
        self._mqtt = mqtt.Client()
        self._mqtt.on_connect = self.mqtt_connected_callback
        self._mqtt.on_message = self.mqtt_message_callback
        self._mqtt.on_disconnect = self.mqtt_disconnect_callback

        remaining_tries = MqttClient.CONNECTION_RETRIES
        while True:
            try:
                self._mqtt.connect(broker_address, broker_port, keepalive=MqttClient.KEEPALIVE_TIME)
                break
            except Exception as e:
                remaining_tries -= 1
                print("Failed to connect to broker: {}.  {} retries remaining.".format(str(e), remaining_tries))
                if remaining_tries == 0:
                    raise
                time.sleep(MqttClient.CONNECTION_RETRY_DELAY)
        self._mqtt.loop_start()

    def __del__(self):
        self.disconnect()
        print("Deleting mqtt client from MQTT broker")

    def disconnect(self):
        self._mqtt.disconnect()

    def publish(self, topic, value, retain=False):
        publish_topic = "{}{}{}".format(self._prefix, "" if self._prefix.endswith("/") else "/", topic)
        print("Publishing {} value of {}".format(publish_topic, value))
        self._mqtt.publish(publish_topic, value, retain=retain)

    def mqtt_connected_callback(self, client, userdata, flags, rc):
        print("MQTT server connect: {}".format(rc))
        pass

    def mqtt_disconnect_callback(self, client, userdata, rc):
        print("Mqtt disconnected: {}".format(rc))
        if rc != 0:
            print("Attempting reconnect...")
            self._mqtt.reconnect()
        else:
            print("Stopping Mqtt loop")
            self._mqtt.loop_stop()

    def mqtt_message_callback(self, client, userdata, msg):
        print("Received message: {}".format(msg))
