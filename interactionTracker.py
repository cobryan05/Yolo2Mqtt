import json
import argparse
import time
import paho.mqtt.client as mqtt
import re
import os
import pathlib
import sys

from dataclasses import dataclass

# fmt: off
submodules_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "submodules")
sys.path.append(submodules_dir)
from src.mqttClient import MqttClient
from src.watchedObject import WatchedObject
# fmt: on

CONFIG_KEY_INTRCTS = "interactions"
CONFIG_KEY_INTRCT_THRESH = "threshold"
CONFIG_KEY_INTRCT_MIN_FRAMES = "min_frames"
CONFIG_KEY_INTRCT_OBJ_A = "first"
CONFIG_KEY_INTRCT_OBJ_B = "second"

RE_GROUP_CAMERA = "camera"
RE_GROUP_OBJID = "objectId"


class InteractionTracker:
    def __init__(self, args: argparse.Namespace):
        config: dict = json.load(open(args.config))

        mqtt = config.get("mqtt", {})
        mqttAddress = mqtt.get("address", "localhost")
        mqttPort = mqtt.get("port", 1883)
        mqttPrefix = mqtt.get("prefix", "myhome/yolo2mqtt/")
        print(f"Connecting to MQTT broker at {mqttAddress}:{mqttPort}...")

        self.mqtt: MqttClient = MqttClient(broker_address=mqttAddress,
                                           broker_port=mqttPort, prefix=mqttPrefix)

        self.mqtt.subscribe("#", self.mqttCallback)

        self._topicRe: re.Pattern = re.compile(rf"{mqttPrefix}(?P<{RE_GROUP_CAMERA}>[^/]+)/(?P<{RE_GROUP_OBJID}>.*)")

    def run(self):
        while True:
            time.sleep(10)

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        objInfo = WatchedObject.fromJson(msg.payload.decode())


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.json")
    parser.add_argument('--debug', help="Show labeled images", action='store_true', default=False)

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    interactionTracker = InteractionTracker(args)
    interactionTracker.run()
