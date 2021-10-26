import argparse
import json
import os
import pathlib
import sys

from src.mqttClient import MqttClient

submodules_dir = os.path.join(pathlib.Path('__file__').parent.resolve(), "..", "submodules")
sys.path.append(submodules_dir)
sys.path.append(os.path.join(submodules_dir, "yolov5"))


class CatTracker:
    def __init__(self, args: argparse.Namespace):
        config: dict = json.load(open(args.config))

        mqtt = config.get("mqtt", {})
        mqttAddress = mqtt.get("address", "localhost")
        mqttPort = mqtt.get("port", 1883)
        mqttPrefix = mqtt.get("prefix", "myhome/")
        print(f"Connecting to MQTT broker at {mqttAddress}:{mqttPort}...")
        self.mqtt = MqttClient(broker_address=mqttAddress,
                               broker_port=mqttPort, prefix=mqttPrefix)

    def run(self):
        pass


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.json")

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    catTracker = CatTracker(args)
    catTracker.run()
