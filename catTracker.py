import sys
import pathlib
import os
import json
import argparse
import threading
import time

# fmt: off
submodules_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "submodules")
sys.path.append(submodules_dir)
sys.path.append(os.path.join(submodules_dir, "yolov5"))
from trackerTools.yoloInference import YoloInference
from src.mqttClient import MqttClient
from src.watcher import Watcher
from src.imgSources.urlSource import UrlSource
from src.imgSources.videoSource import VideoSource
# fmt: on


CONFIG_KEY_SNAPSHOT_URL = "snapshot-url"
CONFIG_KEY_VIDEO_PATH = "video-path"


class CatTracker:
    def __init__(self, args: argparse.Namespace):
        config: dict = json.load(open(args.config))

        mqtt = config.get("mqtt", {})
        mqttAddress = mqtt.get("address", "localhost")
        mqttPort = mqtt.get("port", 1883)
        mqttPrefix = mqtt.get("prefix", "myhome/")
        print(f"Connecting to MQTT broker at {mqttAddress}:{mqttPort}...")

        self.mqtt: MqttClient = MqttClient(broker_address=mqttAddress,
                                           broker_port=mqttPort, prefix=mqttPrefix)

        self.models: dict[str, YoloInference] = {}
        for key, modelInfo in config.get("models", {}).items():
            self.models[key] = YoloInference(weights=modelInfo['path'],
                                             imgSize=int(modelInfo['width']),
                                             labels=modelInfo['labels'])

        self.watchers: dict[str, Watcher] = {}
        for key, cameraInfo in config.get("cameras", {}).items():
            source = CatTracker.getSource(cameraInfo)
            if source is None:
                print("Couldn't create source for [{key}]")
                continue
            modelName = cameraInfo.get("model", None)
            model = self.models[modelName]
            refreshDelay = cameraInfo.get("refresh", 5)
            self.watchers[key] = Watcher(source=source, model=model, refreshDelay=refreshDelay, debug=args.debug)

        print(f"Starting {len(self.watchers)} watchers...")
        threads: list[threading.Thread] = []
        for key, watcher in self.watchers.items():
            thread = threading.Thread(target=watcher.run, name=key)
            threads.append(thread)
            thread.start()

        # Wait until all threads exit (forever?)
        for thread in threads:
            thread.join()

    @staticmethod
    def getSource(cameraConfig: dict):
        ''' Returns a source for the given camera config'''
        if CONFIG_KEY_VIDEO_PATH in cameraConfig:
            return VideoSource(cameraConfig[CONFIG_KEY_VIDEO_PATH])

        if CONFIG_KEY_SNAPSHOT_URL in cameraConfig:
            return UrlSource(cameraConfig[CONFIG_KEY_SNAPSHOT_URL])

        return None

    def run(self):
        pass


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
    catTracker = CatTracker(args)
    catTracker.run()
