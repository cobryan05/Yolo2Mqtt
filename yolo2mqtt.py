import sys
import pathlib
import os
import logging
import json
import argparse
import threading
import time
from dataclasses import dataclass

# fmt: off
submodules_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "submodules")
sys.path.append(submodules_dir)
sys.path.append(os.path.join(submodules_dir, "yolov5"))
from trackerTools.yoloInference import YoloInference
from src.mqttClient import MqttClient
from src.watchedObject import WatchedObject
from src.watcher import Watcher
from src.imgSources.rtspSource import RtspSource
from src.imgSources.urlSource import UrlSource
from src.imgSources.videoSource import VideoSource
# fmt: on


CONFIG_KEY_RTSP_URL = "rtsp-url"
CONFIG_KEY_SNAPSHOT_URL = "snapshot-url"
CONFIG_KEY_VIDEO_PATH = "video-path"
CONFIG_KEY_USER = "user"
CONFIG_KEY_PWD = "password"

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Watcher")


class Yolo2Mqtt:
    @dataclass
    class _WatcherUserData:
        name: str

    def __init__(self, args: argparse.Namespace):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config: dict = json.load(open(args.config))

        mqttCfg = config.get("mqtt", {})
        mqttAddress = mqttCfg.get("address", "localhost")
        mqttPort = mqttCfg.get("port", 1883)
        mqttPrefix = mqttCfg.get("prefix", "myhome/yolo2mqtt").rstrip('/')
        self._mqttDet = mqttCfg.get("detections", "detections").rstrip('/')

        logging.info("Connecting to MQTT broker at {mqttAddress}:{mqttPort}...")

        self.mqtt: MqttClient = MqttClient(broker_address=mqttAddress,
                                           broker_port=mqttPort, prefix=mqttPrefix)

        self.models: dict[str, YoloInference] = {}
        for key, modelInfo in config.get("models", {}).items():
            self.models[key] = YoloInference(weights=modelInfo['path'],
                                             imgSize=int(modelInfo['width']),
                                             labels=modelInfo['labels'])

        self.watchers: dict[str, Watcher] = {}
        for key, cameraInfo in config.get("cameras", {}).items():
            source = Yolo2Mqtt.getSource(cameraInfo)
            if source is None:
                logging.error("Couldn't create source for [{key}]")
                continue
            modelName = cameraInfo.get("model", None)
            model = self.models[modelName]
            refreshDelay = cameraInfo.get("refresh", 5)
            userData = Yolo2Mqtt._WatcherUserData(key)
            watcher: Watcher = Watcher(source=source, model=model, refreshDelay=refreshDelay,
                                       userData=userData, debug=args.debug)
            watcher.connectNewObjSignal(self._objAddedCallback)
            watcher.connectLostObjSignal(self._objRemovedCallback)
            watcher.connectUpdatedObjSignal(self._objUpdatedCallback)
            self.watchers[key] = watcher

        logging.info(f"Starting {len(self.watchers)} watchers...")
        threads: list[threading.Thread] = []
        for key, watcher in self.watchers.items():
            thread = threading.Thread(target=watcher.run, name=key)
            threads.append(thread)
            thread.start()

        # Wait until all threads exit (forever?)
        for thread in threads:
            thread.join()

    def _objAddedCallback(self, obj, userData, **kwargs):
        # SignalSlots doesn't support annotations
        obj: WatchedObject = obj
        userData: Yolo2Mqtt._WatcherUserData = userData
        self.mqtt.publish(self._getDetTopic(obj, userData), obj.json(), retain=False)

    def _objRemovedCallback(self, obj, userData, **kwargs):
        # SignalSlots doesn't support annotations
        obj: WatchedObject = obj
        userData: Yolo2Mqtt._WatcherUserData = userData
        self.mqtt.publish(self._getDetTopic(obj, userData), None, retain=False)

    def _objUpdatedCallback(self, obj, userData, **kwargs):
        # SignalSlots doesn't support annotations
        obj: WatchedObject = obj
        userData: Yolo2Mqtt._WatcherUserData = userData
        self.mqtt.publish(self._getDetTopic(obj, userData), obj.json(), retain=False)

    def _getDetTopic(self, obj: WatchedObject, userData: _WatcherUserData) -> str:
        return f"{self._mqttDet}/{userData.name}/{obj.objId}"

    @staticmethod
    def getSource(cameraConfig: dict):
        ''' Returns a source for the given camera config'''
        if CONFIG_KEY_RTSP_URL in cameraConfig:
            return RtspSource(cameraConfig[CONFIG_KEY_RTSP_URL])
        if CONFIG_KEY_VIDEO_PATH in cameraConfig:
            return VideoSource(cameraConfig[CONFIG_KEY_VIDEO_PATH])

        if CONFIG_KEY_SNAPSHOT_URL in cameraConfig:
            user = cameraConfig.get(CONFIG_KEY_USER, None)
            pwd = cameraConfig.get(CONFIG_KEY_PWD, None)
            return UrlSource(cameraConfig[CONFIG_KEY_SNAPSHOT_URL], user=user, password=pwd)

        return None

    def run(self):
        pass


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.json")
    parser.add_argument('--debug', help="Show labeled images", action='store_true', default=False)
    parser.add_argument('--verbose', '-v', help="Verbose", action='store_true', default=False)

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    Yolo2Mqtt = Yolo2Mqtt(args)
    Yolo2Mqtt.run()
