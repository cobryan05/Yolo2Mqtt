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
scriptDir = pathlib.Path(__file__).parent.resolve()
submodules_dir = os.path.join(scriptDir, "submodules")
sys.path.append( scriptDir )
sys.path.append(submodules_dir)
sys.path.append(os.path.join(submodules_dir, "yolov5"))
from trackerTools.yoloInference import YoloInference
from src.config import Config, Camera
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.watchedObject import WatchedObject
from src.watcher import Watcher
from src.imgSources.rtspSource import RtspSource
from src.imgSources.urlSource import UrlSource
from src.imgSources.videoSource import VideoSource
# fmt: on

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
        self._config: Config = Config(config)

        self._mqttDet = self._config.Mqtt.detections

        logger.info(f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}...")

        self._mqtt: MqttClient = MqttClient(broker_address=self._config.Mqtt.address,
                                            broker_port=self._config.Mqtt.port,
                                            prefix=self._config.Mqtt.prefix)

        try:
            self._rtspApi = RtspSimpleServer(apiHost=self._config.RtspSimpleServer.apiHost,
                                             apiPort=self._config.RtspSimpleServer.apiPort)
        except Exception as e:
            self._rtspApi = None

        self._models: dict[str, YoloInference] = {}
        for key, modelInfo in self._config.models.items():
            self._models[key] = YoloInference(weights=modelInfo.path,
                                              imgSize=modelInfo.width,
                                              labels=modelInfo.labels)

        self.watchers: dict[str, Watcher] = {}
        for key, cameraInfo in self._config.cameras.items():
            source = self.getSource(name=key, cameraConfig=cameraInfo)
            if source is None:
                logging.error("Couldn't create source for [{key}]")
                continue
            modelName = cameraInfo.model
            model = self._models[modelName]
            refreshDelay = cameraInfo.refresh
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
        self._mqtt.publish(self._getDetTopic(obj, userData), obj.json(), retain=False)

    def _objRemovedCallback(self, obj, userData, **kwargs):
        # SignalSlots doesn't support annotations
        obj: WatchedObject = obj
        userData: Yolo2Mqtt._WatcherUserData = userData
        self._mqtt.publish(self._getDetTopic(obj, userData), None, retain=False)

    def _objUpdatedCallback(self, obj, userData, **kwargs):
        # SignalSlots doesn't support annotations
        obj: WatchedObject = obj
        userData: Yolo2Mqtt._WatcherUserData = userData
        self._mqtt.publish(self._getDetTopic(obj, userData), obj.json(), retain=False)

    def _getDetTopic(self, obj: WatchedObject, userData: _WatcherUserData) -> str:
        return f"{self._mqttDet}/{userData.name}/{obj.objId}"

    def getSource(self, name: str, cameraConfig: Camera):
        ''' Returns a source for the given camera config'''
        if cameraConfig.rtspUrl is not None:
            return RtspSource(name=name, rtspUrl=cameraConfig.rtspUrl, rtspApi=self._rtspApi)

        if cameraConfig.videoPath is not None:
            return VideoSource(cameraConfig.videoPath)

        if cameraConfig.imageUrl is not None:
            return UrlSource(cameraConfig.imageUrl, user=cameraConfig.username, password=cameraConfig.password)

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
