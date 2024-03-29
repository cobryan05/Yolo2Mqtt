import sys
import pathlib
import os
import logging
import yaml
import argparse
import io
import multiprocessing as mp
import numpy as np
from PIL import Image
from multiprocessing.queues import Queue
from queue import Empty
from dataclasses import dataclass

# fmt: off
scriptDir = pathlib.Path(__file__).parent.resolve()
submodules_dir = os.path.join(scriptDir, "submodules")
sys.path.append( scriptDir )
sys.path.append(submodules_dir)
from trackerTools.yoloInference import YoloInference
from src.config import Config, Camera
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.watchedObject import WatchedObject
from src.watcher import Watcher
from src.imgSources.source import Source
from src.imgSources.rtspSource import RtspSource
from src.imgSources.urlSource import UrlSource
from src.imgSources.videoSource import VideoSource
# fmt: on

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Watcher")


KEY_ACTION_ADDED = "added"
KEY_ACTION_LOST = "lost"
KEY_ACTION_UPDATED = "updated"
KEY_ACTION_IMAGE_UPDATED = "image_updated"

class Yolo2Mqtt:
    @dataclass
    class _WatcherUserData:
        name: str

    def __init__(self, args: argparse.Namespace):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            config: dict = yaml.load(open(args.config), yaml.Loader)
        except Exception as e:
            logger.error(f"Failed to load config file {args.config}! {e}")
            config = {}
        self._config: Config = Config(config)

        # Switch to mp.dummy if requested
        if self._config.Yolo.multiprocessing:
            from multiprocessing import Process, Queue
        else:
            from multiprocessing.dummy import Process, Queue

        self._mqttDetTopic = self._config.Mqtt.detections
        self._mqttImageTopic = self._config.Mqtt.images

        logger.info(f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}...")

        self._mqtt: MqttClient = MqttClient(broker_address=self._config.Mqtt.address,
                                            broker_port=self._config.Mqtt.port,
                                            prefix=self._config.Mqtt.prefix)
        self._queue: Queue[tuple(str, WatchedObject)] = Queue()

        self._workers: list[Process] = []
        for key, cameraInfo in self._config.cameras.items():
            newWorker = Process(target=Yolo2Mqtt._workerProc, args=(
                key, self._queue, self._config, cameraInfo, args.debug))
            self._workers.append(newWorker)

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

    def _imgUpdatedCallback(self, image: Image, userData, **kwargs):
        userData: Yolo2Mqtt._WatcherUserData = userData
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        self._mqtt.publish(self._getImageTopic(userData), output_buffer.getvalue(), retain=False)

    def _getImageTopic(self, userData: _WatcherUserData ) -> str:
        return f"{self._mqttImageTopic}/{userData.name}/image"

    def _getDetTopic(self, obj: WatchedObject, userData: _WatcherUserData) -> str:
        return f"{self._mqttDetTopic}/{userData.name}/{obj.objId}"

    @staticmethod
    def _getSource(name: str, cameraConfig: Camera, rtspApi: RtspSimpleServer = None) -> Source:
        ''' Returns a source for the given camera config'''
        if cameraConfig.rtspUrl is not None:
            return RtspSource(name=name, rtspUrl=cameraConfig.rtspUrl, rtspApi=rtspApi)

        if cameraConfig.videoPath is not None:
            return VideoSource(cameraConfig.videoPath)

        if cameraConfig.imageUrl is not None:
            return UrlSource(cameraConfig.imageUrl, user=cameraConfig.username, password=cameraConfig.password)

        return None

    def run(self):
        logger.info("Starting workers...")
        for worker in self._workers:
            worker.start()

        while True:
            try:
                data: tuple[str, WatchedObject] = self._queue.get(timeout=1)
            except Empty as e:
                data = None
                # Check if workers are running
                lostProcIdxs = [idx for idx, worker in enumerate(reversed(self._workers)) if not worker.is_alive()]
                for idx in lostProcIdxs:
                    proc = self._workers[idx]
                    logger.warning(f"Worker {idx}:{proc.name} has exited.")
                    del self._workers[idx]

            if data:
                action, action_data = data
                if action == KEY_ACTION_ADDED:
                    obj, userdata = action_data
                    self._objAddedCallback(obj, userdata)
                elif action == KEY_ACTION_LOST:
                    obj, userdata = action_data
                    self._objRemovedCallback(obj, userdata)
                elif action == KEY_ACTION_UPDATED:
                    obj, userdata = action_data
                    self._objUpdatedCallback(obj, userdata)
                elif action == KEY_ACTION_IMAGE_UPDATED:
                    img, userdata = action_data
                    self._imgUpdatedCallback(img, userdata)
                else:
                    logger.warning(f"Unknown action: [{action}]")
            if len(self._workers) == 0:
                logger.info("All workers have exited")
                break

    @staticmethod
    def _workerProc(name: str, queue: Queue, config: Config, camera: Camera, debug: bool = False) -> None:
        print(f"Background thread {name}")
        logger = logging.getLogger(f"Worker_{name}")

        def fatal(msg: str):
            logger.error(msg)
            raise Exception(msg)

        try:
            rtspApi = RtspSimpleServer(apiHost=config.RtspSimpleServer.apiHost,
                                       apiPort=config.RtspSimpleServer.apiPort)
        except Exception as e:
            rtspApi = None

        modelInfo = config.models.get(camera.model, None)
        if modelInfo is None:
            fatal(f"Could not find model configuration [{camera.model}]")

        try:
            model = YoloInference(weights=modelInfo.path, imgSize=modelInfo.width,
                                  labels=modelInfo.labels, device=config.Yolo.device,
                                  yoloVersion=modelInfo.yoloVersion)
        except Exception as e:
            fatal(f"Failed to load model [{modelInfo.path}]: {e}")

        source = Yolo2Mqtt._getSource(name=name, cameraConfig=camera, rtspApi=rtspApi)
        if source is None:
            fatal(f"Could not load configured source")

        def objAddedCallback(obj, userData, **kwargs):
            queue.put((KEY_ACTION_ADDED, (obj, userData)))

        def objLostCallback(obj, userData, **kwargs):
            queue.put((KEY_ACTION_LOST, (obj, userData)))

        def objUpdatedCallback(obj, userData, **kwargs):
            queue.put((KEY_ACTION_UPDATED, (obj, userData)))

        def imageUpdatedCallback(image, userData,  **kwargs):
            queue.put((KEY_ACTION_IMAGE_UPDATED, (image, userData)))


        watcher: Watcher = Watcher(source=source, model=model, refreshDelay=camera.refresh,
                                   userData=Yolo2Mqtt._WatcherUserData(name), timelapseDir=camera.timelapseDir,
                                   timelapseInterval=camera.timelapseInterval, debug=debug)
        watcher.connectNewObjSignal(objAddedCallback)
        watcher.connectLostObjSignal(objLostCallback)
        watcher.connectUpdatedObjSignal(objUpdatedCallback)
        if camera.publishImages:
            watcher.connectImageUpdatedSignal(imageUpdatedCallback)
        watcher.run()


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.yml")
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
