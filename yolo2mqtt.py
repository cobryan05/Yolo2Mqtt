import sys
import pathlib
import os
import logging
import yaml
import argparse
import io
import threading
import time
import numpy as np
import datetime
import cv2
from PIL import Image

# from multiprocessing.queues import Queue
from queue import Empty, Queue
from dataclasses import dataclass, field
from typing import List
from functools import partial

# fmt: off
scriptDir = pathlib.Path(__file__).parent.resolve()
submodules_dir = os.path.join(scriptDir, "submodules")
sys.path.append( scriptDir )
sys.path.append(submodules_dir)
from src.config import Config, Camera
from src.inferenceServer import InferenceServer
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.watchedObject import WatchedObject
from src.watcher import Watcher
from src.valueStatTracker import ValueStatTracker
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


@dataclass
class TimingStats:
    start: int = 0
    end: int = 0


@dataclass
class MqttItem:
    topic: str
    data: str
    retain: bool = False


@dataclass
class DebugImage:
    id: str
    image: np.array


@dataclass
class GrabberThreadCommonArgs:
    mqttQueue: Queue[MqttItem] = field(default_factory=Queue)
    dbgQueue: Queue[DebugImage] = field(default_factory=partial(Queue, 1))
    stopEvent: threading.Event = field(default_factory=threading.Event)
    rtspApi: RtspSimpleServer = None
    inferenceServer: InferenceServer = None
    mqttImagePrefix: str = None
    mqttDetPrefix: str = None
    debug: bool = False


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

        self._mqttDetTopic = self._config.Mqtt.detections
        self._mqttImageTopic = self._config.Mqtt.images

        logger.info(
            f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}..."
        )

        self._mqtt: MqttClient = MqttClient(
            broker_address=self._config.Mqtt.address,
            broker_port=self._config.Mqtt.port,
            prefix=self._config.Mqtt.prefix,
        )
        try:
            rtspApi = RtspSimpleServer(
                apiHost=self._config.RtspSimpleServer.apiHost,
                apiPort=self._config.RtspSimpleServer.apiPort,
            )
        except Exception as e:
            rtspApi = None

        self._grabberArgs: GrabberThreadCommonArgs = GrabberThreadCommonArgs(
            rtspApi=rtspApi,
            inferenceServer=InferenceServer(
                self._config.models, device=self._config.Yolo.device
            ),
            mqttDetPrefix=self._mqttDetTopic,
            mqttImagePrefix=self._mqttImageTopic,
            debug=args.debug,
        )

        self._grabberThreads: List[threading.Thread] = []
        for name, config in self._config.cameras.items():
            grabberThread = threading.Thread(
                target=Yolo2Mqtt._grabberThreadFunc,
                kwargs={
                    "id": name,
                    "config": config,
                    "common": self._grabberArgs,
                },
            )
            self._grabberThreads.append(grabberThread)

    @staticmethod
    def _grabberThreadFunc(id: str, config: Camera, common: GrabberThreadCommonArgs):
        logger.info(f"Starting grabber thread for {id}")
        source = Yolo2Mqtt._getSource(
            name=id, cameraConfig=config, rtspApi=common.rtspApi
        )
        if source is None:
            logger.error(f"Could not load configured source for '{id}'")
            return

        watcher: Watcher = Watcher(
            inferenceServer=common.inferenceServer,
            model=config.model,
            userData=Yolo2Mqtt._WatcherUserData(id),
        )

        detTopic = f"{common.mqttDetPrefix}/{id}"
        imgTopic = f"{common.mqttImagePrefix}/{id}/image"

        dbgWin = None

        nextTimelapse = None
        if config.timelapseDir and config.timelapseInterval > 0:
            try:
                os.makedirs(config.timelapseDir, mode=555, exist_ok=True)
                nextTimelapse = 0
            except Exception as e:
                logger.error(f"Failed to initialize timelapses: {e}")

        def objAddedCb(obj, userData, **kwargs):
            obj: WatchedObject = obj
            item: MqttItem = MqttItem(topic=f"{detTopic}/{obj.objId}", data=obj.json())
            common.mqttQueue.put(item)

        def objRemovedCb(obj, userData, **kwargs):
            obj: WatchedObject = obj
            item: MqttItem = MqttItem(topic=f"{detTopic}/{obj.objId}", data=None)
            common.mqttQueue.put(item)

        def imgUpdatedCb(image, userData, **kwargs):
            pubImg = image.copy()
            watcher.annotateImage(pubImg)

            imgRgb = cv2.cvtColor(pubImg, cv2.COLOR_BGR2RGB)
            pilImg: Image = Image.fromarray(imgRgb)
            output_buffer = io.BytesIO()
            pilImg.save(output_buffer, format="PNG")
            item: MqttItem = MqttItem(
                topic=f"{imgTopic}", data=output_buffer.getvalue()
            )
            common.mqttQueue.put(item)

        watcher.connectNewObjSignal(objAddedCb)
        watcher.connectLostObjSignal(objRemovedCb)
        watcher.connectUpdatedObjSignal(objAddedCb)
        if config.publishImages:
            watcher.connectImageUpdatedSignal(imgUpdatedCb)

        fetchStats: ValueStatTracker = ValueStatTracker()
        while not common.stopEvent.is_set():
            try:
                start: float = time.time()
                nextFrame = source.getNextFrame()
                fetchStats.addValue(time.time() - start)

                # Process the frame
                watcher.pushFrame(nextFrame)

                doTimelapse: bool = (
                    nextTimelapse is not None and time.time() > nextTimelapse
                )

                if common.debug:
                    annotatedImg = nextFrame.copy()
                    prefix = f"Fetch: {fetchStats.lastValue:0.2}|{fetchStats.avg:0.2}"
                    watcher.annotateImage(annotatedImg, prefixInfo=prefix)
                    common.dbgQueue.put(DebugImage(id=id, image=annotatedImg))
                else:
                    annotatedImg = None

                if doTimelapse:
                    Yolo2Mqtt.saveTimelapse(nextFrame, config.timelapseDir)
                    nextTimelapse = time.time() + config.timelapseInterval

            except Exception as e:
                logger.error(f"Exception for '{id}': {e}")
                continue

        logger.info(f"Exiting grabber thread for {id}")

    @staticmethod
    def saveTimelapse(image: np.array, outputDir: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = os.path.join(outputDir, f"{timestamp}.png")
        logger.info(f"Saving timelapse {output_path}")
        imgRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pilImg: Image = Image.fromarray(imgRgb)
        pilImg.save(output_path)

    @staticmethod
    def _getSource(
        name: str, cameraConfig: Camera, rtspApi: RtspSimpleServer = None
    ) -> Source:
        """Returns a source for the given camera config"""
        if cameraConfig.rtspUrl is not None:
            return RtspSource(name=name, rtspUrl=cameraConfig.rtspUrl, rtspApi=rtspApi)

        if cameraConfig.videoPath is not None:
            return VideoSource(cameraConfig.videoPath)

        if cameraConfig.imageUrl is not None:
            return UrlSource(
                cameraConfig.imageUrl,
                user=cameraConfig.username,
                password=cameraConfig.password,
            )

        return None

    def run(self):
        logger.info("Starting workers...")
        for worker in self._grabberThreads:
            worker.start()

        while True:
            try:
                data: MqttItem = self._grabberArgs.mqttQueue.get(timeout=1)
                self._mqtt.publish(
                    topic=data.topic, value=data.data, retain=data.retain
                )
            except Empty as e:
                pass

            try:
                dbgData: DebugImage = self._grabberArgs.dbgQueue.get(timeout=0.1)
                dbgWin = f"DebugWindow {dbgData.id}"
                cv2.imshow(dbgData.id, dbgData.image)
                cv2.waitKey(1)
            except Empty as e:
                pass
            #     # Check if workers are running
            #     lostProcIdxs = [
            #         idx
            #         for idx, worker in enumerate(reversed(self._workers))
            #         if not worker.is_alive()
            #     ]
            #     for idx in lostProcIdxs:
            #         proc = self._workers[idx]
            #         logger.warning(f"Worker {idx}:{proc.name} has exited.")
            #         del self._workers[idx]
            # if len(self._workers) == 0:
            #     logger.info("All workers have exited")
            #     break


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run object tracking on image streams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", help="Configuration file", required=False, default="config.yml"
    )
    parser.add_argument(
        "--debug", help="Show labeled images", action="store_true", default=False
    )
    parser.add_argument(
        "--verbose", "-v", help="Verbose", action="store_true", default=False
    )

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    args = parseArgs()
    Yolo2Mqtt = Yolo2Mqtt(args)
    Yolo2Mqtt.run()
