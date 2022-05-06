import json
import argparse
import logging
import os
import paho.mqtt.client as mqtt
import re
import time
import sys

# fmt: off
from src.config import Config
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.rtspProxyFfmpeg import RtspProxyFfmpeg
from src.rtspRecorder import RtspRecorder
from src.ffmpeg import Ffmpeg
# fmt: on

RE_GROUP_CAMERA = "camera"
RE_GROUP_EVENTNAME = "eventName"
RE_GROUP_EVENTSLOTS = "eventSlots"

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RecordingManager")


class RecordingManager:
    def __init__(self, args: argparse.Namespace):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config: dict = json.load(open(args.config))
        self._config: Config = Config(config)
        self._debug = args.debug
        Ffmpeg.setFFmpegPath(args.ffmpeg)

        self._rtsp = RtspSimpleServer(self._config.RtspSimpleServer.apiHost, self._config.RtspSimpleServer.apiPort)
        self._activeRecs: dict[str, RtspRecorder] = {}

        logger.info(f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}...")

        self._mqttEvents = self._config.Mqtt.events
        self._mqtt: MqttClient = MqttClient(broker_address=self._config.Mqtt.address,
                                            broker_port=self._config.Mqtt.port,
                                            prefix=self._config.Mqtt.prefix)

        self._mqtt.subscribe(f"{self._mqttEvents}/#", self.mqttCallback)
        self._topicRe: re.Pattern = re.compile(
            rf"{self._config.Mqtt.prefix}/{self._mqttEvents}/(?P<{RE_GROUP_CAMERA}>[^/]+)/(?P<{RE_GROUP_EVENTNAME}>[^/]+)/(?P<{RE_GROUP_EVENTSLOTS}>.*)")

    def run(self):
        while True:
            time.sleep(1)

    def stop(self):
        for rec in self._activeRecs:
            rec.stop()

    def __del__(self):
        self.stop()

    @staticmethod
    def _createEventName(cameraName: str, eventName: str, slots: list[str]):
        return f"{cameraName}__{eventName}__{slots.replace('/', '_')}"

    def startRecordingEvent(self, cameraName: str, eventName: str):
        if eventName in self._activeRecs:
            logger.warning(f"There is already an active recording for {eventName}!")
            return

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        fileName = f"{timestamp}__{eventName}.mkv"
        filePath = os.path.join(self._config.recordingManager.mediaRoot, fileName)

        logger.info(f"Starting recording of {eventName} to file {fileName}")
        cameraName = "loft"
        rtspUrl = f"{self._rtsp.rtspProxyUrl}/{cameraName}"
        self._activeRecs[eventName] = RtspRecorder(rtspUrl, filePath)

    def stopRecordingEvent(self, eventName: str):
        if eventName not in self._activeRecs:
            logger.warning(f"Tried to stop recording of non-active event: {eventName}")
            return
        logger.info(f"Stopping recording of {eventName}")
        rec = self._activeRecs.pop(eventName)
        rec.stop()

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        eventName: str = match[RE_GROUP_EVENTNAME]
        eventSlots: str = match[RE_GROUP_EVENTSLOTS]
        cameraName: str = match[RE_GROUP_CAMERA]

        eventName = RecordingManager._createEventName(cameraName=cameraName, eventName=eventName, slots=eventSlots)
        if len(msg.payload) > 0:
            logger.debug(f"Got event: {eventName}")
            self.startRecordingEvent(cameraName, eventName)
        else:
            logger.debug(f"Cleared event: {eventName}")
            self.stopRecordingEvent(eventName)


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.json")
    parser.add_argument('--debug', help="Show labeled images", action='store_true', default=False)
    parser.add_argument('--verbose', '-v', help="Verbose", action='store_true', default=False)
    parser.add_argument('--ffmpeg', help="Path to ffmpeg executable", default="/usr/bin/ffmpeg")
    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    recManager = RecordingManager(parseArgs())
    recManager.run()
    recManager.stop()
