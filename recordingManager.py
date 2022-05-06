import json
import argparse
import logging
import paho.mqtt.client as mqtt
import re
import time
import sys

# fmt: off
from src.config import Config
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.rtspProxyFfmpeg import RtspProxyFfmpeg
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
        self._streams: list[RtspProxyFfmpeg] = []
        for name, cfg in self._config.cameras.items():
            stream = RtspProxyFfmpeg(f"{name}_proxied", f"{self._rtsp.rtspProxyUrl}/{name}",  self._rtsp)
            self._streams.append(stream)

        self._mqttEvents = self._config.Mqtt.events

        print(f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}...")

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
        for stream in self._streams:
            stream.stop()

    def __del__(self):
        self.stop()

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        eventName: str = match[RE_GROUP_EVENTNAME]
        eventSlots: str = match[RE_GROUP_EVENTSLOTS]
        cameraName: str = match[RE_GROUP_CAMERA]
        if len(msg.payload) > 0:
            logger.debug(f"Got event for camera {cameraName} : {eventName}  params [{eventSlots}]")
        else:
            logger.debug(f"Cleared event for camera {cameraName} : {eventName}  params [{eventSlots}]")


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
