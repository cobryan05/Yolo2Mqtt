''' Manages recordings from configured cameras '''
''' Note: Immediately queries RtspSimpleServer for active streams. Give streams
          time to warm up before starting recordingManager '''

# fmt: off
import json
import argparse
import logging
import os
import paho.mqtt.client as mqtt
import re
import time
import sys
from dataclasses import dataclass
from threading import Timer
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


@dataclass
class EventFileWriter:
    ''' Active file recorder. Records from delayed stream to disk'''
    fileRecorder: RtspRecorder = None

    ''' Timer to stop fileRecorder after a delay '''
    stopDelayTimer: Timer = None

    ''' Count of number of outstanding requests to record this event '''
    refCnt: int = 1


class StreamEventRecorder:
    ''' Manages multiple recordings from one delayed stream
        Delays stopping a recording, and will resume the same recording if the same
        event is received before the recording stop delay expires '''

    def __init__(self, delayedStream: RtspProxyFfmpeg):
        self._stream: RtspProxyFfmpeg = delayedStream
        self._recorders: dict[str, EventFileWriter] = {}

    def startEventRecording(self, eventName: str, outputDir: str):
        ''' Starts recording an event, or increments refCnt of an active recording of this event'''
        recording = self._recorders.get(eventName, None)
        if recording is not None:
            if recording.stopDelayTimer is not None:
                recording.stopDelayTimer.cancel()
                recording.stopDelayTimer = None
            recording.refCnt += 1
            logger.info(
                f"Extending recording of {eventName} from {self._stream.rtspUrl}. Refcnt is now {recording.refCnt}")
            return

        logger.info(f"Starting recording of {eventName} from {self._stream.rtspUrl}")

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f"{timestamp}__{eventName}.mkv"
        filePath = os.path.join(outputDir, fileName)
        self._recorders[eventName] = EventFileWriter(fileRecorder=RtspRecorder(self._stream.rtspUrl, filePath))

    def stopEventRecording(self, eventName: str):
        ''' Decrements recording refCnt, and stops the stream if it reaches zero'''
        recording = self._recorders.get(eventName, None)
        if recording is None:
            logger.warning(f"Failed to stopEventRecording. Event not found: {eventName}")
            return

        recording.refCnt -= 1
        if recording.refCnt > 0:
            logger.info(
                f"Decrementing refCnt of {eventName} from {self._stream.rtspUrl}. RefCnt is now {recording.refCnt}")
        else:
            assert(recording.stopDelayTimer is None)
            recording.stopDelayTimer = Timer(self._stream.delay, lambda: self._stopRecording(eventName))
            recording.stopDelayTimer.start()
            logger.info(
                f"Starting timer to stop recording of {eventName} from {self._stream.rtspUrl}")
        return

    def _stopRecording(self, eventName: str):
        ''' Stop an event recording '''
        recording = self._recorders.pop(eventName, None)
        if recording is None:
            logger.error(f"_stopRecording called, but event [{eventName}] not active!")
            return
        logger.info(f"Stopping recording [{eventName}] from [{self._stream.rtspUrl}]")
        recording.fileRecorder.stop()

    def stopAll(self):
        for rec in self._recorders.values():
            if rec.stopDelayTimer is not None:
                rec.stopDelayTimer.cancel()
            if rec.fileRecorder is not None:
                rec.fileRecorder.stop()
        if self._stream is not None:
            self._stream.stop()


class RecordingManager:
    def __init__(self, args: argparse.Namespace):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config: dict = json.load(open(args.config))
        self._config: Config = Config(config)
        Ffmpeg.setFFmpegPath(args.ffmpeg)

        self._rtsp = RtspSimpleServer(
            self._config.RtspSimpleServer.apiHost, self._config.RtspSimpleServer.apiPort)

        self._recs: dict[str, StreamEventRecorder] = {}

        rtspCfg = self._rtsp.GetConfig()
        rtspActiveStreams: dict[str, dict] = rtspCfg["paths"]

        logger.info("Starting up delayed streams...")
        for cameraName, camera in self._config.cameras.items():
            if cameraName in rtspActiveStreams:
                logger.info(f"Delaying {cameraName} by {camera.rewindSec}")
                delayedStream = RtspProxyFfmpeg(
                    publishName=f"{cameraName}_delayed",
                    srcRtspUrl=f"{self._rtsp.rtspProxyUrl}/{cameraName}",
                    rtspApi=self._rtsp,
                    delay=camera.rewindSec,
                    skipExisting=args.dbgSkipExistingDelay
                )
                self._recs[cameraName] = StreamEventRecorder(delayedStream=delayedStream)

        logger.info(
            f"Connecting to MQTT broker at {self._config.Mqtt.address}:{self._config.Mqtt.port}...")

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
        for rec in self._recs.values():
            rec.stopAll()

    def __del__(self):
        self.stop()

    @staticmethod
    def _createEventName(cameraName: str, eventName: str, slots: list[str]):
        return f"{cameraName}__{eventName}__{slots.replace('/', '_')}"

    def startRecordingEvent(self, cameraName: str, eventName: str):
        if cameraName not in self._recs:
            logger.warning(
                f"Failed to startRecordingEvent for {cameraName}|{eventName}. No camera stream found."
            )
            return

        rec = self._recs[cameraName]
        rec.startEventRecording(eventName=eventName, outputDir=self._config.recordingManager.mediaRoot)

    def stopRecordingEvent(self, cameraName: str, eventName: str):
        if cameraName not in self._recs:
            logger.warning(
                f"Failed to stopRecordingEvent for {cameraName}|{eventName}. No camera stream found."
            )
            return

        rec = self._recs[cameraName]
        rec.stopEventRecording(eventName)

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        eventName: str = match[RE_GROUP_EVENTNAME]
        eventSlots: str = match[RE_GROUP_EVENTSLOTS]
        cameraName: str = match[RE_GROUP_CAMERA]

        eventName = RecordingManager._createEventName(
            cameraName=cameraName, eventName=eventName, slots=eventSlots)
        if len(msg.payload) > 0:
            logger.debug(f"Got event: {eventName}")
            self.startRecordingEvent(cameraName, eventName)
        else:
            logger.debug(f"Cleared event: {eventName}")
            self.stopRecordingEvent(cameraName, eventName)


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.json")
    parser.add_argument('--verbose', '-v', help="Verbose",
                        action='store_true', default=False)
    parser.add_argument(
        '--ffmpeg', help="Path to ffmpeg executable", default="/usr/bin/ffmpeg")
    parser.add_argument(
        '--dbgSkipExistingDelay', action='store_true', help="If true, do not overwrite existing delayed streams", default=False)
    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    recManager = RecordingManager(parseArgs())
    recManager.run()
    recManager.stop()
