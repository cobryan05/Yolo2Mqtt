''' Manages recordings from configured cameras '''
''' Note: Immediately queries RtspSimpleServer for active streams. Give streams
          time to warm up before starting recordingManager '''

# fmt: off
import argparse
import logging
import itertools
import os
import paho.mqtt.client as mqtt
import re
import time
import sys
import yaml
from dataclasses import dataclass
from threading import Timer
from src.config import Config
from src.mediaManager import MediaManager
from src.mqttClient import MqttClient
from src.rtspSimpleServer import RtspSimpleServer
from src.rtspDelayedProxy import RtspDelayedProxy
from src.rtspRecorder import RtspRecorder
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

    def __init__(self, delayedStream: RtspDelayedProxy, mediaManager: MediaManager, ffmpegCmd: str = "ffmpeg"):
        self._stream: RtspDelayedProxy = delayedStream
        self._recorders: dict[str, EventFileWriter] = {}
        self._mediaMan = mediaManager
        self._ffmpegCmd = ffmpegCmd

    def startEventRecording(self, eventParams: MediaManager.EventParams) -> str:
        ''' Starts recording an event, or increments refCnt of an active recording of this event'''
        ''' Returns the filename of the recorded event, or None if no recording started '''
        recording = self._recorders.get(eventParams.eventName, None)
        if recording is not None:
            if recording.stopDelayTimer is not None:
                recording.stopDelayTimer.cancel()
                recording.stopDelayTimer = None
            recording.refCnt += 1
            logger.info(
                f"Extending recording of {eventParams.eventName} from {self._stream.rtspUrl}. Refcnt is now {recording.refCnt}")
            if recording.fileRecorder.running():
                return None
            # If FFMPEG wasn't actually running then fallthrough to a new instance
            logger.warning("Recorder wasn't running! Starting new file")
            recording.fileRecorder.stop()

        filePath = self._mediaMan.getRecordingPath(eventParams)

        logger.info(f"Starting recording of {eventParams.eventName} from {self._stream.rtspUrl} to {filePath}")

        self._recorders[eventParams.eventName] = EventFileWriter(fileRecorder=RtspRecorder(
            self._stream.rtspUrl, filePath, ffmpegCmd=self._ffmpegCmd))
        return filePath

    def stopEventRecording(self, eventParams: MediaManager.EventParams):
        ''' Decrements recording refCnt, and stops the stream if it reaches zero'''
        recording = self._recorders.get(eventParams.eventName, None)
        if recording is None:
            logger.warning(f"Failed to stopEventRecording. Event not found: {eventParams.eventName}")
            return

        recording.refCnt -= 1
        if recording.refCnt > 0:
            logger.info(
                f"Decrementing refCnt of {eventParams.eventName} from {self._stream.rtspUrl}. RefCnt is now {recording.refCnt}")
        else:
            if recording.stopDelayTimer is None:
                recording.stopDelayTimer = Timer(self._stream.delay, lambda: self._stopRecording(eventParams.eventName))
                recording.stopDelayTimer.start()
                logger.info(
                    f"Starting timer for {self._stream.delay}s to stop recording of {eventParams.eventName} from {self._stream.rtspUrl}")
            else:
                logger.warning(
                    f"Timer to stop recording of {eventParams.eventName} from {self._stream.rtspUrl} already exists!")
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
                rec.stopDelayTimer = None
            if rec.fileRecorder is not None:
                rec.fileRecorder.stop()
        if self._stream is not None:
            self._stream.stop()


class RecordingManager:
    def __init__(self, args: argparse.Namespace):
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config: dict = yaml.load(open(args.config), yaml.Loader)
        self._config: Config = Config(config)

        self._rtsp = RtspSimpleServer(
            self._config.RtspSimpleServer.apiHost, self._config.RtspSimpleServer.apiPort)

        recSettings = self._config.recordingManager
        self._mediaManager = MediaManager(recSettings.mediaRoot, daysToKeep=recSettings.keepVideosDays)

        self._recs: dict[str, StreamEventRecorder] = {}

        rtspCfg = self._rtsp.GetConfig()
        rtspActiveStreams: dict[str, dict] = rtspCfg["paths"]

        logger.info("Starting up delayed streams...")
        for cameraName, camera in self._config.cameras.items():
            if cameraName in rtspActiveStreams:
                logger.info(f"Delaying {cameraName} by {camera.rewindSec}")
                delayedStream = RtspDelayedProxy(
                    publishName=f"{cameraName}_delayed",
                    srcRtspUrl=f"{self._rtsp.rtspProxyUrl}/{cameraName}",
                    rtspApi=self._rtsp,
                    delay=camera.rewindSec,
                    overwriteExisting=(False == args.dbgSkipExistingDelay),
                    ffmpegCmd=args.ffmpeg
                )
                self._recs[cameraName] = StreamEventRecorder(
                    delayedStream=delayedStream, mediaManager=self._mediaManager, ffmpegCmd=args.ffmpeg)

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

    def startRecordingEvent(self, eventParams: MediaManager.EventParams):
        if eventParams.cameraName not in self._recs:
            logger.warning(
                f"Failed to startRecordingEvent for {eventParams.cameraName}|{eventParams.interactionName}. No camera stream found."
            )
            return

        rec = self._recs[eventParams.cameraName]
        videoPath = rec.startEventRecording(eventParams)
        if videoPath is not None and self._config.recordingManager.makeSymlinks:
            self._mediaManager.createSymlinks(eventParams, videoPath)

    def stopRecordingEvent(self, eventParams: MediaManager.EventParams):
        if eventParams.cameraName not in self._recs:
            logger.warning(
                f"Failed to stopRecordingEvent for {eventParams.cameraName}|{eventParams.interactionName}. No camera stream found."
            )
            return

        rec = self._recs[eventParams.cameraName]
        rec.stopEventRecording(eventParams)

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        eventName: str = match[RE_GROUP_EVENTNAME]
        eventSlots: str = match[RE_GROUP_EVENTSLOTS]
        cameraName: str = match[RE_GROUP_CAMERA]

        eventParams = MediaManager.EventParams(
            cameraName=cameraName, interactionName=eventName, slots=eventSlots.split("/"))
        if len(msg.payload) > 0:
            logger.debug(f"Got event: {eventParams.eventName}")
            self.startRecordingEvent(eventParams)
        else:
            logger.debug(f"Cleared event: {eventParams.eventName}")
            self.stopRecordingEvent(eventParams)


def parseArgs():
    parser = argparse.ArgumentParser(description="Run object tracking on image streams",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help="Configuration file",
                        required=False, default="config.yml")
    parser.add_argument('--verbose', '-v', help="Verbose",
                        action='store_true', default=False)
    parser.add_argument(
        '--ffmpeg', help="Path to ffmpeg executable", default="ffmpeg")
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
