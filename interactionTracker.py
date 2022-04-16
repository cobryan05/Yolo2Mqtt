import json
import argparse
import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import re
import os
import pathlib
import sys

from dataclasses import dataclass, field
from threading import Lock


# fmt: off
submodules_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "submodules")
sys.path.append(submodules_dir)
from src.contextChecker import ContextChecker
from src.mqttClient import MqttClient
from src.watchedObject import WatchedObject
from src.watcher import Watcher
# fmt: on

MQTT_KEY_EVENT_NAME = "name"
MQTT_KEY_FIRST = "first"
MQTT_KEY_SECOND = "second"

CONFIG_KEY_INTRCTS = "interactions"

RE_GROUP_CAMERA = "camera"
RE_GROUP_OBJID = "objectId"


@dataclass
class TrackedObject:
    obj: WatchedObject


@dataclass
class TrackedLabel:
    ids: set[int] = field(default_factory=set)


@dataclass
class EventKey:
    name: str
    first: str
    second: str

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.name == other.name and self.first == other.first and self.second == other.second


@dataclass
class EventValue:
    firstTimestamp: float = 0.0
    lastTimestamp: float = 0.0
    triggered: bool = False


@dataclass
class Context:
    name: str
    checker: ContextChecker
    objectMap: dict[int, TrackedObject] = field(default_factory=dict)
    events: dict[EventKey, EventValue] = field(default_factory=dict)


class InteractionTracker:
    def __init__(self, args: argparse.Namespace):
        self._config: dict = json.load(open(args.config))
        self._debug = args.debug
        self._lock: Lock = Lock()

        mqttCfg = self._config.get("mqtt", {})
        mqttAddress = mqttCfg.get("address", "localhost")
        mqttPort = mqttCfg.get("port", 1883)
        mqttPrefix = mqttCfg.get("prefix", "myhome/yolo2mqtt/").rstrip('/)')
        self._mqttEvents = mqttCfg.get("events", "events").rstrip('/')
        self._mqttDet = mqttCfg.get("detections", "detections").rstrip('/')
        print(f"Connecting to MQTT broker at {mqttAddress}:{mqttPort}...")

        self._mqtt: MqttClient = MqttClient(broker_address=mqttAddress,
                                            broker_port=mqttPort, prefix=mqttPrefix)

        self._mqtt.subscribe(f"{self._mqttDet}/#", self.mqttCallback)

        self._contextConfig: dict = self._config.get(CONFIG_KEY_INTRCTS, {})
        self._contexts: dict[str, Context] = {}
        self._topicRe: re.Pattern = re.compile(
            rf"{mqttPrefix}/{self._mqttDet}/(?P<{RE_GROUP_CAMERA}>[^/]+)/(?P<{RE_GROUP_OBJID}>.*)")

    def run(self):
        while True:
            time.sleep(1)
            self.checkForEvents()

    def checkForEvents(self):
        with self._lock:
            for key, context in self._contexts.items():
                objList = [trackedObj.obj for trackedObj in context.objectMap.values()]
                events = context.checker.getEvents(objList)

                # Track events that were not triggered
                missedEvents = set(context.events.keys())
                for event in events:
                    eventKey: EventKey = EventKey(name=event.event.name,
                                                  first=event.first.label,
                                                  second=event.second.label)
                    trackedEvent = context.events.get(eventKey, None)
                    if trackedEvent is None:
                        trackedEvent = EventValue()
                        trackedEvent.firstTimestamp = time.time()
                        context.events[eventKey] = trackedEvent
                    else:
                        if eventKey in missedEvents:
                            missedEvents.remove(eventKey)
                        else:
                            print(f"{eventKey} was not in missedEvents")

                        if not trackedEvent.triggered and time.time() > trackedEvent.firstTimestamp + event.event.minTime:
                            trackedEvent.triggered = True
                            self.publishEvent(context.name, eventKey)
                    trackedEvent.lastTimestamp = time.time()

                # Check for expired events
                for eventKey in missedEvents:
                    event = context.events[eventKey]
                    eventConfig = self._contextConfig[eventKey.name]
                    if time.time() > event.lastTimestamp + float(eventConfig["expire_time"]):
                        if event.triggered:
                            self.publishEvent(context.name, eventKey, clear=True)
                        context.events.pop(eventKey)

                if self._debug:
                    InteractionTracker.debugContext(context)

    def publishEvent(self, contextName: str, eventKey: EventKey, clear: bool = False):
        data = {}
        data[MQTT_KEY_EVENT_NAME] = eventKey.name
        data[MQTT_KEY_FIRST] = eventKey.first
        data[MQTT_KEY_SECOND] = eventKey.second
        topic = self._getEventTopic(contextName, eventKey)
        if clear:
            self._mqtt.publish(topic, None, True)
        else:
            self._mqtt.publish(topic, json.dumps(data), False)

    def _getEventTopic(self, contextName: str, eventKey: EventKey) -> str:
        return f"{self._mqttEvents}/{contextName}/{eventKey.name}"

    def mqttCallback(self, msg: mqtt.MQTTMessage):
        match = self._topicRe.match(msg.topic)
        objId: int = int(match[RE_GROUP_OBJID])
        cameraName: str = match[RE_GROUP_CAMERA]

        with self._lock:
            context = self._contexts.get(cameraName, None)
            # New context?
            if context is None:
                context = Context(name=cameraName, checker=ContextChecker(self._contextConfig))
                self._contexts[cameraName] = context

            if len(msg.payload) == 0:
                context.objectMap.pop(objId, None)
                print(f"Removed {objId}. Tracking {len(context.objectMap)} objects.")
            else:
                objInfo = WatchedObject.fromJson(msg.payload.decode())
                if objId not in context.objectMap:
                    print(f"Added {objId}. Tracking {len(context.objectMap)} objects.")
                context.objectMap[objId] = TrackedObject(obj=objInfo)

    @staticmethod
    def debugContext(context: Context):
        cv2.namedWindow(context.name, flags=cv2.WINDOW_NORMAL)
        dbgImg = np.zeros((1000, 1000, 3))
        for key, trackedObj in context.objectMap.items():
            obj: WatchedObject = trackedObj.obj
            Watcher.drawBboxOnImage(dbgImg, obj.bbox)
            Watcher.drawBboxLabel(dbgImg, obj.bbox, f"{key} {obj.label}")
        cv2.imshow(context.name, dbgImg)
        cv2.waitKey(1)


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
    interactionTracker = InteractionTracker(args)
    interactionTracker.run()
