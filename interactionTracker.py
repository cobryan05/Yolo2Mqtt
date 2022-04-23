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
MQTT_KEY_SLOTS = "slots"

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
    slots: list[str]

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class EventValue:
    firstTimestamp: float = 0.0
    lastTimestamp: float = 0.0
    published: bool = False


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

        haCfg = self._config.get("homeassistant", {})
        self._discoveryEnabled = bool(haCfg.get("discoveryEnabled", False))
        self._discoveryPrefix = haCfg.get("discovery_prefix", "homeassistant").rstrip("/")
        self._discoveryConfigDone: set[str] = set()
        self._entityPrefix = haCfg.get("entity_prefix", "TRACKER")
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
                newEvents = context.checker.getEvents(objList)

                # Track events that were not triggered
                allKeys: set[EventKey] = set(context.events.keys())
                usedKeys: set[EventKey] = set()
                for event in newEvents:
                    slotLabels = [obj.label for obj in event.slotsObjs]
                    eventKey: EventKey = EventKey(name=event.event.name,
                                                  slots=slotLabels)
                    # Don't process multiple detections of the same event
                    if eventKey in usedKeys:
                        continue
                    usedKeys.add(eventKey)

                    trackedEvent = context.events.get(eventKey, None)
                    if trackedEvent is None:
                        trackedEvent = EventValue()
                        trackedEvent.firstTimestamp = time.time()
                        context.events[eventKey] = trackedEvent
                    else:
                        if not trackedEvent.published and time.time() > trackedEvent.firstTimestamp + event.event.minTime:
                            trackedEvent.published = True
                            self.publishEvent(context, eventKey)
                            if self._discoveryEnabled:
                                self.publishDiscoveryEvent(context, eventKey, "ON")
                    trackedEvent.lastTimestamp = time.time()

                # Check for expired events
                for eventKey in allKeys.difference(usedKeys):
                    trackedEvent = context.events.get(eventKey, None)
                    eventConfig = self._contextConfig[eventKey.name]

                    # If this even expired then clear it from MQTT and remove it from the context
                    if time.time() > trackedEvent.lastTimestamp + float(eventConfig["expire_time"]):
                        if trackedEvent.published:
                            self.publishEvent(context, eventKey, clear=True)
                            if self._discoveryEnabled:
                                self.publishDiscoveryEvent(context, eventKey, "OFF")
                        context.events.pop(eventKey)

                if self._debug:
                    InteractionTracker.debugContext(context)

    def publishEvent(self, context: Context, eventKey: EventKey, clear: bool = False):
        data = {}
        data[MQTT_KEY_EVENT_NAME] = eventKey.name
        data[MQTT_KEY_SLOTS] = eventKey.slots
        topic = self._getEventTopic(context, eventKey)
        if clear:
            self._mqtt.publish(topic, None, True)
        else:
            self._mqtt.publish(topic, json.dumps(data), False)

    def publishDiscoveryEvent(self, context: Context, eventKey: EventKey, state: str):
        entityId = f"{self._entityPrefix}-{context.name}-{eventKey.name}-{'-'.join(eventKey.slots)}"
        mqttConfigTopic = f"{self._discoveryPrefix}/binary_sensor/{entityId}"
        stateTopic = f"{mqttConfigTopic}/state"
        self._mqtt.publish(stateTopic, state, retain=True, absoluteTopic=True)
        if entityId not in self._discoveryConfigDone:
            self._discoveryConfigDone.add(entityId)
            configTopic = f"{mqttConfigTopic}/config"
            friendlyName = f"{self._entityPrefix} - [{eventKey.name}] [{context.name}] [{'|'.join(eventKey.slots)}]"
            entityCfg = {"name": friendlyName, "friendly_name": friendlyName,
                         "unique_id": entityId, "state_topic": stateTopic}
            self._mqtt.publish(configTopic, json.dumps(entityCfg), retain=True, absoluteTopic=True)

        configTopic = f"{mqttConfigTopic}/config"

    def _getEventTopic(self, context: Context, eventKey: EventKey) -> str:

        topicStr = f"{self._mqttEvents}/{context.name}/{eventKey.name}/{'/'.join(eventKey.slots)}"
        return topicStr

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
                print(f"{cameraName} Removed {objId}. Tracking {len(context.objectMap)} objects.")
            else:
                objInfo = WatchedObject.fromJson(msg.payload.decode())
                if objId not in context.objectMap:
                    print(f"{cameraName} Added {objId}. Tracking {len(context.objectMap)} objects.")
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
