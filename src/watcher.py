''' Class to watch for objects in an image stream '''

from threading import Event
import urllib.request
import time

import cv2
import numpy as np

from trackerTools.yoloInference import YoloInference
from trackerTools.bboxTracker import BBoxTracker


METAKEY_LOST_TIMESTAMP = "losttime"
METAKEY_LABEL = "label"
METAKEY_CONFIDENCE = "conf"

# How long must an object be lost before removed
LOST_OBJ_REMOVE_DELAY = 10


class Watcher:

    def __init__(self, url: str, model: YoloInference, refreshDelay: int = 1):
        self._url: str = url
        self._model: YoloInference = model
        self._delay: int = refreshDelay
        self._stopEvent: Event = Event()
        self._bboxTracker: BBoxTracker = BBoxTracker()

    def stop(self):
        self._stopEvent.set()

    def run(self):
        print(f"Starting Watcher on URL [{self._url}], refreshing every {self._delay} seconds")
        while True:
            if self._stopEvent.wait(timeout=self._delay):
                break

            img = self._downloadImage()
            res = self._model.runInference(img)

            detections = [det[0] for det in res]
            metadata = [{METAKEY_LABEL: self._model.getLabel(det[2]),
                         METAKEY_CONFIDENCE: det[1]} for det in res]

            def metaCompare(left: dict, right: dict):
                if left.get(METAKEY_LABEL, "") == right.get(METAKEY_LABEL, ""):
                    return 1.0
                return 0.0

            trackedObjs, newObjs, lostObjs = self._bboxTracker.update(
                detections, metadata=metadata, metadataComp=metaCompare)

            for key, obj in trackedObjs.items():
                label = ""

                # TODO: Minimum acquisition time, similar to loss?
                # If we have new objects and lost objects
                if key in newObjs and len(lostObjs) > 0:
                    print("HI")
                    pass

                if key in newObjs:
                    label = "NEW "
                elif key in lostObjs:
                    label = "LOST "
                    if METAKEY_LOST_TIMESTAMP not in obj.metadata:
                        obj.metadata[METAKEY_LOST_TIMESTAMP] = time.time()
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)
                    else:
                        lostDuration = time.time() - obj.metadata[METAKEY_LOST_TIMESTAMP]
                        if lostDuration > LOST_OBJ_REMOVE_DELAY:
                            print(f"{obj.metadata[METAKEY_LABEL]} lost for {lostDuration}, removing.")
                            self._bboxTracker.removeBox(key)
                else:
                    # Ensure object isn't marked as lost
                    if METAKEY_LOST_TIMESTAMP in obj.metadata:
                        obj.metadata.pop(METAKEY_LOST_TIMESTAMP)
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)

                print(f"{key} - {obj.metadata} {label}")
            print("------")

        print("Exit")

    def _downloadImage(self):
        req = urllib.request.urlopen(self._url)
        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
