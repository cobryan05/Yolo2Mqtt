''' Class to watch for objects in an image stream '''

from threading import Event
import time

from trackerTools.yoloInference import YoloInference
from trackerTools.bboxTracker import BBoxTracker
from . imgSources.source import Source

METAKEY_LOST_TIMESTAMP = "losttime"
METAKEY_LABEL = "label"
METAKEY_CONFIDENCE = "conf"

# How long must an object be lost before removed
LOST_OBJ_REMOVE_DELAY = 10


class Watcher:

    def __init__(self, source: Source, model: YoloInference, refreshDelay: float = 1.0, debug: bool = False):
        self._source: Source = source
        self._model: YoloInference = model
        self._delay: float = refreshDelay
        self._stopEvent: Event = Event()
        self._bboxTracker: BBoxTracker = BBoxTracker()

    def stop(self):
        self._stopEvent.set()

    def run(self):
        print(f"Starting Watcher with [{self._source}], refreshing every {self._delay} seconds")
        while True:
            if self._stopEvent.wait(timeout=self._delay):
                break

            img = self._source.getNextFrame()
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
