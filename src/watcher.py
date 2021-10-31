''' Class to watch for objects in an image stream '''

from threading import Event
import time
import cv2
import numpy as np

from trackerTools.yoloInference import YoloInference
from trackerTools.bboxTracker import BBoxTracker
from . imgSources.source import Source

METAKEY_LOST_FRAMES = "losttime"
METAKEY_LABEL = "label"
METAKEY_CONF = "conf"
METAKEY_CONF_DICT = "confdict"

# How long must an object be lost before removed
LOST_OBJ_REMOVE_FRAME_CNT = 10
BBOX_TRACKER_MAX_DIST_THRESH = 0.5


class Watcher:
    class ConfDictEntry:
        def __init__(self, conf: float = None):
            if conf is not None:
                self._sum = conf
                self._count = 1
            else:
                self._sum = 0
                self._count = 0

        def __repr__(self):
            return f"AvgConf: {self.avg}   Cnt: {self._count}"

        def addConf(self, conf: float):
            self._sum += conf
            self._count += 1

        @property
        def avg(self) -> float:
            return self._sum / self._count

    def __init__(self, source: Source, model: YoloInference, refreshDelay: float = 1.0, debug: bool = False):
        self._source: Source = source
        self._model: YoloInference = model
        self._delay: float = refreshDelay
        self._stopEvent: Event = Event()
        self._bboxTracker: BBoxTracker = BBoxTracker(distThresh=BBOX_TRACKER_MAX_DIST_THRESH)
        self._debug = debug

    def stop(self):
        self._stopEvent.set()

    def run(self):
        print(f"Starting Watcher with [{self._source}], refreshing every {self._delay} seconds")

        if self._debug:
            dbgWin = f"DebugWindow {self._source}"
            cv2.namedWindow(dbgWin, flags=cv2.WINDOW_NORMAL)
        else:
            dbgWin = None

        while True:
            if self._stopEvent.wait(timeout=self._delay):
                break

            img = self._source.getNextFrame()
            res = self._model.runInference(img)

            detections = []
            metadata = []
            SAME_BOX_THRESH = 0.01
            for bbox, conf, objClass, label in res:

                # Check if this may be a second detection of the same object
                dupIdx = -1
                for idx, prevDet in enumerate(detections):
                    if bbox.similar(prevDet, SAME_BOX_THRESH):
                        dupIdx = idx
                        break

                # Merge any duplicate boxes into one
                if dupIdx != -1:
                    metadatum = metadata[dupIdx]
                    metadatum[METAKEY_CONF_DICT][label] = Watcher.ConfDictEntry(conf)
                    # Label the detection as the higher confidence label
                    if conf > metadatum[METAKEY_CONF]:
                        metadatum[METAKEY_CONF] = conf
                        metadatum[METAKEY_LABEL] = label
                else:
                    detections.append(bbox)
                    metadata.append({METAKEY_LABEL: label,
                                    METAKEY_CONF: conf,
                                    METAKEY_CONF_DICT: {label: Watcher.ConfDictEntry(conf)}})

            def metaCompare(left: dict, right: dict):
                if left.get(METAKEY_LABEL, "") == right.get(METAKEY_LABEL, ""):
                    return 1.0
                return 0.5

            trackedObjs, newObjs, lostObjs, detectedKeys = self._bboxTracker.update(
                detections, metadata=metadata, metadataComp=metaCompare)

            for key, obj in trackedObjs.items():
                newOrLostLabel = ""

                # TODO: Minimum acquisition time, similar to loss?

                if key in newObjs:
                    newOrLostLabel = "NEW "
                elif key in lostObjs:
                    newOrLostLabel = "LOST "
                    if METAKEY_LOST_FRAMES not in obj.metadata:
                        obj.metadata[METAKEY_LOST_FRAMES] = 0
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)
                    else:
                        lostDuration = obj.metadata[METAKEY_LOST_FRAMES] + 1
                        obj.metadata[METAKEY_LOST_FRAMES] = lostDuration
                        if lostDuration > LOST_OBJ_REMOVE_FRAME_CNT:
                            print(f"{obj.metadata[METAKEY_LABEL]} lost for {lostDuration}, removing.")
                            self._bboxTracker.removeBox(key)
                else:
                    # Ensure object isn't marked as lost
                    if METAKEY_LOST_FRAMES in obj.metadata:
                        obj.metadata.pop(METAKEY_LOST_FRAMES)
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)

                    # Was this item part of our last detection update?
                    if key in detectedKeys:
                        objDetIdx = detectedKeys.index(key)
                        detMeta: dict = metadata[objDetIdx]
                        detConfDict: dict[str, Watcher.ConfDictEntry] = detMeta[METAKEY_CONF_DICT]
                        objConfDict: dict[str, Watcher.ConfDictEntry] = obj.metadata[METAKEY_CONF_DICT]

                        # Add the current detection confidences to the tracked confidences
                        for label, entry in detConfDict.items():
                            objConfEntry: Watcher.ConfDictEntry = objConfDict.setdefault(label, Watcher.ConfDictEntry())
                            objConfEntry.addConf(entry.avg)

                        # Take the highest average confidence
                        highConfKey: str = max(objConfDict, key=lambda key: objConfDict[key].avg)
                        obj.metadata[METAKEY_LABEL] = highConfKey
                        obj.metadata[METAKEY_CONF] = objConfDict[highConfKey].avg
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)

                print(f"{key} - {obj.metadata} {newOrLostLabel}")

            if self._debug:
                dbgImg = img.copy()
                for key, tracker in self._bboxTracker.getTrackedObjects().items():
                    Watcher.drawTrackerOnImage(dbgImg, tracker)
                cv2.imshow(dbgWin, dbgImg)
                cv2.waitKey(1)

            print("------")

        print("Exit")

    @staticmethod
    def drawTrackerOnImage(img: np.array, tracker: BBoxTracker.Tracker, color: tuple[int, int, int] = (255, 255, 255)):
        if METAKEY_LOST_FRAMES in tracker.metadata:
            lostFrames = tracker.metadata[METAKEY_LOST_FRAMES]
            red = 255 * (LOST_OBJ_REMOVE_FRAME_CNT - lostFrames)/LOST_OBJ_REMOVE_FRAME_CNT
            color = (0, 0, red)

        imgY, imgX = img.shape[:2]
        x1, y1, x2, y2 = tracker.bbox.asX1Y1X2Y2(imgX, imgY)
        cv2.rectangle(img, (x1, y1), (x2, y2), color)

        objClass = tracker.metadata.get(METAKEY_LABEL, "")
        objConf = tracker.metadata.get(METAKEY_CONF, 0.0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{tracker.key} - {objClass} {objConf:0.2}"
        cv2.putText(img, label, (x1, y1 + 16), font, 0.6, color, 1, cv2.LINE_AA)
