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
METAKEY_FRAME_CNT = "frameCnt"


LOST_OBJ_REMOVE_FRAME_CNT = 20  # How long must an object be lost before removed
NEW_OBJ_MIN_FRAME_CNT = 2  # How many frames must a new object be present in before considered new
BBOX_TRACKER_MAX_DIST_THRESH = 0.5  # Percent of image a box can move and still be matched


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

            # Process each tracked item
            for key, obj in trackedObjs.items():

                if key in newObjs:
                    # Initialize new object metadata
                    obj.metadata[METAKEY_FRAME_CNT] = 0
                    obj.metadata[METAKEY_LOST_FRAMES] = 0
                    self._bboxTracker.updateBox(key, metadata=obj.metadata)
                elif key in lostObjs:
                    # If it was lost before reaching the minimum frame count then remove it
                    if obj.metadata[METAKEY_FRAME_CNT] < NEW_OBJ_MIN_FRAME_CNT:
                        print(f"{obj.metadata[METAKEY_LABEL]} lost before minimum frame count")
                        self._bboxTracker.removeBox(key)
                    else:
                        lostFrames = obj.metadata[METAKEY_LOST_FRAMES] + 1
                        obj.metadata[METAKEY_LOST_FRAMES] = lostFrames
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)

                        if lostFrames > LOST_OBJ_REMOVE_FRAME_CNT:
                            print(f"{obj.metadata[METAKEY_LABEL]} lost for {lostFrames}, removing.")
                            self._bboxTracker.removeBox(key)
                else:
                    # A previously tracked object, ensure it isn't marked as lost
                    obj.metadata[METAKEY_LOST_FRAMES] = 0
                    obj.metadata[METAKEY_FRAME_CNT] += 1

                    # Update the object if it was present in our most recent detection
                    if key in detectedKeys:
                        objDetIdx = detectedKeys.index(key)
                        detMeta: dict = metadata[objDetIdx]
                        detConfDict: dict[str, Watcher.ConfDictEntry] = detMeta[METAKEY_CONF_DICT]
                        objConfDict: dict[str, Watcher.ConfDictEntry] = obj.metadata[METAKEY_CONF_DICT]

                        # Add the current detection confidences to the tracked confidences
                        for label, entry in detConfDict.items():
                            objConfEntry: Watcher.ConfDictEntry = objConfDict.setdefault(label, Watcher.ConfDictEntry())
                            objConfEntry.addConf(entry.avg)

                        # Take the highest average confidence as the 'overall' confidence and label
                        highConfKey: str = max(objConfDict, key=lambda key: objConfDict[key].avg)
                        obj.metadata[METAKEY_LABEL] = highConfKey
                        obj.metadata[METAKEY_CONF] = objConfDict[highConfKey].avg
                        self._bboxTracker.updateBox(key, metadata=obj.metadata)

                print(f"{key} - {obj.metadata}")

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
        frameCnt = tracker.metadata[METAKEY_FRAME_CNT]
        if frameCnt < NEW_OBJ_MIN_FRAME_CNT:
            brightness = 255 * (1-(NEW_OBJ_MIN_FRAME_CNT - frameCnt)/NEW_OBJ_MIN_FRAME_CNT)
            color = 3*(brightness,)

        lostFrames = tracker.metadata[METAKEY_LOST_FRAMES]
        if lostFrames > 0:
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
