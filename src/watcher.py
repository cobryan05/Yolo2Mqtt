''' Class to watch for objects in an image stream '''

from threading import Event
import cv2
import numpy as np
import math
import time

from trackerTools.yoloInference import YoloInference
from trackerTools.bboxTracker import BBoxTracker
from trackerTools.objectTracker import ObjectTracker
from . imgSources.source import Source

METAKEY_LOST_FRAMES = "losttime"
METAKEY_LABEL = "label"
METAKEY_CONF = "conf"
METAKEY_CONF_DICT = "confdict"
METAKEY_FRAME_CNT = "frameCnt"


LOST_OBJ_REMOVE_FRAME_CNT = 20  # How long must an object be lost before removed
NEW_OBJ_MIN_FRAME_CNT = 2  # How many frames must a new object be present in before considered new
BBOX_TRACKER_MAX_DIST_THRESH = 0.5  # Percent of image a box can move and still be matched
MAX_DETECT_INTERVAL = 10  # Maximum amount of frames without full detection
MIN_CONF_THRESH = 0.6  # Minimum confidence threshold for display


class Watcher:
    class ConfDictEntry:
        def __init__(self, conf: float = None):
            self._sum: float = 0.0
            self._sum_sq: float = 0.0
            self._count: float = 0.0
            self._min: float = 0.0
            self._max: float = 0.0
            self._avg: float = 0.0
            self._history: list(float) = []
            if conf is not None:
                self.addConf(conf)

        def __repr__(self):
            return f"{self.avg - self.stdev:.4}|{self.avg:.4}|{self.avg + self.stdev:.4}   Cnt: {self._count}"

        def addConf(self, conf: float):
            self._count += 1
            self._history.append(conf)
            if self._count == 1:
                self._min = conf
                self._max = conf
                self._avg = conf
            else:
                prevAvg = self._avg
                self._avg += (conf - prevAvg) / self._count
                self._sum_sq += (conf - prevAvg)*(conf-self._avg)

                if conf < self._min:
                    self._min = conf
                if conf > self._max:
                    self._max = conf

        @property
        def n(self) -> int:
            return self._count

        @property
        def avg(self) -> float:
            return self._avg

        @property
        def max(self) -> float:
            return self._max

        @property
        def min(self) -> float:
            return self._min

        @property
        def variance(self) -> float:
            return self._sum_sq / (self._count - 1) if self._count > 1 else 0.0

        @property
        def stdev(self) -> float:
            return math.sqrt(self.variance)

    def __init__(self, source: Source, model: YoloInference, refreshDelay: float = 1.0, debug: bool = False):
        self._source: Source = source
        self._model: YoloInference = model
        self._delay: float = refreshDelay
        self._stopEvent: Event = Event()
        self._objTracker: ObjectTracker = ObjectTracker(distThresh=BBOX_TRACKER_MAX_DIST_THRESH)
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

        runDetectCntdwn = 0
        loopStart = time.time()
        while True:
            timeElapsed = time.time() - loopStart
            if self._stopEvent.wait(timeout=max(0, self._delay - timeElapsed)):
                break
            loopStart = time.time()
            try:
                img = self._source.getNextFrame()
            except Exception as e:
                print(f"Exception getting image for {self._source}: {str(e)}")
                continue

            # First try object tracking on the new image
            trackedObjs, newObjs, lostObjs, detectedKeys = self._objTracker.update(image=img)
            runDetectCntdwn -= 1

            if runDetectCntdwn <= 0 or len(lostObjs) > 0:
                # If tracking lost an object then run yolo
                print("Running inference")
                res = self._model.runInference(img=img)
                runDetectCntdwn = MAX_DETECT_INTERVAL

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

                trackedObjs, newObjs, lostObjs, detectedKeys = self._objTracker.update(image=img, detections=detections,
                                                                                       metadata=metadata, metadataComp=metaCompare)

                # Process each tracked item
                for key, obj in trackedObjs.items():

                    if key in newObjs:
                        # Initialize new object metadata
                        obj.metadata[METAKEY_FRAME_CNT] = 0
                        obj.metadata[METAKEY_LOST_FRAMES] = 0
                        self._objTracker.updateBox(key, metadata=obj.metadata)
                    elif key in lostObjs:
                        # If it was lost before reaching the minimum frame count then remove it
                        if obj.metadata[METAKEY_FRAME_CNT] < NEW_OBJ_MIN_FRAME_CNT:
                            print(f"{obj.metadata[METAKEY_LABEL]} lost before minimum frame count")
                            self._objTracker.removeBox(key)
                        else:
                            lostFrames = obj.metadata[METAKEY_LOST_FRAMES] + 1
                            obj.metadata[METAKEY_LOST_FRAMES] = lostFrames
                            self._objTracker.updateBox(key, metadata=obj.metadata)

                            if lostFrames > LOST_OBJ_REMOVE_FRAME_CNT:
                                print(f"{obj.metadata[METAKEY_LABEL]} lost for {lostFrames}, removing.")
                                self._objTracker.removeBox(key)
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
                                objConfEntry: Watcher.ConfDictEntry = objConfDict.setdefault(
                                    label, Watcher.ConfDictEntry())
                                objConfEntry.addConf(entry.avg)

                            # Select best label by comparing bottom of confidence intervals
                            def calcConf(entry: Watcher.ConfDictEntry):
                                if entry.n >= NEW_OBJ_MIN_FRAME_CNT:
                                    return entry.avg - entry.stdev
                                else:
                                    return 0

                            highConfKey: str = max(objConfDict, key=lambda key, d=objConfDict: calcConf(d[key]))
                            highConfEntry: Watcher.ConfDictEntry = objConfDict[highConfKey]
                            obj.metadata[METAKEY_LABEL] = highConfKey
                            obj.metadata[METAKEY_CONF] = calcConf(highConfEntry)
                            self._objTracker.updateBox(key, metadata=obj.metadata)

                    print(f"{key} - {obj.metadata}")

            if self._debug:
                dbgImg = img.copy()
                for key, tracker in self._objTracker.getTrackedObjects().items():
                    if tracker.metadata[METAKEY_CONF] >= MIN_CONF_THRESH:
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
