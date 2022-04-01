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
from . valueStatTracker import ValueStatTracker
from . watchedObject import WatchedObject

METAKEY_DET_LABEL = "detLabel"
METAKEY_DET_CONF = "detConf"

METAKEY_TRACKED_WATCHED_OBJ = "trackedWatchedObj"
METAKEY_DETECTION_WATCHED_OBJ = "detWatchedObj"


LOST_OBJ_REMOVE_FRAME_CNT = 20  # How long must an object be lost before removed
NEW_OBJ_MIN_FRAME_CNT = 5  # How many frames must a new object be present in before considered new
BBOX_TRACKER_MAX_DIST_THRESH = 0.5  # Percent of image a box can move and still be matched
MAX_DETECT_INTERVAL = 10  # Maximum amount of frames without full detection
MIN_CONF_THRESH = 0.1  # Minimum confidence threshold for display


class Watcher:

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
        fetchTimeStats: ValueStatTracker = ValueStatTracker()
        trackTimeStats: ValueStatTracker = ValueStatTracker()
        inferTimeStats: ValueStatTracker = ValueStatTracker()
        forceInference: bool = False
        while True:
            timeElapsed = time.time() - loopStart
            if self._stopEvent.wait(timeout=max(0, self._delay - timeElapsed)):
                break
            loopStart = time.time()
            try:
                startTime = time.time()
                img = self._source.getNextFrame()
                fetchTimeStats.addValue(time.time() - startTime)
            except Exception as e:
                print(f"Exception getting image for {self._source}: {str(e)}")
                continue

            # First try object tracking on the new image
            startTime = time.time()
            trackedObjs, _, lostObjs, detectedKeys = self._objTracker.update(image=img)
            trackTimeStats.addValue(time.time() - startTime)
            runDetectCntdwn -= 1

            if forceInference or runDetectCntdwn <= 0 or len(lostObjs) > 0:
                forceInference = False

                startTime = time.time()
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
                        dupObj = metadata[dupIdx][METAKEY_DETECTION_WATCHED_OBJ]
                        dupObj.markSeen(WatchedObject.Detection(label, conf), newFrame=False)
                    else:
                        detections.append(bbox)
                        metadata.append({METAKEY_DETECTION_WATCHED_OBJ: WatchedObject(
                            initialDetection=WatchedObject.Detection(label, conf)
                        )})

                def metaCompare(left: dict, right: dict):
                    if left.get(METAKEY_DET_LABEL, "") == right.get(METAKEY_DET_LABEL, ""):
                        return 1.0
                    return 0.5

                trackedObjs, newObjs, lostObjs, detectedKeys = self._objTracker.update(image=img, detections=detections,
                                                                                       metadata=metadata, metadataComp=metaCompare, mergeMetadata=True)

                # Process each tracked item
                for key, obj in trackedObjs.items():
                    trackedObj: WatchedObject = obj.metadata.get(METAKEY_TRACKED_WATCHED_OBJ, None)

                    # Pop any detection info off that may be on the tracked object
                    detObj: WatchedObject = obj.metadata.pop(METAKEY_DETECTION_WATCHED_OBJ, None)
                    if detObj:
                        self._objTracker.updateBox(key, metadata=obj.metadata)

                    if key in newObjs:
                        assert(trackedObj is None)
                        obj.metadata[METAKEY_TRACKED_WATCHED_OBJ] = detObj
                        self._objTracker.updateBox(key, metadata=obj.metadata)
                    elif key in lostObjs:
                        # If it was lost before reaching the minimum frame count then remove it
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            print(f"{trackedObj.label} lost before minimum frame count")
                            self._objTracker.removeBox(key)
                        else:
                            trackedObj.markMissing()
                            if trackedObj.framesSinceSeen > LOST_OBJ_REMOVE_FRAME_CNT:
                                print(f"{trackedObj.label} lost for {trackedObj.framesSinceSeen}, removing")
                                self._objTracker.removeBox(key)
                    else:
                        # A previously tracked object, ensure it isn't marked as lost and add any new detection
                        if detObj:
                            trackedObj.extend(detObj)
                        trackedObj.markSeen()

                        # Run inference every frame when there is a new object
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            forceInference = True

                    print(f"{key} - {obj.metadata}")

                inferTimeStats.addValue(time.time() - startTime)

            if self._debug:
                dbgImg = img.copy()
                for key, tracker in self._objTracker.getTrackedObjects().items():
                    trackedObj: WatchedObject = tracker.metadata[METAKEY_TRACKED_WATCHED_OBJ]

                    if trackedObj.conf >= MIN_CONF_THRESH:
                        Watcher.drawTrackerOnImage(dbgImg, tracker)
                dbgInfo = f"Fetch: {fetchTimeStats.lastValue:0.2}|{fetchTimeStats.avg:0.2}  Track: {trackTimeStats.lastValue:0.2}|{trackTimeStats.avg:0.2}  Infer: {inferTimeStats.lastValue:0.2}|{inferTimeStats.avg:0.2}"
                cv2.putText(dbgImg, dbgInfo, (0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow(dbgWin, dbgImg)
                cv2.waitKey(1)

            print("------")

        print("Exit")

    @ staticmethod
    def drawTrackerOnImage(img: np.array, tracker: BBoxTracker.Tracker, color: tuple[int, int, int] = (255, 255, 255)):
        watchedObj: WatchedObject = tracker.metadata[METAKEY_TRACKED_WATCHED_OBJ]

        if watchedObj.age < NEW_OBJ_MIN_FRAME_CNT:
            brightness = 255 * (1-(NEW_OBJ_MIN_FRAME_CNT - watchedObj.age)/NEW_OBJ_MIN_FRAME_CNT)
            color = 3*(brightness,)

        if watchedObj.framesSinceSeen > 0:
            red = 255 * (LOST_OBJ_REMOVE_FRAME_CNT - watchedObj.framesSinceSeen)/LOST_OBJ_REMOVE_FRAME_CNT
            color = (0, 0, red)

        imgY, imgX = img.shape[:2]
        x1, y1, x2, y2 = tracker.bbox.asX1Y1X2Y2(imgX, imgY)
        cv2.rectangle(img, (x1, y1), (x2, y2), color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{tracker.key} - {watchedObj.label} {watchedObj.conf:0.2}"
        if watchedObj.framesSinceSeen > 0:
            label += f" [missing {watchedObj.framesSinceSeen}]"
        cv2.putText(img, label, (x1, y1 + 16), font, 0.4, color, 1, cv2.LINE_AA)
