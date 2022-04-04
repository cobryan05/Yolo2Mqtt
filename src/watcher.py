''' Class to watch for objects in an image stream '''

from dataclasses import dataclass
from threading import Event
import cv2
import numpy as np
import math
import time

from trackerTools.yoloInference import YoloInference
from trackerTools.bbox import BBox
from trackerTools.bboxTracker import BBoxTracker
from trackerTools.objectTracker import ObjectTracker
from . imgSources.source import Source
from . valueStatTracker import ValueStatTracker
from . watchedObject import WatchedObject

METAKEY_TRACKED_WATCHED_OBJ = "trackedWatchedObj"
METAKEY_DETECTIONS = "detections"


LOST_OBJ_REMOVE_FRAME_CNT = 20  # How long must an object be lost before removed
NEW_OBJ_MIN_FRAME_CNT = 5  # How many frames must a new object be present in before considered new
BBOX_TRACKER_MAX_DIST_THRESH = 0.5  # Percent of image a box can move and still be matched
MAX_DETECT_INTERVAL = 10  # Maximum amount of frames without full detection
MIN_CONF_THRESH = 0.1  # Minimum confidence threshold for display


class Watcher:
    @dataclass
    class _DetectionInfo:
        detection: WatchedObject.Detection
        bbox: BBox

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
        frameCnt: int = 0
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

            yoloRes = None
            if forceInference or runDetectCntdwn <= 0 or len(lostObjs) > 0:
                forceInference = False

                startTime = time.time()
                # If tracking lost an object then run yolo
                print("Running inference")
                yoloRes = self._model.runInference(img=img)
                runDetectCntdwn = MAX_DETECT_INTERVAL

                detBboxes: list[BBox] = []
                metadata: list[WatchedObject.Detection] = []
                SAME_BOX_DIST_THRESH = 0.03
                SAME_BOX_SIZE_THRESH = 0.9
                for bbox, conf, objClass, label in yoloRes:
                    detectInfo = Watcher._DetectionInfo(bbox=bbox, detection=WatchedObject.Detection(label, conf))

                    # Check if this may be a second detection of the same object
                    dupIdx = -1
                    for idx, prevDet in enumerate(detBboxes):
                        if detectInfo.bbox.similar(prevDet, SAME_BOX_DIST_THRESH, SAME_BOX_SIZE_THRESH):
                            dupIdx = idx
                            break

                    # Merge any duplicate boxes into one
                    if dupIdx != -1:
                        # Add this detection to the original detection list
                        metadata[dupIdx][METAKEY_DETECTIONS].append(detectInfo)
                    else:
                        # Add new detection
                        detBboxes.append(bbox)
                        metadata.append({METAKEY_DETECTIONS: [detectInfo]})

                def metaCompare(trackedInfo: tuple[BBox, dict], detectedInfo: tuple[BBox, dict]) -> float:
                    trackedBbox, trackedMeta = trackedInfo
                    detectedBbox, detectMeta = detectedInfo
                    assert(METAKEY_DETECTIONS in detectMeta)
                    assert(METAKEY_TRACKED_WATCHED_OBJ in trackedMeta)

                    newDetInfos: list[Watcher._DetectionInfo] = detectMeta[METAKEY_DETECTIONS]
                    trackedObj: WatchedObject = trackedMeta[METAKEY_TRACKED_WATCHED_OBJ]

                    bestLabelConf: float = 0.0
                    for detInfo in newDetInfos:
                        bestLabelConf = max(trackedObj.labelConf(detInfo.detection.label), bestLabelConf)

                    if not trackedBbox.similar(detectedBbox):
                        bestLabelConf *= 0.5
                    return bestLabelConf

                trackedObjs, newObjs, lostObjs, detectedKeys = self._objTracker.update(image=img, detections=detBboxes,
                                                                                       metadata=metadata, metadataComp=metaCompare, mergeMetadata=True)

                # Process each tracked item
                for key, obj in trackedObjs.items():
                    trackedObj: WatchedObject = obj.metadata.get(METAKEY_TRACKED_WATCHED_OBJ, None)

                    # Pop any detection info off that may be on the tracked object
                    detBboxes: list[Watcher._DetectionInfo] = obj.metadata.pop(METAKEY_DETECTIONS, [])
                    if len(detBboxes) > 0:
                        # Update objTracker with the 'detections removed' metadata
                        self._objTracker.updateBox(key, metadata=obj.metadata)

                    if key in newObjs:
                        assert(trackedObj is None)
                        trackedObj: WatchedObject = WatchedObject()
                        for detectInfo in detBboxes:
                            trackedObj.markSeen(detectInfo.detection, newFrame=False)

                        obj.metadata[METAKEY_TRACKED_WATCHED_OBJ] = trackedObj
                        self._objTracker.updateBox(key, metadata=obj.metadata)
                    elif key in lostObjs:
                        # If it was lost before reaching the minimum frame count then remove it
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            print(f"{trackedObj.label} lost before minimum frame count")
                            self._objTracker.removeBox(key)
                        else:
                            trackedObj.markMissing()
                            forceInference = True
                            if trackedObj.framesSinceSeen > LOST_OBJ_REMOVE_FRAME_CNT:
                                print(f"{trackedObj.label} lost for {trackedObj.framesSinceSeen}, removing")
                                self._objTracker.removeBox(key)
                    else:
                        # A previously tracked object, ensure it isn't marked as lost and add any new detections
                        for detectInfo in detBboxes:
                            trackedObj.markSeen(detectInfo.detection, newFrame=False)
                        trackedObj.markSeen()

                        # Run inference every frame when there is a new object
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            forceInference = True

                    print(f"{key} - {obj.metadata}")

                inferTimeStats.addValue(time.time() - startTime)

            if self._debug:
                dbgImg = img.copy()
                if yoloRes:
                    for bbox, conf, classIdx, label in yoloRes:
                        Watcher.drawBboxOnImage(dbgImg, bbox, color=(0, 255, 0), thickness=2)
                        Watcher.drawBboxLabel(dbgImg, bbox, f"{label}: {conf:0.2}", color=(0, 255, 0), align=7)
                for key, tracker in self._objTracker.getTrackedObjects().items():
                    trackedObj: WatchedObject = tracker.metadata[METAKEY_TRACKED_WATCHED_OBJ]

                    if trackedObj.conf >= MIN_CONF_THRESH:
                        Watcher.drawTrackerOnImage(dbgImg, tracker)
                dbgInfo = f"Fetch: {fetchTimeStats.lastValue:0.2}|{fetchTimeStats.avg:0.2}  Track: {trackTimeStats.lastValue:0.2}|{trackTimeStats.avg:0.2}  Infer: {inferTimeStats.lastValue:0.2}|{inferTimeStats.avg:0.2}"
                cv2.putText(dbgImg, dbgInfo, (0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow(dbgWin, dbgImg)
                cv2.waitKey(1)

            print(f"---End of frame {frameCnt}---")
            frameCnt += 1

        print("Exit")

    @ staticmethod
    def drawTrackerOnImage(img: np.array, tracker: BBoxTracker.Tracker, color: tuple[int, int, int] = (255, 255, 255)):
        watchedObj: WatchedObject = tracker.metadata[METAKEY_TRACKED_WATCHED_OBJ]

        if watchedObj.age < NEW_OBJ_MIN_FRAME_CNT:
            brightness = 255 * (1-(NEW_OBJ_MIN_FRAME_CNT - watchedObj.age)/NEW_OBJ_MIN_FRAME_CNT)
            color = (0, brightness, 0)

        if watchedObj.framesSinceSeen > 0:
            red = 255 * (LOST_OBJ_REMOVE_FRAME_CNT - watchedObj.framesSinceSeen)/LOST_OBJ_REMOVE_FRAME_CNT
            color = (0, 0, red)

        label = f"{tracker.key} - {watchedObj.label} {watchedObj.conf:0.2}"
        if watchedObj.framesSinceSeen > 0:
            label += f" [missing {watchedObj.framesSinceSeen}|{watchedObj.age}]"
        Watcher.drawBboxOnImage(img, tracker.bbox, color=color)
        Watcher.drawBboxLabel(img, tracker.bbox, label, color=color)
        for idx, (key, entry) in enumerate(watchedObj._confDict.items()):
            Watcher.drawBboxLabel(img, tracker.bbox, f"{key}: {entry.tracker}", color=color, line=idx+1, size=0.3)

    @ staticmethod
    def drawBboxOnImage(img: np.array, bbox: BBox, color: tuple[int, int, int] = (255, 255, 255), thickness=1):
        imgY, imgX = img.shape[:2]
        x1, y1, x2, y2 = bbox.asX1Y1X2Y2(imgX, imgY)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    @ staticmethod
    def drawBboxLabel(img: np.array,
                      bbox: BBox,
                      label: str,
                      color: tuple[int, int, int] = (255, 255, 255),
                      line: int = 0,
                      font: int = cv2.FONT_HERSHEY_SIMPLEX,
                      size: float = 0.4,
                      align: int = 0):
        imgY, imgX = img.shape[:2]
        x1, y1, x2, y2 = bbox.asX1Y1X2Y2(imgX, imgY)
        if align == 7:
            yPos = y2 - line*16
        else:
            yPos = y1 + (1+line)*16
        cv2.putText(img, label, (x1, yPos), font, size, color, 1, cv2.LINE_AA)
