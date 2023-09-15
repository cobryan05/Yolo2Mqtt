''' Class to watch for objects in an image stream '''

from dataclasses import dataclass
from threading import Event
import cv2
import datetime
import os
import logging
import sys
import numpy as np
import time
from signalslot import Signal
from PIL import Image

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

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Watcher")


class Watcher:
    @dataclass
    class _DetectionInfo:
        detection: WatchedObject.Detection

    def __init__(self, source: Source, model: YoloInference, refreshDelay: float = 1.0, userData=None, timelapseDir: str = None, timelapseInterval: int = -1, debug: bool = False, maxNoFrameSec: int = 30):
        self._source: Source = source
        self._model: YoloInference = model
        self._delay: float = refreshDelay
        self._stopEvent: Event = Event()
        self._objTracker: ObjectTracker = ObjectTracker(distThresh=BBOX_TRACKER_MAX_DIST_THRESH)
        self._userData = userData
        self._debug = debug
        self._timelapseDir: str = timelapseDir
        self._timelapseInterval: int = timelapseInterval
        self._maxNoFrameInterval: int = maxNoFrameSec
        self._newObjSignal: Signal = Signal(args=['obj', 'userData'])
        self._lostObjSignal: Signal = Signal(args=['obj', 'userData'])
        self._updatedObjSignal: Signal = Signal(args=['obj', 'userData'])
        self._imgUpdatedSignal: Signal = Signal(args=['image', 'userData'])

    def stop(self):
        self._stopEvent.set()

    def connectNewObjSignal(self, slot):
        return self._newObjSignal.connect(slot)

    def connectLostObjSignal(self, slot):
        return self._lostObjSignal.connect(slot)

    def connectUpdatedObjSignal(self, slot):
        return self._updatedObjSignal.connect(slot)

    def connectImageUpdatedSignal(self, slot):
        return self._imgUpdatedSignal.connect(slot)

    def disconnectNewObjSignal(self, slot):
        return self._newObjSignal.disconnect(slot)

    def disconnectLostObjSignal(self, slot):
        return self._lostObjSignal.disconnect(slot)

    def disconnectUpdatedObjSignal(self, slot):
        return self._updatedObjSignal.disconnect(slot)

    def disconnectImageUpdatedSignal(self, slot):
        return self._imgUpdatedSignal.disconnect(slot)

    def run(self):
        logger.info(f"Starting Watcher with [{self._source}], refreshing every {self._delay} seconds")

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
        forceInference: bool = True  # First loop always runs inference
        lastFrameTime = time.time()
        lastTimelapse = time.time()
        nextTimelapse = float("inf")
        if self._timelapseDir is not None and self._timelapseInterval > 0:
            try:
                os.makedirs(self._timelapseDir, mode=555, exist_ok=True)
                nextTimelapse = lastTimelapse + self._timelapseInterval
            except Exception as e:
                logger.error(f"Failed to initialize timelapses: {e}")

        while True:
            timeElapsed = time.time() - loopStart
            if self._stopEvent.wait(timeout=max(0, self._delay - timeElapsed)):
                break
            loopStart = time.time()
            try:
                startTime = time.time()
                img = self._source.getNextFrame()
                lastFrameTime = time.time()
                fetchTimeStats.addValue(lastFrameTime - startTime)
            except Exception as e:
                logger.error(f"Exception getting image for {self._source}: {str(e)}")
                if time.time() - lastFrameTime > self._maxNoFrameInterval:
                    logger.error(f"Timeout exceeded! It has been {time.time() - lastFrameTime} since last frame! Restarting source...")
                    self._source.restart()
                continue

            if time.time() > nextTimelapse:
                try:
                    self.saveTimelapse(img)
                    nextTimelapse = time.time() + self._timelapseInterval
                except Exception as e:
                    logger.error(f"Failed to save timelapse for {self._source}: {str(e)}")


            # Just run object tracking if not scheduled to run inference
            runInference: bool = forceInference or runDetectCntdwn <= 0
            if not runInference:
                startTime = time.time()
                trackedObjs, _, lostObjs, detectedKeys = self._objTracker.update(image=img)
                trackTimeStats.addValue(time.time() - startTime)
                runDetectCntdwn -= 1
                # If an object was lost then run inference
                if len(lostObjs) > 0:
                    runInference = True
                else:
                    # If not going to run inference then be sure to update location of tracked objects.
                    # TODO: Clean this up. It's a quick hack to just put this here
                    for key, obj in trackedObjs.items():
                        trackedObj: WatchedObject = obj.metadata.get(METAKEY_TRACKED_WATCHED_OBJ, None)
                        if trackedObj.bbox != obj.bbox:
                            trackedObj.updateBbox(obj.bbox)
                            self._updatedObjSignal.emit(obj=trackedObj, userData=self._userData)

            # Run inference if required
            yoloRes = []
            if runInference:
                forceInference = False
                runDetectCntdwn = MAX_DETECT_INTERVAL

                # Run the image through the model
                startTime = time.time()
                logger.debug("Running inference")
                yoloRes = self._model.runInference(img=img)

                # Now merge any duplicate boxes from the inference
                detBboxes: list[BBox] = []
                metadata: list[WatchedObject.Detection] = []
                SAME_BOX_DIST_THRESH = 0.03
                SAME_BOX_SIZE_THRESH = 0.9
                for bbox, conf, objClass, label in yoloRes:
                    detectInfo = Watcher._DetectionInfo(detection=WatchedObject.Detection(label, conf, bbox))

                    # Check if this may be a second detection of the same object
                    dupIdx = -1
                    for idx, prevDet in enumerate(detBboxes):
                        if detectInfo.detection.bbox.similar(prevDet, SAME_BOX_DIST_THRESH, SAME_BOX_SIZE_THRESH):
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
                    ''' This function returns confidence that two objects are the same object.
                        This will influence matching detected objects with already-tracked objects '''
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

                # Run the object tracker, updating it with the current inference detections
                trackedObjs, newObjs, lostObjs, detectedKeys = self._objTracker.update(image=img, detections=detBboxes,
                                                                                       metadata=metadata, metadataComp=metaCompare, mergeMetadata=True)

                # Process each tracked item
                for key, obj in trackedObjs.items():
                    trackedObj: WatchedObject = obj.metadata.get(METAKEY_TRACKED_WATCHED_OBJ, None)

                    # Pop any temporary detection info off that may be on the tracked object
                    detBboxes: list[Watcher._DetectionInfo] = obj.metadata.pop(METAKEY_DETECTIONS, [])
                    if len(detBboxes) > 0:
                        # Update objTracker with the metadata from which we popped off the temporary detection info
                        self._objTracker.updateBox(key, metadata=obj.metadata)

                    if key in newObjs:
                        assert(trackedObj is None)
                        trackedObj: WatchedObject = WatchedObject(objId=key)
                        for detectInfo in detBboxes:
                            trackedObj.markSeen(detectInfo.detection, newFrame=False)
                        forceInference = True
                        obj.metadata[METAKEY_TRACKED_WATCHED_OBJ] = trackedObj
                        self._objTracker.updateBox(key, metadata=obj.metadata)
                    elif key in lostObjs:
                        # If it was lost before reaching the minimum frame count then remove it
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            logger.debug(f"{trackedObj.label} lost before minimum frame count")
                            self._objTracker.removeBox(key)
                        else:
                            trackedObj.markMissing()
                            forceInference = True
                            if trackedObj.framesSinceSeen > LOST_OBJ_REMOVE_FRAME_CNT:
                                logger.debug(f"{trackedObj.label} lost for {trackedObj.framesSinceSeen}, removing")
                                self._lostObjSignal.emit(obj=trackedObj, userData=self._userData)
                                self._objTracker.removeBox(key)
                    else:
                        # A previously tracked object, ensure it isn't marked as lost and add any new detections
                        for detectInfo in detBboxes:
                            trackedObj.markSeen(detectInfo.detection, newFrame=False)
                        trackedObj.markSeen()
                        self._objTracker.updateBox(key, bbox=trackedObj.bbox)

                        # Run inference every frame when there is a new object
                        if trackedObj.age < NEW_OBJ_MIN_FRAME_CNT:
                            forceInference = True
                        elif trackedObj.age == NEW_OBJ_MIN_FRAME_CNT:
                            self._newObjSignal.emit(obj=trackedObj, userData=self._userData)
                        else:
                            self._updatedObjSignal.emit(obj=trackedObj, userData=self._userData)

                    logger.debug(f"{key} - {obj.metadata}")

                inferTimeStats.addValue(time.time() - startTime)

            # Only publish if inference was ran and there is a listener for the image
            should_publish_image = yoloRes and len(self._imgUpdatedSignal.slots) > 0

            if self._debug or should_publish_image:
                dbgImg = img.copy()
                for bbox, conf, classIdx, label in yoloRes:
                    Watcher.drawBboxOnImage(dbgImg, bbox, color=(0, 255, 0), thickness=2)
                    Watcher.drawBboxLabel(dbgImg, bbox, f"{label}: {conf:0.2}", color=(0, 255, 0), align=7)

                for key, tracker in self._objTracker.getTrackedObjects().items():
                    trackedObj: WatchedObject = tracker.metadata[METAKEY_TRACKED_WATCHED_OBJ]

                    if trackedObj.conf >= MIN_CONF_THRESH:
                        Watcher.drawTrackerOnImage(dbgImg, tracker)
                dbgInfo = f"Fetch: {fetchTimeStats.lastValue:0.2}|{fetchTimeStats.avg:0.2}  Track: {trackTimeStats.lastValue:0.2}|{trackTimeStats.avg:0.2}  Infer: {inferTimeStats.lastValue:0.2}|{inferTimeStats.avg:0.2}"
                cv2.putText(dbgImg, dbgInfo, (0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                if yoloRes:
                    pilImg: Image = Image.fromarray(cv2.cvtColor(dbgImg, cv2.COLOR_BGR2RGB))
                    self._imgUpdatedSignal.emit(image=pilImg, userData=self._userData)

                if self._debug:
                    cv2.imshow(dbgWin, dbgImg)
                    cv2.waitKey(1)

            logger.debug(f"---End of frame {frameCnt} [{self._source}]---")
            frameCnt += 1

        logger.info(f"!!!Exit frame capture loop for {self._source}!!!")

    def saveTimelapse(self, img: np.array):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = os.path.join(self._timelapseDir, f"{timestamp}.png")
        logger.info(f"Saving timelapse {output_path}")
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilImg: Image = Image.fromarray(imgRgb)
        pilImg.save(output_path)

    @staticmethod
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

    @staticmethod
    def drawBboxOnImage(img: np.array, bbox: BBox, color: tuple[int, int, int] = (255, 255, 255), thickness=1):
        imgY, imgX = img.shape[:2]
        x1, y1, x2, y2 = bbox.asX1Y1X2Y2(imgX, imgY)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    @staticmethod
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
