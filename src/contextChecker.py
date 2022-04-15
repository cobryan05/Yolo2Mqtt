''' Class to check for configured interactions '''
import numpy as np
from dataclasses import dataclass, field

from trackerTools.bbox import BBox

from . watchedObject import WatchedObject

CONFIG_KEY_INTRCT_THRESH = "threshold"
CONFIG_KEY_INTRCT_MIN_FRAMES = "min_frames"
CONFIG_KEY_INTRCT_OBJ_A = "first"
CONFIG_KEY_INTRCT_OBJ_B = "second"


@dataclass
class ConfiguredInteraction:
    name: str
    thresh: float
    minFrames: int
    listA: list[str] = field(default_factory=list)
    listB: list[str] = field(default_factory=list)


@dataclass
class OverlapInfo:
    index_pair: tuple[int, int]
    ios: float


class ContextChecker:

    def __init__(self, config: dict[str, dict]):
        self._events = []
        for event, eventInfo in config.items():
            listA = eventInfo[CONFIG_KEY_INTRCT_OBJ_A]
            listB = eventInfo[CONFIG_KEY_INTRCT_OBJ_B]
            thresh = eventInfo[CONFIG_KEY_INTRCT_THRESH]
            minFrames = eventInfo[CONFIG_KEY_INTRCT_MIN_FRAMES]
            newConfig = ConfiguredInteraction(name=event, listA=listA, listB=listB, thresh=thresh, minFrames=minFrames)
            self._events.append(newConfig)

    def getEvents(self, objects: list[WatchedObject]):
        overlaps = ContextChecker.getOverlaps([obj.bbox for obj in objects])
        for overlap in overlaps:
            idxA, idxB = overlap.index_pair
            objA = objects[idxA]
            objB = objects[idxB]
            print(f"{idxA}:{objA.label} overlaps {idxB}:{objB.label} IoS: {overlap.ios}")
        pass

    @staticmethod
    def getOverlaps(bboxes: list[BBox]) -> list[OverlapInfo]:
        ''' returns a matrix of overlap IoS for BBox pairs'''
        retMatrix = np.zeros((len(bboxes), len(bboxes)), dtype=float)
        for idx1, bbox1 in enumerate(bboxes):
            for idx2, bbox2 in enumerate(bboxes[idx1+1:]):
                idx2 += 1 + idx1
                ios = ContextChecker.calcIoS(bbox1, bbox2)
                retMatrix[idx1][idx2] = ios

        overlaps: list[OverlapInfo] = []
        intersects = np.where(retMatrix != 0.0)
        for idxPair in zip(intersects[0], intersects[1]):
            overlaps.append(OverlapInfo(idxPair, retMatrix[idxPair[0]][idxPair[1]]))
        return overlaps

    @staticmethod
    def calcIoS(boxA: BBox, boxB: BBox) -> float:
        ''' Calculate the area of overlap over the smaller area '''
        aX1, aY1, aX2, aY2 = boxA.asRX1Y1X2Y2()
        bX1, bY1, bX2, bY2 = boxB.asRX1Y1X2Y2()

        oX1, oY1, oX2, oY2 = max(aX1, bX1), max(aY1, bY1), min(aX2, bX2), min(aY2, bY2)

        if oX1 > oX2 or oY1 > oY2:
            return 0.0

        return ((oX2 - oX1) * (oY2 - oY1)) / min(boxA.area, boxB.area)
