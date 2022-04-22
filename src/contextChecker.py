''' Class to check for configured interactions '''
import numpy as np
from dataclasses import dataclass, field

from trackerTools.bbox import BBox

from . watchedObject import WatchedObject

CONFIG_KEY_INTRCT_THRESH = "threshold"
CONFIG_KEY_INTRCT_MIN_TIME = "min_time"
CONFIG_KEY_INTRCT_EXPIRE_TIME = "expire_time"
CONFIG_KEY_INTRCT_SLOTS = "slots"


@dataclass
class ConfiguredInteraction:
    name: str
    thresh: float
    minTime: int
    expireTime: int
    slots: list[list[str]] = field(default_factory=list)


@dataclass
class OverlapInfo:
    idxs: list[int]
    ios: float


class ContextChecker:
    @dataclass
    class EventInfo:
        event: ConfiguredInteraction
        slotsObjs: list[WatchedObject]

    def __init__(self, config: dict[str, dict]):
        self._events: list[ConfiguredInteraction] = []
        for event, eventInfo in config.items():
            slots = eventInfo[CONFIG_KEY_INTRCT_SLOTS]
            thresh = eventInfo.get(CONFIG_KEY_INTRCT_THRESH, 0.7)
            minTime = eventInfo.get(CONFIG_KEY_INTRCT_MIN_TIME, 5)
            expireTime = eventInfo.get(CONFIG_KEY_INTRCT_EXPIRE_TIME, 5)
            newConfig = ConfiguredInteraction(name=event, slots=slots, thresh=thresh,
                                              minTime=minTime, expireTime=expireTime)
            self._events.append(newConfig)

    def getEvents(self, objects: list[WatchedObject]) -> list[EventInfo]:

        # Recursively return list of all possible slot filling combinations
        def findMatches(overlapIdxs: list[list[int]], slots: list[list[str]], maxRecurse: int = 100) -> list[list[int]]:
            assert(len(overlapIdxs) == len(slots))
            if maxRecurse == 0:
                raise RecursionError()

            ret = []
            slot = slots[0]
            for enumIdx, overlapIdx in enumerate(overlapIdxs):
                obj = objects[overlapIdx]
                if obj.label in slot:
                    # This was the final match
                    if len(overlapIdxs) == 1:
                        ret.append([overlapIdx])
                    else:
                        overlapIdxLeft = overlapIdxs.copy()
                        overlapIdxLeft.pop(enumIdx)
                        matches = findMatches(overlapIdxs=overlapIdxLeft, slots=slots[1:], maxRecurse=maxRecurse-1)

                        for matchList in matches:
                            ret.append([overlapIdx] + matchList)
            return ret

        # TODO: Support multiple overlaps
        triggeredEvents: list[ContextChecker.EventInfo] = []
        overlaps = ContextChecker.getOverlaps([obj.bbox for obj in objects])
        for overlap in overlaps:
            overlapIdxs = overlap.idxs

            # Match up overlaps to any configured events
            for event in self._events:
                matches = findMatches(overlapIdxs, event.slots)
                for match in matches:
                    objList = [objects[idx] for idx in match]
                    eventInfo: ContextChecker.EventInfo = ContextChecker.EventInfo(
                        event=event, slotsObjs=objList)
                    triggeredEvents.append(eventInfo)

        return triggeredEvents

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
            overlaps.append(OverlapInfo(list(idxPair), retMatrix[idxPair[0]][idxPair[1]]))
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
