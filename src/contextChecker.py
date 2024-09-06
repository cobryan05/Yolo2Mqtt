""" Class to check for configured interactions """
import numpy as np
import math
from dataclasses import dataclass, field

from trackerTools.bbox import BBox

from .watchedObject import WatchedObject
from .config import Config, Interaction


@dataclass
class PairInfo:
    idxs: list[int]
    ios: float
    dist: float


class ContextChecker:
    @dataclass
    class EventInfo:
        name: str
        event: Interaction
        slotsObjs: list[WatchedObject]
        pairInfo: PairInfo

    def __init__(self, interactions: dict[str, Interaction]):
        self._interactions = interactions.copy()

    def getEvents(self, objects: list[WatchedObject]) -> list[EventInfo]:

        # Recursively return list of all possible slot filling combinations
        def findMatches(
            overlapIdxs: list[list[int]], slots: list[list[str]], maxRecurse: int = 100
        ) -> list[list[int]]:
            assert len(overlapIdxs) == len(slots)
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
                        matches = findMatches(
                            overlapIdxs=overlapIdxLeft,
                            slots=slots[1:],
                            maxRecurse=maxRecurse - 1,
                        )

                        for matchList in matches:
                            ret.append([overlapIdx] + matchList)
            return ret

        # TODO: Support multiple overlaps
        triggeredEvents: list[ContextChecker.EventInfo] = []
        pairInfos = ContextChecker.getPairInfo([obj.bbox for obj in objects])
        for pairInfo in pairInfos:
            pairIdxs = pairInfo.idxs

            # Match up overlaps to any configured events
            for key, interaction in self._interactions.items():
                if pairInfo.ios < interaction.threshold or pairInfo.dist > interaction.maxDist:
                    continue

                matches = findMatches(pairIdxs, interaction.slots)
                for match in matches:
                    objList = [objects[idx] for idx in match]
                    eventInfo: ContextChecker.EventInfo = ContextChecker.EventInfo(
                        name=key, event=interaction, slotsObjs=objList, pairInfo=pairInfo
                    )
                    triggeredEvents.append(eventInfo)

        return triggeredEvents

    @staticmethod
    def getPairInfo(bboxes: list[BBox]) -> list[PairInfo]:
        """returns a matrix of overlap IoS for BBox pairs"""
        iosMatrix = np.zeros((len(bboxes), len(bboxes)), dtype=float)
        distMatrix = np.zeros((len(bboxes), len(bboxes)), dtype=float)
        for idx1, bbox1 in enumerate(bboxes):
            for idx2, bbox2 in enumerate(bboxes[idx1 + 1 :]):
                idx2 += 1 + idx1
                ios, dist = ContextChecker.calcPairInfo(bbox1, bbox2)
                iosMatrix[idx1][idx2] = ios
                distMatrix[idx1][idx2] = dist

        retInfoPairs: list[PairInfo] = []
        pairsWithInfo = np.where(np.logical_or(iosMatrix != 0.0, distMatrix != 0.0))
        for idxPair in zip(pairsWithInfo[0], pairsWithInfo[1]):
            retInfoPairs.append(
                PairInfo(
                    list(idxPair),
                    iosMatrix[idxPair[0]][idxPair[1]],
                    distMatrix[idxPair[0]][idxPair[1]],
                )
            )
        return retInfoPairs

    @staticmethod
    def calcPairInfo(boxA: BBox, boxB: BBox) -> tuple[float, float]:
        """Calculate the area of overlap over the smaller area and the minimum distance"""
        aX1, aY1, aX2, aY2 = boxA.asRX1Y1X2Y2()
        bX1, bY1, bX2, bY2 = boxB.asRX1Y1X2Y2()

        oX1, oY1, oX2, oY2 = max(aX1, bX1), max(aY1, bY1), min(aX2, bX2), min(aY2, bY2)

        if oX1 > oX2 or oY1 > oY2:
            # No overlap, calculate minimum distance
            dx = max(0, bX1 - aX2, aX1 - bX2)
            dy = max(0, bY1 - aY2, aY1 - bY2)
            min_distance = math.hypot(dx, dy)
            return 0.0, min_distance

        # Calculate IoS (Intersection over Smaller area)
        overlap_area = (oX2 - oX1) * (oY2 - oY1)
        ios = overlap_area / min(boxA.area, boxB.area)

        return ios, 0.0  # If they overlap, the minimum distance is zero
