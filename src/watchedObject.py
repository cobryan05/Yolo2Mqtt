''' Class of an object watched by Watcher '''
from __future__ import annotations
from . valueStatTracker import ValueStatTracker


class WatchedObject:
    class Detection:
        def __init__(self, label: str, conf: float):
            self._label: str = label
            self._conf: float = conf

        @property
        def label(self) -> str:
            ''' The label associated with this detection '''
            return self._label

        @property
        def conf(self) -> float:
            ''' The confidence of this detection '''
            return self._conf

    def __init__(self, initialDetection: WatchedObject.Detection = None):
        self._framesCnt: int = 0
        self._framesSeen: int = 0
        self._framesSinceSeen: int = 0
        self._confDict: dict(ValueStatTracker) = {}
        self._bestLabel: str = ""
        self._bestConf: float = 0.0
        if initialDetection is not None:
            self.markSeen(initialDetection)

    def __repr__(self):
        return f"WatchedObject: {self.label}:{self.conf:0.2}"

    def markMissing(self):
        ''' Mark that this object was missing for a frame '''
        self._framesSinceSeen += 1

    def extend(self, other: WatchedObject):
        ''' Adds the observations data from another WatchedObject to this one

        Parameters:
        other (WatchedObject) - the object to add to the current object
        '''
        for key, statTracker in other._confDict.items():
            if key in self._confDict:
                self._confDict[key].merge(statTracker)
            else:
                self._confDict[key] = statTracker.copy()
        self._recalculateBest()

    def markSeen(self, detection: WatchedObject.Detection = None, newFrame: bool = True):
        ''' Mark that this object was seen with a given label and confidence

        Parameters:
        detection (WatchedObject.Detection, optional) - new detection confidence information, if any
        newFrame( bool, optional) - Set to false if this label is an additional label on the same frame
        '''
        self._framesSinceSeen = 0
        if newFrame:
            self._framesCnt += 1
        if detection is not None:
            self._confDict.setdefault(detection.label, ValueStatTracker()).addValue(detection.conf)
            self._recalculateBest()

    def _recalculateBest(self):
        ''' Recalculate the best label for this object '''
        bestConf = 0.0
        bestLabel = None
        bestTracker = None

        # Determine confidence this is the best label among tracked labels
        meanConfSum = sum([tracker.avg*tracker.n for tracker in self._confDict.values()])
        for key, tracker in self._confDict.items():
            conf = tracker.sum / meanConfSum
            if conf > bestConf:
                bestConf = conf
                bestLabel = key
                bestTracker = tracker

        # Now get overall confidence by multiplying the confidence that this
        # is the best label by the confidence of that label
        self._bestLabel = bestLabel
        self._bestConf = bestTracker.avg * bestConf

    @property
    def age(self) -> int:
        ''' The total number of frames existing '''
        return self._framesCnt

    @property
    def label(self) -> str:
        ''' Best label of the object '''
        return self._bestLabel

    @property
    def conf(self) -> float:
        ''' Confidence of best label '''
        return self._bestConf

    @property
    def framesSeen(self) -> int:
        ''' Current number of consecutive frames seen'''
        return self._framesSeen

    @property
    def framesSinceSeen(self) -> int:
        ''' Current number of consecutive frames missing '''
        return self._framesSinceSeen
