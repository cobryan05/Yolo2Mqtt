''' Class of an object watched by Watcher '''
from __future__ import annotations
from dataclasses import dataclass
from . valueStatTracker import ValueStatTracker


class WatchedObject:
    @dataclass
    class Detection:
        label: str
        conf: float

    @dataclass
    class _ConfDictEntry:
        tracker: ValueStatTracker
        conf: float

    def __init__(self, initialDetection: WatchedObject.Detection = None):
        self._framesCnt: int = 0
        self._framesSeen: int = 0
        self._framesSinceSeen: int = 0
        self._confDict: dict[str, WatchedObject._ConfDictEntry] = {}
        self._bestLabel: str = ""
        self._bestConf: float = 0.0
        if initialDetection is not None:
            self.markSeen(initialDetection)

    def __repr__(self):
        return f"WatchedObject: {self.label}:{self.conf:0.2}"

    def collapse(self):
        ''' Collapse the confidence dictionary into a single entry with
            the current best confidence  '''
        resetDetection = WatchedObject.Detection(self.label, self.conf)
        origMissingCnt = self._framesSinceSeen
        self._confDict.clear()
        self.markSeen(resetDetection)
        self._framesSinceSeen = origMissingCnt

    def markMissing(self):
        ''' Mark that this object was missing for a frame '''
        self._framesSinceSeen += 1

    def markSeen(self, detection: WatchedObject.Detection = None, newFrame: bool = True):
        ''' Mark that this object was seen with a given label and confidence

        Parameters:
        detection (WatchedObject.Detection, optional) - new detection confidence information, if any.
                  This may be None if no detection (eg, only tracking) was performed.
        newFrame( bool, optional) - Set to false if this label is an additional label on the same frame
        '''
        self._framesSinceSeen = 0
        if newFrame:
            self._framesCnt += 1
        if detection is not None:
            detectionEntry = self._confDict.setdefault(detection.label, None)
            if detectionEntry is None:
                detectionEntry = WatchedObject._ConfDictEntry(ValueStatTracker(), None)
            detectionEntry.tracker.addValue(detection.conf)
            self._confDict[detection.label] = detectionEntry
            self._recalculateBest()

    def _recalculateBest(self):
        ''' Recalculate the best label for this object '''
        bestConf = 0.0
        bestLabel = None
        bestTracker = None

        # Determine confidence this is the best label among tracked labels
        meanConfSum = sum([entry.tracker.avg*entry.tracker.n for entry in self._confDict.values()])
        for key, entry in self._confDict.items():
            entry.conf = entry.tracker.sum / meanConfSum
            if entry.conf > bestConf:
                bestConf = entry.conf
                bestLabel = key
                bestTracker = entry.tracker

        # Now get overall confidence by multiplying the confidence that this
        # is the best label by the confidence of that label
        self._bestLabel = bestLabel
        self._bestConf = bestTracker.avg * bestConf

    def labelConf(self, label) -> float:
        ''' Check confidence of a given label '''
        entry = self._confDict.get(label, None)
        if entry is not None:
            return entry.conf
        return 0.0

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
