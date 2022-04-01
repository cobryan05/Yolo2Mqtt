''' Class of an object watched by Watcher '''
from . valueStatTracker import ValueStatTracker


class WatchedObject:
    def __init__(self, label: str = None, conf: float = None):
        self._framesCnt: int = 0
        self._framesSeen: int = 0
        self._framesSinceSeen: int = 0
        self._confDict: dict(ValueStatTracker) = {}
        self._bestLabel: str = ""
        self._bestConf: float = 0.0
        if label is not None and conf is not None:
            self.markSeen(label, conf)

    def __repr__(self):
        return f"WatchedObject: {self.label}:{self.conf:0.2}"

    def markMissing(self):
        ''' Mark that this object was missing for a frame '''
        self._framesSinceSeen += 1

    def markSeen(self, label: str, conf: float):
        ''' Mark that this object was seen with a given label and confidence '''
        self._framesSinceSeen = 0
        self._framesCnt += 1
        self._confDict.setdefault(label, ValueStatTracker()).addValue(conf)
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
