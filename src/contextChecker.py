''' Class to check for configured interactions '''
from dataclasses import dataclass, field

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
        pass
