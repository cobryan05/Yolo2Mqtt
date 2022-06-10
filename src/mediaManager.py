''' Helps manage media file paths '''
import logging
import itertools
import os
import sys
import time
from dataclasses import dataclass, field


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("MediaManager")


class MediaManager:
    @dataclass
    class EventParams:
        cameraName: str
        interactionName: str
        slots: list[str]

        @property
        def eventName(self) -> str:
            return f"{self.cameraName}___{self.interactionName}___{'__'.join(self.slots)}"

    def __init__(self, mediaRoot: str, daysToKeep: int):
        ''' Constructor '''
        self._root: str = mediaRoot
        self._keepTime: int = daysToKeep

        os.makedirs(self.videoPath, exist_ok=True)
        os.makedirs(self.symlinkPath, exist_ok=True)

    @property
    def videoPath(self) -> str:
        return os.path.join(self._root, "video")

    @property
    def symlinkPath(self) -> str:
        return os.path.join(self._root, "symlinks")

    def createSymlinks(self, eventParams: EventParams, videoPath: str) -> None:
        ''' Create a hierarchy of symlinks for an EventParam pointing to a vidoe file'''
        rootPath = self.symlinkPath

        videoName = os.path.basename(videoPath)
        parts = [eventParams.cameraName, eventParams.interactionName] + eventParams.slots
        symlinks = []

        for permutation in itertools.permutations(parts):
            curPath = rootPath
            for part in permutation:
                curPath = os.path.join(curPath, part)
                symlinks.append(os.path.join(curPath, videoName))

        try:
            symlinks = sorted(set(symlinks))
            for linkname in symlinks:
                os.makedirs(os.path.dirname(linkname), exist_ok=True)
                if not os.path.exists(linkname):
                    relVidPath = os.path.relpath(videoPath, os.path.dirname(linkname))
                    os.symlink(relVidPath, linkname)
        except Exception as e:
            logger.error(f"Failed to create symlinks for {videoPath}: {e}")

    def getRecordingPath(self, eventParams: EventParams) -> str:
        ''' Returns the filename that an event should be recorded to '''
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f"{timestamp}___{eventParams.eventName}.mp4"
        return os.path.join(self.videoPath, fileName)
