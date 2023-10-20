''' Helps manage media file paths '''
import logging
import itertools
import os
import pathlib
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("MediaManager")


class MediaManager:
    eventNameRe: re.Pattern = re.compile(
        r"(?P<timestamp>\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d)___(?P<cameraName>.*)___(?P<interactionName>.*)___(?P<slotList>.*)\.mp4")
    timestampFmt: str = "%Y-%m-%d_%H-%M-%S"

    @dataclass
    class EventParams:
        cameraName: str
        interactionName: str
        slots: list[str]
        timestamp: datetime = field(default_factory=datetime.now)

        @property
        def eventName(self) -> str:
            return f"{self.cameraName}___{self.interactionName}___{'__'.join(self.slots)}"

    def __init__(self, mediaRoot: str, daysToKeep: int):
        ''' Constructor '''
        self._root: str = mediaRoot
        self._keepTime: int = daysToKeep
        self._clearMediaTimer = threading.Timer(0.0, self._clearOldMedia)

        os.makedirs(self.videoPath, exist_ok=True)
        os.makedirs(self.symlinkPath, exist_ok=True)

        self._clearMediaTimer.start()

    def __del__(self):
        if self._clearMediaTimer:
            self._clearMediaTimer.cancel()

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
        timestamp = eventParams.timestamp.strftime(MediaManager.timestampFmt)
        fileName = f"{timestamp}___{eventParams.eventName}.mp4"
        return os.path.join(self.videoPath, fileName)

    @staticmethod
    def _parseRecordingPath(recordingPath: str) -> EventParams:
        filename = os.path.basename(recordingPath)
        match = MediaManager.eventNameRe.match(filename)
        timestamp = datetime.strptime(match["timestamp"], MediaManager.timestampFmt)
        cameraName = match["cameraName"]
        interactionName = match["interactionName"]
        slots = match["slotList"].split("__")

        return MediaManager.EventParams(cameraName=cameraName, interactionName=interactionName, slots=slots, timestamp=timestamp)

    def _clearOldMedia(self) -> None:
        ''' Remove any expired media files'''
        expiredFiles = MediaManager._getExpiredFiles(self.videoPath, self._keepTime)
        for file in expiredFiles:
            logger.debug(f"Removing expired file {file}")
            os.unlink(file)

        brokenSymlinks = MediaManager._getBrokenSymlinks(self.symlinkPath)
        for sym in brokenSymlinks:
            logger.debug(f"Removing broken symlink: {sym}")
            os.unlink(sym)

        emptyDirs = MediaManager._getEmptyDirs(self.symlinkPath)
        while len(emptyDirs) > 0:
            for dir in emptyDirs:
                logger.debug(f"Removing empty directory: {dir}")
                pathlib.Path.rmdir(pathlib.Path(dir))
            # We may have created new empty dirs, so keep checking
            emptyDirs = MediaManager._getEmptyDirs(self.symlinkPath)

        # Schedule next clean 1 day from now
        self._clearMediaTimer = threading.Timer(timedelta(days=1).total_seconds(), self._clearOldMedia)
        self._clearMediaTimer.start()

    @staticmethod
    def _getExpiredFiles(path: str,  age: int, recurse: bool = True) -> list[str]:
        expiredList: list[str] = []
        expiredTime = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=age)
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    params = MediaManager._parseRecordingPath(os.path.join(root, file))
                    if params.timestamp < expiredTime:
                        expiredList.append(os.path.join(root, file))
                except Exception as e:
                    logger.warning(f"Failed to parse path {file}: {e}")
            if not recurse:
                break
        return expiredList

    @staticmethod
    def _getBrokenSymlinks(path: str, recurse: bool = True) -> list[str]:
        brokenList: list[str] = []
        for root, dirs, files in os.walk(path):
            for file in files:
                fullPath = os.path.join(root, file)
                try:
                    linkTarget = os.readlink(fullPath)
                except OSError:
                    logger.warning(f"Unexpected file in symlinks folder: {fullPath}")
                    continue

                if not os.path.isabs(linkTarget):
                    linkTarget = os.path.abspath(os.path.join(root, linkTarget))

                if not os.path.exists(linkTarget):
                    brokenList.append(fullPath)
            if not recurse:
                break

        return brokenList

    @staticmethod
    def _getEmptyDirs(path: str, recurse: bool = True) -> list[str]:
        emptyDirs: list[str] = []
        for root, dirs, files in os.walk(path):
            if len(files) == 0 and len(dirs) == 0:
                emptyDirs.append(root)
            if not recurse:
                break

        return emptyDirs
