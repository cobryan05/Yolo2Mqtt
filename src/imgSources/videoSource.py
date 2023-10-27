""" URL-backed image source class """

from collections.abc import Iterator
from threading import Event
import urllib.request

import cv2
import numpy as np

from .source import Source


class VideoSource(Source):
    def __init__(self, paths: list[str]):
        if isinstance(paths, str):
            paths = [paths]

        self._paths: list[str] = paths
        self._iter: Iterator = iter(self._paths)
        self._vidPath: str = None
        self._vid: cv2.VideoCapture = None
        self._nextVideo()

    def __repr__(self):
        return f"VideoSource [{self._vidPath}]"

    def _nextVideo(self):
        if self._vid:
            self._vid.release()
        self._vidPath = next(self._iter, None)

        # Reset iterator when we get to the end
        if not self._vidPath:
            self._iter = iter(self._paths)
            self._vidPath = next(self._iter)
        self._vid = cv2.VideoCapture(self._vidPath)

    def getNextFrame(self) -> np.array:
        ret, frame = self._vid.read()
        if not ret:
            self._nextVideo()
            ret, frame = self._vid.read()
            if not ret:
                return None
        return frame
