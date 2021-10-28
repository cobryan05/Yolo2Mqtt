''' URL-backed image source class '''

from threading import Event
import urllib.request

import cv2
import numpy as np

from . source import Source


class VideoSource(Source):

    def __init__(self, path: str):
        self._path: str = path
        self._vid: cv2.VideoCapture = cv2.VideoCapture(self._path)

    def __repr__(self):
        return f"VideoSource [{self._path}]"

    def getNextFrame(self) -> np.array:
        ret, frame = self._vid.read()
        if not ret:
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._vid.read()
            if not ret:
                return None
        return frame
