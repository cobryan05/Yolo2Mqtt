''' URL-backed image source class '''

from threading import Event
import urllib.request
import time
import cv2
import numpy as np

from . source import Source


class UrlSource(Source):

    def __init__(self, url: str):
        self._url: str = url

    def __repr__(self):
        return f"UrlSource [{self._url}]"

    def getNextFrame(self) -> np.array:
        return self._downloadImage()

    def _downloadImage(self):
        req = urllib.request.urlopen(self._url)
        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
