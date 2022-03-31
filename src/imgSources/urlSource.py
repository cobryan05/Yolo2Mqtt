''' URL-backed image source class '''

import urllib.request
import ssl
import base64
import cv2
import numpy as np

from . source import Source


class UrlSource(Source):

    def __init__(self, url: str, user: str = None, password: str = None):
        self._url: str = url
        self._request = urllib.request.Request(url)
        if user:
            base64String = base64.b64encode(bytes(f"{user}:{password}", encoding='utf8'))
            self._request.add_header("Authorization", f"Basic {base64String.decode()}")

    def __repr__(self):
        return f"UrlSource [{self._url}]"

    def getNextFrame(self) -> np.array:
        return self._downloadImage()

    def _downloadImage(self):
        req = urllib.request.urlopen(self._request, context=ssl._create_unverified_context())
        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
