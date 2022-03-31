''' URL-backed image source class '''

import requests
from requests.auth import HTTPBasicAuth
import cv2
import io
import numpy as np
from PIL import Image

from . source import Source


class UrlSource(Source):

    def __init__(self, url: str, user: str = None, password: str = None):
        self._url: str = url
        if user:
            self._auth: HTTPBasicAuth = HTTPBasicAuth(user, password)
            requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        else:
            self._auth = None

    def __repr__(self):
        return f"UrlSource [{self._url}]"

    def getNextFrame(self) -> np.array:
        return self._downloadImage()

    def _downloadImage(self):
        resp = requests.get(self._url, verify=False, auth=self._auth)
        resp.raise_for_status()

        bytesStream = io.BytesIO(resp.content)
        bytesArray = np.array(Image.open(bytesStream))
        return cv2.cvtColor(bytesArray, cv2.COLOR_RGB2BGR)
