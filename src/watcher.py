''' Class to watch for objects in an image stream '''

from threading import Event
import urllib.request

import cv2
import numpy as np

from trackerTools.yoloInference import YoloInference


class Watcher:

    def __init__(self, url: str, model: YoloInference, refreshDelay: int = 1):
        self._url = url
        self._model = model
        self._delay = refreshDelay
        self._stopEvent = Event()

    def stop(self):
        self._stopEvent.set()

    def run(self):
        print(f"Starting Watcher on URL [{self._url}], refreshing every {self._delay} seconds")
        while True:
            if self._stopEvent.wait(timeout=self._delay):
                break

            img = self._downloadImage()
            res = self._model.runInference(img)
            for bbox, conf, objclass, label in res:
                print(f"{label} - {conf:.3f} [{bbox}]")
            print("------")

        print("Exit")

    def _downloadImage(self):
        req = urllib.request.urlopen(self._url)
        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
