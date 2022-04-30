''' Rtsp stream image source class '''
import cv2
import logging
import numpy as np
import sys
import urllib.request

from collections.abc import Iterator
from threading import Thread, Event, Lock

from . source import Source

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Watcher")


class RtspSource(Source):

    def __init__(self, rtspUrl: str):
        self._url: str = rtspUrl
        self._vid: cv2.VideoCapture = cv2.VideoCapture(rtspUrl, cv2.CAP_FFMPEG)
        self._stopEvent: Event = Event()
        self._frameAvail: Event = Event()
        self._thread: Thread = Thread(target=self._captureThread, name="RtspCaptureThread")
        self._thread.start()
        self._lock = Lock()

    def __del__(self):
        logger.info(f"Destroying RtspSource {self._url}")
        self._stopEvent.set()
        self._thread.join()

    def __repr__(self):
        return f"RtspSource [{self._url}]"

    def _captureThread(self):
        logger.info(f"RTSP capture thread started for {self._url}")
        while not self._stopEvent.is_set():
            with self._lock:
                if not self._vid.grab():
                    logger.warning("Failed to grab frame")
                else:
                    self._frameAvail.set()

    def getNextFrame(self) -> np.array:
        frame = None
        if self._frameAvail.wait(100):
            with self._lock:
                self._frameAvail.clear()
                ret, frame = self._vid.retrieve()
        if frame is None:
            raise Exception("Failed to retrieve frame")
        return frame
