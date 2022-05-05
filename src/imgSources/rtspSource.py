''' Rtsp stream image source class, optionally proxied through an RtspSimpleServer '''
import cv2
import logging
import numpy as np
import sys
import urllib.request

from collections.abc import Iterator
from threading import Thread, Event, Lock

from src.rtspSimpleServer import RtspSimpleServer
from . source import Source

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Watcher")


class RtspSource(Source):

    def __init__(self, name: str, rtspUrl: str, rtspApi: RtspSimpleServer = None, rewindBufSec: int = 0):
        ''' RtspSimpleServer will be configured to host proxy stream for rtspUrl '''
        self._name = name
        self._vid = RtspSource._getCapture(name, rtspUrl, rtspApi, rewindBufSec)
        self._stopEvent: Event = Event()
        self._frameAvail: Event = Event()
        self._rtspApi = rtspApi
        self._rtspRewindSec = rewindBufSec
        self._lock = Lock()
        self._thread: Thread = Thread(target=self._captureThread, name="RtspCaptureThread")
        self._thread.start()

    def __del__(self):
        logger.info(f"Destroying RtspSource {self._name}")
        self._stopEvent.set()
        self._thread.join()

        if self._rtspApi is not None:
            proxiedName = f"{self._name}_proxied"
            self._rtspApi.RemoveConfig(proxiedName)

    def __repr__(self):
        return f"RtspSource [{self._name}]"

    @staticmethod
    def _getCapture(name: str, rtspUrl: str, rtspApi: RtspSimpleServer, rewindBufSec: int) -> cv2.VideoCapture:
        cap: cv2.VideoCapture = None

        # Set up proxy if available
        if rtspApi is not None:
            proxiedName = f"{name}_proxied"
            rtspApi.AddConfig(proxiedName, source=rtspUrl)
            rtspUrl = f"{rtspApi.rtspProxyUrl}/{proxiedName}"
        cap = cv2.VideoCapture(rtspUrl, cv2.CAP_FFMPEG)
        return cap

    def _captureThread(self):
        logger.info(f"RTSP capture thread started for {self._name}")
        while not self._stopEvent.is_set():
            with self._lock:
                if not self._vid.grab():
                    logger.warning(f"{self._name}: Failed to grab frame")
                else:
                    self._frameAvail.set()

    def getNextFrame(self) -> np.array:
        frame = None
        if self._frameAvail.wait(100):
            with self._lock:
                self._frameAvail.clear()
                ret, frame = self._vid.retrieve()
        if frame is None:
            raise Exception(f"{self._name}: Failed to retrieve frame")
        return frame
