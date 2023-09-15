''' Rtsp stream image source class, optionally proxied through an RtspSimpleServer '''
import cv2
import logging
import numpy as np
import sys
import time

from collections.abc import Iterator
from threading import Thread, Event, Lock

from src.rtspSimpleServer import RtspSimpleServer
from . source import Source

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspSource")


class RtspSource(Source):

    def __init__(self, name: str, rtspUrl: str, rtspApi: RtspSimpleServer = None, rewindBufSec: int = 0):
        ''' RtspSimpleServer will be configured to host proxy stream for rtspUrl '''
        self._name = name

        self._rtspUrl = rtspUrl
        self._proxyRtspUrl = None
        self._vid: cv2.VideoCapture = self._getCap()
        self._stopEvent: Event = Event()
        self._frameAvail: Event = Event()
        self._rtspApi = rtspApi
        self._rtspRewindSec = rewindBufSec
        self._lock = Lock()
        self._thread: Thread = None

        self._configRtspProxy()
        self._restartThread()

    def __del__(self):
        logger.info(f"Destroying RtspSource {self._name}")
        self._stopEvent.set()
        self._thread.join()

        if self._rtspApi is not None:
            self._rtspApi.RemoveConfig(self._name)

    def __repr__(self):
        return f"RtspSource [{self._name}]"

    def _getCap(self) -> cv2.VideoCapture:
        ''' Returns CV2 video capture object for RTSP stream '''
        return cv2.VideoCapture(self._proxyRtspUrl if self._proxyRtspUrl else self._rtspUrl, cv2.CAP_FFMPEG)

    def _restartThread(self):
        if self._thread:
            self._stopEvent.set()
            self._thread.join()
            self._stopEvent.clear()

        self._thread: Thread = Thread(target=self._captureThread, name=f"RtspCaptureThread_{self._name}")
        self._thread.start()

    def _configRtspProxy(self):
        # Attempt to proxy through RtspSimpleServer, otherwise just direct connect
        self._proxyRtspUrl = RtspSource._getProxyUrl(self._name, self._rtspApi, self._rtspUrl)
        if self._proxyRtspUrl is None:
            logger.info(f"RtspSource {self._name} not using proxy! URL is [{self._proxyRtspUrl}]")
        else:
            logger.info(f"Proxying [{self._rtspUrl}] as [{self._proxyRtspUrl}]")
            time.sleep(0.25)  # Give proxy a chance to initialize

    def restart(self):
        # Attempt to restart the source
        logger.info(f"Attempting to restart {self._name}")
        self._configRtspProxy()
        self._restartThread()

    @staticmethod
    def _getProxyUrl(name: str, rtspApi: RtspSimpleServer, rtspUrl: str) -> str:
        if rtspApi is not None:
            rtspApi.RemoveConfig(name)  # Kick any other streamer off
            rtspApi.AddConfig(name, source=rtspUrl)
            return f"{rtspApi.rtspProxyUrl}/{name}"
        return None

    def _captureThread(self):
        logger.info(f"RTSP capture thread started for {self._name}")
        minRetryTime = 0.05
        maxRetryTime = 30
        retryTime = minRetryTime
        while not self._stopEvent.is_set():
            with self._lock:
                hasFrame = self._vid.grab()

            if hasFrame:
                retryTime = minRetryTime
                self._frameAvail.set()
            else:
                logger.warning(f"{self._name}: Failed to grab frame. Retrying in {retryTime}s")
                if True or not self._vid.isOpened():
                    self._vid = self._getCap()
                time.sleep(retryTime)
                retryTime = min(maxRetryTime, retryTime*2)

    def getNextFrame(self) -> np.array:
        frame = None
        if self._frameAvail.wait(100):
            with self._lock:
                self._frameAvail.clear()
                ret, frame = self._vid.retrieve()
        if frame is None:
            raise Exception(f"{self._name}: Failed to retrieve frame")
        return frame
