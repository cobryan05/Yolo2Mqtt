''' Saves an RTSP stream to disk '''


import logging
import sys

from .ffmpeg import Ffmpeg
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspRecorder")


class RtspRecorder:
    def __init__(self, rtspUrl: str, outputFile: str):
        self._srcUrl: str = rtspUrl
        self._outFile: str = outputFile

        self._ffmpeg = Ffmpeg(["-i", self._srcUrl,
                               "-rtsp_transport", "tcp",  # for some reason UDP was producing no output
                               "-codec", "copy",
                               outputFile
                               ])

    def stop(self, timeout: float = None):
        self._ffmpeg.stop(timeout=timeout)

    def __del__(self):
        self.stop()
