''' Publish a stream to RtspSimpleServer using FFmpeg '''
''' Not really useful except for sample code '''


import logging
import sys
import time
from .rtspSimpleServer import RtspSimpleServer
from .ffmpeg import Ffmpeg
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RtspProxyFfmpeg")


class RtspProxyFfmpeg:
    def __init__(self, publishName: str, srcRtspUrl: str, rtspApi: RtspSimpleServer):
        self._publishName: str = publishName
        self._rtspApi: RtspSimpleServer = rtspApi
        self._srcUrl: str = srcRtspUrl

        # Remove anything existing delayed stream
        self._rtspApi.RemoveConfig(self._publishName)
        # Add a stream to the server and publish to it
        if self._rtspApi.AddConfig(self._publishName, source="publisher", sourceOnDemand=False):
            time.sleep(0.5)  # Give the server a bit before attempting to publish to it

            self._ffmpeg = Ffmpeg(["-i", self._srcUrl,
                                   "-rtsp_transport", "tcp",  # for some reason UDP was producing no output
                                   "-codec", "copy",
                                   "-f", "rtsp",
                                   f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"
                                   ])

    def stop(self, timeout: float = None):
        self._ffmpeg.stop(timeout=timeout)
        self._rtspApi.RemoveConfig(self._publishName)

    def __del__(self):
        self.stop()
