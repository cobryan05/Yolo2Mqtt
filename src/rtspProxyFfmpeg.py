''' Publish a stream to RtspSimpleServer using FFmpeg '''
''' Not really useful except for sample code '''

import logging
import subprocess
import os
import sys
from dataclasses import dataclass
from .rtspSimpleServer import RtspSimpleServer
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RtspProxyFfmpeg")


class RtspProxyFfmpeg:
    def __init__(self, publishName: str, srcRtspUrl: str, rtspApi: RtspSimpleServer, ffmpegPath: str = "/usr/bin/ffmpeg"):
        self._publishName: str = publishName
        self._rtspApi: RtspSimpleServer = rtspApi

        self._srcUrl: str = srcRtspUrl
        self._ffmpegPath: str = ffmpegPath

        # Remove anything existing delayed stream
        self._rtspApi.RemoveConfig(self._publishName)

        # Add a stream to the server and publish to it
        if self._rtspApi.AddConfig(self._publishName, source="publisher", sourceOnDemand=False):
            self._runFfmpeg()

    def __del__(self):
        self._proc.kill()
        self._proc.wait()
        self._rtspApi.RemoveConfig(self._publishName)

    def _runFfmpeg(self):
        cmdline = self._getFfmpegCmdline()
        self._proc = subprocess.Popen(cmdline, shell=True)

    def _getFfmpegCmdline(self) -> list[str]:
        inputStream: str = self._srcUrl
        outputStream: str = f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"
        return [self._ffmpegPath,
                "-i", inputStream,
                "-rtsp_transport", "tcp",  # for some reason UDP was producing no output
                "-codec", "copy",
                "-f", "rtsp",
                outputStream
                ]
