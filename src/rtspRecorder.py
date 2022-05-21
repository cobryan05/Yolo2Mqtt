''' Saves an RTSP stream to disk '''


import logging
import os
import signal
import sys

import ffmpeg
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspRecorder")


class RtspRecorder:
    def __init__(self, rtspUrl: str, outputFile: str, ffmpegCmd: str = "ffmpeg"):
        self._srcUrl: str = rtspUrl
        self._outFile: str = outputFile

        proc = ffmpeg.input(self._srcUrl, rtsp_transport="tcp")
        proc = proc.output(outputFile, codec="copy")
        self._proc = proc.run_async(cmd=ffmpegCmd, quiet=True)

    def stop(self, timeout: float = None):
        if self._proc is not None:
            os.kill(self._proc.pid, signal.SIGINT)
            self._proc = None

    def __del__(self):
        self.stop()
