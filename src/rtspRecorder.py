""" Saves an RTSP stream to disk """


import logging
import os
import psutil
import signal
import subprocess
import sys

# Define SIGKILL on windows
if sys.platform == "win32":
    signal.SIGKILL = signal.SIGTERM
import time

import ffmpeg

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspRecorder")


def killFfmpeg(pid: int) -> None:
    """Try various signals to get ffmpeg to stop"""
    sigs = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    for sig in sigs:
        try:
            logger.debug(f"Sending kill signal {sig} to process {pid}")
            os.kill(pid, sig)
            if not psutil.pid_exists(pid):
                break
        except OSError as e:
            # Success
            break
        except Exception as e:
            logger.error(f"Exception killing process {pid}: {e}")
        time.sleep(1)


class RtspRecorder:
    def __init__(self, rtspUrl: str, outputFile: str, ffmpegCmd: str = "ffmpeg"):
        self._srcUrl: str = rtspUrl
        self._outFile: str = outputFile
        self._ffmpegCmd: str = ffmpegCmd

        factory = ffmpeg.input(self._srcUrl, rtsp_transport="tcp")
        factory = factory.output(outputFile, codec="copy")
        factory = factory.global_args("-nostats")
        self._factory: ffmpeg.nodes.OutputStream = factory

        self._proc: subprocess.Popen = self._start()

    def _start(self) -> subprocess.Popen:
        return self._factory.run_async(cmd=self._ffmpegCmd, quiet=True)

    def stop(self, timeout: float = None):
        if self._proc is not None:
            killFfmpeg(self._proc.pid)
            self._proc = None

    def running(self) -> bool:
        """checks if ffmpeg is actively running"""
        return psutil.pid_exists(self._proc.pid)

    def __del__(self):
        self.stop()
