''' Saves an RTSP stream to disk '''


import logging
import os
import signal
import sys
# Define SIGKILL on windows
if sys.platform == "win32":
    signal.SIGKILL = signal.SIGTERM
import time

import ffmpeg
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspRecorder")


def killFfmpeg(pid: int) -> None:
    sigs = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    for sig in sigs:
        try:
            os.kill(pid, sig)
            os.kill(pid, 0)
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

        proc = ffmpeg.input(self._srcUrl, rtsp_transport="tcp")
        proc = proc.output(outputFile, codec="copy")
        proc = proc.global_args("-nostats")
        self._proc = proc.run_async(cmd=ffmpegCmd, quiet=True)

    def stop(self, timeout: float = None):
        if self._proc is not None:
            killFfmpeg(self._proc.pid)
            self._proc = None

    def __del__(self):
        self.stop()
