''' Helper class to run FFMPEG'''

import logging
import os
import signal
import subprocess
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("Ffmpeg")


class Ffmpeg:
    _ffmpegPath: str = "/usr/bin/ffmpeg"

    def __init__(self, params: list[str]):
        if isinstance(params, str):
            params = [params]

        self._proc = None
        self._runFfmpeg(params)

    def __del__(self):
        self.stop()

    @classmethod
    def setFFmpegPath(cls, path: str):
        Ffmpeg._ffmpegPath = path

    def wait(self, timeout: float = None):
        ''' Waits for ffmpeg to stop.

        Raises subprocess.TimeoutExpired if the timeout is specified and expires '''
        self._proc.wait(timeout)

    def stop(self, timeout: float = None) -> int:
        ''' Stops any ffmpeg process

        Parameters
        timeout (int): how long to wait for the process to exit. -1 to wait forever, None to not wait at all

        Raises subprocess.TimeoutExpired exception if a timeout was specified and the process fails to stop before the timeout

        Returns ffmpeg's retcode, or none if timeout is 0 and the process did not immediately exit '''
        if self._proc.returncode is None:
            # TODO: Figure out how to get FFMPEG to stop
            os.kill(self._proc.pid, signal.SIGINT)
            self._proc.kill()
            self._proc.terminate()
            if timeout is not None:
                self.wait(None if timeout == -1 else timeout)
        return self._proc.returncode

    def _runFfmpeg(self, params: list[str]):
        cmdline = [Ffmpeg._ffmpegPath]
        cmdline.extend(params)
        logger.debug(f"{cmdline}")
        self._proc = subprocess.Popen(cmdline)
