''' Publish a stream to RtspSimpleServer using FFmpeg '''
''' Not really useful except for sample code '''


import logging
import subprocess
import sys
import time
import os
import signal
from .rtspSimpleServer import RtspSimpleServer
from .ffmpeg import Ffmpeg
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RtspProxyFfmpeg")


class RtspProxyFfmpeg:
    def __init__(self, publishName: str, srcRtspUrl: str, rtspApi: RtspSimpleServer, delay: int = 0, skipExisting: bool = False):
        self._publishName: str = publishName
        self._rtspApi: RtspSimpleServer = rtspApi
        self._srcUrl: str = srcRtspUrl
        self._proc: subprocess.Popen = None
        self._delay: int = delay

        # For development, allow using an existing delayed stream
        self._skipStreamReg = False
        if skipExisting:
            activeStreams = self._rtspApi.GetPaths()
            if self._publishName in activeStreams.get("items", {}):
                self._skipStreamReg = True

        if not self._skipStreamReg:
            # Remove anything existing delayed stream
            self._rtspApi.RemoveConfig(self._publishName)
            # Add a stream to the server and publish to it
            if self._rtspApi.AddConfig(self._publishName, source="publisher", sourceOnDemand=False):
                # Give the server a bit before attempting to publish to it
                time.sleep(0.5)
                ffmpegPath = f"\"{Ffmpeg._ffmpegPath}\""
                cmdline = f"{ffmpegPath} -rtsp_transport tcp -use_wallclock_as_timestamps 1 -i {srcRtspUrl} -c copy -f nut pipe:"
                if self._delay > 0:
                    cmdline += f"|delay -b50m {self._delay}s"
                cmdline += f"| {ffmpegPath} -i pipe: -c copy -rtsp_transport tcp -f rtsp {self._rtspApi.rtspProxyUrl}/{self._publishName}"
                self._proc = subprocess.Popen(cmdline, shell=True)

    def __del__(self):
        self.stop()

    @property
    def rtspUrl(self) -> str:
        return f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"

    @property
    def delay(self) -> int:
        return self._delay

    def wait(self, timeout: float = None):
        ''' Waits for ffmpeg to stop.

        Raises subprocess.TimeoutExpired if the timeout is specified and expires '''
        if not self._skipStreamReg:
            self._proc.wait(timeout)

    def stop(self, timeout: float = None):
        ''' Stops the stream

        Parameters
        timeout (int): how long to wait for the process to exit. -1 to wait forever, None to not wait at all

        Raises subprocess.TimeoutExpired exception if a timeout was specified and the process fails to stop before the timeout

        Returns ffmpeg's retcode, or none if timeout is 0 and the process did not immediately exit '''
        if not self._skipStreamReg:
            self._rtspApi.RemoveConfig(self._publishName)
            if self._proc:
                # TODO: Figure out how to get FFMPEG to stop
                os.kill(self._proc.pid, signal.SIGINT)
                self._proc.kill()
                self._proc.terminate()
                if timeout is not None:
                    self.wait(None if timeout == -1 else timeout)
            return self._proc.returncode
        else:
            return 0
