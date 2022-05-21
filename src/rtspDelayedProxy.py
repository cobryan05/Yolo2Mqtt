''' Publish a stream to RtspSimpleServer using FFmpeg '''
''' Not really useful except for sample code '''


import ffmpeg
import logging
import subprocess
import os
import signal
import sys
import time
from collections import deque
from threading import Timer, Event
from dataclasses import dataclass
from .rtspSimpleServer import RtspSimpleServer
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RtspDelayedProxy")


class RtspDelayedProxy:
    # Amount of time to wait after registering a stream on RtspSimpleServer before publishing to it
    PUBLISH_START_DELAY = 1

    def __init__(self, publishName: str, srcRtspUrl: str, rtspApi: RtspSimpleServer, delay: int = 0, overwriteExisting: bool = False, ffmpegCmd: str = "ffmpeg"):
        self._publishName: str = publishName
        self._rtspApi: RtspSimpleServer = rtspApi
        self._srcUrl: str = srcRtspUrl
        self._proc: subprocess.Popen = None
        self._delay: int = delay
        self._ffmpegIn: ffmpeg.nodes.OutputStream = None
        self._ffmpegOut: ffmpeg.nodes.OutputStream = None
        self._timer: Timer = None
        self._cmd = ffmpegCmd
        self._stopEvent: Event = Event()

        streamExists = self._publishName in self._rtspApi.GetPaths().get("items", {})
        if streamExists:
            if overwriteExisting:
                # Remove anything existing delayed stream
                self._rtspApi.RemoveConfig(self._publishName)
            else:
                raise Exception(f"Stream {self._publishName} already exists on Rtsp server")

        publishUrl = f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"
        self._ffmpegIn = ffmpeg.input(srcRtspUrl, rtsp_transport='tcp',
                                      use_wallclock_as_timestamps=1).output("pipe:", codec="copy", format="nut")
        self._ffmpegOut = ffmpeg.input("pipe:").output(publishUrl, codec="copy", rtsp_transport="tcp", format="rtsp")

        # Add a stream to the server and publish to it
        if self._rtspApi.AddConfig(self._publishName, source="publisher", sourceOnDemand=False):
            # Give the server a bit before attempting to publish to it
            self._timer = Timer(RtspDelayedProxy.PUBLISH_START_DELAY, self._ffmpegThreadFunc)
            self._timer.start()

    def __del__(self):
        if self._timer is not None:
            self._timer.cancel()
        self.stop()

    @ property
    def rtspUrl(self) -> str:
        return f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"

    @ property
    def delay(self) -> int:
        return self._delay

    def _ffmpegThreadFunc(self):
        @dataclass
        class DelayedPacket:
            data: bytes
            timestamp: float

        inProc = self._ffmpegIn.run_async(cmd=self._cmd, pipe_stdout=True)
        outProc = self._ffmpegOut.run_async(cmd=self._cmd, pipe_stdin=True)
        delayBuffer: deque[DelayedPacket] = deque()
        while not self._stopEvent.is_set():
            bytesRead = inProc.stdout.read1()
            delayBuffer.append(DelayedPacket(data=bytesRead, timestamp=time.time() + self._delay))
            while len(delayBuffer) > 0:
                packet = delayBuffer[0]
                if time.time() < packet.timestamp:
                    break
                delayBuffer.popleft()
                outProc.stdin.write(packet.data)

        os.kill(outProc.pid, signal.SIGINT)
        os.kill(inProc.pid, signal.SIGINT)
        self._rtspApi.RemoveConfig(self._publishName)

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
        self._stopEvent.set()
