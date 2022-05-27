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
        self._runId: int = 0

        publishUrl = f"{self._rtspApi.rtspProxyUrl}/{self._publishName}"
        self._ffmpegIn = ffmpeg.input(srcRtspUrl, rtsp_transport='tcp',
                                      use_wallclock_as_timestamps=1).output("pipe:", codec="copy", format="nut")
        self._ffmpegOut = ffmpeg.input("pipe:").output(publishUrl, codec="copy", rtsp_transport="tcp", format="rtsp")

        # Add a stream to the server and publish to it
        self._run( overwrite=overwriteExisting)

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

    def _run(self, overwrite: bool ):
        streamExists = self._publishName in self._rtspApi.GetPaths().get("items", {})
        if streamExists:
            if overwrite:
                # Remove anything existing delayed stream
                self._rtspApi.RemoveConfig(self._publishName)
            else:
                raise Exception(f"Stream {self._publishName} already exists on Rtsp server")

        # Add a stream to the server and publish to it
        if self._rtspApi.AddConfig(self._publishName, source="publisher", sourceOnDemand=False):
            # Increment the runId to stop any running threads
            self._runId += 1
            # Give the server a bit before attempting to publish to it
            self._timer = Timer(RtspDelayedProxy.PUBLISH_START_DELAY, self._ffmpegThreadFunc, args=(self._runId,))
            self._timer.start()


    def _ffmpegThreadFunc(self, runId:int):
        @dataclass
        class DelayedPacket:
            data: bytes
            timestamp: float
        BROKEN_PIPE_RETRY_CNT:int =5

        inProc = self._ffmpegIn.run_async(cmd=self._cmd, pipe_stdout=True)
        outProc = self._ffmpegOut.run_async(cmd=self._cmd, pipe_stdin=True)
        delayBuffer: deque[DelayedPacket] = deque()
        retriesLeft = BROKEN_PIPE_RETRY_CNT
        try:
            while not self._stopEvent.is_set() and self._runId == runId:
                bytesRead = inProc.stdout.read1()
                if len(bytesRead) == 0:
                    if retriesLeft == 0:
                        raise BrokenPipeError(f"Failed to read from pipe {BROKEN_PIPE_RETRY_CNT} times")
                    retriesLeft -= 1
                    continue

                delayBuffer.append(DelayedPacket(data=bytesRead, timestamp=time.time() + self._delay))
                while len(delayBuffer) > 0:
                    packet = delayBuffer[0]
                    if time.time() < packet.timestamp:
                        break
                    delayBuffer.popleft()
                    outProc.stdin.write(packet.data)
            self._rtspApi.RemoveConfig(self._publishName)
        except BrokenPipeError as e:
            logger.error(f"Stream failed! Retrying...")
            self._run(overwrite=True)
        except Exception as e:
            logger.error(f"***********Unexpected Exception: {e}")
            self._run(overwrite=True)

        # TODO: This is really ugly and bad
        try:
            os.kill(outProc.pid, signal.SIGINT)
            os.kill(inProc.pid, signal.SIGINT)
            time.sleep(1)
            os.kill(outProc.pid, signal.SIGTERM)
            os.kill(inProc.pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(outProc.pid, signal.SIGKILL)
            os.kill(inProc.pid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Exception killing processes: {e}")

    def wait(self, timeout: float = None):
        ''' Waits for ffmpeg to stop.

        Raises subprocess.TimeoutExpired if the timeout is specified and expires '''
        raise NotImplemented()

    def stop(self, timeout: float = None):
        ''' Stops the stream

        Parameters
        timeout (int): how long to wait for the process to exit. -1 to wait forever, None to not wait at all

        Raises subprocess.TimeoutExpired exception if a timeout was specified and the process fails to stop before the timeout

        Returns ffmpeg's retcode, or none if timeout is 0 and the process did not immediately exit '''
        self._stopEvent.set()
