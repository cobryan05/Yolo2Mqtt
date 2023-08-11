''' Interface with the mediamtx REST API '''
from email.headerregistry import ContentTypeHeader
import logging
import json
import requests
import sys

from dataclasses import dataclass

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspSimpleServer")


class RtspSimpleServer:

    def __init__(self, apiHost: str = "localhost", apiPort: int = 9997):
        self._host = apiHost
        self._apiPort = apiPort
        try:
            self._config = self.GetConfig()
            self._rtspPort = int(self._config.get("rtspAddress", ":8554").lstrip(':'))
        except Exception as e:
            logger.error("Failed to get config from {self.apiUrl}: {e}")
            raise
        logger.debug(f"{self._config}")

    @property
    def hostname(self) -> str:
        return self._host

    @property
    def apiUrl(self) -> str:
        return f"http://{self.hostname}:{self._apiPort}"

    @property
    def rtspProxyUrl(self) -> str:
        return f"rtsp://{self.hostname}:{self._rtspPort}"

    def GetConfig(self) -> dict:
        ''' returns the configuration '''
        return self._Get("v2/config/get")

    def SetConfig(self, config: dict) -> bool:
        return self._Post("v2/config/set", config)

    def GetActiveRtspSessions(self) -> dict:
        ''' returns all active RTSP sessions '''
        return self._Get("v2/rtspsessions/list")

    def KickRtspSession(self, id: str) -> bool:
        ''' kicks out a RTSP session from the server '''
        return self._Post(f"v2/rtspsessions/kick/{id}")

    def GetActiveRtspsSessions(self) -> dict:
        ''' returns all active RTSPS sessions '''
        return self._Get("v2/rtspssessions/list")

    def KickRtspsSession(self, id: str) -> bool:
        ''' kicks out a RTSPS session from the server '''
        return self._Post(f"v2/rtspssessions/kick/{id}")

    def GetActiveRtmpConnections(self) -> dict:
        ''' returns all active RTMP connections '''
        return self._Get("v2/rtmpconns/list")

    def KickRtmpConnection(self, id: str) -> bool:
        ''' kicks out a RTSPS session from the server '''
        return self._Post(f"v2/rtmpconns/kick/{id}")

    def GetPaths(self) -> dict:
        ''' returns all active paths '''
        return self._Get("v2/paths/list")

    def GetHlsMuxers(self) -> dict:
        ''' returns all active HLS muxers. '''
        return self._Get("v2/hlsmuxers/list")

    def AddConfig(self, name: str, **kwargs) -> bool:
        ''' adds the configuration of a path '''
        # See API for possible kwargs: https://bluenviron.github.io/mediamtx/#operation/configPathsAdd
        # Useful:
        # source:
        # * publisher -> the stream is published by a RTSP or RTMP client
        # * rtsp://existing-url -> the stream is pulled from another RTSP server / camera
        # * redirect -> the stream is provided by another path or server
        return self._Post(f"v2/config/paths/add/{name}", kwargs)

    def EditConfig(self, name: str, **kwargs) -> bool:
        ''' changes the configuration of a path '''
        return self._Post(f"v2/config/paths/edit/{name}", kwargs)

    def RemoveConfig(self, name: str, **kwargs) -> bool:
        ''' changes the configuration of a path '''
        return self._Post(f"v2/config/paths/remove/{name}", kwargs)

    def _Get(self, endpoint: str) -> dict:
        resp = requests.get(f"{self.apiUrl}/{endpoint}")
        return json.loads(resp.content.decode())

    def _Post(self, endpoint: str, payload: dict = None) -> bool:
        resp = requests.post(f"{self.apiUrl}/{endpoint}", json=(payload if payload is not None else {}))
        if not resp.ok:
            logger.warning(f"Post to {endpoint} returned {resp.status_code}: {resp.reason}")
        return resp.ok
