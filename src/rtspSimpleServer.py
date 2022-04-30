''' Interface with the rtsp-simple-server REST API '''
from email.headerregistry import ContentTypeHeader
import logging
import json
import requests
import sys

from dataclasses import dataclass

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rtspSimpleServer")


class RtspSimpleServer:

    def __init__(self, apiUrl: str = "http://localhost:9997"):
        self._url: str = apiUrl

        try:
            self._config = self.GetConfig()
        except:
            logger.error("Failed to get config from {apiUrl}")
            raise
        logger.debug(f"{self._config}")

    def GetConfig(self) -> dict:
        ''' returns the configuration '''
        return self._Get("v1/config/get")

    def SetConfig(self, config: dict) -> bool:
        return self._Post("v1/config/set", config)

    def GetActiveRtspSessions(self) -> dict:
        ''' returns all active RTSP sessions '''
        return self._Get("v1/rtspsessions/list")

    def KickRtspSession(self, id: str) -> bool:
        ''' kicks out a RTSP session from the server '''
        return self._Post(f"v1/rtspsessions/kick/{id}")

    def GetActiveRtspsSessions(self) -> dict:
        ''' returns all active RTSPS sessions '''
        return self._Get("v1/rtspssessions/list")

    def KickRtspsSession(self, id: str) -> bool:
        ''' kicks out a RTSPS session from the server '''
        return self._Post(f"v1/rtspssessions/kick/{id}")

    def GetActiveRtmpConnections(self) -> dict:
        ''' returns all active RTMP connections '''
        return self._Get("v1/rtmpconns/list")

    def KickRtmpConnection(self, id: str) -> bool:
        ''' kicks out a RTSPS session from the server '''
        return self._Post(f"v1/rtmpconns/kick/{id}")

    def GetPaths(self) -> dict:
        ''' returns all active paths '''
        return self._Get("v1/paths/list")

    def GetHlsMuxers(self) -> dict:
        ''' returns all active HLS muxers. '''
        return self._Get("v1/hlsmuxers/list")

    def AddConfig(self, name: str, **kwargs) -> bool:
        ''' adds the configuration of a path '''
        # See API for possible kwargs: https://aler9.github.io/rtsp-simple-server/#operation/configPathsAdd
        # Useful:
        # source:
        # * publisher -> the stream is published by a RTSP or RTMP client
        # * rtsp://existing-url -> the stream is pulled from another RTSP server / camera
        # * redirect -> the stream is provided by another path or server
        return self._Post(f"v1/config/paths/add/{name}", kwargs)

    def EditConfig(self, name: str, **kwargs) -> bool:
        ''' changes the configuration of a path '''
        return self._Post(f"v1/config/paths/edit/{name}", kwargs)

    def RemoveConfig(self, name: str, **kwargs) -> bool:
        ''' changes the configuration of a path '''
        return self._Post(f"v1/config/paths/remove/{name}", kwargs)

    def _Get(self, endpoint: str) -> dict:
        resp = requests.get(f"{self._url}/{endpoint}")
        return json.loads(resp.content.decode())

    def _Post(self, endpoint: str, payload: dict = None) -> bool:
        resp = requests.post(f"{self._url}/{endpoint}", json=(payload if payload is not None else {}))
        return resp.ok
