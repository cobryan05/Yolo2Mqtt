''' Class to handle configuration '''
from dataclasses import dataclass, field

# Only keys in this list will be in the config. Casing must match dataclasses
validKeys = ["mqtt", "address", "port", "prefix", "events", "detections",
             "rtspSimpleServer", "apiHost", "apiPort",
             "homeAssistant", "discoveryEnabled", "discoveryPrefix", "entityPrefix",
             "interactions", "slots", "threshold", "minTime", "expireTime",
             "cameras", "rtspUrl", "videoPath", "imageUrl", "refresh", "model", "username", "password",
             "models", "path", "width", "labels"]


@dataclass
class Mqtt:
    address: str
    port: int = 1883
    prefix: str = "myhome/ObjectTrackers"
    events: str = "events"
    detections: str = "detections"


@dataclass
class RtspSimpleServer:
    apiHost: str = "localhost"
    apiPort: int = 9997


@dataclass
class HomeAssistant:
    discoveryEnabled: bool = False
    discoveryPrefix: str = "homeassistant"
    entityPrefix: str = "Tracker"


@dataclass
class Interaction:
    slots: list[list[str]]
    threshold: float = 0.5
    minTime: int = 3
    expireTime: int = 5


@dataclass
class Camera:
    rtspUrl: str = None
    videoPath: str = None
    imageUrl: str = None
    username: str = None
    password: str = None
    model: str = "default"
    refresh: int = 1


@dataclass
class Model:
    path: str = "models/default.pt"
    width: int = 640
    labels: list[str] = field(default_factory=list)


class Config:
    def __init__(self, config: dict):

        config = Config.validKeys(config)

        self._models: dict[str, Model] = {}
        for key, cfg in config.get("models", {}).items():
            self._models[key] = Model(**Config.validKeys(cfg))

        self._cameras: dict[str, Model] = {}
        for key, cfg in config.get("cameras", {}).items():
            self._cameras[key] = Camera(**Config.validKeys(cfg))

        self._interactions: dict[str, Interaction] = {}
        for key, cfg in config.get("interactions", {}).items():
            self._interactions[key] = Interaction(**Config.validKeys(cfg))

        cfg = config.get("homeAssistant", {})
        self._homeAssistant: HomeAssistant = HomeAssistant(**Config.validKeys(cfg))
        self._homeAssistant.discoveryPrefix = self.homeAssistant.discoveryPrefix.rstrip('/')

        cfg = config.get("rtspSimpleServer", {})
        self._rtspSimpleServer: RtspSimpleServer = RtspSimpleServer(**Config.validKeys(cfg))

        cfg = config.get("mqtt", {})
        self._mqtt: Mqtt = Mqtt(**Config.validKeys(cfg))
        self._mqtt.prefix = self._mqtt.prefix.rstrip('/')

    @staticmethod
    def validKeys(cfg: dict) -> dict:
        newDict = {}
        for key, value in cfg.items():
            for validKey in validKeys:
                if validKey.lower() == key.lower():
                    newDict[validKey] = value
                    break
        return newDict

    @property
    def models(self) -> dict[str, Model]:
        return self._models

    @property
    def cameras(self) -> dict[str, Camera]:
        return self._cameras

    @property
    def interactions(self) -> dict[str, Interaction]:
        return self._interactions

    @property
    def homeAssistant(self) -> HomeAssistant:
        return self._homeAssistant

    @property
    def RtspSimpleServer(self) -> RtspSimpleServer:
        return self._rtspSimpleServer

    @property
    def Mqtt(self) -> Mqtt:
        return self._mqtt