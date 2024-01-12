""" Class to handle configuration """
from dataclasses import dataclass, field

# Only keys in this list will be in the config. Casing must match dataclasses
validKeys = [
    "mqtt",
    "address",
    "port",
    "prefix",
    "events",
    "detections",
    "images",
    "rtspSimpleServer",
    "apiHost",
    "apiPort",
    "homeAssistant",
    "discoveryEnabled",
    "discoveryPrefix",
    "entityPrefix",
    "deviceName",
    "interactions",
    "slots",
    "threshold",
    "minTime",
    "expireTime",
    "cameras",
    "rtspUrl",
    "videoPath",
    "imageUrl",
    "refresh",
    "model",
    "username",
    "password",
    "rewindSec",
    "timelapseDir",
    "timelapseInterval",
    "publishImages",
    "maxNoFrameSec",
    "models",
    "path",
    "width",
    "labels",
    "yoloVersion",
    "recordingManager",
    "mediaRoot",
    "makeSymlinks",
    "keepVideosDays",
    "yolo",
    "device",
    "multiprocessing",
]


@dataclass
class Yolo:
    device: str = "cpu"
    multiprocessing: bool = True


@dataclass
class Mqtt:
    address: str = "mqtt"
    port: int = 1883
    prefix: str = "myhome/ObjectTrackers"
    events: str = "events"
    detections: str = "detections"
    images: str = "images"


@dataclass
class RecordingManager:
    mediaRoot: str = "/media"
    makeSymlinks: bool = True
    keepVideosDays: int = 14


@dataclass
class RtspSimpleServer:
    apiHost: str = "localhost"
    apiPort: int = 9997


@dataclass
class HomeAssistant:
    discoveryEnabled: bool = False
    discoveryPrefix: str = "homeassistant"
    entityPrefix: str = "Tracker"
    deviceName: str = "Yolo2Mqtt"


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
    refresh: float = 1.0
    rewindSec: int = 20
    timelapseDir: str = None
    timelapseInterval: int = 0
    publishImages: bool = False
    maxNoFrameSec: int = 30


@dataclass
class Model:
    path: str = "models/yolov5s.pt"
    width: int = 640
    labels: list[str] = field(default_factory=list)
    yoloVersion: int = 8


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

        cfg = config.get("recordingManager", {})
        self._recordingManager: RecordingManager = RecordingManager(
            **Config.validKeys(cfg)
        )

        cfg = config.get("homeAssistant", {})
        self._homeAssistant: HomeAssistant = HomeAssistant(**Config.validKeys(cfg))
        self._homeAssistant.discoveryPrefix = self.homeAssistant.discoveryPrefix.rstrip(
            "/"
        )

        cfg = config.get("rtspSimpleServer", {})
        self._rtspSimpleServer: RtspSimpleServer = RtspSimpleServer(
            **Config.validKeys(cfg)
        )

        cfg = config.get("mqtt", {})
        self._mqtt: Mqtt = Mqtt(**Config.validKeys(cfg))
        self._mqtt.prefix = self._mqtt.prefix.rstrip("/")

        cfg = config.get("yolo", {})
        self._yolo: Yolo = Yolo(**Config.validKeys(cfg))

    @staticmethod
    def validKeys(cfg: dict) -> dict:
        if cfg is None:
            cfg = {}
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
    def recordingManager(self) -> RecordingManager:
        return self._recordingManager

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

    @property
    def Yolo(self) -> Yolo:
        return self._yolo
