""" URL-backed image source class """


import numpy as np
import io
from PIL import Image
from src.mqttClient import MqttClient
from .source import Source
from queue import Queue



class MqttSource(Source):
    def __init__(self, mqtt_client: MqttClient, topic: str):
        self._topic: str = topic
        self._mqtt_client: MqttClient = mqtt_client
        self._frameQueue: Queue = Queue()
        mqtt_client.subscribe(self._topic, self._pushFrame, absoluteTopic=True)

    def __del__(self):
        self._mqtt_client.unsubscribe(self._topic)

    def __repr__(self):
        return f"MqttSource [{self._topic}]"

    def _pushFrame(self, mqttMsg):
      png_data = mqttMsg.payload
      image = Image.open(io.BytesIO(png_data))
      image_bytes = np.array(image)
      self._frameQueue.put(image_bytes)

    def getForceInference(self) -> bool:
        return True

    def getNextFrame(self) -> np.array:
        return self._frameQueue.get()
