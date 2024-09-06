""" Class to run YOLO Inference as a separate process """
from __future__ import annotations
from typing import Dict
import numpy as np
from threading import Lock
from dataclasses import dataclass, field
import config
from .valueStatTracker import ValueStatTracker
import time


@dataclass
class ModelInfo:
    config: config.Model
    model: YoloInference = None
    refCnt: int = 0
    inferStats: ValueStatTracker = field(default_factory=ValueStatTracker)


class InferenceServer:
    def __init__(self, models: Dict, device: str = "cpu"):
        self._lock: Lock = Lock()
        self._device: str = device
        self._models: Dict[str, ModelInfo] = {
            name: ModelInfo(config=config) for name, config in models.items()
        }

    def detect(self, model: str, image: np.array):
        with self._lock:
            yolo = self._getYoloModel(model)
            yolo.refCnt += 1

        start: float = time.time()
        results = yolo.model.runInference(image)
        yolo.inferStats.addValue(time.time() - start)

        with self._lock:
            yolo.refCnt -= 1
        return results

    def _getYoloModel(self, model) -> ModelInfo:
        from trackerTools.yoloInference import YoloInference

        model = self._models[model]
        if model.model is None:
            # TODO: Limit number of models?
            model.model = YoloInference(
                weights=model.config.path,
                labels=model.config.labels,
                imgSize=model.config.width,
                yoloVersion=model.config.yoloVersion,
                device=self._device,
            )
        return model
