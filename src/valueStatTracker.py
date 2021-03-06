''' Class to track statistics about a series of floats'''
from __future__ import annotations
import math
import copy


class ValueStatTracker:
    def __init__(self, value: float = None):
        self._lastValue: float = 0.0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._count: int = 0
        self._min: float = 0.0
        self._max: float = 0.0
        self._avg: float = 0.0
        if value is not None:
            self.addValue(value)

    def __repr__(self):
        return f"{self.avg - self.stdev:.4}|{self.avg:.4}|{self.avg + self.stdev:.4}   Cnt: {self._count}"

    def addValue(self, value: float):
        self._lastValue = value
        self._count += 1

        if self._count == 1:
            self._min = value

        prevAvg = self._avg
        self._sum += value
        self._avg += (value - prevAvg) / self._count
        self._sum_sq += (value - prevAvg)*(value-self._avg)

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def copy(self):
        return copy.copy(self)

    @property
    def lastValue(self) -> float:
        return self._lastValue

    @property
    def n(self) -> int:
        return self._count

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def max(self) -> float:
        return self._max

    @property
    def min(self) -> float:
        return self._min

    @property
    def variance(self) -> float:
        return self._sum_sq / (self._count - 1) if self._count > 1 else 0.0

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def sum(self) -> float:
        return self._sum
