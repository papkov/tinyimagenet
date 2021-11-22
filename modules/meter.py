from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass()
class Meter:
    name: str
    history: List[float]
    sum: float = 0
    avg: float = 0
    last: float = 0
    min: float = np.inf
    max: float = -np.inf
    extremum: str = ""
    monitor_min: bool = False

    def __init__(self, name: str) -> None:
        """
        Stores all the incremented elements, their sum and average
        Meter with name {}_loss will monitor min values in history
        :param name: {train, val, test}_{loss, acc, ...} for saving and monitoring
        """
        self.name = name
        self.monitor_min = name.endswith("loss")
        self.reset()

    def reset(self) -> None:
        """
        Restore default values
        :return:
        """
        self.history = []
        self.sum = 0
        self.avg = 0
        self.last = 0
        self.min = np.inf
        self.max = -np.inf
        self.extremum = ""

    def add(self, value: Union[int, float]) -> None:
        """
        Add a value in history and check extrema
        :param value: monitored value
        :return:
        """
        self.last = value
        self.extremum = ""

        if self.monitor_min and value < self.min:
            self.min = value
            self.extremum = "min"
        elif not self.monitor_min and value > self.max:
            self.max = value
            self.extremum = "max"

        self.history.append(value)
        self.sum += value
        self.avg = self.sum / len(self.history)

    def is_best(self) -> bool:
        """
        Check if the last epoch was the best according to the meter
        :return: whether last value added was the best
        """
        is_best = (self.monitor_min and self.extremum == "min") or (
            (not self.monitor_min) and self.extremum == "max"
        )
        return is_best
