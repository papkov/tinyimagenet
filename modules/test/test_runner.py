import unittest
import warnings
from typing import List, Optional

import numpy as np

from runner import Meter

warnings.simplefilter("ignore", category=DeprecationWarning)


class TestMeter(unittest.TestCase):
    def test_reset(self) -> None:
        meter_updated = Meter("val_loss")
        meter_static = Meter("val_loss")
        meter_updated.add(1)
        meter_updated.reset()
        assert meter_updated == meter_static

    def test_add(self, values: Optional[List[float]] = None) -> None:
        if not values:
            values = [1, 2, 3]
        meter = Meter("val_loss")
        for v in values:
            meter.add(v)
        assert meter.history == values
        assert np.isclose(meter.sum, np.sum(values))
        assert np.isclose(meter.min, np.min(values))
        assert np.isclose(meter.max, -np.inf)  # loss monitor
        assert np.isclose(meter.avg, np.mean(values))

    def test_monitor_min(self) -> None:
        meter = Meter("val_loss")
        assert meter.monitor_min  # minimize loss

    def test_monitor_max(self) -> None:
        meter = Meter("val_acc")
        assert not meter.monitor_min  # maximize acc

    def test_min_extremum(self) -> None:
        meter = Meter("val_loss")
        meter.add(0.2)
        assert meter.extremum == "min"
        assert np.isclose(meter.min, 0.2)
        assert meter.is_best()
        meter.add(0.4)
        assert meter.extremum == ""
        assert np.isclose(meter.min, 0.2)
        assert not meter.is_best()
        meter.add(0.1)
        assert meter.extremum == "min"
        assert np.isclose(meter.min, 0.1)
        assert meter.is_best()

    def test_max_extremum(self) -> None:
        meter = Meter("val_acc")
        meter.add(0.4)
        assert meter.extremum == "max"
        assert np.isclose(meter.max, 0.4)
        assert meter.is_best()
        meter.add(0.4)
        assert meter.extremum == ""
        assert np.isclose(meter.max, 0.4)
        assert not meter.is_best()
        meter.add(0.6)
        assert meter.extremum == "max"
        assert np.isclose(meter.max, 0.6)
        assert meter.is_best()


if __name__ == "__main__":
    unittest.main()
