from __future__ import annotations

from CuPyNLO.light.PulseBase_v2 import Pulse

from typing import Callable
import numpy as np
from scipy import constants


class Crystal:
    def __init__(self, *, length: float = 1.0, dmg_threshold_GW_per_sqcm: float = 1.0):
        self._c_nm_ps = constants.speed_of_light * 1e9 / 1e12
        self._dmg_threshold_GW_per_sqcm = dmg_threshold_GW_per_sqcm

        self._pp = self.x
        self._length_m = length

    def x(self, start: float, stop: float, x: float) -> float:
        return start + (stop - start) * x / self._length_m
    
    @property
    def pp(self) -> Callable[[float, float, float], float]:
        return self._pp
    
    @pp.setter
    def pp(self, fn: Callable[[float, float, float], float]):
        self._pp = fn

    def pulse_k(self, Pulse: Pulse, axis: float | None = None) -> np.ndarray:
        # if axis is None:
        #     return 2.0 * np.pi * 
        raise NotImplementedError()
