from __future__ import annotations

from CuPyNLO.light.PulseBase_v2 import Pulse
from CuPyNLO.media.crystals.CrystalContainer_v2 import Crystal

import logging

import numpy.typing as npt
from scipy import constants

logger = logging.getLogger(__name__)

try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np


class Beam:
    def __init__(self, *, waist_m: float = 1.0, pulse: Pulse | None = None,
                 axis: str | None = None):
        self._c_nm_ps = constants.speed_of_light * 1e9 / 1e12
        self._c = constants.speed_of_light
        self._e0 = constants.epsilon_0
        self._crystal_id = None

        self._lambda0 = pulse.wavelength_m if pulse is not None else None
        self._axis = axis
        self._w0 = waist_m

    @property
    def lambda0(self) -> npt.NDArray[np.float64]:
        if self._lambda0 is None:
            raise ValueError("Lambda0 not set.")
        return self._lambda0
    
    @lambda0.setter
    def lambda0(self, arr: npt.NDArray[np.float64]):
        self._lambda0 = arr

    @property
    def w0(self) -> float:
        return self._w0
    
    @w0.setter
    def w0(self, val: float):
        self._w0 = val

    def calculate_waist(self, z: float, *, n_s: float = 1.0) -> float:
        b = self.confocal(n_s)
        return self._w0 * np.sqrt(1.0 + (2.0 * z / b)**2)

    def confocal(self, n_s: float = 1.0) -> float:
        if self._lambda0 is None:
            raise ValueError
        return (2.0 * np.pi) * self._w0**2 * (n_s / self._lambda0[0])
    
    def calculate_zR(self, n_s: float = 1.0) -> float:
        return self.confocal(n_s) / 2.0
    
    def calculate_R(self, z: float, n_s: float = 1.0) -> float:
        z_r = self.calculate_zR(n_s)
        return z * (1 + (z_r / z)**2)
    
    def calculate_gouy_phase(self, z: float, n_s: float) -> float:
        z_r = self.calculate_zR(n_s)
        psi_guoy = np.arctan2(z, z_r)
        return np.exp(1j * psi_guoy)
    
    def _rtP_to_a(self, n_s: float, z: float, *, waist: float | None = None) -> float:
        if waist is None:
            waist = self.calculate_waist(z, n_s=n_s)
        return 1.0 / np.sqrt(np.pi * waist**2 * n_s * \
                             self._e0 * self._c)
    
    def rtP_to_a(self, n_s: float, z: float) -> float:
        return self._rtP_to_a(n_s, z, waist=self._w0)
    
    def rtP_to_a_2(self, pulse: Pulse, crystal: Crystal) -> float:
        raise NotImplementedError()
