from __future__ import annotations

from CuPyNLO.light.PulseBase_v2 import Pulse

import numpy as np


class SechPulse(Pulse):
    def __init__(self, power: float, t0_ps: float, center_wavelength_nm: float, *,
                 time_window_ps: float = 10.0, frep_MHz: float = 100.0, n: int = 2**10,
                 gdd: float = 0.0, tod: float = 0.0, chirp2: float = 0.0, chirp3: float = 0.0,
                 power_is_avg: bool = False):
        super().__init__(frep_MHz=frep_MHz, n=n)

        if center_wavelength_nm <= 1.0 or time_window_ps <= 1.0:
            raise ValueError("Check that mks units were not passed.")
        self.center_wavelength_nm = center_wavelength_nm
        self.time_window_ps = time_window_ps

        if not power_is_avg:
            self.at = np.sqrt(power) / np.cosh(self.T_ps / t0_ps)
        else:
            self.at = 1 /np.cosh(self.T_ps / t0_ps)
            self.at *= np.sqrt(power / (frep_MHz * 1e6 * self.calc_epp()))

        self.chirp_pulse_W(gdd, tod=tod)
        self.chirp_pulse_T(chirp2, chirp3, t0_ps)

