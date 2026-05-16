from __future__ import annotations

from CuPyNLO.interactions.FourWaveMixing import global_variables
from CuPyNLO.light.PulseBase_v2 import Pulse
from CuPyNLO.media.fibers.fiber_v2 import FiberInstance

import gc
from typing import Any

import numpy as np
import pyfftw
from scipy.fft import fftshift, ifftshift


class SSFM:
    def __init__(self, *, local_error: float = 0.001, dz: float = 1e-5,
                 disable_raman: bool = False, disable_self_steepening: bool = False,
                 suppress_iteration: bool = True, use_simple_raman: bool = False,
                 f_r: float = 0.18, f_r0: float = 0.18, tau_1: float = 0.0122,
                 tau_2: float = 0.0320):
        
        self._local_error = local_error
        self.method, self.method_rk4ip = range(2)
        self._disable_raman = disable_raman
        self._disable_self_steepening = disable_self_steepening
        self._use_simple_raman = use_simple_raman

        self.f_r = f_r
        self.f_r0 = f_r0
        
        self.tau1 = tau_1
        self.tau2 = tau_2
        self.dz = dz
        self.dz_min = 1e-12
        self.suppress_iteration = suppress_iteration
        self.n = None

        self._gamma = None

        self.fft_setup: list[Any] = [pyfftw.empty_aligned(self.n, dtype="complex128") for _ in range(10)]

    def setup_fftw(self, pulse: Pulse, fiber: FiberInstance, output_power: float, *, raman_plots: bool = False):
        self.n = pulse.n

