from __future__ import annotations

import logging

import numpy as np
from scipy import constants, integrate, signal # type: ignore

from CuPyNLO.light.PulseBase_v2 import Pulse

logger = logging.getLogger(__name__)

class TracyCompressor:
    def __init__(self, lines_per_mm: float, incident_angle_degrees: float):
        self.d = 1e-3 / lines_per_mm
        self.g = incident_angle_degrees * 2.0 * np.pi / 360

        self.c = constants.speed_of_light

    def l_of_w(self, x: float | np.ndarray) -> float | np.ndarray:
        return 1e9 * 2.0 * np.pi * self.c / x
    
    def w_of_l(self, x: float | np.ndarray) -> float | np.ndarray:
        return 2.0 * np.pi * self.c / (x * 1e-9)
    
    def fn(self, x: float | np.ndarray, grating_separation_m: float) -> float | np.ndarray:
        return self.calc_compressor_gdd(self.l_of_w(x), grating_separation_m)
        

    def calc_theta(self, wavelength_nm: float | np.ndarray, *, display_angle: bool = False) -> float | np.ndarray:
        wavelength_m = wavelength_nm * 1e-9
        value = wavelength_m / self.d - np.sin(self.g)
        if value < -1 or value > 1:
            raise ValueError("Arcsin invalid. You are probably asking for diffraction of an impossible color.")
        
        if type(value) is np.ndarray:
            value[value > 1] = 1
            value[value < -1] = -1

        alpha = np.arcsin(value)

        if display_angle:
            logger.info(f"Diffraction angle: {alpha * 360 / (2.0 * np.pi)}.")

        return self.g - alpha
    
    def dt_dw_singlepass(self, wavelength_nm: float | np.ndarray, grating_separation_m: float, *, verbose: bool = False) -> float | np.ndarray:
        w = 2.0 * np.pi * self.c / (wavelength_nm * 1e-9)
        theta = self.calc_theta(wavelength_nm, display_angle=verbose)
        b = grating_separation_m / np.cos(self.g - theta)

        return (-4.0 * np.pi**2 * self.c * b) / (w**3 * self.d**2 * (1.0 - (2.0 * np.pi * self.c / (w * self.d) - np.sin(self.g))**2))
    
    def dphi_domega(self, omega: float, grating_separation_m: float, *, verbose: bool = False) -> float | np.ndarray:
        wavelength_nm = 1e9 * 2.0 * np.pi * self.c / omega
        theta = self.calc_theta(wavelength_nm, display_angle=verbose)
        b = grating_separation_m / np.cos(self.g - theta)
        p = b * (1 + np.cos(theta))

        return p / self.c

    def calc_compressor_gdd(self, wavelength_nm: float | np.ndarray, grating_separation_m: float) -> float | np.ndarray:
        return 2.0 * self.dt_dw_singlepass(wavelength_nm, grating_separation_m)

    def calc_compressor_HOD(self, wavelength_nm: float | np.ndarray, grating_separation_m: float, dispersion_order: int) -> float | np.ndarray:
        if dispersion_order < 3:
            raise ValueError("Order must be > 2. For TOD, specify 3.")
        
        w0 = self.w_of_l(wavelength_nm)
        ws = np.linspace(w0 - 10e12, w0 + 10e12, 101)
        dphidw = self.fn(ws, grating_separation_m)

        y = signal.savgol_filter(dphidw, window_length=11, # type: ignore
                                 polyorder=7, deriv= dispersion_order)
        
        dw = (ws[1] - ws[0]) * 10**(-15 * (1 + dispersion_order))

        return y[50] / (dw**dispersion_order)
    
    def apply_phase_to_pulse(self, grating_separation_m: float, pulse: Pulse) -> None:
        w0 = pulse.center_frequency_THz * 2.0 * np.pi * 1e12
        def integrand(x: float):
            return 2.0 * self.dphi_domega(x, grating_separation_m)
        
        def calc_phase(x: float) -> float:
            return integrate.quad(integrand, w0, x, epsabs=1e-8, epsrel=1e-8)[0] # type: ignore
        
        vec_calc_phase = np.vectorize(calc_phase)
        phase: float = vec_calc_phase(pulse.W_Hz)
        groupdelay: float = np.polyder(np.polyfit(pulse.W_Hz, phase, 2))[0]

        pulse.apply_phase_W(phase + pulse.V_Hz * groupdelay) 