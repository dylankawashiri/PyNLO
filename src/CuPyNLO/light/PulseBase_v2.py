from __future__ import annotations

from enum import Enum
import logging

import numpy as np
import numpy.typing as npt
from scipy import constants # type: ignore
from scipy.interpolate import interp1d # type: ignore
from scipy.fft import fft, fftfreq # type: ignore

from CuPyNLO.util.pynlo_ffts import IFFT_t, FFT_t

logger = logging.getLogger(__name__)

class Noise(Enum):
    SQRT_N_FREQ = "sqrt_N_freq"
    ONE_PHOTON_FREQ = "one_photon_freq"

class Gate(Enum):
    XFROG = "xfrog"
    FROG = "frog"

class Pulse:
    def __init__(self, *,
                 center_frequency_THz: float = 1.0,
                 time_window_ps: float = 1.0,
                 frep_MHz: float = 100.0, n: int = 1024):
        self._frep_MHz = frep_MHz
        if self._frep_MHz > 1e6:
            logger.warning("Frep should be specified in MHz; large value given.")
        self._frep_Hz = self._frep_MHz * 1e6

        self._n: int = n

        self._c_nmps = constants.speed_of_light * 1e9 / 1e12
        self._c_mks = constants.speed_of_light
        self._h = constants.Planck

        self._center_frequency_THz = center_frequency_THz
        self._time_window_ps = time_window_ps

        self._T_ps: npt.NDArray[np.float64] = np.linspace(-self._time_window_ps / 2.0, self._time_window_ps / 2.0, self._n, endpoint=False)

        self._ready = False
        self._external_units = None

        self._aw: np.ndarray = np.zeros((self._n,), dtype = np.complex128)
        self._at: np.ndarray | None = None
        self._V_THz: npt.NDArray[np.float64] = 2.0 * np.pi * np.transpose(np.arange(-self._n / 2, self._n / 2)) / (self._n * self.dT_ps)


    @property
    def frep_MHz(self) -> float:
        return self._frep_MHz
    
    @frep_MHz.setter
    def frep_MHz(self, val: float):
        if val > 1e6:
            logger.warning("Frep should be specified in MHz; large value given.")
        self._frep_MHz = val
    
    @property
    def frep_Hz(self) -> float:
        return self._frep_MHz * 1e6
    
    @frep_Hz.setter
    def frep_Hz(self, val: float):
        self._frep_MHz = val * 1e-6
    
    @property
    def n(self) -> int:
        return self._n
    
    @n.setter
    def n(self, val: int):
        self._n = val

    @property
    def center_frequency_THz(self) -> float:
        return self._center_frequency_THz
    
    @center_frequency_THz.setter
    def center_frequency_THz(self, val: float):
        self._center_frequency_THz = val
    
    @property
    def center_frequency_mks(self) -> float:
        return self._center_frequency_THz * 1e12
    
    @center_frequency_mks.setter
    def center_frequency_mks(self, val: float):
        self._center_frequency_THz = val * 1e-12
    
    @property
    def time_window_ps(self) -> float:
        return self._time_window_ps
    
    @time_window_ps.setter
    def time_window_ps(self, val: float):
        self._time_window_ps = val
    
    @property
    def time_window_s(self) -> float:
        return self._time_window_ps * 1e-12

    @time_window_s.setter
    def time_window_s(self, val: float):
        self._time_window_ps = val * 1e12

    @property
    def T_ps(self) -> npt.NDArray[np.float64]:
        return np.linspace(-self._time_window_ps / 2.0, self._time_window_ps / 2.0, self._n, endpoint=False)

    @T_ps.setter
    def T_ps(self, val: npt.NDArray[np.float64] | list[float]):
        self._T_ps = np.array(val)

    @property
    def T_s(self) -> npt.NDArray[np.float64]:
        return self._T_ps * 1e-12
    
    @T_s.setter
    def T_s(self, val: npt.NDArray[np.float64] | list[float]):
        self._T_ps = np.array(val) * 1e12

    @property
    def dT_ps(self) -> float:
        return self.dT_ps

    @property
    def dT_s(self) -> float:
        return self.dT_ps * 1e-12

    @property
    def V_THz(self) -> npt.NDArray[np.float64]:
        self._V_THz = 2.0 * np.pi * np.transpose(np.arange(-self._n / 2, self._n / 2)) / (self._n * self.dT_ps)
        return self._V_THz

    @property
    def V_Hz(self) -> npt.NDArray[np.float64]:
        return self.V_THz * 1e12

    @property
    def aw(self) -> np.ndarray:
        return self._aw.copy()
    
    @aw.setter
    def aw(self, arr: np.ndarray):
        self._aw = arr

    @property
    def at(self) -> np.ndarray:
        return IFFT_t(self.aw)

    @at.setter
    def at(self, arr: np.ndarray | list[float]):
        self.aw = FFT_t(np.array(arr))

    @property
    def w0(self) -> float:
        return 2.0 * np.pi * self._center_frequency_THz
    
    @w0.setter
    def w0(self, val: float):
        self._center_frequency_THz = val / (2.0 * np.pi)

    @property
    def W_THz(self) -> npt.NDArray[np.float64]:
        return self.V_THz + self.w0
    
    @property
    def W_Hz(self) -> np.ndarray:
        return self.W_THz * 1e12

    @property
    def wavelength_nm(self) -> npt.NDArray[np.float64]:
        return 2.0 * np.pi * self._c_nmps / self.W_THz
    
    @property
    def wavelength_m(self) -> npt.NDArray[np.float64]:
        return self.wavelength_nm * 1e-9
    
    @property
    def center_wavelength_nm(self) -> float:
        return self._c_nmps / self._center_frequency_THz
    
    @center_wavelength_nm.setter
    def center_wavelength_nm(self, val: float):
        self._center_frequency_THz = self._c_nmps / val

    @property
    def center_wavelength_mks(self) -> float:
        return self.center_wavelength_nm * 1e9
    
    @center_wavelength_mks.setter
    def center_wavelength_mks(self, val: float):
        self.center_wavelength_nm = val * 1e9

    @property
    def F_Hz(self) -> np.ndarray:
        return self.W_Hz / (2.0 * np.pi)

    @property
    def F_THz(self) -> np.ndarray:
        return self.F_Hz * 1e-12

    @property
    def dF_THz(self) -> float:
        return np.abs(self.W_THz[1] - self.W_THz[0]) / (2.0 * np.pi)
    
    @property
    def dF_Hz(self) -> float:
        return self.dF_THz * 1e12

    @property
    def frequency_window_THz(self) -> float:
        return self._n / self._time_window_ps
    
    @frequency_window_THz.setter
    def frequency_window_THz(self, val: float):
        self.time_window_ps = self._n / val

    @property
    def frequency_window_mks(self) -> float:
        return self.frequency_window_THz * 1e12
    
    @frequency_window_mks.setter
    def frequency_window_mks(self, val: float):
        self.time_window_ps = (self._n / val) * 1e12

    def calc_epp(self) -> npt.NDArray[np.float64]:
        if self._at is None:
            self._at = self.at
        return self.dT_s * np.trapezoid(np.abs(self._at)**2)
    
    def set_epp(self, epp_J: float):
        self.at = self._at * np.sqrt(epp_J / self.calc_epp())

    def add_noise(self, noise_type: Noise):
        power_per_bin = np.abs(self._aw)**2
        photon_energy = self._h * self.F_Hz
        photons_per_bin = power_per_bin / photon_energy
        photons_per_bin[photons_per_bin < 0] = 0

        size: int = self._aw.shape[0]

        random_intensity: npt.NDArray[np.float64] = np.random.normal(size=size)
        random_phase: npt.NDArray[np.float64] = np.random.uniform(size=size) * 2.0 * np.pi

        if noise_type is Noise.SQRT_N_FREQ:
            noise = random_intensity * np.sqrt(photons_per_bin) * photon_energy * self.dF_Hz * np.exp(1j * random_phase)
        else:
            noise = random_intensity * photon_energy * self.dF_Hz * np.exp(1j * random_phase)

        self.aw += noise

    def chirp_pulse_W(self, gdd: float, *, tod: float = 0.0, fod: float = 0.0, w0_THz: float | None = None):
        if w0_THz is None:
            v = self._V_THz
        else:
            v = self.W_THz - w0_THz
            self.aw *= np.exp(1j * (gdd / 2.0) * v**2 +
                             1j * (tod / 6.0) * v**3 +
                             1j * (fod / 24.0) * v**4)
    
    def apply_phase_W(self, phase: np.ndarray) -> None:
        self.aw *= np.exp(1j * phase)

    def chirp_pulse_T(self, chirp2: float, chirp3: float, t0: float):
        self.at *= np.exp(-1j * (chirp2 / 2.0) * (self._T_ps / t0)**2 +
                     -1j * (chirp3 / 3.0) * (self._T_ps/t0)**3)
        
    def dechirp_pulse(self, *, intensity_threshold: float = 0.05) -> None:
        phase = np.unwrap(np.angle(self.aw))
        ampl = np.abs(self.aw)
        mask = ampl**2 > intensity_threshold * np.max(ampl)**2
        gdd = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 2))
        self.aw = ampl * np.exp(1j * (phase - gdd(self.W_THz)))

    def remove_time_delay(self, *, intensity_threshold: float = 0.05) -> None:
        phase = np.unwrap(np.angle(self.aw))
        ampl = np.abs(self.aw)
        mask = ampl**2 > intensity_threshold * np.max(ampl)**2
        ld = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 1))
        self.aw = ampl * np.exp(1j * (phase-ld(self.W_THz)))

    def add_time_offset(self, offset_ps: float):
        phase_ramp = np.exp(-1j * self.W_THz * offset_ps)
        self.aw *= phase_ramp

    def expand_time_window(self, factor_log2: int, *, new_pts_loc: str = "before"):
        new_pts_loc = new_pts_loc.lower().strip()
        if new_pts_loc not in ["before", "after", "even"]:
            raise ValueError("new_pts_loc param  must be either 'before', 'after', or 'even'")
        num_new_pts = self._n * (2**factor_log2 - 1)
        self.n *= 2**factor_log2
        self.time_window_ps *= 2**factor_log2

        if self._at is not None:
            at = self._at
        else:
            at = self.at
        
        if new_pts_loc == "before":
            self.at = np.hstack((np.zeros(num_new_pts, ), at))
        elif new_pts_loc == "after":
            self.at = np.hstack((at, np.zeros(num_new_pts)))
        else:
            pts_before = int(np.floor(num_new_pts * 0.5))
            pts_after = num_new_pts - pts_before

            self.at = np.hstack((np.zeros(pts_before, ), at, np.zeros(pts_after, )))

    def rotate_spectrum(self, center_wl_nm: float) -> None:
        center_THz = self._c_nmps / center_wl_nm
        rotation = (self._center_frequency_THz - center_THz) / self.dF_THz
        self.aw = np.roll(self.aw, -1 * round(rotation))

    def interpolate(self, wavelength_nm: float) -> Pulse:
        pulse = self.create_cloned_pulse()
        pulse.center_wavelength_nm = wavelength_nm
        interpolator = interp1d(self.W_Hz, self._aw, bounds_error=False, fill_value=0.0)
        pulse.aw = interpolator(pulse.W_Hz)
        return pulse
    
    def clone_pulse(self, pulse: Pulse):
        self.n = pulse.n
        self.time_window_ps = pulse.time_window_ps
        self.center_wavelength_nm = pulse.center_wavelength_nm
        self.frep_MHz = pulse.frep_MHz
        self.at = pulse.at

    def create_cloned_pulse(self) -> Pulse:
        pulse = Pulse()
        pulse.clone_pulse(self)
        return pulse

    def filter_by_wavelength_nm(self, lower_wavelength_nm: float, upper_wavelength_nm: float):
        aw = self._aw
        aw[self.wavelength_nm < lower_wavelength_nm] = 0.0
        aw[self.wavelength_nm > upper_wavelength_nm] = 0.0
        self.aw = aw

    def create_subset_pulse(self, center_wavelength_nm: float, n: int):
        if n >= self._n:
            raise ValueError("New pulse must have fewer points than existing one.")
        pulse = Pulse()
        center_idx = np.argmin(np.abs(self.wavelength_nm - center_wavelength_nm))
        
        pulse.frep_MHz = self._frep_MHz
        pulse.center_wavelength_nm = self.wavelength_nm[center_idx]
        pulse.time_window_ps = self._time_window_ps
        pulse.n = n

        idx1 = center_idx - (n >> 1)
        idx2 = center_idx = (n >> 1)
        
        pulse.aw = self._aw[idx1:idx2]

        return pulse
    
    def weighted_avg_frequency_Hz(self):
        avg = np.sum(np.abs(self._aw)**2 * self.W_Hz)
        weights = np.sum(np.abs(self._aw)**2)
        return avg / (weights * 2.0 * np.pi)
    
    def weighted_avg_wavelength_nm(self):
        return 1e9 * self._c_mks / self.weighted_avg_frequency_Hz()
    
    def intensity_autocorrelation(self):
        if self._at is None:
            self._at = self.at
        return np.correlate(np.abs(self._at)**2, np.abs(self._at), mode="same")
        
    def spectrogram(self, *, gate_type: Gate = Gate.XFROG, gate_function_width_ps: float = 0.02, time_steps: int = 500):
        def gauss(x: np.ndarray, *, mu: np.ndarray, A: int = 1,  sigma: float = 1.0) -> npt.NDArray[np.float64]:
            return A * np.exp(-(x - mu)**2 / (2.0 * sigma**2))
        
        delay = np.linspace(np.min(self._T_ps), np.max(self._T_ps), time_steps)

        if self._at is None:
            self._at = self.at

        d, t = np.meshgrid(delay, self._T_ps)
        d, at = np.meshgrid(delay, self._at)
        
        phase = np.unwrap(np.angle(at))
        amp = np.abs(at)

        if gate_type is Gate.XFROG:
            gate_function = gauss(t, mu=d, sigma=gate_function_width_ps)
        else:
            raise NotImplementedError("")
            # tstep = float(self._T_ps[1] - self._T_ps[0])
            # dcoord = d * 0
            # tcoord = (t - d - np.min(t)) / tstep

        e = amp * gate_function * np.exp(1j * (2.0 * np.pi * t * self._center_frequency_THz + phase))

        spectrogram = np.array(fft(e, axis=0))
        freqs: npt.NDArray[np.float64] = np.asarray(fftfreq(e.shape[0], t[1] - t[0]), dtype=float)

        delays, freqs = np.meshgrid(delay, freqs)

        h = spectrogram.shape[0]
        spectrogram = spectrogram[:h//2]
        delays = delays[:h//2]
        freqs = freqs[:h//2]

        extent = (np.min(delays), np.max(delays), np.min(freqs), np.max(freqs))

        return delays, freqs, extent, np.abs(spectrogram)
