from __future__ import annotations

from CuPyNLO.interactions.FourWaveMixing import global_variables as gv
from CuPyNLO.light.PulseBase_v2 import Pulse, Noise
from CuPyNLO.media.fibers.fiber_v2 import FiberInstance

from enum import IntEnum

import numpy as np
from scipy.fft import fftshift, ifftshift, ifft, fft

from CuPyNLO.util.checker import checker

# NOTE: I am not using pyfftw, CuPy is faster for accelerated, scipy is easier to use

class Methods(IntEnum):
    SSFM = 1
    RK4IP = 2

class SSFM:
    def __init__(self, *, local_error: float = 0.001, dz: float = 1e-5,
                 disable_raman: bool = False, disable_self_steepening: bool = False,
                 suppress_iteration: bool = True, use_simple_raman: bool = False,
                 f_r: float = 0.18, f_r0: float = 0.18, tau_1: float = 0.0122,
                 tau_2: float = 0.0320, eta: float = 5.0, dz_min: float = 1e-12):
        
        self._local_error = local_error
        self._disable_raman = disable_raman
        self._disable_self_steepening = disable_self_steepening
        self._use_simple_raman = use_simple_raman

        self.f_r = f_r
        self.f_r0 = f_r0
        
        self.tau1 = tau_1
        self.tau2 = tau_2
        self.dz = dz
        self.dz_min = dz_min
        self.suppress_iteration = suppress_iteration
        self.eta = eta

        self._gamma = None
        self._n = 1024

        self.fft_inputs: np.ndarray
        self.fft_outputs: np.ndarray

        self.ifft_inputs: np.ndarray
        self.ifft_outputs: np.ndarray

        self.vars: np.ndarray

        self.h = None
        self.direction = None

        self.iter = 0

    def setup_fftw(self, pulse: Pulse, fiber: FiberInstance, output_power: float, *, raman_plots: bool = False):
        self._n = pulse.n

        self.fft_inputs = np.ndarray((self._n, 3), dtype=complex)
        self.fft_outputs = np.ndarray((self._n, 3), dtype=complex)
        
        self.ifft_inputs = np.ndarray((self._n, 3), dtype=complex)
        self.ifft_outputs = np.ndarray((self._n, 3), dtype=complex)

        self.a_i    = np.ndarray((self._n,), dtype=complex)

        self.a2     = np.ndarray((self._n,), dtype=complex)
        self.exp_D  = np.ndarray((self._n,), dtype=complex)
        self.k1     = np.ndarray((self._n,), dtype=complex)
        self.k2     = np.ndarray((self._n,), dtype=complex)
        self.k3     = np.ndarray((self._n,), dtype=complex)
        self.k4     = np.ndarray((self._n,), dtype=complex)
        self.temp   = np.ndarray((self._n,), dtype=complex)
        self.Aw     = np.ndarray((self._n,), dtype=complex)
        self.A2w    = np.ndarray((self._n,), dtype=complex)
        self.dA     = np.ndarray((self._n,), dtype=complex)
        self.dA2    = np.ndarray((self._n,), dtype=complex)
        self.r_a2   = np.ndarray((self._n,), dtype=complex)
        self.dR_A2  = np.ndarray((self._n,), dtype=complex)
        self.omegas = np.ndarray((self._n,), dtype=complex)
        self.alpha  = np.ndarray((self._n,), dtype=complex)
        self.betas  = np.ndarray((self._n,), dtype=complex)
        self._linear_step = np.ndarray((self._n,), dtype=complex)
        self.a      = np.ndarray((self._n,), dtype=complex)
        self.r      = np.ndarray((self._n,), dtype=complex)
        self.r0     = np.ndarray((self._n,), dtype=complex)
        self.af     = np.ndarray((self._n,), dtype=complex)      
        self.ac     = np.ndarray((self._n,), dtype=complex)

        self.a_i[:] = 0.0        
        self.a2[:]  = 0.0
        self.af[:]  = 0.0
        self.ac[:]  = 0.0
        self.a[:]   = 0.0
        self.r[:]   = 0.0
        self.r0[:]  = 0.0

        self.omegas[:] = pulse.V_THz
        self.alpha[:] = -fiber.get_gain(pulse, output_power=output_power)
        self._gamma = fiber.gamma
        self._w0 = pulse.center_frequency_THz * 2.0 * np.pi

        self.CalculateRamanResponseFT(pulse)

        self.a[:] = fftshift(pulse.at)
        self.omegas[:] = fftshift(self.omegas)
        self.alpha[:] = fftshift(self.alpha)
        self.r[:] = fftshift(self.r)
        self.r0[:] = fftshift(self.r0)
        

    def CalculateRamanResponseFT(self, pulse: Pulse):
        tau1 = self.tau1
        tau2 = self.tau2
        c = (self.tau1**2 + tau2**2) / (tau1 * tau2**2) 
        self.r0 = np.array([(1.0 - self.f_r) + (self.f_r * (c * tau1 * tau2**2 / (tau1**2 + tau2**2 - 2j * self.omegas[i] * tau1**2 * tau2 - tau1**2 * tau2**2 * self.omegas[i]**2))) for i in range(pulse.n)])

        t = pulse.T_ps
        rt = np.zeros(pulse.n, dtype=complex)

        if self._use_simple_raman: 
            rt = (tau1**2 + tau2**2) / (tau1 * tau2**2) *  np.exp(-t / tau2) * np.sin(t / tau1)
            rt[0:pulse.n >> 1] = 0
            rt[:] = rt / np.trapezoid(rt, t)
            self.r[:] = (1.0 - self.f_r) + pulse.time_window_ps * self.FFT_t_shift(self.f_r * rt)

        else:
            taub = 0.096
            fa = 0.75
            fb = 0.21
            fc = 0.04
            self.f_r = 0.245
            
            ha = tau1 / (tau1**2 + tau2**2) * np.exp(-t / tau2) * np.sin(t / tau1)
            hb = (2 * taub - t) / taub**2 * np.exp(-t / taub)
            rt = (fa + fc) * ha + fb * hb
            rt[0:pulse.n >> 1] = 0
            rt[:] = rt / np.trapezoid(rt, t)
            self.r[:] = rt / np.trapezoid(rt, t)
            self.r[:] = ((1.0 - self.f_r) + pulse.time_window_ps * self.FFT_t_shift(self.f_r * rt))
        if gv.USE_FREQUENCY_DOMAIN_RAMAN:
            self.r[:] = self.r0

    def integrate_over_dz(self, delta_z: float, direction: int = 1):
        dz = self.dz
        factor = 2**(1.0 / self.eta)
        dist = delta_z

        return_dz = None
        force = False

        if 2.0 * dz > dist:
            dz = dist / 2.0
        
        while dist > 0:
            self.ac[:] = self.a
            self.af[:] = self.a

            self.ac[:] = self.advance(self.ac, 2.0  * dz, direction)
            self.af[:] = self.advance(self.af, dz, direction)
            self.af[:] = self.advance(self.af, dz, direction)

            delta = self.calculate_local_error()

            old_dz = dz
            new_dz = dz

            if delta > 2.0 * self._local_error:
                new_dz = dz / 2.0
                if new_dz >= self.dz_min:
                    dz = new_dz
                    continue
            elif delta >= self._local_error and delta <= 2.0 * self._local_error:
                new_dz = dz / factor
                if new_dz >= self.dz_min:
                    dz = new_dz
            elif delta >= 0.5 * self._local_error and delta <= self._local_error:
                new_dz = new_dz
            else:
                new_dz = dz * factor
                dz = new_dz
            if self.eta == 3:
                self.a[:] = (4 / 3) * self.af - (1.0 / 3.0) * self.ac
            elif self.eta == 5:
                self.a[:] = (16 / 15) * self.af - (1 / 15) * self.ac
            else:
                p = 2 ** (self.eta - 1)
                self.a[:] = (p / (p - 1)) * self.af - (1 / (p - 1)) * self.ac

            dist -= 2 * old_dz
            self.iter+=1

            if 2 * dz > dist and dist > 2 * self.dz_min:
                force = True
                return_dz = dz
                dz = dist / 2

        if force:
            if return_dz is None:
                raise ValueError("return_dz not set.")
            dz = return_dz
        self.dz = dz


    def advance(self, a: np.ndarray, dz: float, direction: int, method: Methods = Methods.RK4IP):
        if method is Methods.SSFM:
            if direction == 1:
                a[:] = self.linear_step(a, dz, direction)
                return np.exp(dz * direction * self.nonlinear_operator(a)) * a
            else:
                a[:] = np.exp(dz * direction * self.nonlinear_operator(a)) * a
                return self.linear_step(a, dz, direction)
        else:
            return self.rk4ip(a, dz, direction)

    def linear_step(self, a: np.ndarray, h: float, direction: int):
        self.calculate_expD(h=h, direction=direction)
        self._linear_step[:] = self.IFFT_t(self.exp_D * self.FFT_t(a))
        return self._linear_step

    @checker
    def calculate_expD(self, *, h: float, direction: int):
        self.exp_D[:] = np.exp(direction * h * 0.5 * (1j * self.betas - self.alpha / 2.0))

    def deriv(self, aw: np.ndarray) -> np.ndarray:
        return self.IFFT_t(-1j * self.omegas * aw)
    
    def nonlinear_operator(self, a: np.ndarray) -> np.ndarray:
        self.a2[:] = np.abs(a)**2
        self.A2w[:] = self.FFT_t(self.a2)

        if self._gamma is None:
            raise ValueError("Gamma is not defined.")
        
        if self._disable_self_steepening:
            return 1j * self._gamma * self.IFFT_t(self.r * self.A2w)
        self.Aw[:] = self.FFT_t(a)
        self.r_a2[:] = self.IFFT_t(self.r * self.A2w)
        self.dA[:] = self.deriv(self.Aw)
        self.dA2[:] = self.deriv(self.A2w)
        self.dR_A2[:] = self.IFFT_t(self.r * self.FFT_t(self.dA2))

        return 1j * self._gamma * self.r_a2 - (self._gamma / self._w0) *\
                (self.dR_A2 + np.where(np.abs(a) > 1e-15, self.dA * self.r_a2 / (1e-20 + a), 0))

    def rk4ip(self, a: np.ndarray, h: float, direction: int) -> np.ndarray:
        self.a_i[:] = self.linear_step(a, h, direction)
        self.k1[:] = self.IFFT_t(self.exp_D * self.FFT_t(h * direction * self.nonlinear_operator(a) * a))
        self.k2[:] = h * direction * self.nonlinear_operator(self.a_i + self.k1 / 2.0) *\
                        (self.a_i + self.k1 / 2.0)
        self.k3[:] = h * direction * self.nonlinear_operator(self.a_i + self.k2 / 2.0) *\
                        (self.a_i + self.k2 / 2.0)
        self.temp[:] = self.IFFT_t(self.exp_D * self.FFT_t(self.a_i + self.k3))
        self.k4[:] = h * direction * self.nonlinear_operator(self.temp) * self.temp
        
        return self.IFFT_t(self.exp_D * self.FFT_t(self.a_i + self.k1 / 6.0 + self.k2 / 3.0 + self.k3 / 3.0)) + self.k4 / 6.0
    
    def calculate_local_error(self):
        denom = np.linalg.norm(self.af)
        if denom != 0:
            return np.linalg.norm(self.af - self.ac) / np.linalg.norm(self.af)
        else:
            return np.linalg.norm(self.af - self.ac)
        
    def load_fiber_parameters(self, pulse: Pulse, fiber: FiberInstance, z: float = 0.0):
        self.betas[:] = fiber.get_betas(pulse, z)
        self._gamma = fiber.gamma(z)
        self.betas[:] = fftshift(self.betas)

    def propagate(self, pulse: Pulse, fiber: FiberInstance, n_steps: int, *, output_power: float = 1.0, reload: bool = False, thread: bool = False):
        z_pos = np.linspace(0, fiber.length, n_steps + 1)
        
        if n_steps == 1:
            delta_z = fiber.length
        else:
            delta_z = z_pos[1] - z_pos[0]

        aw = np.zeros((pulse.n, n_steps), dtype=complex)
        at = np.zeros((pulse.n, n_steps), dtype=complex)

        pulse_out = Pulse()
        pulse_out.clone_pulse(pulse)
        self.setup_fftw(pulse, fiber, output_power)
        self.load_fiber_parameters(pulse, fiber, output_power)

        for i in range(n_steps):
            self.integrate_over_dz(delta_z)
            aw[:,i] = ifftshift(self.FFT_t(self.a))
            at[:,i] = ifftshift(self.a)
            pulse_out.at = ifftshift(self.a)
        pulse_out.at = ifftshift(self.a)
        
        return z_pos, aw, at, pulse_out
    
    def calculate_coherance(self, pulse: Pulse, fiber: FiberInstance, *,
                            num_trials: int = 5, noise_type: Noise = Noise.ONE_PHOTON_FREQ, n_steps: int = 50):
        
        results = []
        
        for _ in range(0, num_trials):
            _pulse = pulse.create_cloned_pulse()
            _pulse.add_noise(noise_type=noise_type)

            result = self.propagate(_pulse, fiber, n_steps)

            results.append(result)

        for n1, (y, e1, at, pulse_in, pulse_out) in enumerate(results):
            for n2, (y, e2, at, pulse_in, pulse_out) in enumerate(results):
                if n1 == n2:
                    continue
                g12 = np.conj(e1) * e2 / np.sqrt(np.abs(e1)**2 * np.abs(e2)**2)
                if "g12_stack" not in locals():
                    g12_stack = g12
                else:
                    g12_stack = np.dstack((g12, g12_stack))

    def FFT_t(self, A: np.ndarray) -> np.ndarray:
        if gv.PRE_FFTSHIFT:
            return ifft(A)
        else:
            return ifftshift(ifft(fftshift(A)))
    
    def FFT_t_shift(self, A: np.ndarray) -> np.ndarray:
        return ifftshift(ifft(fftshift(A)))
    
    def IFFT_t(self, A: np.ndarray) -> np.ndarray:
        if gv.PRE_FFTSHIFT:
            return fft(A)
        return ifftshift(fft(fftshift(A)))
    
    def IFFT_t_shift(self, A: np.ndarray) -> np.ndarray:
        return ifftshift(fft(fftshift(A)))
