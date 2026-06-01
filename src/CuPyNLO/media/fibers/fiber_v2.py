from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import interpolate, constants # type: ignore
from scipy.special import factorial # type: ignore
from scipy.optimize import minimize, OptimizeResult # type: ignore
from typing import Any, Callable, cast

from CuPyNLO.light.PulseBase_v2 import Pulse
from CuPyNLO.media.fibers.calculators_v2 import DTabulationToBetas
from CuPyNLO.util.fft import IFFT_t
from CuPyNLO.util.util import to_numpy
from CuPyNLO.media.fibers.fiber_loader import Collection, Fibers, FiberLoader

class FiberInstance:
    def __init__(self, *,
                fiber_db: Collection = Collection.GENERAL_FIBERS,
                fiber_db_dir: str | None = None,
                is_simple_fiber: bool = False,
                dispersion_changes_with_z: bool = False,
                gamma_changes_with_z: bool = False,
                betas: npt.NDArray[np.float64] | None = None,
                length: float | None = None,
                fiber_type: str | None = None,
                fiber_specs: dict[str, Any] | None = None,
                poly_order: int | None = None,
                gamma: float | None = None):

        self._c_mks = constants.speed_of_light
        self._c = constants.speed_of_light * 1e9 / 1e12
        self.is_simple_fiber = is_simple_fiber

        self.dispersion_changes_with_z = dispersion_changes_with_z
        self.gamma_changes_with_z = gamma_changes_with_z

        self.dispersion_function = None
        self.gamma_function = None

        self._betas = betas
        self._length = length
        self._fiber_type = fiber_type
        self._fiber_specs: dict[str, str | bool | float | None] = fiber_specs if fiber_specs is not None else {}
        self._poly_order = poly_order
        self._gamma = gamma
        self._gain: float = 0.0
        self._gain_x_units = None

        self._center_wavelength_nm: float | None = None

        self.x = None
        self.y = None

        self.fiberloader = FiberLoader()

    def set_dispersion_function(self, dispersion_function: Callable[[float], float] | Callable[[float], npt.NDArray[np.float64]], dispersion_format: str = "GVD"):
        self.dispersion_changes_with_z = True
        self._fiber_specs["dispersion_format"] = dispersion_format
        self.dispersion_function = dispersion_function

    def set_gamma_function(self, gamma_function: Callable[[float], float]) -> None:
        self.gamma_function = gamma_function
        self.gamma_changes_with_z = True

    def load_from_db(self, length: float, fibertype: str, poly_order: int = 2):
        self._fiber_type = fibertype
        self._fiber_specs = self.fiberloader.get_fiber(fibertype)
        self._length = length
        self._betas = np.array([0])
        self._gamma = self._fiber_specs["gamma"]
        self._poly_order = poly_order
        self.load_dispersion()

    def load_dispersion(self):
        """This is typically called by the "load_from_db" function. 
        It takes the values from the self.fiberspecs dict and transfers them into the appropriate variables. """
        
        if self._fiber_specs["dispersion_format"] == "D":
            self.dispersion_x_units = self._fiber_specs["dispersion_x_units"]
            self.dispersion_y_units = self._fiber_specs["dispersion_y_units"]
            self.x = self._fiber_specs["dispersion_x_data"]
            self.y = self._fiber_specs["dispersion_y_data"]
            return 1
            
        elif self._fiber_specs["dispersion_format"] == "GVD":
            self.dispersion_gvd_units = self._fiber_specs["dispersion_gvd_units"]
            self._center_wavelength_nm = self._fiber_specs["dispersion_gvd_center_wavelength"]
            # If in km^-1 units, scale to m^-1
            if self.dispersion_gvd_units == 'ps^n/km':
                self._betas = np.array(self._fiber_specs["dispersion_data"]) / 1e3
            return 1
        else:
            print( "Error: no dispersion found.")
            return None   

    def gamma(self, z: float = 0.0) -> float:
        if self.gamma_changes_with_z:
            if self.gamma_function is None:
                raise ValueError("Gamma function not set.")
            return self.gamma_function(z)
        if self._gamma is None:
            raise ValueError("Gamma is not set.")
        return self._gamma

    def set_gamma(self, val: float):
        self._gamma = val

    @property
    def length(self) -> float:
        if self._length is None:
            raise ValueError("Length not set.")
        return self._length
    
    @length.setter
    def length(self, val: float):
        self._length = val

    def get_betas(self, pulse: Pulse, z: float = 0.0) -> np.ndarray:
        b: npt.NDArray[np.float64] = np.zeros((pulse.n, ), dtype=float)
        pulse_w = pulse.W_THz
        pulse_v = pulse.V_THz
        if self.dispersion_changes_with_z:
            if self.dispersion_function is None:
                raise ValueError("Dispersion function not set.")
            if self._fiber_specs["dispersion_format"] == "D" or self._fiber_specs["dispersion_format"] == "n":
                output = self.dispersion_function(z)
                if type(output) is np.ndarray:
                    self.x, self.y = output
                    if self._fiber_specs["dispersion_format"] == "D":
                        if self._poly_order is None:
                            raise ValueError("Poly order not set.")
                        self._betas = DTabulationToBetas(pulse.center_wavelength_nm,
                                                        np.transpose(np.vstack((self.x, self.y))),
                                                        self._poly_order,
                                                        data_is_file=False)
                        b = np.array([self._betas[i] / factorial(i+2) * pulse_v**(i+2) for i in range(len(self._betas))], dtype=float)
                        return b
                    if self._fiber_specs["dispersion_format"] == "n":
                        supplied_W_THz = 2.0 * np.pi * 1e-12 * 3e8 / (self.x * 1e-9)
                        supplied_betas = self.y * 2.0 * np.pi / (self.x * 1e-9)
                        interpolator = interpolate.InterpolatedUnivariateSpline(supplied_W_THz[::-1], supplied_betas[::-1])
                        b = np.array(interpolator(pulse_w), dtype=float)
            else:
                self._betas = np.array(self.dispersion_function(z))

        if self._fiber_specs["dispersion_format"] == "GVD":
            if self._center_wavelength_nm is None:
                raise ValueError("Center wavelength not set.")
            if self._betas is None:
                raise ValueError("Betas not set.")
            fiber_omega0 = 2.0 * np.pi * self._c / self._center_wavelength_nm
            betas = self._betas
            for i in range(len(betas)):
                betas[i] = betas[i]
                b += betas[i] / factorial(i + 2) * (pulse_w - fiber_omega0)**(i + 2)

        if self._fiber_specs["dispersion_format"] == "GVD" or self._fiber_specs["dispersion_format"] == "n":
            center_idx = np.argmin(np.abs(pulse_v))
            slope = np.gradient(b) / np.gradient(pulse_w)
            b = b - slope[center_idx] * pulse_v - b[center_idx]

        return b

    def get_gain(self, pulse: Pulse, *, output_power: float = 1.0) -> np.ndarray:
        if self._fiber_specs["is_gain"]:
            if self.is_simple_fiber:
                raise TypeError("Fiber is not gain fiber.")
            else:
                if self._fiber_specs["gain_x_data"] is not None:
                    self._gain_x_units = self._fiber_specs["gain_x_units"]

                    x: npt.NDArray[np.float64] = np.asarray(
                        self._fiber_specs["gain_x_data"],
                        dtype=float,
                    )

                    y: npt.NDArray[np.float64] = np.asarray(
                        self._fiber_specs["gain_y_data"],
                        dtype=float,
                    )

                    f = interpolate.interp1d(
                        self._c_mks / x[::-1],
                        y[::-1],
                        kind="cubic",
                        bounds_error=False,
                        fill_value=0,
                    )

                    gain_spec: npt.NDArray[np.float64] = np.asarray(
                        f((pulse.W_Hz) / (2.0 * np.pi)),
                        dtype=float,
                    )

                    def g(k: npt.NDArray[np.float64]) -> float:
                        if self._length is None:
                            raise ValueError("Length not set.")

                        val = np.abs(
                            output_power
                            - pulse.frep_Hz
                            * pulse.dT_s
                            * np.trapezoid(
                                np.abs(
                                    IFFT_t(
                                        pulse.aw
                                        * np.exp(k[0] * gain_spec * self._length / 2.0)
                                    )
                                ) ** 2
                            )
                        )

                        return float(val)

                    x0: npt.NDArray[np.float64] = np.array([1.0], dtype=float)

                    scale_factor = cast(
                        OptimizeResult,
                        minimize(g, x0=x0, method="Powell"),
                    )

                    return gain_spec * float(scale_factor.x[0])
                else:
                    return np.ones((pulse.n, )) * self._gain
        return np.zeros((pulse.n, )) * self._gain

    def beta2_to_d(self, pulse: Pulse):
        wavelength_nm = pulse.wavelength_nm
        return -2.0 * np.pi * self._c / wavelength_nm**2 * self.beta2(pulse) * 1000
    
    def beta2(self, pulse: Pulse):
        V_THz = pulse.V_THz
        dw = V_THz[1] - V_THz[0]
        out = np.diff(self.get_betas(pulse, 2), 2) / dw**2
        out = np.append(out[0], out)
        out = np.append(out, out[-1])
        return out
    
    def generate_fiber(self, length: float, center_wavelength_nm: float, betas: np.ndarray, gamma_W_m: float, *,
                       gain: float = 0.0, gvd_units: str = "ps^n/km", label: Fibers = Fibers.SIMPLE_FIBER):
        self.length = length
        self._fiber_specs = {"dispersion_format": "GVD", "is_gain": bool(gain),
                             "gain_x_data": None}
        self._fiber_type = label
        self._gain = gain

        self._center_wavelength_nm = center_wavelength_nm
        self._betas = np.array(betas, dtype=float, copy=True)
        self.set_gamma(gamma_W_m) 

        if gvd_units == "ps^n/km":
            self._betas *= 1e-3
