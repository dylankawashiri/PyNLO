from __future__ import annotations

import numpy as np
from scipy import interpolate, constants
from scipy.special import factorial
from scipy.optimize import minimize
from typing import Any, Callable

from CuPyNLO.light.PulseBase_v2 import Pulse
from CuPyNLO.media.fibers.calculators_v2 import DTabulationToBetas
from CuPyNLO.util.pynlo_ffts import IFFT_t
from CuPyNLO.media.fibers.JSONFiberLoader import JSONFiberLoader, Collection


class FiberInstance:
    def __init__(self, *,
                fiber_db: Collection = Collection.GENERAL_FIBERS,
                fiber_db_dir: str | None = None,
                is_simple_fiber: bool = False,
                dispersion_changes_with_z: bool = False,
                gamma_changes_with_z: bool = False,
                betas: np.ndarray | None = None,
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

        self._betas = betas
        self._length = length
        self._fiber_type = fiber_type
        self._fiber_specs = fiber_specs if fiber_specs is not None else {}
        self._poly_order = poly_order
        self._gamma = gamma

    def set_dispersion_function(self, dispersion_function: Callable[[float], float], dispersion_format: str = "GVD"):
        self.dispersion_changes_with_z = True
        self._fiber_specs["dispersion_format"] = dispersion_format
        self.dispersion_function = dispersion_function

    
