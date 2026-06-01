from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import constants # type: ignore
from typing import Any

from CuPyNLO.media.fibers.fiber_loader import Collection, Fibers, FiberLoader

class Fiber:
    def __init__(self,
             length: float, 
             center_wavelength_nm: float, *,
             fiber_db: Collection = Collection.GENERAL_FIBERS,
             fiber_db_dir: str | None = None,
             is_simple_fiber: bool = False,
             dispersion_changes_with_z: bool = False,
             gamma_changes_with_z: bool = False,
             betas: npt.NDArray[np.float64] | None = None,
             fiber_type: str | None = None,
             fiber_specs: dict[str, Any] | None = None,
             poly_order: int | None = None,
             gamma: float | None = None,
             gain: float = 0.0,
             gvd_units: str = "ps^n/km",
             label: Fibers = Fibers.SIMPLE_FIBER
             ):

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
        self._fiber_specs = fiber_specs if fiber_specs is not None else {"dispersion_format": "GVD",
                                                                         "is_gain": bool(gain),
                                                                         "gain_x_data": None}
        self._poly_order = poly_order
        self._gamma = gamma
        self._gain = gain
        self._gain_x_units = None
        self._fiber_type = label

        self._center_wavelength_nm = center_wavelength_nm
        self._betas = np.array(betas, dtype=float, copy=True)

        if gvd_units == "ps^n/km":
                self._betas *= 1e-3

        self.x = None
        self.y = None

        self.fiberloader = FiberLoader()
