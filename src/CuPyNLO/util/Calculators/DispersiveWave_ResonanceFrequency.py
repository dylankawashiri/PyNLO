from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import math
from scipy import optimize # type: ignore

from CuPyNLO.media.fibers import fiber
from CuPyNLO.media.fibers.calculators import DTabulationToBetas
###############################################################################
""" 
Solve the dispersive wave resonance equation, (2) from "Dispersive wave 
blue-shift in supercontinuum generation", Dane R. Austin, C. Martijn de Sterke,
Benjamin J. Eggleton, Thomas G. Brown
"""
###############################################################################
## Fiber parameters
fiber1 = fiber.FiberInstance()
fiber1.fiberloader.print_fiber_list()
fibername = 'PMHNLF_2_2_FASTAXIS_LOWER_D'
fiber1.load_from_db( 1, fibername)

center_wavelength_nm = 1560.0
poly_order = 2

betas, omegaAxis, data, fit = DTabulationToBetas(center_wavelength_nm,
                           np.transpose(np.vstack((fiber1.x, fiber1.y))),
                            poly_order,
                            DDataIsFile = False,
                            return_diagnostics = True)
plt.figure(figsize = (12, 6)) # type: ignore
plt.title(fibername) # type: ignore
plt.subplot(121) # type: ignore
plt.plot(omegaAxis / (2.0*np.pi), data* 1.0e6, label = 'OFS Data' ) # type: ignore
plt.plot(omegaAxis / (2.0*np.pi), fit* 1.0e6, label = 'Fit' ) # type: ignore
plt.ylabel('GVD (fs^2 / m)') # type: ignore
plt.xlabel('Frequency from 1560 nm (THz)') # type: ignore
plt.legend(loc=2) # type: ignore
plt.subplot(122) # type: ignore
plt.plot(omegaAxis / (2.0*np.pi), (data - fit)* 1.0e6) # type: ignore
plt.ylabel('Fit Residuals (fs^2 / m)') # type: ignore
plt.xlabel('Frequency from 1560 nm (THz)') # type: ignore

###############################################################################
## Pulse parameters
# Calculate P0 from frep, Pavg, and pulse length

Pavg = 200.0e-3
fr   = 160.0e6
t0   = 100e-15

EPP  = Pavg / fr

P0 = 0.94 * EPP / t0 # Gaussian pulse

###############################################################################
## Solve
def fn(x: float):
    eqn = 0
    for n in range(len(betas)):
       print(betas[n] * np.power(x, n+2) / math.factorial(n+2))
       eqn += betas[n] * np.power(x, n+2) / math.factorial(n+2)
    eqn -= fiber1.gamma * P0 / 2.0
    return abs(eqn)

result = optimize.minimize(fn, -betas[0]/betas[1])