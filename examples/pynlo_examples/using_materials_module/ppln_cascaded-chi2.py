# -*- coding: utf-8 -*-
"""
This example demonstrates supercontinuum generation due to phase-mismatched
second harmonic generation (SHG) in periodically poled lithium niobate (PPLN).
The simulation uses the unidirectional propagation equation (UPE) and is based
on the parameters given for the second example in part III of Conforti et al.
(2010).

References
----------
Conforti M, Baronio F, De Angelis C. Nonlinear envelope equation for broadband
 optical pulses in quadratic media. Physical review. A, Atomic, molecular, and
 optical physics. 2010;81(5).
 https://doi.org/10.1103/PhysRevA.81.053841

"""

# %% Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt
import pynlo
from pynlo import utility as ut
from pynlo.utility import clipboard


# %% Pulse Properties
"""
We start by initializing a Gaussian pulse using one of the built-in pulse
shapes of the `Pulse` class. The first few parameters constrain the frequency
grid, the number of points and the frequency range, while the last three set
the initial pulse properties, its center frequency, pulse energy, and pulse
width. Since we are going to be simulating 2nd-order effects, the `alias`
parameter has been increased to support two alias-free Nyquist zones.

"""
n_points = 2**13
v_min = c / 3500e-9  # c / 3500 nm
v_max = c / 450e-9  # c / 450 nm

v0 = c / 1580e-9  # c / 1580 nm
e_p = 1e-9  # 1 nJ
t_fwhm = 50e-15  # 50 fs

pulse = pynlo.light.Pulse.Gaussian(n_points, v_min, v_max, v0, e_p, t_fwhm, 10e-12)


# %% Mode Properties
"""
In this example we use the refractive index, effective area, and nonlinear
susceptibility to calculate the linear and nonlinear properties of the mode.
The material used in this case is congruent lithium niobate, and we use the
Sellmeier equations from Jundt (1997) for the refractive index. The generalized
2nd-order nonlinear parameter is weighted for second-harmonic generation using
the `g2_shg` function from the `utility.chi2` submodule. Poling is implemented
by indicating the location of each domain inversion. This is calculated using
the `domain_inversion` function of the `utility.chi2` submodule. This example
only uses a constant poling period, but the function also supports the
generation of arbitrarily chirped poling periods.

"""
length = 7e-3  # 7 mm
a_eff = 15e-6 * 15e-6  # 15 um * 15 um


# poling
p0 = 30e-6  # 30 um poling period
z_invs, domains, poled = ut.chi2.domain_inversions(length, 2 * pi / p0)
LN = pynlo.materials.cLN(T=24.5, axis="e")

# %% Model
"""
The UPE model is initialized with the pulse and mode objects defined above. At
this stage we also use the target local error to estimate the optimal initial
step size.

"""

model = LN.generate_model(
    pulse,
    a_eff,
    length,
    g2_inv=z_invs,
    beta=None,
    is_gaussian_beam=False,
)
model.g3 = 0 
model.alpha = 0

# ---- Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(local_error=local_error)

# %% Simulate
"""
The model's `simulate` method runs the simulation. We input the total
propagation length, the initial step size, local error, and the number of
simulation steps we wish to record. We receive the output pulse and the
propagations distance, pulse spectrum, and complex envelope at each record
point. To view real-time simulation results (updated whenever the simulation
reaches a record point), set the `plot` keyword to "frq", "wvl", or "time".

"""
sim = model.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)

# %% Plot Results
"""
The results are plotted in the time and frequency domains. Although the
3rd-order nonlinearity is not directly included in the propagation model we see
several 3rd-order phenomena due to cascaded 2nd-order effects. Phase-mismatched
SHG yields an effective Kerr effect (see DeSalvo et al. (1992)), which leads to
soliton and dispersive wave generation. Additionally, the high frequency spike
is third-harmonic generation due to cascaded second-harmonic and sum-frequency
generation from the pump frequency. The other spike is due to SHG quasi-phase
matched off of the 3rd-order spatial harmonic of the poling structure.

References
----------
R. DeSalvo, D. J. Hagan, M. Sheik-Bahae, G. Stegeman, E. W. Van Stryland, and
 H. Vanherzeele, "Self-focusing and self-defocusing by cascaded second-order
 effects in KTP," Opt. Lett. 17, 28-30 (1992)

"""
sim.plot("frq")
