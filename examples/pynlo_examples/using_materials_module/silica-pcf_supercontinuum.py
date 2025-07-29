# -*- coding: utf-8 -*-
"""
This example demonstrates supercontinuum generation due to soliton effects in
silica-based photonic crystal fiber. The simulation uses the nonlinear
Schrödinger equation (NLSE) and is based on the parameters given in part V-A of
Dudley et al. (2006).

References
----------
Dudley JM, Genty G, Coen S. Supercontinuum generation in photonic crystal
 fiber. Reviews of modern physics. 2006;78(4):1135-84.
 https://doi.org/10.1103/RevModPhys.78.1135

"""


# %% Imports
import numpy as np
from scipy.constants import c
from matplotlib import pyplot as plt
import pynlo
from pynlo.utility import clipboard


# %% Pulse Properties
"""
We start by initializing a hyperbolic secant pulse using one of the built-in
pulse shapes of the `Pulse` class. The first few parameters constrain the
frequency grid, the number of points and the frequency range, while the last
three set the initial pulse properties, its center frequency, pulse energy, and
pulse width.

"""
n_points = 2**12
v_min = c / 1500e-9  # c / 1400 nm
v_max = c / 500e-9  # c / 475 nm

v0 = c / 835e-9  # c / 835 nm
e_p = 285e-12  # 285 pJ
t_fwhm = 50e-15  # 50 fs
time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm, time_window)


# %% Mode Properties
"""
We need to define both the linear and nonlinear properties of the waveguide. In
this example, we are only given the waveguide properties at a single frequency
so we must extrapolate to the rest of the frequency grid. For the beta
parameter this is accomplished using a Taylor series, but for the nonlinearity
we use the `gamma_to_g3` conversion function from the `utility.chi3` submodule.
This function calculates the generalized 3rd-order nonlinear parameter
(required by the PyNLO propagation models) from the gamma parameter and the
optical shock time scale. If available, the nonlinear parameter can also be
generated from the refractive index, effective area, and nonlinear
susceptibility, see `utility.chi3` for more details. The Raman effect is
implemented using the Raman response function given in section 2.3.3 of
Agrawal.

References
----------
Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
 CA;: Academic Press; 2019.
 https://doi.org/10.1016/B978-0-12-817042-7.00009-9

"""
length = 15e-2  # 15 cm

# ---- Phase Coefficient, starting from the gdd term
beta_n = 9 * [0]
beta_n[0] = -11.830 * 1e-12**2 / 1e3  # -11.830 ps**2 / km
beta_n[1] = 8.1038e-2 * 1e-12**3 / 1e3  # 8.1038e-2 ps**3 / km
beta_n[2] = -9.5205e-5 * 1e-12**4 / 1e3  # -9.5205e-5 ps**4 / km
beta_n[3] = 2.0737e-7 * 1e-12**5 / 1e3  # 2.0737e-7 ps**5 / km
beta_n[4] = -5.3943e-10 * 1e-12**6 / 1e3  # -5.3943e-10 ps**6 / km
beta_n[5] = 1.3486e-12 * 1e-12**7 / 1e3  # 1.3486e-12 ps**7 / km
beta_n[6] = -2.5495e-15 * 1e-12**8 / 1e3  # -2.5495e-15 ps**8 / km
beta_n[7] = 3.0524e-18 * 1e-12**9 / 1e3  # 3.0524e-18 ps**9 / km
beta_n[8] = -1.7140e-21 * 1e-12**10 / 1e3  # -1.7140e-21 ps**10 / km

fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(pulse.v0, beta_n)

# ---- 3rd-Order Nonlinearity
gamma = 0.11  # 0.11 / W * m
t_shock = 0.56e-15  # 0.56 fs
fiber.gamma = gamma

# Raman effect
r_weights = [0.245 * (1 - 0.21), 12.2e-15, 32e-15]  # resonant contribution
b_weights = [0.245 * 0.21, 96e-15]  # boson contribution
fiber.r_weights = r_weights
fiber.b_weights = b_weights

# %% Model
"""
The NLSE model is initialized with the pulse and mode objects defined above. At
this stage we also estimate the optimal initial step size given a target local
error.

"""
model = fiber.generate_model(pulse, t_shock=t_shock, method="nlse")

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
For comparison with Dudley, we plot the evolution in the time and wavelength
domains. For accurate representation of the density, plotting over wavelength
requires converting the power from a per Hz basis to a per m basis. This is
accomplished by multiplying the frequency domain power spectral density by the
ratio of the frequency and wavelength differentials. The power spectral density
is then converted to decibel (dB) scale to increase the visible dynamic range.

"""
sim.plot("wvl")
